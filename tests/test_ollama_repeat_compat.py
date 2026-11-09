"""Regression for the actual first-request HTTP 400 reported by the user."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import ollama_teacher as ot
import qwen15_cds_ollama as job
import qwen15_cds_ollama_compat as compat
from teacher_length_retry import diagnosis_length_retry
from teacher_generation_retry import generation_length_retry
from test_ollama_teacher import config, metadata, reply
import test_qwen15_cds_ollama as local_tests

ERROR = "Field 'repeat_last_n': Value must be between 0 <= value <= 2147483647, but got -1"
REQUEST_KEY = '1a63b8f78eebcbfa9ff2ce3735701d7e464464800283136f76320ef32fdcacaa'


def first_request():
    prompt = cp.prompt_module().prompt4kps.format(
        X=5, kps='- Mixed Operations\n', DIFFICULTY='junior high school')
    return {'model': ot.MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.5, 'top_p': 0.8, 'max_tokens': 4096, 'stream': False}


def http_error(text=ERROR):
    return urllib.error.HTTPError(ot.DEFAULT_BASE + '/api/chat', 400, 'Bad Request', {},
                                 io.BytesIO(json.dumps({'error': text}).encode()))


class RepeatCompatibilityTests(unittest.TestCase):
    def test_exact_reported_request_seed_and_only_repeat_window_changes(self):
        cfg = config()
        payload = first_request()
        original = ot.OllamaTeacher(cfg, Path('/tmp/not-used')).native_request(payload)
        corrected = compat.ContextRepeatTeacher(cfg, Path('/tmp/not-used')).native_request(payload)
        self.assertEqual(cp.digest({'url': ot.DEFAULT_BASE, 'payload': payload}), REQUEST_KEY)
        self.assertEqual(corrected['options']['seed'], 2103518819)
        expected = copy.deepcopy(original)
        expected['options']['repeat_last_n'] = 32768
        self.assertEqual(corrected, expected)
        self.assertEqual(original['options']['repeat_last_n'], -1)
        self.assertTrue(0 <= corrected['options']['repeat_last_n'] <= 2147483647)

    def test_server_range_check_reproduces_error_and_fixed_cache_replays(self):
        cfg, calls = config(), []

        def fake(base, route, payload=None, timeout=30):
            if route != '/api/chat':
                return metadata(base, route, payload, timeout)
            calls.append(payload)
            if payload['options']['repeat_last_n'] < 0:
                raise http_error()
            return reply('complete teacher output')

        with tempfile.TemporaryDirectory() as tmp, patch.object(ot, 'rpc', side_effect=fake):
            run = Path(tmp)
            request = first_request()
            with self.assertRaisesRegex(RuntimeError, 'Ollama HTTP 400'):
                ot.OllamaTeacher(cfg, run).call(request['messages'])
            self.assertFalse((run / 'teacher_cache').exists())
            old_transport, old_rpc, old_teacher = ot.transport, ot.rpc, cp.Teacher
            with compat.compatibility(cfg, run), ot.transport():
                teacher = cp.Teacher(cfg, run)
                result = teacher.call(request['messages'])
                saved = (run / 'teacher_cache' / (REQUEST_KEY + '.json')).read_bytes()
                self.assertEqual(result, ('complete teacher output', REQUEST_KEY))
                self.assertEqual(teacher.call(request['messages']), result)
                self.assertEqual(saved, (run / 'teacher_cache' / (REQUEST_KEY + '.json')).read_bytes())
            self.assertEqual(len(calls), 2)  # One rejected legacy call, one successful corrected call.
            self.assertEqual(calls[1]['options']['repeat_last_n'], 32768)
            self.assertIs(ot.transport, old_transport)
            self.assertIs(ot.rpc, old_rpc)
            self.assertIs(cp.Teacher, old_teacher)
            policy = cp.read(run / 'teacher_repeat_scope_policy.json')
            self.assertEqual(policy['effective_repeat_last_n'], 32768)
            audit = cp.read(run / 'teacher_repeat_scope_requests' / (REQUEST_KEY + '.json'))
            self.assertEqual(audit['finish_reason'], 'stop')
            audit_path = run / 'teacher_repeat_scope_requests' / (REQUEST_KEY + '.json')
            audit_bytes = audit_path.read_bytes()
            cache_path = run / 'teacher_cache' / (REQUEST_KEY + '.json')
            corrupted = cp.read(cache_path)
            corrupted['content'] = 'tampered cached content'
            cache_path.write_text(json.dumps(corrupted))
            with compat.compatibility(cfg, run), ot.transport():
                with self.assertRaisesRegex(ValueError, 'cache/configuration mismatch'):
                    cp.Teacher(cfg, run).call(request['messages'])
            self.assertEqual(audit_path.read_bytes(), audit_bytes)

    def test_other_http_errors_retain_body_and_actual_options_without_retrying_400(self):
        cfg, calls = config(), []

        def fake(base, route, payload=None, timeout=30):
            if route != '/api/chat':
                return metadata(base, route, payload, timeout)
            calls.append(payload)
            raise http_error('offline simulated other request error')

        with tempfile.TemporaryDirectory() as tmp, patch.object(ot, 'rpc', side_effect=fake), \
             redirect_stdout(io.StringIO()) as out:
            run = Path(tmp)
            with compat.compatibility(cfg, run), ot.transport():
                with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
                    cp.Teacher(cfg, run).call(first_request()['messages'])
            self.assertEqual(len(calls), 1)
            self.assertIn('offline simulated other request error', out.getvalue())
            errors = list((run / 'teacher_http_errors').glob('*.json'))
            self.assertEqual(len(errors), 1)
            error = cp.read(errors[0])
            self.assertEqual(error['status'], 400)
            self.assertEqual(error['native_request']['options']['repeat_last_n'], 32768)
            self.assertFalse((run / 'teacher_cache').exists())

    def test_length_retries_keep_positive_scope_and_audit_each_cached_attempt(self):
        for limits in [[1024, 2048], [4096, 8192]]:
            cfg, calls = config(), []

            def fake(base, route, payload=None, timeout=30):
                if route != '/api/chat':
                    return metadata(base, route, payload, timeout)
                self.assertEqual(payload['options']['repeat_last_n'], 32768)
                calls.append(payload['options']['num_predict'])
                return reply('partial' if len(calls) == 1 else 'complete',
                             'length' if len(calls) == 1 else 'stop')

            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(ot, 'rpc', side_effect=fake), ExitStack() as stack:
                run = Path(tmp)
                stack.enter_context(compat.compatibility(cfg, run))
                stack.enter_context(ot.transport())
                stack.enter_context(diagnosis_length_retry(cfg, run))
                stack.enter_context(generation_length_retry(cfg, run))
                cp.Teacher(cfg, run).call([{'role': 'user', 'content': 'original prompt'}],
                                          max_tokens=limits[0])
                self.assertEqual(calls, limits)
                audits = [cp.read(p) for p in (run / 'teacher_repeat_scope_requests').glob('*.json')]
                self.assertEqual(sorted(r['finish_reason'] for r in audits), ['length', 'stop'])

    def test_legacy_native_cache_is_not_silently_rewritten(self):
        cfg = config()

        def fake(base, route, payload=None, timeout=30):
            return reply() if route == '/api/chat' else metadata(base, route, payload, timeout)

        with tempfile.TemporaryDirectory() as tmp, patch.object(ot, 'rpc', side_effect=fake):
            run = Path(tmp)
            messages = first_request()['messages']
            ot.OllamaTeacher(cfg, run).call(messages)
            path = run / 'teacher_cache' / (REQUEST_KEY + '.json')
            before = path.read_bytes()
            with compat.compatibility(cfg, run), ot.transport():
                with self.assertRaisesRegex(ValueError, 'cache/configuration mismatch'):
                    cp.Teacher(cfg, run).call(messages)
            self.assertEqual(path.read_bytes(), before)

    def test_frozen_local_run_loads_and_all_original_files_stay_unchanged(self):
        fixture = local_tests.LocalCdsTests('test_route_preserves_other_experiments')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = fixture.fixture(root)
            with fixture.context(root, source):
                cfg, run = job.prepare()
                original_files = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}

                def fake(base, route, payload=None, timeout=30):
                    return reply() if route == '/api/chat' else metadata(base, route, payload, timeout)

                def synthesize(current_cfg, current_run, args):
                    self.assertIs(cp.Teacher, compat.ContextRepeatTeacher)
                    cp.Teacher(current_cfg, current_run).call(first_request()['messages'])

                with patch.object(ot, 'rpc', side_effect=fake), \
                     patch.object(cp, 'synthesize', side_effect=synthesize), \
                     compat.compatibility(cfg, run):
                    job.teacher_stage('synthesize')
                job.load()
                for path, data in original_files.items():
                    self.assertEqual(path.read_bytes(), data)
                policy_bytes = (run / 'teacher_repeat_scope_policy.json').read_bytes()
                with compat.compatibility(cfg, run):
                    pass
                self.assertEqual((run / 'teacher_repeat_scope_policy.json').read_bytes(), policy_bytes)


if __name__ == '__main__':
    unittest.main()
