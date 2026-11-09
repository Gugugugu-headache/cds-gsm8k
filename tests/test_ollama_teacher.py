"""Mock native Ollama endpoints; never contact or load a real model."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import ollama_teacher as ot
from teacher_length_retry import diagnosis_length_retry
from teacher_generation_retry import generation_length_retry
from teacher_sample_descriptive_footer import sample_format_compatibility
from test_teacher_sample_format import batch

DIGEST = ot.DIGEST_PREFIX + '0' * (64 - len(ot.DIGEST_PREFIX))
DETAILS = {'family': 'qwen2', 'quantization_level': 'Q4_K_M', 'parameter_size': '72.7B'}
SHOWN = {'details': DETAILS, 'model_info': {'qwen2.context_length': 32768},
         'template': 'model chat template', 'system': 'model default system', 'parameters': ''}


def metadata(base, route, payload=None, timeout=30):
    if route == '/api/version':
        return {'version': '0.40.8'}
    if route == '/api/tags':
        return {'models': [{'name': ot.MODEL, 'digest': DIGEST, 'size': 47_000_000_000,
                            'details': DETAILS}]}
    if route == '/api/show':
        return SHOWN
    if route == '/api/ps':
        return {'models': []}
    raise AssertionError('Unexpected network/generation route: ' + route)


def config():
    with patch.object(ot, 'rpc', side_effect=metadata):
        identity = ot.inspect()
    return {'seed': 42, 'teacher': {'provider': 'ollama_native', 'model': ot.MODEL,
            'base_url_default': ot.DEFAULT_BASE, 'identity': identity,
            'retries': 0, 'timeout_seconds': 1800}}


def reply(content='teacher text', reason='stop'):
    return {'model': ot.MODEL, 'message': {'role': 'assistant', 'content': content},
            'done': True, 'done_reason': reason, 'prompt_eval_count': 100,
            'eval_count': 30, 'eval_duration': 2_000_000_000}


class OllamaTeacherTests(unittest.TestCase):
    def test_check_only_reads_metadata_and_does_not_generate(self):
        with patch.object(ot, 'rpc', side_effect=metadata) as rpc, redirect_stdout(io.StringIO()) as out:
            ot.check()
        self.assertEqual([c.args[1] for c in rpc.call_args_list],
                         ['/api/version', '/api/tags', '/api/show', '/api/ps'])
        self.assertFalse(json.loads(out.getvalue())['generation_called'])

    def test_rpc_bypasses_proxy_and_sends_only_native_json(self):
        response = io.BytesIO(json.dumps(reply()).encode())
        opener = SimpleNamespace(open=lambda request, timeout: response)
        with patch.object(ot.urllib.request, 'build_opener', return_value=opener) as build:
            result = ot.rpc(ot.DEFAULT_BASE, '/api/chat', {'model': ot.MODEL})
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertEqual(result['model'], ot.MODEL)
        for value in ['https://user:pass@host', 'http://host/v1', 'http://host?token=value']:
            with self.assertRaises(ValueError):
                ot.base_url(value)

    def test_native_options_normalized_cache_reuse_and_parser_audit(self):
        cfg = config()
        chats = []

        def fake(base, route, payload=None, timeout=30):
            if route != '/api/chat':
                return metadata(base, route, payload, timeout)
            chats.append(payload)
            return reply(batch())

        with tempfile.TemporaryDirectory() as tmp, patch.object(ot, 'rpc', side_effect=fake):
            run = Path(tmp)
            original = cp.Teacher
            with ot.transport(), sample_format_compatibility(cfg, run):
                teacher = cp.Teacher(cfg, run)
                messages = [{'role': 'user', 'content': 'original math prompt'}]
                text, key = teacher.call(messages)
                saved = (run / 'teacher_cache' / (key + '.json')).read_bytes()
                self.assertEqual(teacher.call(messages), (text, key))
                rows = cp.parse_samples(text, 3, ['Multiplication'], {'stage': 'fine', 'request': key})
                self.assertEqual(len(rows), 3)
                self.assertEqual(saved, (run / 'teacher_cache' / (key + '.json')).read_bytes())
            self.assertIs(cp.Teacher, original)
            self.assertEqual(len(chats), 1)
            native = chats[0]
            self.assertEqual(native['messages'], messages)
            self.assertEqual(native['options']['num_ctx'], 32768)
            self.assertEqual(native['options']['num_predict'], 4096)
            self.assertEqual(native['options']['temperature'], 0.5)
            self.assertEqual(native['options']['top_p'], 0.8)
            self.assertEqual(native['options']['top_k'], 0)
            self.assertEqual(native['options']['repeat_penalty'], 1.05)
            cached = cp.read(run / 'teacher_cache' / (key + '.json'))
            self.assertEqual(cached['native_request'], native)
            self.assertEqual(cached['native_response']['message']['content'], text)
            self.assertEqual(cached['usage']['total_tokens'], 130)
            self.assertEqual(key, cp.digest({'url': ot.DEFAULT_BASE, 'payload': cached['request']}))

    def test_scoring_options_and_seed_are_explicit(self):
        cfg = config()
        teacher = ot.OllamaTeacher(cfg, Path('/tmp/not-used'))
        payload = {'model': ot.MODEL, 'messages': [{'role': 'user', 'content': 'score this'}],
                   'temperature': 0, 'top_p': 1, 'top_k': 1, 'max_tokens': 512, 'stream': False}
        request = teacher.native_request(payload)
        self.assertEqual(request, teacher.native_request(payload))
        self.assertEqual(request['options']['num_predict'], 512)
        self.assertEqual(request['options']['top_k'], 1)
        self.assertEqual(request['options']['temperature'], 0)
        other = teacher.native_request({**payload, 'messages': [{'role': 'user', 'content': 'different'}]})
        self.assertNotEqual(request['options']['seed'], other['options']['seed'])

    def test_length_retry_helpers_keep_native_attempts_and_caps(self):
        for diagnosis, expected in [(True, [1024, 2048]), (False, [4096, 8192])]:
            cfg = config()
            calls = []

            def fake(base, route, payload=None, timeout=30):
                if route != '/api/chat':
                    return metadata(base, route, payload, timeout)
                calls.append(payload['options']['num_predict'])
                return reply('partial' if len(calls) == 1 else 'complete',
                             'length' if len(calls) == 1 else 'stop')

            with self.subTest(diagnosis=diagnosis), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(ot, 'rpc', side_effect=fake), ExitStack() as stack:
                run = Path(tmp)
                stack.enter_context(ot.transport())
                stack.enter_context(diagnosis_length_retry(cfg, run))
                stack.enter_context(generation_length_retry(cfg, run))
                text, key = cp.Teacher(cfg, run).call([{'role': 'user', 'content': 'diagnose'}],
                                                    max_tokens=expected[0])
                self.assertEqual(calls, expected)
                self.assertEqual(text, 'complete')
                caches = [cp.read(p) for p in (run / 'teacher_cache').glob('*.json')]
                self.assertEqual(sorted(c['finish_reason'] for c in caches), ['length', 'stop'])

    def test_changed_digest_empty_or_incomplete_response_and_tampered_cache_fail(self):
        for mode in ['digest', 'empty', 'incomplete', 'tampered']:
            cfg = config()

            def fake(base, route, payload=None, timeout=30):
                if route != '/api/chat':
                    result = metadata(base, route, payload, timeout)
                    if mode == 'digest' and route == '/api/tags':
                        return {'models': [{**result['models'][0], 'digest': ot.DIGEST_PREFIX + '1' * 52}]}
                    return result
                return {**reply('' if mode == 'empty' else 'text'), 'done': mode != 'incomplete'}

            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(ot, 'rpc', side_effect=fake):
                run = Path(tmp)
                teacher = ot.OllamaTeacher(cfg, run)
                messages = [{'role': 'user', 'content': 'prompt'}]
                if mode == 'tampered':
                    _, key = teacher.call(messages)
                    path = run / 'teacher_cache' / (key + '.json')
                    data = cp.read(path)
                    data['content'] = 'altered response'
                    path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    teacher.call(messages)


if __name__ == '__main__':
    unittest.main()
