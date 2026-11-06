"""Offline generation-retry, diagnosis-boundary and cache-inspection tests."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import teacher_env
import teacher_generation_retry as generation
import teacher_length_retry as diagnosis
import raw_cds18
from test_teacher_length_retry import CFG

MESSAGES = [{'role': 'user', 'content': 'diagnose'}, {'role': 'assistant', 'content': 'diagnosis'},
            {'role': 'user', 'content': 'generate three samples'}]


def seed(teacher, messages, limit, finish='length'):
    payload = {'model': teacher.cfg['model'], 'messages': messages,
               'temperature': 0.5, 'top_p': 0.8, 'max_tokens': limit, 'stream': False}
    key = cp.digest({'url': teacher.url, 'payload': payload})
    path = teacher.cache / (key + '.json')
    cp.write(path, {'request': payload, 'url': teacher.url, 'content': 'sample response',
                    'finish_reason': finish, 'usage': {'completion_tokens': limit}})
    return key, path


class GenerationRetryTests(unittest.TestCase):
    def network(self, status, seen):
        def urlopen(request, timeout):
            payload = json.loads(request.data)
            seen.append(payload)
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'complete samples'},
                       'finish_reason': status}], 'usage': {'completion_tokens': 5000}}).encode())
        return urlopen

    def test_generation_recovers_once_and_resume_reuses_both_caches(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            old, path = seed(teacher, MESSAGES, 4096)
            content = path.read_bytes()
            seen = []
            with patch.object(cp.urllib.request, 'urlopen', side_effect=self.network('stop', seen)), \
                 diagnosis.diagnosis_length_retry(CFG, run), generation.generation_length_retry(CFG, run):
                result = teacher.call(MESSAGES)
                self.assertEqual(teacher.call(MESSAGES), result)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual([p['max_tokens'] for p in seen], [8192])
            self.assertEqual(seen[0]['messages'], MESSAGES)
            self.assertEqual(seen[0]['model'], CFG['teacher']['model'])
            report = cp.read(run / 'teacher_generation_retries' / (old + '.json'))
            self.assertEqual(report['accepted_request'], result[1])
            self.assertEqual(report['accepted_max_tokens'], 8192)
            self.assertEqual([a['max_tokens'] for a in report['attempts']], [4096, 8192])

    def test_generation_exhaustion_stops_without_partial_result(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            old, _ = seed(teacher, MESSAGES, 4096)
            seen = []
            with patch.object(cp.urllib.request, 'urlopen', side_effect=self.network('length', seen)), \
                 generation.generation_length_retry(CFG, run):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, 'finish normally'):
                        teacher.call(MESSAGES)
            self.assertEqual(len(seen), 1)
            self.assertEqual(cp.read(run / 'teacher_generation_retries' / (old + '.json'))['status'], 'exhausted')

    def test_diagnosis_escalation_still_stops_at_4096_not_8192(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            messages = MESSAGES[:1]
            for limit in [1024, 2048, 4096]:
                seed(teacher, messages, limit)
            with patch.object(cp.urllib.request, 'urlopen') as network, \
                 diagnosis.diagnosis_length_retry(CFG, run), generation.generation_length_retry(CFG, run):
                with self.assertRaisesRegex(ValueError, 'finish normally'):
                    teacher.call(messages, max_tokens=1024)
                network.assert_not_called()
            self.assertFalse((run / 'teacher_generation_retries').exists())

    def test_successful_generation_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            key, _ = seed(teacher, MESSAGES, 4096, 'stop')
            with patch.object(cp.urllib.request, 'urlopen') as network, generation.generation_length_retry(CFG, run):
                self.assertEqual(teacher.call(MESSAGES)[1], key)
                network.assert_not_called()

    def test_scoring_and_unrelated_errors_are_not_escalated(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = []

            def fail(teacher, messages, max_tokens=4096, scoring=False):
                seen.append((max_tokens, scoring))
                raise ValueError('Unrelated failure')

            with patch.object(cp.Teacher, 'call', fail), generation.generation_length_retry(CFG, Path(tmp)):
                teacher = cp.Teacher(CFG, Path(tmp))
                for tokens, scoring in [(4096, False), (512, True)]:
                    with self.assertRaisesRegex(ValueError, 'Unrelated failure'):
                        teacher.call(MESSAGES, max_tokens=tokens, scoring=scoring)
            self.assertEqual(seen, [(4096, False), (512, True)])

    def test_policy_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with generation.generation_length_retry(CFG, run):
                pass
            with self.assertRaises(ValueError):
                with generation.generation_length_retry({**CFG, 'changed': True}, run):
                    self.fail('Changed policy accepted')

    def test_only_large_output_requests_receive_longer_curl_timeout(self):
        with patch.object(teacher_env, 'curl_request', return_value='{}') as curl, \
             teacher_env.curl_transport('fake'):
            for limit, expected in [(4096, 180), (8192, 600)]:
                request = urllib.request.Request('https://example.com/v1/chat/completions',
                                                data=json.dumps({'max_tokens': limit}).encode())
                with urllib.request.urlopen(request, timeout=180) as response:
                    self.assertEqual(json.load(response), {})
                self.assertEqual(curl.call_args.args[2], expected)

    def test_inspect_does_not_load_dotenv_or_make_network_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            key, path = seed(teacher, MESSAGES, 4096)
            config = run / 'config.json'
            cp.write(config, {'run_dir': str(run)})
            output = io.StringIO()
            with patch.object(raw_cds18, 'CONFIG_PATH', config), \
                 patch.object(teacher_env, 'load_teacher_env') as load_env, \
                 patch.object(teacher_env, 'check_models') as network, \
                 patch.object(sys, 'argv', ['teacher_env.py', 'inspect', '--request-key', key]), \
                 contextlib.redirect_stdout(output):
                teacher_env.main()
                load_env.assert_not_called()
                network.assert_not_called()
            result = json.loads(output.getvalue())
            self.assertEqual(result['max_tokens'], 4096)
            self.assertEqual(result['messages'], 3)
            self.assertNotIn('content', result)
            cache = cp.read(path)
            cache['request']['model'] = 'tampered'
            path.write_text(json.dumps(cache))
            with patch.object(raw_cds18, 'CONFIG_PATH', config), self.assertRaises(ValueError):
                teacher_env.inspect_request(key)


if __name__ == '__main__':
    unittest.main()
