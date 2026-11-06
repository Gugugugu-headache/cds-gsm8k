"""No paid calls: exercise real Teacher caching with fake network responses."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import teacher_length_retry as retry
import teacher_env

CFG = {'teacher': {'model': 'teacher/model', 'api_key_env': 'TEST_TEACHER_KEY',
                  'base_url_env': 'TEST_TEACHER_URL', 'base_url_default': 'https://example.com/v1',
                  'retries': 0, 'timeout_seconds': 30}}
MESSAGES = [{'role': 'user', 'content': 'Diagnose this wrong answer.'}]


def seed(teacher, limit, finish='length', content='unfinished diagnosis'):
    payload = retry.payload_for(teacher, MESSAGES, limit)
    key = cp.digest({'url': teacher.url, 'payload': payload})
    path = teacher.cache / (key + '.json')
    cp.write(path, {'request': payload, 'url': teacher.url, 'content': content,
                    'finish_reason': finish, 'usage': {'completion_tokens': limit}})
    return key, path


class DiagnosisRetryTests(unittest.TestCase):
    def network(self, statuses, observed):
        def urlopen(request, timeout):
            payload = json.loads(request.data)
            observed.append(payload)
            finish = statuses[payload['max_tokens']]
            result = {'choices': [{'message': {'content': 'diagnosis at ' + str(payload['max_tokens'])},
                                    'finish_reason': finish}],
                      'usage': {'completion_tokens': payload['max_tokens']}}
            return io.BytesIO(json.dumps(result).encode())
        return urlopen

    def test_cached_truncation_recovers_and_resume_costs_no_new_requests(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            old_key, old_path = seed(teacher, 1024)
            original_bytes = old_path.read_bytes()
            observed = []
            original_method = cp.Teacher.call
            with patch.object(cp.urllib.request, 'urlopen', side_effect=self.network({2048: 'stop'}, observed)):
                with retry.diagnosis_length_retry(CFG, run):
                    content, accepted_key = teacher.call(MESSAGES, max_tokens=1024)
                    first_record = (run / 'teacher_diagnosis_retries' / (old_key + '.json')).read_bytes()
                    self.assertEqual(teacher.call(MESSAGES, max_tokens=1024), (content, accepted_key))
            self.assertIs(cp.Teacher.call, original_method)
            self.assertEqual(old_path.read_bytes(), original_bytes)
            self.assertEqual([p['max_tokens'] for p in observed], [2048])
            self.assertNotEqual(accepted_key, old_key)
            mapping = cp.read(run / 'teacher_diagnosis_retries' / (old_key + '.json'))
            self.assertEqual(mapping['accepted_request'], accepted_key)
            self.assertEqual(mapping['accepted_max_tokens'], 2048)
            self.assertEqual([r['finish_reason'] for r in mapping['attempts']], ['length', 'stop'])
            self.assertEqual((run / 'teacher_diagnosis_retries' / (old_key + '.json')).read_bytes(), first_record)
            payload = observed[0]
            self.assertEqual(payload['messages'], MESSAGES)
            self.assertEqual(payload['temperature'], 0.5)
            self.assertEqual(payload['top_p'], 0.8)

    def test_second_escalation_is_bounded_and_records_all_usage(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            old_key, _ = seed(teacher, 1024)
            observed = []
            with patch.object(cp.urllib.request, 'urlopen', side_effect=self.network(
                    {2048: 'length', 4096: 'stop'}, observed)), retry.diagnosis_length_retry(CFG, run):
                content, key = teacher.call(MESSAGES, max_tokens=1024)
            mapping = cp.read(run / 'teacher_diagnosis_retries' / (old_key + '.json'))
            self.assertEqual([p['max_tokens'] for p in observed], [2048, 4096])
            self.assertEqual(mapping['accepted_max_tokens'], 4096)
            self.assertEqual(mapping['accepted_request'], key)
            self.assertEqual(sum(a['usage']['completion_tokens'] for a in mapping['attempts']), 7168)
            self.assertEqual(content, 'diagnosis at 4096')

    def test_exhaustion_stops_and_never_returns_partial_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            old_key, _ = seed(teacher, 1024)
            observed = []
            with patch.object(cp.urllib.request, 'urlopen', side_effect=self.network(
                    {2048: 'length', 4096: 'length'}, observed)), retry.diagnosis_length_retry(CFG, run):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, 'finish normally'):
                        teacher.call(MESSAGES, max_tokens=1024)
            self.assertEqual(len(observed), 2)
            mapping = cp.read(run / 'teacher_diagnosis_retries' / (old_key + '.json'))
            self.assertEqual(mapping['status'], 'exhausted')
            self.assertNotIn('accepted_request', mapping)

    def test_successful_original_diagnosis_is_reused_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            key, path = seed(teacher, 1024, 'stop', 'complete diagnosis')
            data = path.read_bytes()
            with patch.object(cp.urllib.request, 'urlopen') as network, retry.diagnosis_length_retry(CFG, run):
                self.assertEqual(teacher.call(MESSAGES, max_tokens=1024), ('complete diagnosis', key))
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse((run / 'teacher_diagnosis_retries').exists())

    def test_generation_scoring_and_other_errors_do_not_trigger_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fail(teacher, messages, max_tokens=4096, scoring=False):
                calls.append((max_tokens, scoring))
                raise ValueError('Unrelated parse error')

            with patch.object(cp.Teacher, 'call', fail), retry.diagnosis_length_retry(CFG, Path(tmp)):
                teacher = cp.Teacher(CFG, Path(tmp))
                for limit, scoring in [(4096, False), (512, True), (1024, False)]:
                    with self.assertRaisesRegex(ValueError, 'Unrelated parse error'):
                        teacher.call(MESSAGES, max_tokens=limit, scoring=scoring)
            self.assertEqual(calls, [(4096, False), (512, True), (1024, False)])

    def test_cache_payload_mismatch_is_not_treated_as_valid_truncation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            _, path = seed(teacher, 1024)
            cache = cp.read(path)
            cache['request']['model'] = 'wrong-model'
            path.write_text(json.dumps(cache))
            with patch.object(cp.urllib.request, 'urlopen') as network, retry.diagnosis_length_retry(CFG, run):
                with self.assertRaisesRegex(ValueError, 'cache request differs'):
                    teacher.call(MESSAGES, max_tokens=1024)
                network.assert_not_called()

    def test_retry_policy_cannot_silently_change_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            original = cp.Teacher.call
            with retry.diagnosis_length_retry(CFG, run):
                pass
            with self.assertRaises(ValueError):
                with retry.diagnosis_length_retry({**CFG, 'different': True}, run):
                    self.fail('Changed policy was accepted')
            self.assertIs(cp.Teacher.call, original)

    def test_retry_integrates_with_dotenv_curl_transport_and_uses_new_real_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_TEACHER_KEY': 'fake'}):
            run = Path(tmp)
            teacher = cp.Teacher(CFG, run)
            old_key, _ = seed(teacher, 1024)
            response = {'choices': [{'message': {'content': 'complete diagnosis'}, 'finish_reason': 'stop'}],
                        'usage': {'completion_tokens': 1500}}
            curl_result = SimpleNamespace(returncode=0, stderr='',
                stdout=json.dumps(response) + '\nCDS_HTTP:200')
            with patch.object(teacher_env.subprocess, 'run', return_value=curl_result) as curl, \
                 teacher_env.curl_transport('fake'), retry.diagnosis_length_retry(CFG, run):
                content, key = teacher.call(MESSAGES, max_tokens=1024)
                self.assertEqual(content, 'complete diagnosis')
                self.assertNotEqual(key, old_key)
                curl.assert_called_once()
                options = dict((line.split(' = ', 1)[0], json.loads(line.split(' = ', 1)[1]))
                               for line in curl.call_args.kwargs['input'].splitlines())
                self.assertEqual(json.loads(options['data-binary'])['max_tokens'], 2048)
            self.assertEqual(cp.read(teacher.cache / (key + '.json'))['finish_reason'], 'stop')


if __name__ == '__main__':
    unittest.main()
