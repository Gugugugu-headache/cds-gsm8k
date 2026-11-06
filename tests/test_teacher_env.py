"""Credential handling and curl transport tests use temporary fake secrets only."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import teacher_env as env
import cds_pipeline as cp

TEACHER = {'api_key_env': 'Siliconflow_API_KEY', 'base_url_env': 'Siliconflow_BASE_URL',
           'base_url_default': 'https://api.siliconflow.cn/v1', 'model': 'teacher/model',
           'timeout_seconds': 30, 'retries': 0}


class TeacherEnvTests(unittest.TestCase):
    def test_file_values_override_stale_teacher_only_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, clear=True):
            root = Path(tmp)
            path = root / '.env'
            path.write_text('export Siliconflow_API_KEY="fake-test-value"\n'
                            'Siliconflow_BASE_URL="https://api.siliconflow.cn/v1/"\n'
                            'OTHER_SERVICE_TOKEN=unrelated-value\n'
                            'COMMAND=$(touch never-execute)\n')
            os.environ['Siliconflow_API_KEY'] = 'stale-value'
            key, base = env.load_teacher_env(path, TEACHER)
            self.assertEqual(key, 'fake-test-value')
            self.assertEqual(os.environ['Siliconflow_API_KEY'], key)
            self.assertEqual(base, 'https://api.siliconflow.cn/v1')
            self.assertNotIn('OTHER_SERVICE_TOKEN', os.environ)
            self.assertNotIn('COMMAND', os.environ)
            self.assertFalse((root / 'never-execute').exists())

    def test_missing_malformed_key_or_unsafe_url_rejected_without_secret_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, clear=True):
            path = Path(tmp) / '.env'
            for content in ['', 'Siliconflow_API_KEY="fake value"',
                            'Siliconflow_API_KEY="测试"',
                            'Siliconflow_API_KEY=fake\nSiliconflow_BASE_URL=http://example.com',
                            'Siliconflow_API_KEY=fake\nSiliconflow_BASE_URL=https://user:pass@example.com']:
                path.write_text(content)
                with self.assertRaises(ValueError) as error:
                    env.load_teacher_env(path, TEACHER)
                self.assertNotIn('fake value', str(error.exception))
                self.assertNotIn('user:pass', str(error.exception))

    def test_curl_keeps_key_and_payload_off_argv_and_preserves_json(self):
        key = 'fake-test-secret'
        payload = {'model': 'teacher/model', 'messages': [{'role': 'user', 'content': '数学\n"q"'}],
                   'max_tokens': 4096, 'temperature': 0.5}
        request = urllib.request.Request('https://api.siliconflow.cn/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key,
                                                       'Content-Type': 'application/json'})
        result = SimpleNamespace(returncode=0, stdout='{"choices": []}\nCDS_HTTP:200', stderr='')
        with patch.object(env.subprocess, 'run', return_value=result) as run:
            self.assertEqual(json.loads(env.curl_request(request, key)), {'choices': []})
            argv = run.call_args.args[0]
            self.assertFalse(any(key in value for value in argv))
            self.assertFalse(any('数学' in value for value in argv))
            config = run.call_args.kwargs['input']
            options = [(line.split(' = ', 1)[0], json.loads(line.split(' = ', 1)[1]))
                       for line in config.splitlines()]
            self.assertEqual(json.loads(dict(options)['data-binary']), payload)
            self.assertIn(('header', 'Authorization: Bearer ' + key), options)

    def test_error_response_redacts_key_before_output_and_exception(self):
        key = 'fake-test-secret'
        result = SimpleNamespace(returncode=0,
            stdout='Token is invalid: ' + key + '\nCDS_HTTP:401', stderr='')
        stream = io.StringIO()
        with patch.object(env.subprocess, 'run', return_value=result), contextlib.redirect_stderr(stream):
            with self.assertRaises(urllib.error.HTTPError) as error:
                env.curl_request(urllib.request.Request('https://api.siliconflow.cn/v1/models'), key)
        self.assertNotIn(key, stream.getvalue())
        self.assertNotIn(key, error.exception.read().decode())
        self.assertIn('[REDACTED]', stream.getvalue())

    def test_teacher_payload_cache_key_and_cache_behavior_are_unchanged(self):
        key, message = 'fake-test-secret', [{'role': 'user', 'content': 'a math prompt'}]
        response = {'choices': [{'message': {'content': 'a sample'}, 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 10}, 'model': TEACHER['model']}
        result = SimpleNamespace(returncode=0, stdout=json.dumps(response) + '\nCDS_HTTP:200', stderr='')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                TEACHER['api_key_env']: key, TEACHER['base_url_env']: TEACHER['base_url_default']}):
            teacher = cp.Teacher({'teacher': TEACHER}, Path(tmp))
            original = urllib.request.urlopen
            with patch.object(env.subprocess, 'run', return_value=result) as run:
                with env.curl_transport(key):
                    content, cache_key = teacher.call(message)
                    self.assertEqual(teacher.call(message), (content, cache_key))
                run.assert_called_once()
            self.assertIs(urllib.request.urlopen, original)
            cache = cp.read(Path(tmp) / 'teacher_cache' / (cache_key + '.json'))
            self.assertEqual(cache['request'], {
                'model': TEACHER['model'], 'messages': message, 'temperature': 0.5,
                'top_p': 0.8, 'max_tokens': 4096, 'stream': False})
            self.assertEqual(cache_key, cp.digest({'url': TEACHER['base_url_default'],
                                                 'payload': cache['request']}))
            self.assertNotIn(key, json.dumps(cache))

    def test_check_only_queries_model_list_and_does_not_generate(self):
        with patch.object(env, 'curl_request', return_value=json.dumps({
                'data': [{'id': TEACHER['model']}]})) as call:
            env.check_models(TEACHER, 'fake', TEACHER['base_url_default'])
            call.assert_called_once()
            self.assertTrue(call.call_args.args[0].full_url.endswith('/models?sub_type=chat'))
            self.assertEqual(call.call_args.args[0].get_method(), 'GET')


if __name__ == '__main__':
    unittest.main()
