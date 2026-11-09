#!/usr/bin/env python3
"""Run the frozen local CDS entry with an audited repeat_last_n compatibility fix.

The reported Ollama 0.40.8 server rejects the documented -1 sentinel. Send the
positive num_ctx value instead. Existing frozen source files are not modified.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import urllib.error

import cds_pipeline as cp
import ollama_teacher as ot
import qwen15_cds_ollama as job

POLICY_VERSION = 'ollama_positive_repeat_scope_v1'
TEACHER_COMMANDS = {'synthesize', 'quality', 'augment'}


class ContextRepeatTeacher(ot.OllamaTeacher):
    def native_request(self, payload):
        native = super().native_request(payload)
        if native['options']['repeat_last_n'] != -1:
            raise ValueError('Unexpected source repeat scope; review this compatibility policy.')
        native['options']['repeat_last_n'] = native['options']['num_ctx']
        return native

    def call(self, messages, max_tokens=4096, scoring=False):
        payload = {'model': ot.MODEL, 'messages': messages,
                   'temperature': 0 if scoring else 0.5, 'top_p': 1 if scoring else 0.8,
                   'max_tokens': max_tokens, 'stream': False}
        if scoring:
            payload['top_k'] = 1
        key = cp.digest({'url': self.url, 'payload': payload})
        path = self.cache / (key + '.json')
        auditable = False
        try:
            result = super().call(messages, max_tokens=max_tokens, scoring=scoring)
            auditable = True
            return result
        except ValueError as error:
            auditable = str(error) == f'Teacher did not finish normally (length): {key}'
            raise
        finally:
            # Record both complete and length-truncated native responses. Never
            # rewrite the original cache or bless an incompatible older request.
            if auditable and path.is_file():
                cached = cp.read(path)
                native = self.native_request(payload)
                if (cached.get('native_request') == native
                        and cached.get('config_hash') == self.config_hash
                        and cached.get('model_digest') == self.cfg['identity']['digest']):
                    cp.write(self.cache.parent / 'teacher_repeat_scope_requests' / (key + '.json'), {
                        'policy': POLICY_VERSION, 'request': key,
                        'config_hash': self.config_hash, 'cache_sha256': cp.file_hash(path),
                        'native_request_sha256': cp.digest(native),
                        'source_repeat_last_n': -1,
                        'effective_repeat_last_n': native['options']['repeat_last_n'],
                        'context_tokens': native['options']['num_ctx'],
                        'finish_reason': cached['finish_reason']})


@contextmanager
def compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_repeat_scope_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__),
        'source_entry_sha256': cp.file_hash(job.__file__),
        'source_transport_sha256': cp.file_hash(ot.__file__),
        'source_repeat_last_n': -1, 'effective_repeat_last_n': ot.CONTEXT,
        'reason': 'Observed HTTP 400: repeat_last_n must be an integer in [0, 2147483647].',
        'intent': 'Explicit positive context-sized repetition window instead of the rejected sentinel.',
        'preserved': ['original frozen config/protocol', 'model and full digest',
                      'messages', 'temperature/top_p/top_k/min_p', 'seed',
                      'repeat_penalty=1.05', 'num_ctx', 'num_predict',
                      'quality threshold', 'student responses and allocations'],
        'cache': 'Existing normalized keys; actual native request retained and verified; '
                 'incompatible pre-fix native caches are rejected, not overwritten.'})
    original_transport, original_rpc = ot.transport, ot.rpc

    @contextmanager
    def fixed_transport():
        previous = cp.Teacher
        cp.Teacher = ContextRepeatTeacher
        try:
            yield
        finally:
            cp.Teacher = previous

    def detailed_rpc(base, route, payload=None, timeout=30):
        try:
            return original_rpc(base, route, payload, timeout)
        except urllib.error.HTTPError as error:
            body = error.read(8192).decode('utf-8', errors='replace')
            try:
                detail = json.loads(body).get('error', body)
            except (ValueError, AttributeError):
                detail = body
            record = {'status': error.code, 'endpoint': base + route,
                      'native_request': payload, 'response_body': body,
                      'maximum_body_bytes': 8192, 'policy': POLICY_VERSION}
            cp.write(run / 'teacher_http_errors' / (cp.digest(record) + '.json'), record)
            print(f'Ollama HTTP {error.code}; endpoint={route}; error={detail}', flush=True)
            raise

    ot.transport, ot.rpc = fixed_transport, detailed_rpc
    try:
        yield
    finally:
        ot.transport, ot.rpc = original_transport, original_rpc


def main():
    if len(sys.argv) > 1 and sys.argv[1] in TEACHER_COMMANDS:
        cfg, run, _ = job.load()
        with compatibility(cfg, run):
            job.main()
    else:
        job.main()


if __name__ == '__main__':
    main()
