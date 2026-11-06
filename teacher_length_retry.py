"""Bounded, auditable retries for truncated teacher diagnoses only.

Successful historical requests retain their original payloads and cache keys.
Only a nonempty diagnosis cached with finish_reason=length is retried at a
larger limit. Never delete/overwrite truncated caches or accept partial text.
"""
from contextlib import contextmanager
from pathlib import Path

import cds_pipeline as cp

LIMITS = (1024, 2048, 4096)
POLICY_VERSION = 'diagnosis_length_retry_v1'


def payload_for(teacher, messages, limit):
    return {'model': teacher.cfg['model'], 'messages': messages,
            'temperature': 0.5, 'top_p': 0.8, 'max_tokens': limit, 'stream': False}


def cached_attempt(teacher, messages, limit):
    payload = payload_for(teacher, messages, limit)
    key = cp.digest({'url': teacher.url, 'payload': payload})
    path = teacher.cache / (key + '.json')
    cached = cp.read(path)
    if cached['request'] != payload or cached['url'] != teacher.url:
        raise ValueError('Diagnosis cache request differs from its expected payload.')
    attempt = {'request': key, 'max_tokens': limit,
               'finish_reason': cached['finish_reason'],
               'cache_sha256': cp.file_hash(path), 'usage': cached.get('usage')}
    return key, cached, attempt


@contextmanager
def diagnosis_length_retry(cfg, run):
    run = Path(run)
    policy = {'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
              'helper_sha256': cp.file_hash(__file__), 'limits': list(LIMITS),
              'scope': 'Single-user-message non-scoring diagnosis calls in synthesize, nominal max_tokens=1024.',
              'trigger': 'Nonempty cached output rejected with finish_reason=length.',
              'unchanged': ['model', 'messages', 'temperature', 'top_p', 'generation/scoring budgets',
                            'student inference budget'],
              'reuse': 'Preserve every original cache; higher-limit requests have distinct real cache keys.',
              'cost': 'Count all attempts, including truncated responses; no best-answer selection.'}
    cp.write(run / 'teacher_diagnosis_retry_policy.json', policy)
    original = cp.Teacher.call

    def call(teacher, messages, max_tokens=4096, scoring=False):
        eligible = (max_tokens == LIMITS[0] and not scoring and len(messages) == 1
                    and messages[0].get('role') == 'user')
        if not eligible:
            return original(teacher, messages, max_tokens=max_tokens, scoring=scoring)
        attempts = []
        for index, limit in enumerate(LIMITS):
            try:
                result = original(teacher, messages, max_tokens=limit, scoring=False)
            except ValueError as error:
                if not str(error).startswith('Teacher did not finish normally (length): '):
                    raise
                key, cached, attempt = cached_attempt(teacher, messages, limit)
                if cached['finish_reason'] != 'length' or not cached['content'].strip():
                    raise ValueError('Retry trigger disagrees with the cached diagnosis.') from error
                if str(error) != f'Teacher did not finish normally (length): {key}':
                    raise ValueError('Truncation error does not identify the expected cache.') from error
                attempts.append(attempt)
                if index == len(LIMITS) - 1:
                    cp.write(run / 'teacher_diagnosis_retries' / (attempts[0]['request'] + '.json'), {
                        'policy': POLICY_VERSION, 'status': 'exhausted',
                        'messages_sha256': cp.digest(messages), 'attempts': attempts})
                    raise
                print(f'Teacher diagnosis truncated at {limit}; retrying with '
                      f'max_tokens={LIMITS[index + 1]}. Request: {key}', flush=True)
            else:
                if attempts:
                    key, cached, attempt = cached_attempt(teacher, messages, limit)
                    if cached['finish_reason'] != 'stop' or result[1] != key:
                        raise ValueError('Recovered diagnosis/cache is not complete or consistent.')
                    attempts.append(attempt)
                    cp.write(run / 'teacher_diagnosis_retries' / (attempts[0]['request'] + '.json'), {
                        'policy': POLICY_VERSION, 'status': 'recovered', 'accepted_request': key,
                        'accepted_max_tokens': limit, 'messages_sha256': cp.digest(messages),
                        'attempts': attempts})
                    print(f'Teacher diagnosis recovered at max_tokens={limit}. Request: {key}', flush=True)
                return result

    cp.Teacher.call = call
    try:
        yield
    finally:
        cp.Teacher.call = original
