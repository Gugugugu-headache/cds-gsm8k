"""Retry length-truncated synthesis/augmentation at 8192; retain original caches.

Enter this context AFTER diagnosis_length_retry so diagnosis escalation calls
keep their separate 4096 cap. No prompt/model/scoring/student changes.
"""
from contextlib import contextmanager
from pathlib import Path

import cds_pipeline as cp

LIMITS = (4096, 8192)
POLICY_VERSION = 'generation_length_retry_v1'
LARGE_OUTPUT_TIMEOUT = 600


def cached_attempt(teacher, messages, limit):
    payload = {'model': teacher.cfg['model'], 'messages': messages,
               'temperature': 0.5, 'top_p': 0.8, 'max_tokens': limit, 'stream': False}
    key = cp.digest({'url': teacher.url, 'payload': payload})
    path = teacher.cache / (key + '.json')
    cached = cp.read(path)
    if cached['request'] != payload or cached['url'] != teacher.url:
        raise ValueError('Generation cache request differs from its expected payload.')
    return key, cached, {'request': key, 'max_tokens': limit,
        'finish_reason': cached['finish_reason'], 'cache_sha256': cp.file_hash(path),
        'usage': cached.get('usage')}


@contextmanager
def generation_length_retry(cfg, run):
    run = Path(run)
    policy = {'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__), 'limits': list(LIMITS),
        'scope': 'Non-scoring generation calls with nominal max_tokens=4096 in synthesize/augment.',
        'curl_timeout_seconds_at_8192': LARGE_OUTPUT_TIMEOUT,
        'trigger': 'Nonempty cached output rejected with finish_reason=length.',
        'preserved': ['messages', 'model', 'temperature', 'top_p', 'original caches',
                      'diagnosis retry limits', 'scoring budget', 'student budget'],
        'cost': 'Record all attempts including truncated responses; accept first stop, no answer selection.'}
    cp.write(run / 'teacher_generation_retry_policy.json', policy)
    original = cp.Teacher.call

    def call(teacher, messages, max_tokens=4096, scoring=False):
        if max_tokens != LIMITS[0] or scoring:
            return original(teacher, messages, max_tokens=max_tokens, scoring=scoring)
        attempts = []
        for index, limit in enumerate(LIMITS):
            try:
                result = original(teacher, messages, max_tokens=limit, scoring=False)
            except ValueError as error:
                if not str(error).startswith('Teacher did not finish normally (length): '):
                    raise
                key, cached, attempt = cached_attempt(teacher, messages, limit)
                if (cached['finish_reason'] != 'length' or not cached['content'].strip()
                        or str(error) != f'Teacher did not finish normally (length): {key}'):
                    raise ValueError('Retry trigger disagrees with the generation cache.') from error
                attempts.append(attempt)
                if index == len(LIMITS) - 1:
                    cp.write(run / 'teacher_generation_retries' / (attempts[0]['request'] + '.json'), {
                        'policy': POLICY_VERSION, 'status': 'exhausted',
                        'messages_sha256': cp.digest(messages), 'attempts': attempts})
                    raise
                print(f'Teacher generation truncated at {limit}; retrying with '
                      f'max_tokens={LIMITS[index + 1]}. Request: {key}', flush=True)
            else:
                if attempts:
                    key, cached, attempt = cached_attempt(teacher, messages, limit)
                    if cached['finish_reason'] != 'stop' or result[1] != key:
                        raise ValueError('Recovered generation/cache is not complete or consistent.')
                    attempts.append(attempt)
                    cp.write(run / 'teacher_generation_retries' / (attempts[0]['request'] + '.json'), {
                        'policy': POLICY_VERSION, 'status': 'recovered', 'accepted_request': key,
                        'accepted_max_tokens': limit, 'messages_sha256': cp.digest(messages),
                        'attempts': attempts})
                    print(f'Teacher generation recovered at max_tokens={limit}. Request: {key}', flush=True)
                return result

    cp.Teacher.call = call
    try:
        yield
    finally:
        cp.Teacher.call = original
