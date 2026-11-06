#!/usr/bin/env python3
"""Load only teacher settings from a chosen .env and use curl for API calls.

Defaults to the project-root .env. Never prints its values or sources it as shell
code. Existing nominal payloads/cache keys stay the same; synthesis adds a
recorded 1024->2048->4096 diagnosis retry and 4096->8192 generation retry.
`inspect` is offline; `check` only queries models. Generation stages are explicit.
"""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


def load_teacher_env(path, teacher):
    from dotenv import dotenv_values
    path = Path(path)
    if not path.is_file():
        raise ValueError('The specified .env file does not exist.')
    values = dotenv_values(path, interpolate=False, encoding='utf-8-sig')
    key_name, url_name = teacher['api_key_env'], teacher['base_url_env']
    key = values.get(key_name)
    base = values.get(url_name) or teacher['base_url_default']
    if not key or not key.isascii() or any(c.isspace() for c in key):
        raise ValueError(f'The .env must contain a nonempty ASCII {key_name} without whitespace.')
    base = base.rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or any(c.isspace() for c in base)):
        raise ValueError('Teacher base URL must be an HTTPS API URL without credentials/query/fragment.')
    # Explicit file values replace stale teacher variables only.
    os.environ[key_name], os.environ[url_name] = key, base
    return key, base


def redact(text, key):
    text = text.replace(key, '[REDACTED]')
    return re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', text)


def curl_request(request, key, timeout=180):
    """Curl config on stdin keeps authorization and payload out of process argv."""
    options = [('url', request.full_url), ('request', request.get_method())]
    options += [('header', name + ': ' + value) for name, value in request.header_items()]
    if request.data is not None:
        options.append(('data-binary', request.data.decode('utf-8')))
    config = ''.join(name + ' = ' + json.dumps(value, ensure_ascii=False) + '\n'
                     for name, value in options)
    result = subprocess.run(
        ['curl', '-q', '--http1.1', '--config', '-', '--connect-timeout', '10',
         '--max-time', str(timeout), '--silent', '--show-error',
         '--write-out', '\nCDS_HTTP:%{http_code}'],
        input=config, text=True, capture_output=True, timeout=timeout + 10)
    if result.returncode:
        raise urllib.error.URLError(f'curl exit {result.returncode}: '
                                    + redact(result.stderr, key)[:1000])
    body, marker, status = result.stdout.rpartition('\nCDS_HTTP:')
    if not marker or not status.strip().isdigit():
        raise urllib.error.URLError('curl returned no HTTP status.')
    status = int(status.strip())
    if not 200 <= status < 300:
        # Only a redacted response body is exposed, never request headers/config.
        print(f'Teacher HTTP {status}: {redact(body, key)[:2000]}', file=sys.stderr)
        raise urllib.error.HTTPError(request.full_url, status, 'Teacher request failed',
                                     {}, io.BytesIO(redact(body, key).encode()))
    return body


def check_models(teacher, key, base):
    request = urllib.request.Request(base + '/models?sub_type=chat',
                                    headers={'Authorization': 'Bearer ' + key})
    result = json.loads(curl_request(request, key, timeout=30))
    ids = [r['id'] for r in result.get('data', [])]
    print('模型列表 HTTP: 200')
    print('配置的教师模型:', teacher['model'])
    print('配置模型出现在列表中:', teacher['model'] in ids)
    if teacher['model'] not in ids:
        print('Qwen2.5-72B相关模型:', [m for m in ids if 'Qwen2.5' in m and '72B' in m])
        raise ValueError('Configured teacher is absent from the model list; no automatic model substitution.')


@contextmanager
def curl_transport(key):
    original = urllib.request.urlopen

    def urlopen(request, timeout=180):
        from teacher_generation_retry import LARGE_OUTPUT_TIMEOUT
        try:
            limit = json.loads(request.data or b'{}').get('max_tokens', 0)
        except (ValueError, AttributeError, TypeError):
            limit = 0
        effective_timeout = max(timeout, LARGE_OUTPUT_TIMEOUT) if limit >= 8192 else timeout
        return io.BytesIO(curl_request(request, key, effective_timeout).encode('utf-8'))

    urllib.request.urlopen = urlopen
    try:
        yield
    finally:
        urllib.request.urlopen = original


def inspect_request(key):
    import raw_cds18
    import cds_pipeline as cp
    if not re.fullmatch(r'[0-9a-f]{64}', key or ''):
        raise ValueError('--request-key must be a 64-character cache hash.')
    cfg = cp.read(raw_cds18.CONFIG_PATH)
    path = cp.resolve(cfg['run_dir']) / 'teacher_cache' / (key + '.json')
    cached = cp.read(path)
    request = cached['request']
    if cp.digest({'url': cached['url'], 'payload': request}) != key:
        raise ValueError('Cache payload does not match the requested hash.')
    print(json.dumps({'request': key, 'model': request['model'],
        'max_tokens': request['max_tokens'], 'messages': len(request['messages']),
        'scoring': 'top_k' in request, 'finish_reason': cached['finish_reason'],
        'usage': cached.get('usage')}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path,
                        default=Path(__file__).resolve().parents[1] / '.env')
    parser.add_argument('command', choices=['inspect', 'check', 'synthesize', 'quality', 'augment'])
    parser.add_argument('--pool', choices=['seed', 'augmented'])
    parser.add_argument('--request-key')
    args = parser.parse_args()
    if (args.command == 'quality') != (args.pool is not None):
        parser.error('--pool is required for quality and only accepted for quality.')
    if (args.command == 'inspect') != (args.request_key is not None):
        parser.error('--request-key is required for inspect and only accepted for inspect.')
    if args.command == 'inspect':
        inspect_request(args.request_key)
        return
    import raw_cds18
    import cds_pipeline as cp
    from teacher_length_retry import diagnosis_length_retry, POLICY_VERSION
    from teacher_generation_retry import (generation_length_retry, LARGE_OUTPUT_TIMEOUT,
                                           POLICY_VERSION as GENERATION_POLICY)
    cfg = cp.read(raw_cds18.CONFIG_PATH)
    teacher = cfg['teacher']
    key, base = load_teacher_env(args.env_file, teacher)
    print('已从指定.env加载教师配置；密钥未显示。')
    print('API地址:', base)
    check_models(teacher, key, base)
    if args.command == 'check':
        print('鉴权与模型列表检查通过；尚未调用生成接口。')
        return
    # Keep a transport-only provenance record without credentials or .env content.
    _, run, _ = raw_cds18.load()
    event = {'time_utc': datetime.now(timezone.utc).isoformat(), 'command': args.command,
             'pool': args.pool, 'transport': 'curl HTTP/1.1 via stdin config',
             'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'config_hash': cp.digest(cfg), 'teacher': teacher['model'], 'base_url': base,
             'diagnosis_retry_policy': POLICY_VERSION if args.command == 'synthesize' else None,
             'generation_retry_policy': GENERATION_POLICY if args.command in ['synthesize', 'augment'] else None,
             'curl_timeout_seconds_at_8192': LARGE_OUTPUT_TIMEOUT}
    with (run / 'teacher_transport.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(event) + '\n')
    previous = sys.argv
    try:
        sys.argv = ['raw_cds18.py', args.command] + (['--pool', args.pool] if args.pool else [])
        retry = diagnosis_length_retry(cfg, run) if args.command == 'synthesize' else nullcontext()
        generation = (generation_length_retry(cfg, run)
                      if args.command in ['synthesize', 'augment'] else nullcontext())
        # Diagnosis captures the original call before the generation wrapper is
        # installed, so its internal 4096 attempt cannot escalate to 8192.
        with curl_transport(key), retry, generation:
            raw_cds18.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    try:
        main()
    except (ValueError, urllib.error.HTTPError, urllib.error.URLError,
            subprocess.TimeoutExpired, FileNotFoundError) as error:
        # Do not print arbitrary exception strings which might contain process input.
        if isinstance(error, ValueError):
            print(str(error), file=sys.stderr)
        else:
            print(f'教师检查/调用未完成：{type(error).__name__}。请查看上方已脱敏信息。', file=sys.stderr)
        raise SystemExit(1)
