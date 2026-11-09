"""Native Ollama transport with immutable, source-compatible CDS caches.

Only /api/chat generates text. Metadata inspection does not preload models.
The cache's request field is the normalized CDS request used by the existing
retry/parser helpers; native_request/native_response retain the actual exchange.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import cds_pipeline as cp

MODEL = 'qwen2.5:72b'
DIGEST_PREFIX = '424bad2cc13f'
DEFAULT_BASE = 'http://127.0.0.1:11434'
CONTEXT = 32768
TRANSIENT = {429, 500, 502, 503, 504}


def base_url(value):
    value = value.rstrip('/')
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme not in ['http', 'https'] or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path or any(c.isspace() for c in value)):
        raise ValueError('Use an Ollama server origin such as http://127.0.0.1:11434.')
    return value


def rpc(base, route, payload=None, timeout=30):
    base = base_url(base)
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(base + route, data=data,
                                     headers={'Content-Type': 'application/json'})
    # Local Ollama calls do not use shell proxy settings or API credentials.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get('error'):
        raise ValueError('Ollama returned an invalid/error response; no text accepted.')
    return result


def tagged_model(base):
    matches = [m for m in rpc(base, '/api/tags').get('models', [])
               if m.get('name') == MODEL or m.get('model') == MODEL]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one installed model named {MODEL}.')
    model = matches[0]
    digest = model.get('digest', '').removeprefix('sha256:')
    if not re.fullmatch(r'[0-9a-f]{64}', digest) or not digest.startswith(DIGEST_PREFIX):
        raise ValueError('Installed model digest differs from the reported Qwen2.5-72B Q4_K_M.')
    return model, digest


def inspect(base=DEFAULT_BASE):
    base = base_url(base)
    version = rpc(base, '/api/version').get('version')
    model, digest = tagged_model(base)
    shown = rpc(base, '/api/show', {'model': MODEL})
    details = shown.get('details', {})
    info = shown.get('model_info', {})
    if (details.get('quantization_level') != 'Q4_K_M'
            or details.get('family') != 'qwen2'
            or info.get('qwen2.context_length', 0) < CONTEXT
            or not isinstance(version, str) or not version):
        raise ValueError('Unexpected Ollama model quantization, architecture, context limit or version.')
    identity = {'model': MODEL, 'digest': digest, 'ollama_version': version,
                'details': details, 'model_size_bytes': model.get('size'),
                'template_sha256': cp.digest(shown.get('template', '')),
                'system_sha256': cp.digest(shown.get('system', '')),
                'parameters_sha256': cp.digest(shown.get('parameters', '')),
                'model_info_sha256': cp.digest(info)}
    return identity


def check(base=DEFAULT_BASE):
    identity = inspect(base)
    loaded = rpc(base, '/api/ps').get('models', [])
    print(json.dumps({'base_url': base_url(base), 'identity': identity,
                      'currently_loaded': [{k: r.get(k) for k in
                          ['name', 'digest', 'size', 'size_vram', 'context_length']}
                          for r in loaded],
                      'generation_called': False}, ensure_ascii=False, indent=2))
    return identity


class OllamaTeacher:
    def __init__(self, cfg, run):
        self.cfg = cfg['teacher']
        if self.cfg.get('provider') != 'ollama_native' or self.cfg['model'] != MODEL:
            raise ValueError('This transport accepts the frozen local Ollama teacher only.')
        self.config_hash = cp.digest(cfg)
        self.seed = cfg['seed']
        self.url = base_url(self.cfg['base_url_default'])
        self.cache = Path(run) / 'teacher_cache'

    def native_request(self, payload):
        seed = int(cp.digest({'seed': self.seed, 'request': payload})[:8], 16) & 0x7fffffff
        return {'model': MODEL, 'messages': payload['messages'], 'stream': False,
                'keep_alive': '5m', 'options': {
                    'temperature': payload['temperature'], 'top_p': payload['top_p'],
                    'top_k': payload.get('top_k', 0), 'min_p': 0,
                    'repeat_penalty': 1.05, 'repeat_last_n': -1,
                    'num_predict': payload['max_tokens'], 'num_ctx': CONTEXT, 'seed': seed}}

    def verify_digest(self):
        _, digest = tagged_model(self.url)
        if digest != self.cfg['identity']['digest']:
            raise ValueError('Ollama model tag changed after the run was frozen.')

    def call(self, messages, max_tokens=4096, scoring=False):
        payload = {'model': MODEL, 'messages': messages, 'temperature': 0 if scoring else 0.5,
                   'top_p': 1 if scoring else 0.8, 'max_tokens': max_tokens, 'stream': False}
        if scoring:
            payload['top_k'] = 1
        key = cp.digest({'url': self.url, 'payload': payload})
        path = self.cache / (key + '.json')
        native = self.native_request(payload)
        if path.exists():
            cached = cp.read(path)
            if (cached.get('request') != payload or cached.get('url') != self.url
                    or cached.get('config_hash') != self.config_hash
                    or cached.get('native_request') != native
                    or cached.get('model_digest') != self.cfg['identity']['digest']
                    or cached.get('native_response', {}).get('done') is not True
                    or cached['native_response'].get('done_reason') != cached.get('finish_reason')
                    or cached['native_response'].get('message', {}).get('content') != cached.get('content')):
                raise ValueError(f'Ollama cache/configuration mismatch: {key}')
        else:
            self.verify_digest()
            print(f'Ollama {"scoring" if scoring else "generation"}: '
                  f'max_tokens={max_tokens}; request={key}', flush=True)
            for attempt in range(self.cfg['retries'] + 1):
                try:
                    result = rpc(self.url, '/api/chat', native,
                                 timeout=self.cfg['timeout_seconds'])
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in TRANSIENT or attempt == self.cfg['retries']:
                        raise RuntimeError(f'Ollama HTTP {error.code}; request {key}') from None
                except (urllib.error.URLError, TimeoutError):
                    if attempt == self.cfg['retries']:
                        raise RuntimeError(f'Ollama transport failure; request {key}') from None
                time.sleep(min(2 ** attempt, 8))
            self.verify_digest()
            if (result.get('done') is not True or result.get('model') != MODEL
                    or not isinstance(result.get('message'), dict)):
                raise ValueError(f'Ollama did not return a completed chat response: {key}')
            prompt_tokens, output_tokens = result.get('prompt_eval_count'), result.get('eval_count')
            usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': output_tokens,
                     'total_tokens': prompt_tokens + output_tokens
                     if isinstance(prompt_tokens, int) and isinstance(output_tokens, int) else None}
            cached = {'request': payload, 'url': self.url,
                      'content': result['message'].get('content'),
                      'finish_reason': result.get('done_reason'), 'usage': usage,
                      'model': result.get('model'), 'model_digest': self.cfg['identity']['digest'],
                      'config_hash': self.config_hash, 'native_endpoint': self.url + '/api/chat',
                      'native_request': native, 'native_response': result,
                      'cache_schema': 'Normalized CDS request plus actual native Ollama exchange.'}
            cp.write(path, cached)
        if not isinstance(cached['content'], str) or not cached['content'].strip():
            raise ValueError(f'Empty teacher response; request {key}')
        if cached['finish_reason'] != 'stop':
            raise ValueError(f'Teacher did not finish normally ({cached["finish_reason"]}): {key}')
        return cached['content'], key


@contextmanager
def transport():
    previous = cp.Teacher
    cp.Teacher = OllamaTeacher
    try:
        yield
    finally:
        cp.Teacher = previous
