#!/usr/bin/env python3
"""Download a small reference-model pool without changing any CDS experiment.

Public repositories only, anonymous access, sequential models and bounded file
concurrency. Pin each repository commit before downloading so reruns resume the
same revision. This command neither imports torch nor loads model weights.
"""
import argparse
import fnmatch
import json
from pathlib import Path, PurePosixPath

MODELS = {
    'qwen25_05b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen25_15b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen15_18b': 'Qwen/Qwen1.5-1.8B-Chat',
    'qwen15_4b': 'Qwen/Qwen1.5-4B-Chat',
    'smollm2_17b': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    'phi35_mini': 'microsoft/Phi-3.5-mini-instruct',
    'mistral7b_v03': 'mistralai/Mistral-7B-Instruct-v0.3',
}
INCLUDE = ['*.json', '*.safetensors', '*.model', '*.model.v3', '*.txt', '*.tiktoken', '*.jinja']
EXCLUDE = ['consolidated*.safetensors', 'original/*', 'onnx/*', 'onnx/**']
PIN = 'reference_download.pin.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def wanted(name):
    return (any(fnmatch.fnmatchcase(name, pattern) for pattern in INCLUDE)
            and not any(fnmatch.fnmatchcase(name, pattern) for pattern in EXCLUDE))


def safe_file(destination, name):
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or not part.parts:
        raise ValueError('Invalid repository file path.')
    path = destination.joinpath(*part.parts)
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Repository file points outside its model directory.')
    return path


def pin_revision(api, repo, destination):
    path = destination / PIN
    if path.is_file():
        pin = read(path)
        if pin['repo_id'] != repo or not pin.get('revision') or not pin.get('files'):
            raise ValueError('Existing download pin differs; use a different model directory.')
        for name in pin['files']:
            safe_file(destination, name)
        return pin
    # Do not attach a new revision to unrelated existing or manually downloaded weights.
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Model directory is not empty and has no reference download pin.')
    info = api.model_info(repo, revision='main', files_metadata=True)
    files = {sibling.rfilename: getattr(sibling, 'size', None)
             for sibling in info.siblings if wanted(sibling.rfilename)}
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Repository lacks the expected Transformers config and safetensors.')
    for name in files:
        safe_file(destination, name)
    pin = {'repo_id': repo, 'revision': info.sha, 'files': files}
    destination.mkdir(parents=True, exist_ok=True)
    write(path, pin)
    return pin


def validate_download(destination, pin):
    for name, size in pin['files'].items():
        path = safe_file(destination, name)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty downloaded file: {name}')
        if size is not None and path.stat().st_size != size:
            raise ValueError(f'Downloaded file size differs from pinned metadata: {name}')
    for index in destination.glob('*.safetensors.index.json'):
        shards = set(read(index)['weight_map'].values())
        if not shards:
            raise ValueError('Empty safetensors index.')
        for name in shards:
            path = safe_file(destination, name)
            if name not in pin['files'] or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Shard missing from pinned download: {name}')
    return {'repo_id': pin['repo_id'], 'revision': pin['revision'],
            'files_checked': len(pin['files']),
            'bytes_on_disk': sum(safe_file(destination, name).stat().st_size for name in pin['files']),
            'validation': 'Pinned file presence/size and safetensors-index shard checks; not a weight hash audit.'}


def download_one(repo, root, api, snapshot, workers):
    destination = root / repo.split('/')[-1]
    pin = pin_revision(api, repo, destination)
    print(f'\nDownloading {repo}@{pin["revision"]}\nDestination: {destination}', flush=True)
    snapshot(repo_id=repo, revision=pin['revision'], local_dir=str(destination),
             allow_patterns=list(pin['files']), token=False, max_workers=workers)
    report = validate_download(destination, pin)
    write(destination / 'download_provenance.json',
          {'repo_id': repo, 'revision': pin['revision']})
    write(destination / 'reference_download.validation.json', report)
    print(f'Model ready: {destination}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--root', type=Path, default=Path('/media/data/gezhenjiang/model_list'))
    parser.add_argument('--max-workers', type=int, default=2)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--list', action='store_true', help='Print the catalogue offline; download nothing.')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(MODELS, ensure_ascii=False, indent=2))
        return
    if args.max_workers < 1 or args.attempts < 1:
        parser.error('--max-workers and --attempts must be positive.')
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=False)
    root = args.root.expanduser().resolve()
    failed = []
    for alias in dict.fromkeys(args.models):
        repo = MODELS[alias]
        for attempt in range(1, args.attempts + 1):
            try:
                download_one(repo, root, api, snapshot_download, args.max_workers)
                break
            except Exception as error:
                print(f'{repo}: attempt {attempt}/{args.attempts} failed: '
                      f'{type(error).__name__}: {error}', flush=True)
        else:
            failed.append(alias)
    if failed:
        print('Retry these models with --models: ' + ' '.join(failed), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
