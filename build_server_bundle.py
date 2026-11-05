#!/usr/bin/env python3
"""Build an allowlisted, relocatable upload bundle without models or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def build(output):
    files = {}
    for path in CODE.rglob('*'):
        relative = path.relative_to(CODE)
        if any(part in {'dist', 'runs', '__pycache__', '.git'} or part.startswith('.venv') for part in relative.parts):
            continue
        if path.is_file() and (path.suffix in {'.py', '.md', '.txt', '.json'} or path.name == '.gitignore'):
            files['src/' + relative.as_posix()] = path
    for path in (ROOT / 'DP-CDI/data').rglob('*'):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    manifest = json.loads((ROOT / 'DP-CDI/data/split_manifest.json').read_text())
    upstream = {'cds-04D1/4_Q_score/Qscore.py', 'cds-04D1/finetune/code/config.py',
                'cds-04D1/finetune/code/dataset.py', 'cds-04D1/infer/code/infer_math.py'}
    for name in set(manifest['source_sha256']) | upstream:
        files[name] = ROOT / name
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
        import io
        content = (json.dumps({'files': hashes}, indent=2, ensure_ascii=False) + '\n').encode()
        info = tarfile.TarInfo('bundle_manifest.json')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{checksum}  {output.name}\n')
    print(f'{output}\n{len(files)} files; {output.stat().st_size / 1024**2:.2f} MiB\nSHA256: {checksum}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CODE / 'dist/cds_server_bundle.tar.gz')
    args = parser.parse_args()
    build(args.output)


if __name__ == '__main__':
    main()
