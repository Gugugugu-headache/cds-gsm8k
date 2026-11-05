#!/usr/bin/env python3
"""Adopt a sibling Git checkout as server src, preserving config and frozen runs.

Run from the newly cloned checkout. The previous src directory becomes a dated
backup. Data, models and run artifacts in the project root are not moved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def adopt(project_root, checkout):
    project_root = Path(project_root).resolve()
    checkout = Path(checkout).resolve()
    live = project_root / 'src'
    if checkout.parent != project_root or checkout == live:
        raise ValueError('Clone into a sibling of src inside the existing project root.')
    if not (checkout / '.git').is_dir():
        raise ValueError('The new checkout must be a Git clone.')
    if not live.is_dir() or live.is_symlink() or (live / '.git').exists():
        raise ValueError('Expected an existing non-Git src directory; already adopted repositories use git pull.')
    config = live / 'configs/cds_gsm8k.json'
    cfg = read(config)
    run = (project_root / cfg['run_dir']).resolve()
    if not run.is_relative_to(project_root):
        raise ValueError('The configured run must be inside this project root.')
    # Preserve the exact two code files whose hashes are frozen by the pipeline.
    for name in ['cds_pipeline.py', 'train_lora.py']:
        if (live / name).read_bytes() != (checkout / name).read_bytes():
            raise ValueError(f'{name} differs; resolve the frozen implementation before adopting the checkout.')
    frozen = run / 'config.frozen.json'
    if frozen.exists() and read(frozen) != cfg:
        raise ValueError('Live configuration differs from its frozen run.')
    implementation = run / 'implementation.frozen.json'
    if implementation.exists():
        for name, expected in read(implementation).items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid frozen implementation path.')
            path = checkout / Path(*relative.parts[1:]) if relative.parts[0] == 'src' else project_root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen implementation mismatch: {name}')
    # Collect and validate before copying any machine-local configuration.
    copies = []
    for source in (live / 'configs').rglob('*.json'):
        if source.name.endswith('.example.json'):
            continue
        target = checkout / 'configs' / source.relative_to(live / 'configs')
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f'Configuration collision: {target.name}')
        copies.append((source, target))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = project_root / ('src.before_git_' + suffix)
    live.rename(backup)
    try:
        checkout.rename(live)
    except OSError:
        backup.rename(live)
        raise
    print(f'Git checkout: {live}\nPrevious src backup: {backup}')
    print('Live configuration preserved; project data, models and runs unchanged.')
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    adopt(args.project_root, Path(__file__).resolve().parent)


if __name__ == '__main__':
    main()
