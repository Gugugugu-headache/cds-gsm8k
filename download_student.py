#!/usr/bin/env python3
"""Download the configured student at an exact HF commit, then record provenance.

Run before prepare freezes the configuration. Requires huggingface_hub, not torch.
HF_TOKEN, when needed, is read by huggingface_hub from the environment/login state.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('configs') / 'cds_gsm8k.json')
    parser.add_argument('--repo-id', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    project_root = Path(__file__).resolve().parents[1]
    run = (project_root / cfg['run_dir']).resolve()
    if (run / 'config.frozen.json').exists():
        raise ValueError('This run is already frozen. Choose a new run_dir before downloading/recording a model.')
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(args.repo_id, revision=args.revision).sha
    destination = Path(cfg['student_model']).expanduser()
    if not destination.is_absolute():
        raise ValueError('student_model must be an absolute path on this machine.')
    stamp = destination / 'download_provenance.json'
    expected = {'repo_id': args.repo_id, 'revision': revision}
    if stamp.exists() and json.loads(stamp.read_text()) != expected:
        raise ValueError('Destination contains a different model revision; use a fresh directory.')
    snapshot_download(repo_id=args.repo_id, revision=revision, local_dir=str(destination),
                      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken', '*.jinja'])
    if not (destination / 'config.json').is_file() or not list(destination.glob('*.safetensors')):
        raise ValueError('Model download incomplete: config.json or safetensors missing.')
    stamp.write_text(json.dumps(expected, indent=2) + '\n', encoding='utf-8')
    cfg['student_revision_note'] = f'{args.repo_id}@{revision}'
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Model ready: {destination}\nHF revision: {revision}\nConfig updated: {config_path}')


if __name__ == '__main__':
    main()
