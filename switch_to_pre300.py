#!/usr/bin/env python3
"""Migrate the server's live config without replacing its recorded model revision.

Only run_dir, pre_finetune_size and development_size are changed. Old runs remain
available; a backup is written before changing the live configuration.
"""
import argparse
import json
from pathlib import Path

from cds_pipeline import file_hash, read, write

SETTINGS = {'run_dir': 'runs/cds_qwen25_3b_pre300',
            'pre_finetune_size': 300, 'development_size': 0}


def update_config(config_path):
    config_path = Path(config_path).resolve()
    current = read(config_path)
    updated = {**current, **SETTINGS}
    project_root = config_path.parents[2]  # project/src/configs/file.json
    frozen = project_root / updated['run_dir'] / 'config.frozen.json'
    if frozen.exists() and read(frozen) != updated:
        raise ValueError('The pre300 run is already frozen with different settings; do not overwrite it.')
    if current == updated:
        print('Configuration already uses pre300; model path/revision preserved.')
        return updated
    backup = config_path.with_name(config_path.stem + '.before_pre300.' + file_hash(config_path)[:12] + '.json')
    write(backup, current)
    temporary = config_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(config_path)
    print(f'Updated: {config_path}\nBackup: {backup}\nSettings: {SETTINGS}')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('configs') / 'cds_gsm8k.json')
    args = parser.parse_args()
    update_config(args.config)


if __name__ == '__main__':
    main()
