#!/usr/bin/env python3
"""Create a new run for the native Torch AdamW deployment adaptation."""
import argparse
import json
from pathlib import Path

from cds_pipeline import file_hash, read, write

RUN_DIR = 'runs/cds_qwen25_3b_pre300_torchadam'


def update_config(config_path):
    config_path = Path(config_path).resolve()
    current = read(config_path)
    if current.get('pre_finetune_size') != 300 or current.get('development_size') != 0:
        raise ValueError('Complete the pre300 migration first.')
    updated = {**current, 'run_dir': RUN_DIR}
    project_root = config_path.parents[2]
    frozen = project_root / RUN_DIR / 'config.frozen.json'
    if frozen.exists() and read(frozen) != updated:
        raise ValueError('The torchadam run is frozen with different settings.')
    if current == updated:
        print('Already configured for the torchadam run.')
        return updated
    backup = config_path.with_name(config_path.stem + '.before_torchadam.' + file_hash(config_path)[:12] + '.json')
    write(backup, current)
    temporary = config_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(config_path)
    print(f'Updated run_dir: {RUN_DIR}\nBackup: {backup}')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('configs') / 'cds_gsm8k.json')
    args = parser.parse_args()
    update_config(args.config)


if __name__ == '__main__':
    main()
