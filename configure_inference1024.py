#!/usr/bin/env python3
"""Move the current experiment to a separate 1024-token run.

Preserve the server's live model revision and all settings except the student
generation limit and run directory. Existing runs and results are never copied
or rewritten; run prepare after migration to reproduce the same data split.
"""
import argparse
import json
from pathlib import Path

from cds_pipeline import file_hash, read, write

PREVIOUS_RUN_DIR = 'runs/cds_qwen25_3b_pre300_torchadam'
RUN_DIR = PREVIOUS_RUN_DIR + '_tok1024'
SETTINGS = {'run_dir': RUN_DIR, 'inference_max_tokens': 1024}


def update_config(config_path):
    config_path = Path(config_path).resolve()
    current = read(config_path)
    if (current.get('pre_finetune_size') != 300
            or current.get('development_size') != 0
            or current.get('run_dir') not in {PREVIOUS_RUN_DIR, RUN_DIR}
            or current.get('inference_max_tokens') not in {512, 1024}):
        raise ValueError('Expected the pre300/torchadam protocol with a 512 or 1024 token limit.')
    project_root = config_path.parents[2]
    source_frozen = project_root / current['run_dir'] / 'config.frozen.json'
    if source_frozen.exists() and read(source_frozen) != current:
        raise ValueError('Current config differs from its frozen run; restore it before migration.')
    updated = {**current, **SETTINGS}
    target_frozen = project_root / RUN_DIR / 'config.frozen.json'
    if target_frozen.exists() and read(target_frozen) != updated:
        raise ValueError('The 1024-token run is frozen with different settings; do not overwrite it.')
    if updated == current:
        print('Already configured for 1024 tokens; existing results preserved.')
        return updated
    backup = config_path.with_name(
        config_path.stem + '.before_tok1024.' + file_hash(config_path)[:12] + '.json')
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
