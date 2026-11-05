#!/usr/bin/env python3
"""Create an independent Qwen2-7B-Instruct candidate config, preserving the 3B run."""
import argparse
from pathlib import Path

from cds_pipeline import read, write

REPO_ID = 'Qwen/Qwen2-7B-Instruct'
MODEL_NAME = 'Qwen2-7B-Instruct'
CONFIG_NAME = 'cds_gsm8k_qwen2_7b.json'
RUN_DIR = 'runs/cds_qwen2_7b_pre300_torchadam_tok1024'
REVISION_PENDING = f'Pending download: {REPO_ID}; record exact HF commit before prepare.'


def create_config(source_path, output_path=None, model_dir=None):
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve() if output_path else source_path.with_name(CONFIG_NAME)
    if source_path == output_path:
        raise ValueError('Use a separate configuration file; do not replace the previous model config.')
    current = read(source_path)
    if (current.get('pre_finetune_size') != 300 or current.get('development_size') != 0
            or current.get('inference_max_tokens') != 1024 or current.get('inference_prompt') != 'chat'):
        raise ValueError('Expected the current pre300/no-dev/chat/1024 protocol.')
    project_root = source_path.parents[2]
    previous_run = (project_root / current['run_dir']).resolve()
    if previous_run == (project_root / RUN_DIR).resolve():
        raise ValueError('Candidate and source must use different run directories.')
    source_frozen = previous_run / 'config.frozen.json'
    if source_frozen.exists() and read(source_frozen) != current:
        raise ValueError('Source config differs from its frozen run.')
    previous_model = Path(current['student_model']).expanduser()
    if not previous_model.is_absolute():
        raise ValueError('The source student_model must be an absolute model path.')
    destination = Path(model_dir).expanduser() if model_dir else previous_model.parent / MODEL_NAME
    if not destination.is_absolute() or destination.resolve() == previous_model.resolve():
        raise ValueError('Use a distinct, absolute directory for the Qwen2-7B-Instruct weights.')
    updated = {**current, 'run_dir': RUN_DIR, 'student_repo_id': REPO_ID,
               'student_model': str(destination), 'student_revision_note': REVISION_PENDING}
    frozen = project_root / RUN_DIR / 'config.frozen.json'
    if output_path.exists():
        existing = read(output_path)
        # Downloading records a new revision note; preserve that provenance on rerun.
        if {**existing, 'student_revision_note': REVISION_PENDING} != updated:
            raise ValueError('Candidate config already exists with different settings; do not overwrite it.')
        if frozen.exists() and read(frozen) != existing:
            raise ValueError('Candidate configuration differs from its frozen run.')
        print(f'Candidate config already exists; recorded revision preserved: {output_path}')
        return existing
    if frozen.exists():
        raise ValueError('Candidate run already frozen; restore its exact config rather than recreating it.')
    write(output_path, updated)
    print(f'Candidate config: {output_path}\nModel repository: {REPO_ID}\nRun directory: {RUN_DIR}')
    print('Download and record the HF revision before prepare. Previous model config unchanged.')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path,
                        default=Path(__file__).with_name('configs') / 'cds_gsm8k.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    create_config(args.source_config, args.output, args.model_dir)


if __name__ == '__main__':
    main()
