#!/usr/bin/env python3
"""Evaluate downloaded Qwen1.5-7B-Chat Raw on the existing frozen allocation.

This is an evaluation-only run. It does not choose an initialization, synthesis
budget, or training recipe for a subsequent CDS experiment.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cds_pipeline as cp
import raw_cds18

REPO_ID = 'Qwen/Qwen1.5-7B-Chat'
REVISION = '5f4f5e69ac7f1d508f8369e977de208b4803444b'
MODEL_NAME = 'Qwen1.5-7B-Chat'
RUN_DIR = 'runs/qwen15_7b_raw_eval_kc18_dev500_tok1024'
CONFIG_PATH = Path(__file__).with_name('configs') / 'cds_gsm8k_qwen15_7b_raw_eval.json'
COUNTS = {'diag': 3500, 'test': 1319, 'dev': 500, 'reserve': 3173,
          'warmup': 0, 'initialization_holdout': 300}


def model_record(destination):
    destination = Path(destination)
    if not destination.is_absolute():
        raise ValueError('Use an absolute local model directory.')
    stamp = cp.read(destination / 'download_provenance.json')
    if stamp != {'repo_id': REPO_ID, 'revision': REVISION}:
        raise ValueError('Downloaded model repository/revision differs from the recorded Qwen1.5 version.')
    index = cp.read(destination / 'model.safetensors.index.json')
    shards = sorted(set(index['weight_map'].values()))
    if len(shards) != 4 or any(Path(name).name != name for name in shards):
        raise ValueError('Expected four local Qwen1.5 safetensors shards.')
    for name in shards:
        path = destination / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty model shard: {name}')
    metadata = ['download_provenance.json', 'config.json', 'generation_config.json',
                'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
                'vocab.json', 'merges.txt']
    return {'repo_id': REPO_ID, 'revision': REVISION,
            'metadata_sha256': {name: cp.file_hash(destination / name) for name in metadata},
            'shard_sizes': {name: (destination / name).stat().st_size for name in shards},
            'verification': 'Metadata hashes and shard sizes; weights are loaded by the inference engine.'}


def definition(model_dir=None):
    source_cfg, source, _ = raw_cds18.load()
    if (source_cfg.get('initialization_mode') != 'raw'
            or source_cfg['pre_finetune_size'] != 0
            or source_cfg['kc_scope'] != 'diagnosis_kcs'
            or source_cfg['inference_prompt'] != 'chat'
            or source_cfg['inference_max_tokens'] != 1024):
        raise ValueError('Expected the existing Raw/18-KC/chat/1024 evaluation protocol.')
    destination = (Path(model_dir).expanduser() if model_dir else
                   Path(source_cfg['student_model']).parent / MODEL_NAME)
    if not destination.is_absolute() or destination.resolve() == Path(source_cfg['student_model']).resolve():
        raise ValueError('Use a distinct absolute directory for Qwen1.5 weights.')
    run = cp.resolve(RUN_DIR)
    source = Path(source).resolve()
    if run == source or run in source.parents or source in run.parents:
        raise ValueError('Use an independent evaluation directory.')
    exports = {name: cp.read(source / 'data' / (name + '.json')) for name in COUNTS}
    if {name: len(rows) for name, rows in exports.items()} != COUNTS:
        raise ValueError('Source allocation counts differ from the frozen experiment.')
    identifiers = [r['item_id'] for rows in exports.values() for r in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Source splits contain repeated item IDs.')
    if any(r['instruction'] != cp.INSTRUCTION for rows in exports.values() for r in rows):
        raise ValueError('Source evaluation instructions differ from the shared prompt.')
    exports.update(vocab=cp.read(source / 'data/vocab.json'),
                   manifest=cp.read(source / 'data/manifest.json'))
    if len(exports['vocab']) != 18:
        raise ValueError('Expected 18 diagnostic KCs.')
    cfg = {**copy.deepcopy(source_cfg), 'run_dir': RUN_DIR,
           'student_repo_id': REPO_ID, 'student_model': str(destination),
           'student_revision_note': f'{REPO_ID}@{REVISION}', 'evaluation_only': True,
           'allocation_entrypoint': 'qwen15_raw_eval.py prepare'}
    protocol = {
        'purpose': 'Raw baseline evaluation only; downstream CDS protocol remains undecided.',
        'config_hash': cp.digest(cfg), 'source_run': str(source),
        'source_config_hash': cp.digest(source_cfg),
        'data_sha256': {name: cp.digest(rows) for name, rows in exports.items()},
        'source_data_file_sha256': {name: cp.file_hash(source / 'data' / (name + '.json'))
                                    for name in exports},
        'model': model_record(destination),
        'helper_sha256': cp.file_hash(__file__),
        'pipeline_sha256': cp.file_hash(cp.__file__),
        'evaluation': {'adapter': None, 'prompt': 'shared instruction with model chat template',
                       'max_output_tokens': 1024, 'temperature': 0,
                       'top_p': 1, 'top_k': 1,
                       'scorer': 'Existing CDS-compatible numerical extractor, unchanged.'},
    }
    artifacts = {run / 'data' / (name + '.json'): rows for name, rows in exports.items()}
    artifacts.update({CONFIG_PATH: cfg, run / 'config.frozen.json': cfg,
                      run / 'evaluation_protocol.json': protocol})
    return cfg, run, artifacts


def validate_artifacts(artifacts, must_exist=False):
    for path, expected in artifacts.items():
        if path.exists():
            if cp.read(path) != expected:
                raise ValueError(f'Evaluation artifact differs: {path}. Preserve the original run.')
        elif must_exist:
            raise ValueError(f'Missing evaluation artifact: {path}. Run prepare first.')


def prepare(model_dir=None):
    cfg, run, artifacts = definition(model_dir)
    validate_artifacts(artifacts)
    for path, value in artifacts.items():
        cp.write(path, value)
    cp.init_run(CONFIG_PATH)
    print(json.dumps({'model': REPO_ID, 'revision': REVISION, 'counts': COUNTS,
                      'adapter': None, 'run_dir': str(run)}, ensure_ascii=False))
    return cfg, run


def evaluate(split, limit=None):
    if split not in ['diag', 'dev', 'test'] or (limit is not None and limit <= 0):
        raise ValueError('Use diag/dev/test and a positive limit, or omit limit for the full split.')
    existing = cp.read(CONFIG_PATH)
    cfg, run, artifacts = definition(existing['student_model'])
    validate_artifacts(artifacts, must_exist=True)
    cp.init_run(CONFIG_PATH)
    tag = f'raw_{split}' if limit is None else f'raw_smoke_{split}_{limit}'
    cp.infer(cfg, run, SimpleNamespace(split=split, tag=tag, adapter=None, tp=1, limit=limit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--model-dir', type=Path)
    p = commands.add_parser('infer')
    p.add_argument('--split', choices=['diag', 'dev', 'test'], required=True)
    p.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.model_dir)
    else:
        evaluate(args.split, args.limit)


if __name__ == '__main__':
    main()
