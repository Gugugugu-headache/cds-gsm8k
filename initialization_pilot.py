#!/usr/bin/env python3
"""Fixed, development-only initialization pilot; preserve all previous runs.

Use `prepare` once, then `train` (under single-GPU torchrun), `eval`, and
`select`. GPU operations are explicit. No teacher calls or test evaluation.
"""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import cds_pipeline as cp

PILOT_RUN = 'runs/qwen25_3b_init_dev500_v1'
ARMS = {'raw': None, 'lr1e5': 1e-5, 'lr3e5': 3e-5}
RULE = ('Maximum dev correct count; ties: maximum hash extraction count; '
        'remaining ties: raw, lr1e5, lr3e5. No test-based selection.')


def implementation():
    return {name: cp.file_hash(Path(__file__).with_name(name)) for name in
            ['initialization_pilot.py', 'cds_pipeline.py', 'train_lora.py']}


def check_equal(path, value):
    if Path(path).exists() and cp.read(path) != value:
        raise ValueError(f'Existing artifact differs: {path}; do not overwrite this pilot.')


def verify_source(cfg):
    if (Path(cfg['student_model']).name != 'Qwen2.5-3B-Instruct'
            or not Path(cfg['student_model']).is_absolute()
            or cfg['seed'] != 42 or cfg['pre_finetune_size'] != 300
            or cfg['development_size'] != 0 or cfg['inference_prompt'] != 'chat'
            or cfg['inference_max_tokens'] != 1024):
        raise ValueError('Expected the existing Qwen2.5-3B pre300/dev0/seed42/chat/1024 config.')
    if not re.fullmatch(r'Qwen/Qwen2\.5-3B-Instruct@[0-9a-f]{40}', cfg['student_revision_note']):
        raise ValueError('Record the actual downloaded 3B HF commit before this pilot.')
    old = cp.resolve(cfg['run_dir'])
    if cp.read(old / 'config.frozen.json') != cfg:
        raise ValueError('Source configuration differs from the frozen run.')
    for path, expected in cp.read(old / 'implementation.frozen.json').items():
        if cp.file_hash(cp.resolve(path)) != expected:
            raise ValueError(f'Source implementation changed: {path}')
    rows = {name: cp.read(old / 'data' / (name + '.json'))
            for name in ['warmup', 'dev', 'reserve', 'diag', 'test']}
    if {k: len(v) for k, v in rows.items()} != {
            'warmup': 300, 'dev': 0, 'reserve': 3673, 'diag': 3500, 'test': 1319}:
        raise ValueError('Unexpected previous allocation.')
    ids = [r['item_id'] for records in rows.values() for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Previous data contain repeated IDs.')
    return old, rows


def prepare(source_config):
    source_config = Path(source_config).resolve()
    source = cp.read(source_config)
    old, previous = verify_source(source)
    base = cp.resolve(PILOT_RUN)
    if base == old or old in base.parents or base in old.parents:
        raise ValueError('Pilot and previous run must be independent.')
    configs = {}
    config_paths = {}
    for arm, rate in ARMS.items():
        cfg = copy.deepcopy(source)
        cfg.update(run_dir=str(Path(PILOT_RUN) / arm), development_size=500)
        # Batch 20 divides 300 exactly: no remainder update in Trainer 4.51.3.
        # Equal min/max gives a genuinely constant LR in the existing WarmupLR.
        rate = rate or 1e-5  # Raw is never trained; keeps the config schema intact.
        cfg['training'].update(epochs=1, global_batch_size=20, micro_batch_size=2,
                               learning_rate=rate, warmup_min_lr=rate,
                               warmup_max_lr=rate, warmup_num_steps=2)
        path = source_config.with_name(f'cds_init_dev500_{arm}.json')
        configs[arm], config_paths[arm] = cfg, str(path)
        check_equal(path, cfg)
        check_equal(cp.resolve(cfg['run_dir']) / 'config.frozen.json', cfg)
    allocation = {**previous, 'dev': previous['reserve'][:500],
                  'reserve': previous['reserve'][500:]}
    plan = {
        'protocol': 'qwen25_3b_init_dev500_v1', 'source_run': str(old),
        'source_config_sha256': cp.file_hash(source_config),
        'implementation_sha256': implementation(), 'configs': configs,
        'config_paths': config_paths,
        'data_sha256': {name: cp.digest(rows) for name, rows in allocation.items()},
        'selection_rule': RULE, 'tie_order': list(ARMS),
        'training': {'samples': 300, 'epochs': 1, 'global_batch': 20,
                     'micro_batch': 2, 'gradient_accumulation': 10,
                     'updates': 15, 'world_size': 1},
        'limitations': [
            'Exploratory follow-up after observing previous test scores; this does not erase that history.',
            'Development comparison uses one training seed; not evidence of seed robustness.',
            'Batch 20 avoids remainder rounding only for this 300-example single-GPU pilot; '
            'it is not a general Trainer remainder fix.',
            'Old batch32/high-LR M0 is not included in selection; changes cannot isolate the LR effect.']}
    check_equal(base / 'protocol.json', plan)
    # Validate exports in memory before creating any new frozen runs/configs.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cp.prepare(configs['raw'], Path(tmp))
        for name, expected in allocation.items():
            if cp.read(Path(tmp) / 'data' / (name + '.json')) != expected:
                raise ValueError(f'New {name} differs from the declared allocation.')
    cp.write(base / 'protocol.json', plan)
    for arm, cfg in configs.items():
        cp.write(config_paths[arm], cfg)
        actual, run = cp.init_run(config_paths[arm])
        cp.prepare(actual, run)
    print(f'Pilot prepared: {base}\nSelection rule: {RULE}')
    return plan


def load_arm(arm):
    plan = cp.read(cp.resolve(PILOT_RUN) / 'protocol.json')
    if (plan['implementation_sha256'] != implementation()
            or plan['selection_rule'] != RULE or plan['tie_order'] != list(ARMS)):
        raise ValueError('Pilot implementation or decision rule changed; keep this run frozen.')
    cfg = cp.read(plan['config_paths'][arm])
    if cfg != plan['configs'][arm]:
        raise ValueError('Arm configuration changed after pilot registration.')
    cfg, run = cp.init_run(plan['config_paths'][arm])
    for name, expected in plan['data_sha256'].items():
        if cp.digest(cp.read(run / 'data' / (name + '.json'))) != expected:
            raise ValueError(f'Pilot data changed: {name}')
    return plan, cfg, run


def validate_training_layout(cfg, rows, world):
    t = cfg['training']
    if (world != 1 or len(rows) != 300 or t['epochs'] != 1
            or t['global_batch_size'] != 20 or t['micro_batch_size'] != 2):
        raise ValueError('Pilot requires one GPU, 300 examples, one epoch, micro2/global20.')
    batches = math.ceil(len(rows) / t['micro_batch_size'])
    accumulation = t['global_batch_size'] // t['micro_batch_size']
    if batches % accumulation or batches // accumulation != 15:
        raise ValueError('Incomplete gradient accumulation group.')
    return batches // accumulation


def checkpoint_files(checkpoint):
    weights = sorted(checkpoint.glob('adapter_model.*'))
    if not weights or not (checkpoint / 'adapter_config.json').is_file():
        raise ValueError('Adapter files are missing.')
    files = weights + [checkpoint / 'adapter_config.json', checkpoint / 'experiment.json']
    return {p.name: cp.file_hash(p) for p in files}


def train(arm):
    if arm == 'raw':
        raise ValueError('Raw arm must not be trained.')
    plan, cfg, run = load_arm(arm)
    validate_training_layout(cfg, cp.read(run / 'data/warmup.json'),
                             int(os.environ.get('WORLD_SIZE', '1')))
    expected_versions = {'transformers': '4.51.3', 'accelerate': '1.6.0', 'deepspeed': '0.16.7'}
    versions = {name: importlib.metadata.version(name) for name in expected_versions}
    if versions != expected_versions:
        raise ValueError(f'Use the validated cdi-train versions: {expected_versions}; found {versions}')
    checkpoint = run / 'checkpoints/warmup'
    import train_lora
    original_argv = sys.argv
    try:
        sys.argv = ['train_lora.py', '--config', plan['config_paths'][arm],
                    '--data', str(run / 'data/warmup.json'), '--output', str(checkpoint)]
        train_lora.main()
    finally:
        sys.argv = original_argv
    metadata = cp.read(checkpoint / 'experiment.json')
    if not math.isclose(metadata['training_metrics']['epoch'], 1.0, abs_tol=1e-9):
        raise ValueError('Training did not complete one epoch; inspect logs before evaluating.')
    cp.write(run / 'training_completion.json', {
        'expected_updates': 15, 'actual_epoch': metadata['training_metrics']['epoch'],
        'checkpoint_sha256': checkpoint_files(checkpoint), 'versions': versions,
        'config_hash': cp.digest(cfg), 'trained_from': 'raw', 'seed': cfg['seed']})
    print('Pilot training completed one epoch. Next: evaluate this arm on dev only.')


def verify_checkpoint(arm, cfg, run):
    if arm == 'raw':
        return None
    checkpoint = run / 'checkpoints/warmup'
    complete = cp.read(run / 'training_completion.json')
    if (complete['config_hash'] != cp.digest(cfg) or complete['actual_epoch'] != 1.0
            or complete['checkpoint_sha256'] != checkpoint_files(checkpoint)):
        raise ValueError('Checkpoint or training completion metadata changed.')
    return str(checkpoint.resolve())


def evaluate(arm):
    _, cfg, run = load_arm(arm)
    adapter = verify_checkpoint(arm, cfg, run)
    cp.infer(cfg, run, SimpleNamespace(split='dev', tag='dev', adapter=adapter, tp=1, limit=None))


def summarize(result, rows, cfg, adapter):
    meta, records = result['meta'], result['records']
    if (meta['split'] != 'dev' or meta['n'] != 500 or len(records) != 500
            or meta['config_hash'] != cp.digest(cfg) or meta['adapter'] != adapter
            or meta['model'] != str(Path(cfg['student_model']).resolve())
            or meta['prompt'] != cfg['inference_prompt']):
        raise ValueError('Expected a full dev evaluation with this arm configuration and checkpoint.')
    for actual, expected in zip(records, rows):
        if any(actual[key] != expected[key] for key in expected):
            raise ValueError('Development identities/prompts/answers differ.')
        score = cp.grade(actual['response'], actual['output'])
        if any(actual[key] != value for key, value in score.items()):
            raise ValueError('Stored scores differ from the frozen compatible scorer.')
    correct = sum(r['correct'] for r in records)
    hashed = sum(r['extraction'] == 'hash' for r in records)
    no_number = sum(r['extraction'] == 'no_number' for r in records)
    truncated = sum(r['finish_reason'] == 'length' for r in records)
    if (meta['correct'] != correct or meta['accuracy'] != correct / 500
            or meta['no_number'] != no_number or meta['truncated'] != truncated):
        raise ValueError('Evaluation summary disagrees with records.')
    return {'n': 500, 'correct': correct, 'accuracy': correct / 500,
            'hash_count': hashed, 'hash_coverage': hashed / 500,
            'no_number': no_number, 'truncated': truncated}


def choose(metrics):
    if set(metrics) != set(ARMS):
        raise ValueError('All three predefined arms must finish before selection.')
    return max(ARMS, key=lambda arm: (metrics[arm]['correct'], metrics[arm]['hash_count'],
                                     -list(ARMS).index(arm)))


def select():
    metrics, evaluation_hashes = {}, {}
    for arm in ARMS:
        _, cfg, run = load_arm(arm)
        adapter = verify_checkpoint(arm, cfg, run)
        path = run / 'eval/dev.json'
        metrics[arm] = summarize(cp.read(path), cp.read(run / 'data/dev.json'), cfg, adapter)
        evaluation_hashes[arm] = cp.file_hash(path)
    report = {'selected_arm': choose(metrics), 'selection_rule': RULE,
              'metrics': metrics, 'evaluation_sha256': evaluation_hashes,
              'scope': 'Development-only initialization selection; not a final test result.'}
    cp.write(cp.resolve(PILOT_RUN) / 'selection.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source-config', type=Path,
                   default=Path(__file__).with_name('configs') / 'cds_gsm8k.json')
    p = commands.add_parser('train')
    p.add_argument('--arm', choices=['lr1e5', 'lr3e5'], required=True)
    p = commands.add_parser('eval')
    p.add_argument('--arm', choices=list(ARMS), required=True)
    commands.add_parser('select')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source_config)
    elif args.command == 'train':
        train(args.arm)
    elif args.command == 'eval':
        evaluate(args.arm)
    else:
        select()


if __name__ == '__main__':
    main()
