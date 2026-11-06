#!/usr/bin/env python3
"""CDS from Raw, preserving the existing development split and prior experiments.

Prepare is CPU-only. Infer/train/teacher stages must be invoked explicitly.
Use this entry point instead of cds_pipeline.py prepare for this allocation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import cds_pipeline as cp
import initialization_pilot as pilot

RUN_DIR = 'runs/cds_qwen25_3b_raw_dev500_tok1024'
CONFIG_PATH = Path(__file__).with_name('configs') / 'cds_gsm8k_raw.json'
EVAL_FIELDS = ['student_model', 'student_revision_note', 'seed',
               'inference_prompt', 'inference_max_tokens']


def implementation():
    return {name: cp.file_hash(Path(__file__).with_name(name)) for name in
            ['raw_cds.py', 'cds_pipeline.py', 'train_lora.py', 'compare.py',
             'initialization_pilot.py']}


def validate_evaluation(result, expected, cfg, split, adapter=None):
    meta, rows = result['meta'], result['records']
    if (meta['split'] != split or meta['n'] != len(expected) or len(rows) != len(expected)
            or meta['config_hash'] != cp.digest(cfg) or meta['adapter'] != adapter
            or meta['model'] != str(Path(cfg['student_model']).resolve())
            or meta['prompt'] != cfg['inference_prompt']):
        raise ValueError('Evaluation has the wrong model, adapter, split, size or configuration.')
    for row, ref in zip(rows, expected):
        if any(row[k] != ref[k] for k in ref):
            raise ValueError('Evaluation question identities/content differ.')
        if any(row[k] != v for k, v in cp.grade(row['response'], row['output']).items()):
            raise ValueError('Evaluation scores differ from the compatible scorer.')
    correct = sum(r['correct'] for r in rows)
    if (meta['correct'] != correct or meta['accuracy'] != correct / len(rows)
            or meta['no_number'] != sum(r['extraction'] == 'no_number' for r in rows)
            or meta['truncated'] != sum(r['finish_reason'] == 'length' for r in rows)):
        raise ValueError('Evaluation summary does not match its records.')


def prepare():
    plan, pilot_cfg, pilot_run = pilot.load_arm('raw')
    old = Path(plan['source_run'])
    source = cp.read(old / 'config.frozen.json')
    if any(source[k] != pilot_cfg[k] for k in EVAL_FIELDS):
        raise ValueError('Pilot and original Raw evaluation protocols differ.')
    cfg = copy.deepcopy(source)
    cfg.update(run_dir=RUN_DIR, pre_finetune_size=0, development_size=500,
               initialization_mode='raw', reserved_initialization_size=300,
               allocation_entrypoint='raw_cds.py prepare')
    run = cp.resolve(RUN_DIR)
    if run == old or run == pilot_run or old in run.parents or pilot_run in run.parents:
        raise ValueError('Use an independent Raw run directory.')
    exports = {name: cp.read(pilot_run / 'data' / (name + '.json'))
               for name in ['diag', 'test', 'dev', 'reserve']}
    exports.update(warmup=[], initialization_holdout=cp.read(pilot_run / 'data/warmup.json'))
    counts = {k: len(v) for k, v in exports.items()}
    if counts != {'diag': 3500, 'test': 1319, 'dev': 500, 'reserve': 3173,
                  'warmup': 0, 'initialization_holdout': 300}:
        raise ValueError('Unexpected inherited data allocation.')
    ids = [r['item_id'] for rows in exports.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated item IDs in Raw allocation.')
    baseline = old / 'eval/raw_test.json'
    validate_evaluation(cp.read(baseline), exports['test'], source, 'test')
    manifest = copy.deepcopy(cp.read(pilot_run / 'data/manifest.json'))
    manifest.update(counts=counts, ids={k: [r['item_id'] for r in v] for k, v in exports.items()},
                    selection_method='Reuse frozen pilot allocation; reserve old warmup as initialization_holdout',
                    notes=['Original diagnostic/test data and pilot dev identities preserved.',
                           'Former 300 initialization examples are reserved, not trained in the Raw route.',
                           'No reshuffle and no test-based data selection.'])
    vocab = cp.read(pilot_run / 'data/vocab.json')
    protocol = {
        'config': cfg, 'implementation_sha256': implementation(),
        'allocation_source': str(pilot_run),
        'data_sha256': {k: cp.digest(v) for k, v in
                       {**exports, 'vocab': vocab, 'manifest': manifest}.items()},
        'raw_baseline_path': str(baseline), 'raw_baseline_sha256': cp.file_hash(baseline),
        'raw_baseline_config': source,
        'initialization_decision': 'User chose common Raw starts for the multi-model research design; '
                                   'the earlier development pilot selected lr3e5 and remains recorded.',
        'training_decision': 'Retain original downstream CDS training settings from the source config; '
                             'do not use initialization-pilot batch20/constant-LR settings.',
        'limitations': ['Model selection and initialization pilots previously observed test results.',
                       'Legacy Trainer may omit a final partial accumulation group; record actual epoch.',
                       'Random control matches example count, not tokens or teacher costs.',
                       'Exact question overlap is filtered; semantic near-duplicates are not detected.']}
    for path, value in [(CONFIG_PATH, cfg), (run / 'config.frozen.json', cfg),
                        (run / 'raw_protocol.json', protocol), (run / 'data/manifest.json', manifest)]:
        pilot.check_equal(path, value)
    for name, rows in exports.items():
        pilot.check_equal(run / 'data' / (name + '.json'), rows)
    cp.write(CONFIG_PATH, cfg)
    cp.init_run(CONFIG_PATH)
    for name, rows in exports.items():
        cp.write(run / 'data' / (name + '.json'), rows)
    cp.write(run / 'data/vocab.json', vocab)
    cp.write(run / 'data/manifest.json', manifest)
    cp.write(run / 'raw_protocol.json', protocol)
    print(json.dumps(counts))
    print(f'Raw config: {CONFIG_PATH}\nRun: {run}\nExisting Raw test referenced: {baseline}')
    return cfg, run


def load():
    cfg, run = cp.init_run(CONFIG_PATH)
    protocol = cp.read(run / 'raw_protocol.json')
    if (cfg != protocol['config'] or cfg['pre_finetune_size'] != 0
            or cfg.get('initialization_mode') != 'raw'
            or protocol['implementation_sha256'] != implementation()):
        raise ValueError('Raw configuration/implementation changed after freezing.')
    for name, expected in protocol['data_sha256'].items():
        if cp.digest(cp.read(run / 'data' / (name + '.json'))) != expected:
            raise ValueError(f'Raw data changed: {name}')
    if cp.file_hash(protocol['raw_baseline_path']) != protocol['raw_baseline_sha256']:
        raise ValueError('Original Raw test artifact changed.')
    return cfg, run, protocol


def verify_raw_diagnosis(cfg, run):
    result = cp.read(run / 'eval/raw_diag.json')
    validate_evaluation(result, cp.read(run / 'data/diag.json'), cfg, 'diag')
    return result


def select_data(cfg, run):
    # Stronger exclusion for this adapted experiment: diagnosis and reserved
    # initialization questions also cannot be copied verbatim into training.
    blocked = {cp.normalized(r['input']) for name in
               ['diag', 'test', 'dev', 'initialization_holdout']
               for r in cp.read(run / 'data' / (name + '.json'))}
    rows = cp.read(run / 'seed_quality.json') + cp.read(run / 'augmented_quality.json')
    seen, unique, removed = set(), [], []
    for row in rows:
        key = cp.normalized(row['input'])
        if key in blocked or key in seen:
            removed.append({'item_id': row['item_id'],
                            'reason': 'held_out_overlap' if key in blocked else 'duplicate'})
        else:
            seen.add(key)
            unique.append(row)
    profile = cp.read(run / 'profile.json')
    if profile['evaluation_sha256'] != cp.file_hash(run / 'eval/raw_diag.json'):
        raise ValueError('Profile and Raw diagnosis differ.')
    kcs = profile['kcs']
    if any(set(r['label']) - set(kcs) for r in unique):
        raise ValueError('Candidate labels outside vocabulary.')
    scored, threshold, weights = cp.score_pool(unique, kcs, cfg)
    selected = [r for r in scored if r['cds_score'] >= threshold]
    if not selected:
        raise ValueError('No CDS training data selected.')
    control = random.Random(cfg['seed']).sample(scored, len(selected))
    for name, value in [('candidate_pool.json', scored), ('train_cds.json', selected),
                        ('train_random.json', control)]:
        cp.write(run / name, value)
    report = {'before_dedup': len(rows), 'after_dedup': len(unique), 'selected': len(selected),
              'threshold': threshold, 'weights': weights, 'removed': removed,
              'training_sha256': {arm: cp.file_hash(run / f'train_{arm}.json')
                                  for arm in ['cds', 'random']},
              'control': 'same quality-filtered pool and example count; tokens/cost not matched',
              'semantic_overlap_checked': False}
    cp.write(run / 'selection.json', report)
    print(json.dumps({'candidates': len(unique), 'selected': len(selected), 'removed': len(removed)}))


def train(cfg, run, arm):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This Raw pilot uses a single GPU for both training arms.')
    selected = cp.read(run / 'selection.json')
    data = run / f'train_{arm}.json'
    if cp.file_hash(data) != selected['training_sha256'][arm]:
        raise ValueError('Selected training data changed.')
    if len(cp.read(data)) < cfg['training']['global_batch_size']:
        raise ValueError('Fewer than one full effective batch; inspect synthesis/selection before training.')
    checkpoint = run / 'checkpoints' / arm
    import train_lora
    original_argv = sys.argv
    try:
        # No --adapter: both CDS and random start from the same Raw checkpoint.
        sys.argv = ['train_lora.py', '--config', str(CONFIG_PATH),
                    '--data', str(data), '--output', str(checkpoint)]
        train_lora.main()
    finally:
        sys.argv = original_argv
    metadata = cp.read(checkpoint / 'experiment.json')
    if metadata['parent_adapter'] is not None:
        raise ValueError('Training did not start from Raw.')
    cp.write(run / f'{arm}_training_completion.json', {
        'config_hash': cp.digest(cfg), 'training_sha256': cp.file_hash(data),
        'actual_epoch': metadata['training_metrics']['epoch'],
        'checkpoint_sha256': pilot.checkpoint_files(checkpoint), 'initialization': 'raw'})
    print(f'Training finished; actual epoch={metadata["training_metrics"]["epoch"]}.')


def adapter_path(cfg, run, arm):
    if arm == 'raw':
        return None
    path = run / 'checkpoints' / arm
    complete = cp.read(run / f'{arm}_training_completion.json')
    if (complete['config_hash'] != cp.digest(cfg) or complete['initialization'] != 'raw'
            or complete['checkpoint_sha256'] != pilot.checkpoint_files(path)
            or complete['training_sha256'] != cp.file_hash(run / f'train_{arm}.json')):
        raise ValueError('Training completion/checkpoint differs.')
    return str(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    p = commands.add_parser('infer')
    p.add_argument('--split', choices=['diag', 'test'], required=True)
    p.add_argument('--arm', choices=['raw', 'cds', 'random'], default='raw')
    for stage in ['profile', 'synthesize', 'augment', 'select']:
        commands.add_parser(stage)
    p = commands.add_parser('quality')
    p.add_argument('--pool', choices=['seed', 'augmented'], required=True)
    for stage in ['train', 'compare']:
        p = commands.add_parser(stage)
        p.add_argument('--arm', choices=['cds', 'random'], required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
        return
    cfg, run, protocol = load()
    if args.command == 'infer':
        if args.split == 'diag' and args.arm != 'raw':
            parser.error('This diagnosis must describe Raw, not a trained model.')
        if args.split == 'test' and args.arm == 'raw':
            parser.error('Reuse the recorded Raw test baseline; no repeat is needed.')
        cp.infer(cfg, run, SimpleNamespace(split=args.split, tag=f'{args.arm}_{args.split}',
                 adapter=adapter_path(cfg, run, args.arm), tp=1, limit=None))
    elif args.command == 'train':
        train(cfg, run, args.arm)
    elif args.command == 'compare':
        from compare import compare
        after = cp.read(run / 'eval' / f'{args.arm}_test.json')
        validate_evaluation(after, cp.read(run / 'data/test.json'), cfg, 'test',
                            adapter_path(cfg, run, args.arm))
        report = compare(cp.read(protocol['raw_baseline_path']), after)
        cp.write(run / f'comparison_raw_vs_{args.arm}.json', report)
        print(json.dumps(report, indent=2))
    else:
        verify_raw_diagnosis(cfg, run)
        if args.command == 'profile':
            cp.profile(cfg, run, SimpleNamespace(tag='raw_diag'))
            rows = cp.read(run / 'eval/raw_diag.json')['records']
            kcs = cp.read(run / 'profile.json')['kcs']
            print(json.dumps({'n': len(rows), 'wrong': sum(not r['correct'] for r in rows),
                              'weak_kcs': sum(r['weak'] for r in kcs.values())}))
        elif args.command == 'select':
            select_data(cfg, run)
        else:
            getattr(cp, args.command)(cfg, run, args)


if __name__ == '__main__':
    main()
