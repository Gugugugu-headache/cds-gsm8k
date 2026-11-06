#!/usr/bin/env python3
"""Migrate Raw CDS to the fixed 18 diagnostic KCs without regenerating answers.

`prepare` imports the verified 3500 Raw responses into an independent run.
`profile` prints selection reasons. Other CDS stages reuse raw_cds.py.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cds_pipeline as cp
import raw_cds as raw
import initialization_pilot as pilot

RUN_DIR = 'runs/cds_qwen25_3b_raw_kc18_dev500_tok1024'
CONFIG_PATH = Path(__file__).with_name('configs') / 'cds_gsm8k_raw_kc18.json'


@contextmanager
def route18():
    previous = raw.CONFIG_PATH, raw.RUN_DIR
    raw.CONFIG_PATH, raw.RUN_DIR = CONFIG_PATH, RUN_DIR
    try:
        yield
    finally:
        raw.CONFIG_PATH, raw.RUN_DIR = previous


def annotation_source(cfg):
    base = cp.resolve(cfg['data_root'])
    manifest = cp.read(base / 'split_manifest.json')
    for name, expected in manifest['source_sha256'].items():
        if cp.file_hash(cp.resolve(name)) != expected:
            raise ValueError(f'Source data changed: {name}')
    for name, expected in manifest['artifacts'].items():
        if cp.file_hash(base / name) != expected:
            raise ValueError(f'Annotation artifact changed: {name}')
    vocab = cp.read(base / 'kc_vocab.json')['diagnosis_kcs']
    if len(vocab) != 18 or len(set(vocab)) != 18:
        raise ValueError('Expected the fixed 18-KC main diagnosis vocabulary.')
    annotations = {}
    for split in ['D_diag', 'D_remaining', 'D_test']:
        for row in cp.read(base / split / 'data.jsonl'):
            if row['item_id'] in annotations or set(row['diagnosis_kcs']) - set(vocab):
                raise ValueError('Invalid annotation identities or KC labels.')
            annotations[row['item_id']] = row
    return vocab, annotations, cp.file_hash(base / 'split_manifest.json')


def relabel(rows, annotations):
    updated = []
    for row in rows:
        source = annotations[row['item_id']]
        if (row['input'] != source['question'] or row['output'] != source['answer']
                or row['label'] != source['kc_labels']):
            raise ValueError('Existing export differs from the original annotated question.')
        updated.append({**copy.deepcopy(row), 'label': list(source['diagnosis_kcs'])})
    return updated


def reuse_answers(evaluation, old_rows, new_rows, old_cfg, new_cfg, source_path):
    raw.validate_evaluation(evaluation, old_rows, old_cfg, 'diag')
    if len(old_rows) != len(new_rows) or any(old_cfg[k] != new_cfg[k] for k in raw.EVAL_FIELDS):
        raise ValueError('Reuse must preserve model and inference settings.')
    for before, after in zip(old_rows, new_rows):
        if {k: v for k, v in before.items() if k != 'label'} != {
                k: v for k, v in after.items() if k != 'label'}:
            raise ValueError('Reuse can change labels only, not questions, prompts or gold answers.')
    result = copy.deepcopy(evaluation)
    for row, reference in zip(result['records'], new_rows):
        row['label'] = reference['label']
    # config_hash describes this annotation binding, not a new generation event.
    result['meta'].update(config_hash=cp.digest(new_cfg),
                          generation_config_hash=evaluation['meta']['config_hash'],
                          responses_reused=True)
    result['reuse_provenance'] = {
        'source_path': str(source_path), 'source_sha256': cp.file_hash(source_path),
        'generation_config': old_cfg, 'generation_config_hash': cp.digest(old_cfg),
        'annotation_config_hash': cp.digest(new_cfg),
        'operation': 'Only labels rebound to diagnosis_kcs; responses, scores and token counts preserved.'}
    raw.validate_evaluation(result, new_rows, new_cfg, 'diag')
    return result


def prepare():
    old_cfg, old, old_protocol = raw.load()
    if old_cfg['kc_scope'] != 'source_kcs':
        raise ValueError('Expected the existing 28-KC Raw run as migration source.')
    evaluation = raw.verify_raw_diagnosis(old_cfg, old)
    vocab, annotations, source_hash = annotation_source(old_cfg)
    run = cp.resolve(RUN_DIR)
    if run == old or old in run.parents or run in old.parents:
        raise ValueError('Use an independent 18-KC run.')
    cfg = {**copy.deepcopy(old_cfg), 'run_dir': RUN_DIR, 'kc_scope': 'diagnosis_kcs',
           'allocation_entrypoint': 'raw_cds18.py prepare'}
    exports = {name: relabel(cp.read(old / 'data' / (name + '.json')), annotations)
               for name in ['diag', 'test', 'dev', 'reserve', 'warmup', 'initialization_holdout']}
    support = {kc: sum(kc in r['label'] for r in exports['diag']) for kc in vocab}
    if len(exports['diag']) != 3500 or min(support.values()) <= 10:
        raise ValueError('The fixed diagnostic set must cover every one of 18 KCs with >10 items.')
    manifest = copy.deepcopy(cp.read(old / 'data/manifest.json'))
    manifest.update(kc_scope='diagnosis_kcs', kc_count=18,
                    notes=manifest['notes'] + ['Only KC labels changed; every split member/order preserved.'])
    source_path = old / 'eval/raw_diag.json'
    imported = reuse_answers(evaluation, cp.read(old / 'data/diag.json'), exports['diag'],
                             old_cfg, cfg, source_path)
    protocol = copy.deepcopy(old_protocol)
    protocol.update(config=cfg, implementation_sha256=raw.implementation(),
                    data_sha256={k: cp.digest(v) for k, v in
                                 {**exports, 'vocab': vocab, 'manifest': manifest}.items()},
                    kc18_migration={
                        'helper_sha256': cp.file_hash(__file__),
                        'source_manifest_sha256': source_hash,
                        'source_run': str(old), 'source_config_hash': cp.digest(old_cfg),
                        'source_evaluation_sha256': cp.file_hash(source_path),
                        'imported_evaluation_sha256': cp.digest(imported),
                        'support': support,
                        'thresholds_unchanged': True,
                        'note': 'Zero-support and low-coverage are evidence limitations, not demonstrated inability.'})
    artifacts = {run / 'data' / (k + '.json'): v for k, v in exports.items()}
    artifacts.update({CONFIG_PATH: cfg, run / 'config.frozen.json': cfg,
                      run / 'data/vocab.json': vocab, run / 'data/manifest.json': manifest,
                      run / 'raw_protocol.json': protocol, run / 'eval/raw_diag.json': imported})
    for path, value in artifacts.items():
        pilot.check_equal(path, value)
    cp.write(CONFIG_PATH, cfg)
    cp.init_run(CONFIG_PATH)
    for path, value in artifacts.items():
        cp.write(path, value)
    print(json.dumps({'kc_scope': cfg['kc_scope'], 'kc_count': 18, 'responses_reused': 3500,
                      'correct': imported['meta']['correct'], 'run_dir': str(run)}, ensure_ascii=False))
    return cfg, run


def load():
    with route18():
        cfg, run, protocol = raw.load()
    migration = protocol['kc18_migration']
    source = Path(migration['source_run']) / 'eval/raw_diag.json'
    if (cfg['kc_scope'] != 'diagnosis_kcs'
            or migration['helper_sha256'] != cp.file_hash(__file__)
            or migration['source_evaluation_sha256'] != cp.file_hash(source)
            or migration['imported_evaluation_sha256'] != cp.digest(cp.read(run / 'eval/raw_diag.json'))):
        raise ValueError('18-KC migration code or imported/source response artifact changed.')
    return cfg, run, protocol


def explain_profile(kcs, cfg):
    details = {}
    for kc, s in kcs.items():
        zero = s['support'] == 0
        low_acc = not zero and s['accuracy'] <= cfg['weak_accuracy_threshold']
        low_coverage = not zero and s['frequency'] <= cfg['weak_frequency_threshold']
        reasons = ([name for name, flag in [('low_accuracy', low_acc),
                   ('low_coverage', low_coverage), ('zero_support', zero)] if flag])
        expected = (low_acc or low_coverage or zero)
        if zero and cfg['zero_support_policy'] == 'exclude':
            expected = False
        if s['weak'] != expected:
            raise ValueError('Selection reason disagrees with the frozen profile rule.')
        details[kc] = {**s, 'selection_reasons': reasons}
    counts = {reason: sum(reason in s['selection_reasons'] for s in details.values())
              for reason in ['low_accuracy', 'low_coverage', 'zero_support']}
    return {'kc_count': len(kcs), 'selected_for_synthesis': sum(s['weak'] for s in kcs.values()),
            'reason_counts': counts, 'kcs': details,
            'interpretation': 'Low coverage is not low accuracy; zero support is not evidence of inability. '
                              'Reason counts may overlap.'}


def profile():
    cfg, run, _ = load()
    evaluation = raw.verify_raw_diagnosis(cfg, run)
    cp.profile(cfg, run, SimpleNamespace(tag='raw_diag'))
    report = explain_profile(cp.read(run / 'profile.json')['kcs'], cfg)
    report.update(n=len(evaluation['records']), correct=evaluation['meta']['correct'],
                  evaluation_sha256=cp.file_hash(run / 'eval/raw_diag.json'))
    cp.write(run / 'profile_explained.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['kcs', 'interpretation']},
                     ensure_ascii=False, indent=2))
    print('\nSelected KC | support | accuracy | coverage | reasons')
    for kc, s in report['kcs'].items():
        if s['weak']:
            accuracy = 'N/A' if s['accuracy'] is None else f'{s["accuracy"]:.2%}'
            print(f'{kc} | {s["support"]} | {accuracy} | {s["frequency"]:.2%} | '
                  + ', '.join(s['selection_reasons']))
    return report


def main():
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['profile']:
        profile()
    elif len(sys.argv) == 1 or sys.argv[1:] in [['-h'], ['--help']]:
        print(__doc__)
        print('Other stages: synthesize; quality --pool seed|augmented; augment; select; '
              'train --arm cds|random; infer --split test --arm cds|random; compare --arm cds|random')
    else:
        # Reused diagnosis is immutable; never overwrite it with another inference.
        if sys.argv[1] == 'infer' and '--split' in sys.argv:
            pos = sys.argv.index('--split')
            if pos + 1 < len(sys.argv) and sys.argv[pos + 1] == 'diag':
                raise ValueError('The full Raw diagnosis was imported; do not regenerate it.')
        load()
        with route18():
            raw.main()


if __name__ == '__main__':
    main()
