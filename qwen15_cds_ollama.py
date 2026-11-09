#!/usr/bin/env python3
"""CDS baseline from Qwen1.5 Raw, using the installed Ollama Q4 teacher.

check/prepare/profile do not call a generation endpoint. Existing Raw responses
and allocations are imported with provenance into a separate run.
"""
import argparse
import copy
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

import cds_pipeline as cp
import ollama_teacher as ot
import qwen15_raw_eval as evaluation
import raw_cds as raw
import raw_cds18

RUN_DIR = 'runs/cds_qwen15_7b_raw_kc18_ollamaq4_q8_tok1024'
CONFIG_PATH = Path(__file__).with_name('configs') / 'cds_gsm8k_qwen15_7b_ollama.json'


@contextmanager
def route():
    previous = raw.CONFIG_PATH, raw.RUN_DIR
    raw.CONFIG_PATH, raw.RUN_DIR = CONFIG_PATH, RUN_DIR
    try:
        yield
    finally:
        raw.CONFIG_PATH, raw.RUN_DIR = previous


def source():
    existing = cp.read(evaluation.CONFIG_PATH)
    cfg, run, artifacts = evaluation.definition(existing['student_model'])
    evaluation.validate_artifacts(artifacts, must_exist=True)
    diagnosis = cp.read(run / 'eval/raw_diag.json')
    baseline = cp.read(run / 'eval/raw_test.json')
    raw.validate_evaluation(diagnosis, cp.read(run / 'data/diag.json'), cfg, 'diag')
    raw.validate_evaluation(baseline, cp.read(run / 'data/test.json'), cfg, 'test')
    return cfg, run, diagnosis


def rebound_diagnosis(result, before_cfg, cfg, path):
    if any(before_cfg[k] != cfg[k] for k in raw.EVAL_FIELDS):
        raise ValueError('Response reuse requires the same student, revision, seed, prompt and token budget.')
    imported = copy.deepcopy(result)
    imported['meta'].update(config_hash=cp.digest(cfg),
                            generation_config_hash=result['meta']['config_hash'],
                            responses_reused=True)
    imported['reuse_provenance'] = {
        'source_path': str(path), 'source_sha256': cp.file_hash(path),
        'generation_config': before_cfg, 'generation_config_hash': cp.digest(before_cfg),
        'operation': 'Rebind inference-irrelevant CDS configuration; preserve every record, response and score.'}
    return imported


def prepare(base=ot.DEFAULT_BASE):
    old_cfg, old, diagnosis = source()
    base = ot.base_url(base)
    identity = ot.inspect(base)  # Metadata only: no /api/chat or GPU preload.
    run = cp.resolve(RUN_DIR)
    if run == old or run in old.parents or old in run.parents:
        raise ValueError('Use an independent CDS run directory.')
    cfg = {**copy.deepcopy(old_cfg), 'run_dir': RUN_DIR, 'evaluation_only': False,
           'quality_min_score': 8, 'allocation_entrypoint': 'qwen15_cds_ollama.py prepare',
           'teacher': {'provider': 'ollama_native', 'model': ot.MODEL,
                       'base_url_default': base, 'timeout_seconds': 1800, 'retries': 2,
                       'identity': identity}}
    exports = {name: cp.read(old / 'data' / (name + '.json'))
               for name in [*evaluation.COUNTS, 'vocab', 'manifest']}
    imported = rebound_diagnosis(diagnosis, old_cfg, cfg, old / 'eval/raw_diag.json')
    raw.validate_evaluation(imported, exports['diag'], cfg, 'diag')
    baseline_path = old / 'eval/raw_test.json'
    protocol = {
        'config': cfg, 'implementation_sha256': raw.implementation(),
        'data_sha256': {name: cp.digest(rows) for name, rows in exports.items()},
        'raw_baseline_path': str(baseline_path),
        'raw_baseline_sha256': cp.file_hash(baseline_path), 'raw_baseline_config': old_cfg,
        'local_cds': {'source_run': str(old), 'source_config_hash': cp.digest(old_cfg),
                      'source_diag_sha256': cp.file_hash(old / 'eval/raw_diag.json'),
                      'source_evaluation_protocol_sha256': cp.file_hash(old / 'evaluation_protocol.json'),
                      'helper_sha256': cp.file_hash(__file__),
                      'transport_sha256': cp.file_hash(ot.__file__)},
        'method': 'CDS global OR rule, per-error teacher diagnosis, synthesis, rewrite/fusion, '
                  'quality filtering, released-code CDSscore; no proposed diagnostic method.',
        'parameter_sources': {
            'quality_min_score': '8, paper Table 9; previous 3B source-oriented run used 10.',
            'weak_thresholds': 'Existing explicit 0.5 accuracy / 0.05 frequency adaptation, OR unchanged.',
            'sample_counts': 'Released math code: 5/global KC and 3/error.',
            'scoring': 'Released-code inverse-log score, smoothing 0.01 and max normalization.',
            'training': 'Inherited frozen shared training settings, 1 nominal epoch; record actual epoch.'},
        'teacher_transport': {'api': 'native /api/chat', 'context_tokens': ot.CONTEXT,
                              'generation_top_k': 0, 'scoring_top_k': 1,
                              'repeat_penalty': 1.05, 'repeat_last_n': -1,
                              'seed': 'Deterministic per normalized request from the experiment seed.',
                              'keep_alive': '5m'},
        'limitations': ['Raw start, 18 KCs, official 1319 test and local quantized teacher are declared adaptations.',
                       'Per-KC accuracy is a conditional outcome statistic, not isolated latent mastery.',
                       'Diagnosis-label fallback to source question labels remains explicitly recorded.',
                       'Quality scores are teacher judgments; no guarantee of mathematical correctness.',
                       'Exact text overlap only; semantic near-duplicates are not detected.',
                       'Random control matches example count, not tokens/cost.',
                       'Legacy Trainer may omit a final partial accumulation group.']}
    artifacts = {run / 'data' / (name + '.json'): rows for name, rows in exports.items()}
    artifacts.update({CONFIG_PATH: cfg, run / 'config.frozen.json': cfg,
                      run / 'raw_protocol.json': protocol, run / 'eval/raw_diag.json': imported})
    evaluation.validate_artifacts(artifacts)
    for path, value in artifacts.items():
        cp.write(path, value)
    cp.init_run(CONFIG_PATH)
    wrong = len(diagnosis['records']) - diagnosis['meta']['correct']
    print(json.dumps({'run_dir': str(run), 'responses_reused': len(diagnosis['records']),
                      'correct': diagnosis['meta']['correct'], 'wrong': wrong,
                      'teacher': ot.MODEL, 'quantization': 'Q4_K_M',
                      'quality_min_score': 8, 'generation_called': False}, ensure_ascii=False))
    return cfg, run


def load():
    with route():
        cfg, run, protocol = raw.load()
    info = protocol['local_cds']
    old_cfg, old, _ = source()
    if (cfg['teacher'].get('provider') != 'ollama_native'
            or cfg['quality_min_score'] != 8
            or info['helper_sha256'] != cp.file_hash(__file__)
            or info['transport_sha256'] != cp.file_hash(ot.__file__)
            or str(old) != info['source_run'] or cp.digest(old_cfg) != info['source_config_hash']
            or cp.file_hash(old / 'eval/raw_diag.json') != info['source_diag_sha256']
            or cp.file_hash(old / 'evaluation_protocol.json') != info['source_evaluation_protocol_sha256']):
        raise ValueError('Local CDS implementation or source evaluation changed.')
    current = cp.read(run / 'eval/raw_diag.json')
    expected = rebound_diagnosis(cp.read(old / 'eval/raw_diag.json'), old_cfg, cfg,
                                 old / 'eval/raw_diag.json')
    if current != expected:
        raise ValueError('Imported Raw responses differ from their verified source.')
    return cfg, run, protocol


def profile():
    cfg, run, _ = load()
    result = raw.verify_raw_diagnosis(cfg, run)
    cp.profile(cfg, run, SimpleNamespace(tag='raw_diag'))
    report = raw_cds18.explain_profile(cp.read(run / 'profile.json')['kcs'], cfg)
    report.update(n=len(result['records']), correct=result['meta']['correct'],
                  evaluation_sha256=cp.file_hash(run / 'eval/raw_diag.json'))
    cp.write(run / 'profile_explained.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['kcs', 'interpretation']},
                     ensure_ascii=False, indent=2))
    return report


def teacher_stage(command, pool=None):
    cfg, run, _ = load()
    if ot.inspect(cfg['teacher']['base_url_default']) != cfg['teacher']['identity']:
        raise ValueError('Ollama server/model metadata changed after prepare.')
    raw.verify_raw_diagnosis(cfg, run)
    from teacher_length_retry import diagnosis_length_retry
    from teacher_generation_retry import generation_length_retry
    from teacher_sample_descriptive_footer import sample_format_compatibility
    with ExitStack() as stack:
        stack.enter_context(ot.transport())
        if command == 'synthesize':
            stack.enter_context(diagnosis_length_retry(cfg, run))
        if command in ['synthesize', 'augment']:
            stack.enter_context(generation_length_retry(cfg, run))
            stack.enter_context(sample_format_compatibility(cfg, run))
        getattr(cp, command)(cfg, run, SimpleNamespace(pool=pool))
    if ot.inspect(cfg['teacher']['base_url_default']) != cfg['teacher']['identity']:
        raise ValueError('Ollama server/model metadata changed during this stage.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['check', 'prepare']:
        p = commands.add_parser(command)
        p.add_argument('--base-url', default=ot.DEFAULT_BASE)
    for command in ['profile', 'synthesize', 'augment', 'select']:
        commands.add_parser(command)
    p = commands.add_parser('quality')
    p.add_argument('--pool', choices=['seed', 'augmented'], required=True)
    for command in ['train', 'compare']:
        p = commands.add_parser(command)
        p.add_argument('--arm', choices=['cds', 'random'], required=True)
    p = commands.add_parser('infer')
    p.add_argument('--split', choices=['test'], required=True)
    p.add_argument('--arm', choices=['cds', 'random'], required=True)
    args = parser.parse_args()
    if args.command == 'check':
        ot.check(args.base_url)
    elif args.command == 'prepare':
        prepare(args.base_url)
    elif args.command == 'profile':
        profile()
    elif args.command in ['synthesize', 'augment', 'quality']:
        teacher_stage(args.command, getattr(args, 'pool', None))
    else:
        load()
        with route():
            raw.main()


if __name__ == '__main__':
    main()
