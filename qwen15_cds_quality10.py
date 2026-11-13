#!/usr/bin/env python3
"""Reuse verified Q8 seed scores in a separate Q10 run; discard only reported augmentation caches."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import re

import cds_pipeline as cp
import ollama_teacher as ot
import qwen15_cds_ollama as job
import qwen15_cds_ollama_compat as repeat_fix
import qwen15_cds_ollama_samples as entry
import qwen15_raw_eval as evaluation
import raw_cds as raw

SOURCE_CONFIG = job.CONFIG_PATH
SOURCE_RUN = job.RUN_DIR
SOURCE_LOAD = job.load
RUN_DIR = 'runs/cds_qwen15_7b_raw_kc18_ollamaq4_q10_tok1024'
CONFIG_PATH = Path(__file__).with_name('configs') / 'cds_gsm8k_qwen15_7b_ollama_q10.json'
INTERRUPTED_KEYS = {
    '3d77736b44ce32cea118eea28d3c74477a37c98500465d01b0bc65ca5b55e2af',
    '753e0d6bf93ab38535b1d14d0e0d89481d75713e202683387021c7ea9e411e77',
}


@contextmanager
def locations(config, run):
    old = job.CONFIG_PATH, job.RUN_DIR
    job.CONFIG_PATH, job.RUN_DIR = config, run
    try:
        yield
    finally:
        job.CONFIG_PATH, job.RUN_DIR = old


def source():
    with locations(SOURCE_CONFIG, SOURCE_RUN):
        return SOURCE_LOAD()


def payload(cfg, messages, scoring=False):
    result = {'model': cfg['teacher']['model'], 'messages': messages,
              'temperature': 0 if scoring else 0.5, 'top_p': 1 if scoring else 0.8,
              'max_tokens': 512 if scoring else 4096, 'stream': False}
    if scoring:
        result['top_k'] = 1
    return result


def cache_key(cfg, request):
    return cp.digest({'url': cfg['teacher']['base_url_default'], 'payload': request})


def verified_cache(cfg, run, key, request, complete=True):
    path = run / 'teacher_cache' / (key + '.json')
    data = cp.read(path)
    native = repeat_fix.ContextRepeatTeacher(cfg, run).native_request(request)
    response = data.get('native_response', {})
    if (cache_key(cfg, request) != key or data.get('request') != request
            or data.get('url') != cfg['teacher']['base_url_default']
            or data.get('native_request') != native
            or data.get('config_hash') != cp.digest(cfg)
            or data.get('model_digest') != cfg['teacher']['identity']['digest']
            or response.get('done') is not True
            or response.get('message', {}).get('content') != data.get('content')
            or response.get('done_reason') != data.get('finish_reason')
            or (complete and data.get('finish_reason') != 'stop')):
        raise ValueError(f'Source teacher cache differs: {key}')
    return data, path


def verified_scores(cfg, run, samples):
    accepted = cp.read(run / 'seed_quality.json')
    rejected = cp.read(run / 'seed_quality.rejected.json')
    by_id = {}
    for row in accepted + rejected:
        if row['item_id'] in by_id:
            raise ValueError('Repeated quality item ID.')
        by_id[row['item_id']] = row
    if len({row['item_id'] for row in samples}) != len(samples) or len(by_id) != len(samples):
        raise ValueError('Seed synthesis and quality records are not one-to-one.')
    hashes, ordered = {}, []
    prompts = cp.prompt_module()
    for sample in samples:
        row = by_id.get(sample['item_id'])
        if row is None or {k: v for k, v in row.items()
                           if k not in ['quality_score', 'quality_request']} != sample:
            raise ValueError('Quality sample content differs from synthesis.')
        prompt = prompts.prompt4filter.format(
            question=sample['input'], answer=sample['output'], label=sample['label'])
        request = payload(cfg, [cp.user_message(prompt)], scoring=True)
        key = row['quality_request']
        cached, path = verified_cache(cfg, run, key, request)
        matches = re.findall(r'\bScore:\s*(10|[0-9])(?![\d.])', cached['content'])
        score = int(matches[-1]) if len(matches) == 1 else None
        if row['quality_score'] != score:
            raise ValueError('Saved quality score differs from teacher response.')
        hashes[str(path)] = cp.file_hash(path)
        ordered.append(row)
    expected = [r for r in ordered if r['quality_score'] is not None
                and r['quality_score'] >= cfg['quality_min_score']]
    accepted_ids = {r['item_id'] for r in expected}
    if accepted != expected or rejected != [r for r in ordered if r['item_id'] not in accepted_ids]:
        raise ValueError('Source Q8 acceptance or ordering differs.')
    return ordered, hashes


def file_hash_for_json(value):
    data = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def definition():
    old_cfg, old, old_protocol = source()
    run = cp.resolve(RUN_DIR)
    if old_cfg['quality_min_score'] != 8 or run == old or run in old.parents or old in run.parents:
        raise ValueError('Use an independent Q10 run derived from the existing Q8 run.')
    samples = cp.read(old / 'synthesis.json')
    scored, cache_hashes = verified_scores(old_cfg, old, samples)
    cfg = {**copy.deepcopy(old_cfg), 'run_dir': RUN_DIR, 'quality_min_score': 10,
           'allocation_entrypoint': 'qwen15_cds_quality10.py prepare'}
    accepted = [r for r in scored if r['quality_score'] == 10]
    if not accepted:
        raise ValueError('No score-10 seeds remain.')
    exports = {name: cp.read(old / 'data' / (name + '.json')) for name in old_protocol['data_sha256']}
    diagnosis = copy.deepcopy(cp.read(old / 'eval/raw_diag.json'))
    diagnosis['meta']['config_hash'] = cp.digest(cfg)
    raw.validate_evaluation(diagnosis, exports['diag'], cfg, 'diag')
    profile = copy.deepcopy(cp.read(old / 'profile.json'))
    if profile['evaluation_sha256'] != cp.file_hash(old / 'eval/raw_diag.json'):
        raise ValueError('Source diagnosis profile changed.')
    if profile['kcs'] != cp.profile_rows(diagnosis['records'], exports['vocab'], cfg):
        raise ValueError('Source knowledge-component statistics differ.')
    profile['evaluation_sha256'] = file_hash_for_json(diagnosis)
    protocol = copy.deepcopy(old_protocol)
    protocol['config'] = cfg
    protocol['parameter_sources']['quality_min_score'] = (
        '10, user decision before Q10 augmentation; retain only score 10 at both quality stages.')
    reused = ['config.frozen.json', 'raw_protocol.json', 'eval/raw_diag.json', 'profile.json',
              'synthesis.json', 'diagnostics.json', 'seed_quality.json', 'seed_quality.rejected.json']
    reused += ['data/' + name + '.json' for name in exports]
    migration = {
        'version': 'qwen15_quality10_from_verified_q8_v1',
        'helper_sha256': cp.file_hash(__file__), 'source_run': str(old),
        'source_config_hash': cp.digest(old_cfg), 'config_hash': cp.digest(cfg),
        'source_artifact_sha256': {name: cp.file_hash(old / name) for name in reused},
        'source_scoring_cache_sha256': cache_hashes,
        'changed_fields': ['run_dir', 'quality_min_score', 'allocation_entrypoint'],
        'seed_quality_reused': len(scored), 'seed_accepted': len(accepted),
        'seed_rejected': len(scored) - len(accepted),
        'augmentation_requests': int(len(accepted) * cfg['augmentation_request_ratio']),
        'augmentation_samples': int(len(accepted) * cfg['augmentation_request_ratio'])
                                * cfg['augmentation_samples_per_request'],
        'teacher_generation_called': False,
        'scope': 'Reuse original synthesis, labels, scores and model. No old augmentation or teacher cache copied.'}
    artifacts = {run / 'data' / (name + '.json'): rows for name, rows in exports.items()}
    artifacts.update({
        CONFIG_PATH: cfg, run / 'config.frozen.json': cfg, run / 'raw_protocol.json': protocol,
        run / 'eval/raw_diag.json': diagnosis, run / 'profile.json': profile,
        run / 'synthesis.json': samples, run / 'diagnostics.json': cp.read(old / 'diagnostics.json'),
        run / 'seed_quality.json': accepted,
        run / 'seed_quality.rejected.json': [r for r in scored if r['quality_score'] != 10],
        run / 'quality10_migration.json': migration})
    return cfg, run, protocol, artifacts


def prepare():
    cfg, run, _, artifacts = definition()
    evaluation.validate_artifacts(artifacts)
    for path, value in artifacts.items():
        cp.write(path, value)
    cp.init_run(CONFIG_PATH)
    report = cp.read(run / 'quality10_migration.json')
    print(json.dumps({k: report[k] for k in ['seed_quality_reused', 'seed_accepted',
                     'seed_rejected', 'augmentation_requests', 'augmentation_samples',
                     'teacher_generation_called']},
                     ensure_ascii=False, indent=2))
    print(f'Q10 run: {run}', flush=True)


def load():
    cfg, run, _, artifacts = definition()
    evaluation.validate_artifacts(artifacts, must_exist=True)
    with locations(CONFIG_PATH, RUN_DIR), job.route():
        loaded = raw.load()
    if loaded[:2] != (cfg, run):
        raise ValueError('Q10 runtime routing differs from frozen migration.')
    return loaded


def first_augmentation_requests(cfg, seeds):
    rows = list(seeds)
    rng = random.Random(cfg['seed'])
    rng.shuffle(rows)
    prompts, requests = cp.prompt_module(), {}
    for _ in range(2):
        if rng.random() < 0.5:
            example = rng.choice(rows)
            prompt = prompts.prompt4more.format(question=example['input'], answer=example['output'],
                label=example['label'], X=cfg['augmentation_samples_per_request'])
        else:
            prompt = prompts.prompt4exp([rng.choice(rows), rng.choice(rows)],
                                       cfg['augmentation_samples_per_request'])
        request = payload(cfg, [cp.user_message(prompt)])
        requests[cache_key(cfg, request)] = request
    return requests


def discard_interrupted():
    cfg, run, _ = source()
    requests = first_augmentation_requests(cfg, cp.read(run / 'seed_quality.json'))
    if set(requests) != INTERRUPTED_KEYS:
        raise ValueError('First two augmentation requests differ from the user-reported hashes; nothing deleted.')
    # Refuse to touch anything referenced by initial synthesis/diagnosis/scoring.
    for name in ['synthesis.json', 'diagnostics.json', 'seed_quality.json', 'seed_quality.rejected.json']:
        text = (run / name).read_text(encoding='utf-8')
        if any(key in text for key in requests):
            raise ValueError('An interrupted key is referenced by a seed-stage artifact; nothing deleted.')
    report_path = run / 'interrupted_augmentation_discard.json'
    paths = set()
    for key, request in requests.items():
        path = run / 'teacher_cache' / (key + '.json')
        if path.is_file():
            verified_cache(cfg, run, key, request, complete=False)
            paths.add(path)
        audit = run / 'teacher_repeat_scope_requests' / (key + '.json')
        if audit.is_file():
            if cp.read(audit).get('request') != key:
                raise ValueError('Unexpected repeat audit; nothing deleted.')
            paths.add(audit)
        for directory in run.glob('teacher_sample*_parsing'):
            for audit in directory.glob(key + '.*.json'):
                record = cp.read(audit)
                if record.get('request') != key or record.get('stage') not in ['rewrite', 'fusion']:
                    raise ValueError('Unexpected parsing audit; nothing deleted.')
                paths.add(audit)
    if any((run / name).exists() for name in ['augmented.json', 'augmented_quality.json',
                                            'augmented_quality.rejected.json', 'selection.json']):
        raise ValueError('Complete downstream artifacts exist; do not use this two-request cleanup.')
    for path in paths:
        if not path.resolve().is_relative_to(run.resolve()):
            raise ValueError('Unexpected cache symlink outside run; nothing deleted.')
    if report_path.exists():
        report = cp.read(report_path)
        if report['requests'] != sorted(requests) or any(
                report['removed_sha256'].get(str(p.relative_to(run))) != cp.file_hash(p) for p in paths):
            raise ValueError('Discard report or remaining response bytes differ; nothing deleted.')
        for path in sorted(paths):
            path.unlink()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    report = {'requests': sorted(requests),
              'removed_sha256': {str(p.relative_to(run)): cp.file_hash(p) for p in sorted(paths)},
              'scope': 'Only the two reported interrupted augmentation responses and their request audits.',
              'preserved': 'Seed synthesis, diagnosis, scoring, frozen config and all other teacher caches.'}
    cp.write(report_path, report)
    for path in sorted(paths):
        path.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))


@contextmanager
def route():
    original = job.load
    with locations(CONFIG_PATH, RUN_DIR):
        job.load = load
        try:
            yield
        finally:
            job.load = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'discard-augmentation', 'augment', 'select', 'profile']:
        commands.add_parser(command)
    commands.add_parser('quality').add_argument('--pool', choices=['augmented'], required=True)
    for command in ['train', 'compare']:
        commands.add_parser(command).add_argument('--arm', choices=['cds', 'random'], required=True)
    infer = commands.add_parser('infer')
    infer.add_argument('--split', choices=['test'], required=True)
    infer.add_argument('--arm', choices=['cds', 'random'], required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'discard-augmentation':
        discard_interrupted()
    else:
        with route():
            entry.main()


if __name__ == '__main__':
    main()
