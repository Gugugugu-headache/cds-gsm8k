#!/usr/bin/env python3
"""CDS math adaptation. CPU stages use stdlib + the upstream Jinja templates.

Paths in config are relative to the repository, never to the shell cwd.
No model downloads or API calls occur during prepare, profile, or select.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
import unicodedata
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "cds-04D1"
INSTRUCTION = ('You are a highly intelligent mathematical reasoning assistant. Your task is '
               'to solve math problems step-by-step in a clear and logical manner. For each '
               'problem, provide detailed reasoning and calculations, then give the final '
               'answer after "####" with only the numerical value.')
SYN_INSTRUCTION = INSTRUCTION.split(', then give')[0] + '.'


def read(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    # str.splitlines also splits Unicode U+2028 inside valid JSON string values.
    return [json.loads(line) for line in text.split('\n') if line.strip()] if path.suffix == '.jsonl' else json.loads(text)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Do not silently replace an experiment artifact with different content."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Artifact differs: {path}. Use a fresh run/output directory.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def resolve(path):
    return (ROOT / path).resolve()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def prompt_module():
    spec = importlib.util.spec_from_file_location('cds_math_prompts', UPSTREAM / 'prompts/prompt_math.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_run(config_path):
    cfg = read(config_path)
    run = resolve(cfg['run_dir'])
    write(run / 'config.frozen.json', cfg)
    source_files = ['prompts/prompt_math.py', '4_Q_score/Qscore.py', 'finetune/code/config.py',
                    'finetune/code/dataset.py', 'infer/code/infer_math.py']
    # Store paths relative to the project root; the code directory may be renamed
    # locally or deployed as "src" on the server.
    hashes = {'cds-04D1/' + f: file_hash(UPSTREAM / f) for f in source_files}
    for filename in ['cds_pipeline.py', 'train_lora.py']:
        path = Path(__file__).with_name(filename)
        hashes[path.relative_to(ROOT).as_posix()] = file_hash(path)
    write(run / 'implementation.frozen.json', hashes)
    return cfg, run


def prepare(cfg, run):
    base = resolve(cfg['data_root'])
    manifest = read(base / 'split_manifest.json')
    # Existing source/artifact hashes are checked before any export.
    for name, expected in manifest['source_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    for name, expected in manifest['artifacts'].items():
        if file_hash(base / name) != expected:
            raise ValueError(f'Split artifact hash mismatch: {name}')
    splits = {name: read(base / name / 'data.jsonl') for name in ['D_diag', 'D_remaining', 'D_test']}
    expected = {'D_diag': 3500, 'D_remaining': 3973, 'D_test': 1319}
    if {k: len(v) for k, v in splits.items()} != expected:
        raise ValueError('Unexpected split sizes')
    seen_ids, seen_questions = set(), set()
    official = {s: read(ROOT / 'GSM8K' / (s + '.jsonl')) for s in ['train', 'test']}
    for name, rows in splits.items():
        split = 'test' if name == 'D_test' else 'train'
        local_questions = set()
        for row in rows:
            raw = official[split][row['official_row_index']]
            if (row['official_split'] != split or raw['question'] != row['question']
                    or raw['answer'] != row['answer']):
                raise ValueError('Official data mismatch')
            if row['item_id'] in seen_ids:
                raise ValueError('Repeated item ID')
            seen_ids.add(row['item_id'])
            question = normalized(row['question'])
            if question in seen_questions:
                raise ValueError('Question overlap between splits')
            local_questions.add(question)
        seen_questions.update(local_questions)
    vocab = read(base / 'kc_vocab.json')[cfg['kc_scope']]
    label_key = 'kc_labels' if cfg['kc_scope'] == 'source_kcs' else 'diagnosis_kcs'

    def convert(row):
        return {'item_id': row['item_id'], 'instruction': INSTRUCTION,
                'input': row['question'], 'output': row['answer'],
                'label': row[label_key], 'label_status': row['label_status']}

    remainder = list(splits['D_remaining'])
    random.Random(cfg['seed']).shuffle(remainder)
    n, m = cfg['pre_finetune_size'], cfg['development_size']
    if n < 0 or m < 0 or n + m > len(remainder):
        raise ValueError('Invalid remainder allocation')
    exports = {'diag': splits['D_diag'], 'test': splits['D_test'],
               'warmup': remainder[:n], 'dev': remainder[n:n+m], 'reserve': remainder[n+m:]}
    for name, rows in exports.items():
        write(run / 'data' / (name + '.json'), [convert(row) for row in rows])
    write(run / 'data/vocab.json', vocab)
    write(run / 'data/manifest.json', {
        'source_manifest_sha256': file_hash(base / 'split_manifest.json'),
        'selection_seed': cfg['seed'],
        'selection_method': 'seeded shuffle without replacement within D_remaining',
        'counts': {k: len(v) for k, v in exports.items()},
        'ids': {k: [r['item_id'] for r in rows] for k, rows in exports.items()},
        'kc_scope': cfg['kc_scope'], 'kc_count': len(vocab),
        'notes': ['Existing splits preserved; only D_remaining is newly allocated.',
                  'No development set allocated; dev.json is an empty compatibility file.' if not m
                  else 'A separate development subset is allocated from D_remaining.',
                  'Exact normalized overlap checked; semantic near-duplicates not checked.',
                  'Rare KCs deliberately oversampled in diagnosis; frequency reflects this split.']})
    print(json.dumps({k: len(v) for k, v in exports.items()}))


NUMBER = r'-?[\d,]+(?:\.\d+)?'


def grade(response, gold):
    """Mirror upstream numerical scoring; retain no-number errors explicitly."""
    expected = re.findall(r'#### (' + NUMBER + ')', gold)
    if not expected:
        raise ValueError('Gold answer has no #### numeric answer')
    target = float(expected[-1].replace(',', ''))
    patterns = [('hash', r'#### (' + NUMBER + ')'),
                ('answer_is', r'the answer is (' + NUMBER + ')'),
                ('last_number', r'-?\d[\d,]*(?:\.\d+)?')]
    for method, pattern in patterns:
        matches = re.findall(pattern, response, flags=re.I)
        if matches:
            predicted = float(matches[-1].replace(',', ''))
            return {'correct': math.isclose(target, predicted, rel_tol=1e-5, abs_tol=1e-5),
                    'predicted': predicted, 'gold': target, 'extraction': method}
    return {'correct': False, 'predicted': None, 'gold': target, 'extraction': 'no_number'}


def infer(cfg, run, args):
    rows = read(run / 'data' / (args.split + '.json'))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError(f'No items in {args.split}. No development set is allocated when development_size=0.')
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    model_path = Path(cfg['student_model']).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f'Download model first: {model_path}')
    output = run / 'eval' / (args.tag + '.json')
    if output.exists():
        raise ValueError('Evaluation tag already exists; choose a new tag.')
    adapter = str(Path(args.adapter).resolve()) if args.adapter else None
    llm = LLM(model=str(model_path), tensor_parallel_size=args.tp,
              enable_lora=bool(adapter), max_lora_rank=cfg['training']['lora_rank'],
              dtype='bfloat16', seed=cfg['seed'], gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True)
    if cfg['inference_prompt'] == 'source_plain':
        prompts = [r['instruction'] + '\n' + r['input'] + '\nAnswer: ' for r in rows]
    elif cfg['inference_prompt'] == 'chat':
        tokenizer = llm.get_tokenizer()
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': r['instruction'] + '\n' + r['input']}],
            tokenize=False, add_generation_prompt=True) for r in rows]
    else:
        raise ValueError('Unknown inference_prompt')
    request = LoRARequest('cds_adapter', 1, adapter) if adapter else None
    outputs = llm.generate(prompts, SamplingParams(temperature=0, top_p=1, top_k=1,
                           max_tokens=cfg['inference_max_tokens']), lora_request=request)
    records = []
    for row, output_item in zip(rows, outputs):
        generation = output_item.outputs[0]
        records.append({**row, 'response': generation.text, **grade(generation.text, row['output']),
                        'finish_reason': generation.finish_reason, 'output_tokens': len(generation.token_ids)})
    if len(records) != len(rows):
        raise RuntimeError('Incomplete inference')
    meta = {'split': args.split, 'model': str(model_path), 'adapter': adapter,
            'n': len(records), 'correct': sum(r['correct'] for r in records),
            'accuracy': sum(r['correct'] for r in records) / len(records),
            'no_number': sum(r['extraction'] == 'no_number' for r in records),
            'truncated': sum(r['finish_reason'] == 'length' for r in records),
            'prompt': cfg['inference_prompt'], 'config_hash': digest(cfg)}
    write(output, {'meta': meta, 'records': records})
    print(json.dumps(meta))


def profile_rows(rows, vocab, cfg):
    if not rows:
        raise ValueError('Empty diagnosis')
    result = {}
    for kc in vocab:
        subset = [r for r in rows if kc in set(r['label'])]
        correct = sum(r['correct'] for r in subset)
        accuracy = correct / len(subset) if subset else None
        frequency = len(subset) / len(rows)
        weak = ((accuracy is not None and accuracy <= cfg['weak_accuracy_threshold'])
                or frequency <= cfg['weak_frequency_threshold'])
        if not subset and cfg['zero_support_policy'] == 'exclude':
            weak = False
        result[kc] = {'support': len(subset), 'correct': correct, 'accuracy': accuracy,
                      'frequency': frequency, 'weak': weak}
    return result


def profile(cfg, run, args):
    evaluation = read(run / 'eval' / (args.tag + '.json'))
    expected = read(run / 'data/diag.json')
    rows = evaluation['records']
    if evaluation['meta']['split'] != 'diag' or len(rows) != len(expected):
        raise ValueError('Diagnosis requires the full diagnostic set, never test/dev/subset.')
    for row, reference in zip(rows, expected):
        if any(row[key] != reference[key] for key in reference):
            raise ValueError('Diagnostic identity/content mismatch')
        if row['correct'] != grade(row['response'], row['output'])['correct']:
            raise ValueError('Incorrect evaluation labels')
    write(run / 'profile.json', {'evaluation': args.tag,
          'evaluation_sha256': file_hash(run / 'eval' / (args.tag + '.json')),
          'kcs': profile_rows(rows, read(run / 'data/vocab.json'), cfg)})


class Teacher:
    """Sequential, resumable OpenAI-compatible calls. Cache contains no credentials."""
    def __init__(self, cfg, run):
        self.cfg = cfg['teacher']
        self.cache = run / 'teacher_cache'
        self.url = os.environ.get(self.cfg['base_url_env'], self.cfg['base_url_default']).rstrip('/')

    def call(self, messages, max_tokens=4096, scoring=False):
        payload = {'model': self.cfg['model'], 'messages': messages,
                   'temperature': 0 if scoring else 0.5, 'top_p': 1 if scoring else 0.8,
                   'max_tokens': max_tokens, 'stream': False}
        if scoring:
            payload['top_k'] = 1
        # SiliconFlow's documented API has frequency_penalty, not repetition_penalty.
        # Omit the latter rather than substituting a different penalty mechanism.
        key = digest({'url': self.url, 'payload': payload})
        path = self.cache / (key + '.json')
        if path.exists():
            cached = read(path)
        else:
            api_key = os.environ.get(self.cfg['api_key_env'])
            if not api_key:
                raise ValueError('Export the teacher API key environment variable first.')
            request = urllib.request.Request(self.url + '/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key})
            for attempt in range(self.cfg['retries'] + 1):
                try:
                    with urllib.request.urlopen(request, timeout=self.cfg['timeout_seconds']) as response:
                        result = json.load(response)
                    choice = result['choices'][0]
                    cached = {'request': payload, 'url': self.url, 'content': choice['message']['content'],
                              'finish_reason': choice.get('finish_reason'), 'usage': result.get('usage'),
                              'model': result.get('model'), 'id': result.get('id'),
                              'created': result.get('created')}
                    write(path, cached)
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in (429, 500, 502, 503, 504) or attempt == self.cfg['retries']:
                        raise RuntimeError(f'Teacher HTTP {error.code}; request {key}') from None
                except (urllib.error.URLError, TimeoutError):
                    if attempt == self.cfg['retries']:
                        raise RuntimeError(f'Teacher transport failure; request {key}') from None
                time.sleep(min(2 ** attempt, 30))
        if not isinstance(cached['content'], str) or not cached['content'].strip():
            raise ValueError(f'Empty teacher output: {key}')
        if cached['finish_reason'] != 'stop':
            raise ValueError(f'Teacher did not finish normally ({cached["finish_reason"]}): {key}')
        return cached['content'], key


def user_message(text):
    return {'role': 'user', 'content': text}


def parse_samples(text, count, labels, provenance):
    pattern = (r'\*\*Question(?:\s*\d+)?\*\*:\s*(.*?)'
               r'\*\*Answer\*\*:\s*(.*?)(?=\*\*Question(?:\s*\d+)?\*\*|\*\*Assessed Knowledge Points\*\*|\Z)')
    matches = re.findall(pattern, text, re.S)
    if len(matches) != count:
        raise ValueError(f'Expected {count} synthesized questions, parsed {len(matches)}; inspect cache.')
    result = []
    for i, (question, answer) in enumerate(matches):
        wrapped = re.search(r'>>\s*\n(.*?)\n\s*<<', answer, re.S)
        answer = wrapped.group(1) if wrapped else answer
        answer = re.sub(r'(?im)^.*sample.*$', '', answer).strip()
        if not question.strip() or not answer:
            raise ValueError('Empty synthesized question or answer')
        result.append({'item_id': digest([provenance, i])[:24], 'instruction': SYN_INSTRUCTION,
                       'input': question.strip(), 'output': answer, 'label': list(dict.fromkeys(labels)),
                       'provenance': provenance})
    return result


def parse_unmastered(text, vocab, fallback):
    blocks = re.findall(r'Unmastered Knowledge Points(.*?)Mastered Knowledge Points', text, re.S)
    if blocks:
        points = [kc for kc in vocab if kc in blocks[-1]]
        if 0 < len(points) <= 6:
            return points, False
    return fallback, True


def synthesize(cfg, run, args):
    prompts, teacher = prompt_module(), Teacher(cfg, run)
    profile_data = read(run / 'profile.json')
    evaluation_path = run / 'eval' / (profile_data['evaluation'] + '.json')
    if file_hash(evaluation_path) != profile_data['evaluation_sha256']:
        raise ValueError('Diagnosis evaluation changed')
    rows = read(evaluation_path)['records']
    vocab = read(run / 'data/vocab.json')
    records, diagnostics = [], []
    weak = [kc for kc in vocab if profile_data['kcs'][kc]['weak']]
    for kc in weak:
        prompt = prompts.prompt4kps.format(X=cfg['global_samples_per_kc'],
                    kps='- ' + kc + '\n', DIFFICULTY='junior high school')
        response, key = teacher.call([user_message(prompt)])
        records.extend(parse_samples(response, cfg['global_samples_per_kc'], [kc],
                                     {'stage': 'global', 'request': key}))
    wrongs = [r for r in rows if not r['correct']]
    for index, wrong in enumerate(wrongs):
        diagnostic_prompt = prompts.template4cognitive.render(KNOWLEDGE_POINTS=vocab,
            QUESTION=wrong['input'], WRONG_ANSWER=wrong['response'].replace('####', 'Therefore, the answer is') + '.')
        messages = [user_message(diagnostic_prompt)]
        diagnosis, diag_key = teacher.call(messages, max_tokens=1024)
        labels, fallback = parse_unmastered(diagnosis, vocab, wrong['label'])
        if not labels:
            raise ValueError('No labels available for synthesis')
        messages.extend([{'role': 'assistant', 'content': diagnosis}, user_message(
            prompts.template4cogsynthesis.render(DIFFICULTY='Junior high school',
            UNMASTERED=labels, X=cfg['fine_samples_per_error']))])
        response, key = teacher.call(messages)
        records.extend(parse_samples(response, cfg['fine_samples_per_error'], labels,
            {'stage': 'fine', 'source_item': wrong['item_id'], 'diagnosis_request': diag_key,
             'request': key, 'fallback': fallback}))
        diagnostics.append({'item_id': wrong['item_id'], 'labels': labels, 'fallback': fallback,
                            'diagnosis_request': diag_key})
        print(f'Fine synthesis {index + 1}/{len(wrongs)}', flush=True)
    write(run / 'synthesis.json', records)
    write(run / 'diagnostics.json', diagnostics)


def augment(cfg, run, args):
    # Quality-filter the seed pool before augmentation, as in the released math script.
    rows = read(run / 'seed_quality.json')
    if not rows:
        raise ValueError('Empty seed pool')
    prompts, teacher = prompt_module(), Teacher(cfg, run)
    rng = random.Random(cfg['seed'])
    rows = list(rows)
    rng.shuffle(rows)
    generated = []
    for index in range(int(len(rows) * cfg['augmentation_request_ratio'])):
        if rng.random() < 0.5:
            example = rng.choice(rows)
            labels = example['label']
            prompt = prompts.prompt4more.format(question=example['input'], answer=example['output'],
                label=labels, X=cfg['augmentation_samples_per_request'])
            parents, mode = [example['item_id']], 'rewrite'
        else:
            a, b = rng.choice(rows), rng.choice(rows)
            labels = list(dict.fromkeys(a['label'] + b['label']))
            prompt = prompts.prompt4exp([a, b], cfg['augmentation_samples_per_request'])
            parents, mode = [a['item_id'], b['item_id']], 'fusion'
        response, key = teacher.call([user_message(prompt)])
        generated.extend(parse_samples(response, cfg['augmentation_samples_per_request'], labels,
                        {'stage': mode, 'parents': parents, 'request': key, 'draw': index}))
        print(f'Augmentation request {index + 1}', flush=True)
    write(run / 'augmented.json', generated)


def quality(cfg, run, args):
    source = 'synthesis.json' if args.pool == 'seed' else 'augmented.json'
    output = 'seed_quality.json' if args.pool == 'seed' else 'augmented_quality.json'
    rows = read(run / source)
    prompts, teacher = prompt_module(), Teacher(cfg, run)
    accepted, rejected = [], []
    for index, row in enumerate(rows):
        prompt = prompts.prompt4filter.format(question=row['input'], answer=row['output'], label=row['label'])
        response, key = teacher.call([user_message(prompt)], max_tokens=512, scoring=True)
        scores = re.findall(r'\bScore:\s*(10|[0-9])(?![\d.])', response)
        score = int(scores[-1]) if len(scores) == 1 else None
        result = {**row, 'quality_score': score, 'quality_request': key}
        (accepted if score is not None and score >= cfg['quality_min_score'] else rejected).append(result)
        print(f'Quality {args.pool} {index + 1}/{len(rows)}', flush=True)
    write(run / output, accepted)
    write(run / (output.replace('.json', '.rejected.json')), rejected)


def score_pool(rows, kcs, cfg):
    if not rows:
        raise ValueError('Empty candidate pool')
    weight, small = cfg['score_accuracy_weight'], cfg['score_smoothing']
    values = {}
    for kc in kcs:
        freq = round(sum(kc in set(row['label']) for row in rows) / len(rows), 4)
        if freq == 0:
            continue
        accuracy = kcs[kc]['accuracy']
        if accuracy is None:
            if cfg['zero_support_policy'] == 'exclude':
                continue
            accuracy = 0.0
        # Source reports percent accuracy to 2 decimals, then uses fractions in Qscore.
        accuracy = round(accuracy * 100, 2) / 100
        values[kc] = weight * math.log(1 / (accuracy + small)) + (1-weight) * math.log(1 / (freq + small))
    maximum = max(values.values(), default=0)
    if maximum < 0:
        raise ValueError('Source max-normalization would reverse negative weights; inspect saturated profile.')
    values = {kc: v / maximum if maximum else 0 for kc, v in values.items()}
    scored = [{**r, 'cds_score': sum(values.get(kc, 0) for kc in r['label'])} for r in rows]
    scores = [r['cds_score'] for r in scored]
    threshold = statistics.mean(scores) - cfg['score_sigma'] * statistics.pstdev(scores)
    return scored, threshold, values


def select(cfg, run, args):
    rows = read(run / 'seed_quality.json') + read(run / 'augmented_quality.json')
    # Fixed text-only exclusion; no test answers enter generation, scoring, or diagnosis.
    blocked = {normalized(r['input']) for name in ['test', 'dev', 'warmup']
               for r in read(run / 'data' / (name + '.json'))}
    seen, unique, removed = set(), [], []
    for row in rows:
        key = normalized(row['input'])
        if key in blocked or key in seen:
            removed.append({'item_id': row['item_id'], 'reason': 'held_out_overlap' if key in blocked else 'duplicate'})
        else:
            unique.append(row)
            seen.add(key)
    kcs = read(run / 'profile.json')['kcs']
    if any(set(r['label']) - set(kcs) for r in unique):
        raise ValueError('Candidate labels outside vocabulary')
    scored, threshold, weights = score_pool(unique, kcs, cfg)
    selected = [r for r in scored if r['cds_score'] >= threshold]
    if not selected:
        raise ValueError('No samples selected')
    control = random.Random(cfg['seed']).sample(scored, len(selected))
    write(run / 'candidate_pool.json', scored)
    write(run / 'train_cds.json', selected)
    write(run / 'train_random.json', control)
    write(run / 'selection.json', {'before_dedup': len(rows), 'after_dedup': len(unique),
          'selected': len(selected), 'threshold': threshold, 'weights': weights, 'removed': removed,
          'control': 'same quality-filtered pool and example count; token count not matched',
          'semantic_overlap_checked': False})
    print(f'Selected {len(selected)} / {len(unique)}; random control exported.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('configs') / 'cds_gsm8k.json'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    p = commands.add_parser('infer')
    p.add_argument('--split', choices=['diag', 'test', 'dev'], required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--adapter')
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--limit', type=int)
    p = commands.add_parser('profile')
    p.add_argument('--tag', required=True)
    commands.add_parser('synthesize')
    commands.add_parser('augment')
    p = commands.add_parser('quality')
    p.add_argument('--pool', choices=['seed', 'augmented'], required=True)
    commands.add_parser('select')
    args = parser.parse_args()
    if getattr(args, 'tag', None) and not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Use alphanumeric, underscore or hyphen in tag.')
    if getattr(args, 'limit', None) is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    cfg, run = init_run(args.config)
    if args.command == 'prepare':
        prepare(cfg, run)
    else:
        globals()[args.command](cfg, run, args)


if __name__ == '__main__':
    main()
