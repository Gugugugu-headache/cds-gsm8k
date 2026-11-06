"""Conservative fallback for numbered Markdown question headings in cached samples.

The original CDS parser runs first. The fallback only changes heading markup;
it requires exactly N sequential questions and one complete >>...<< answer per
question. No re-generation or partial-batch acceptance.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp

POLICY_VERSION = 'markdown_question_headings_v1'
MARKDOWN = re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+Question[ \t]+([1-9][0-9]*)[ \t]*:[ \t]*(.*)$')
CANONICAL = re.compile(r'^\*\*Question ([1-9][0-9]*)\*\*:', re.M)


def normalize_markdown_questions(text, count):
    lines, numbers, inside_answer = [], [], False
    for line in text.splitlines(keepends=True):
        content = line.rstrip('\r\n')
        ending = line[len(content):]
        stripped = content.strip()
        if stripped.startswith(('```', '~~~')):
            raise ValueError('Markdown fallback does not accept fenced code blocks.')
        if stripped == '>>':
            if inside_answer:
                raise ValueError('Nested answer delimiters.')
            inside_answer = True
        elif stripped == '<<':
            if not inside_answer:
                raise ValueError('Closing answer delimiter without an opening delimiter.')
            inside_answer = False
        elif not inside_answer:
            match = MARKDOWN.fullmatch(content)
            if match:
                number, tail = match.groups()
                numbers.append(int(number))
                line = f'**Question {number}**:' + (' ' + tail if tail else '') + ending
        lines.append(line)
    if inside_answer:
        raise ValueError('Unclosed answer delimiter.')
    if numbers != list(range(1, count + 1)):
        raise ValueError(f'Markdown fallback requires exactly {count} questions numbered 1 through {count}.')
    normalized = ''.join(lines)
    headings = list(CANONICAL.finditer(normalized))
    if len(headings) != count:
        raise ValueError('Ambiguous mixture of question heading formats.')
    for index, header in enumerate(headings):
        section = normalized[header.start():headings[index + 1].start() if index + 1 < count else len(normalized)]
        if len(re.findall(r'^[ \t]*\*\*Answer\*\*:', section, re.M)) != 1:
            raise ValueError('Each Markdown question must have exactly one Answer heading.')
        delimiters = re.findall(r'^[ \t]*(>>|<<)[ \t]*\r?$', section, re.M)
        if delimiters != ['>>', '<<']:
            raise ValueError('Each Markdown question needs one complete answer block.')
        match = re.fullmatch(
            r'\*\*Question [1-9][0-9]*\*\*:[ \t]*(.*?)'
            r'^[ \t]*\*\*Answer\*\*:[ \t]*\r?\n[ \t]*>>[ \t]*\r?\n'
            r'(.*?)\r?\n[ \t]*<<[ \t]*\s*', section, re.M | re.S)
        if not match or not all(part.strip() for part in match.groups()):
            raise ValueError('Markdown question/answer is empty, incomplete or has unexpected trailing text.')
    return normalized


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return parser(text, count, labels, provenance), 'source'
    except ValueError as error:
        if not str(error).startswith(f'Expected {count} synthesized questions, parsed '):
            raise
    normalized = normalize_markdown_questions(text, count)
    return parser(normalized, count, labels, provenance), POLICY_VERSION


def validate_cached_text(run, key, text):
    if not re.fullmatch(r'[0-9a-f]{64}', key or ''):
        raise ValueError('Sample provenance must identify a real cache request hash.')
    path = Path(run) / 'teacher_cache' / (key + '.json')
    cached = cp.read(path)
    if (cp.digest({'url': cached['url'], 'payload': cached['request']}) != key
            or cached['content'] != text or cached['finish_reason'] != 'stop'):
        raise ValueError('Sample response differs from a complete, verified cache entry.')
    return path


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_parser_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__),
        'order': 'Original parser first, then numbered Markdown heading fallback.',
        'required': ['Question numbering exactly 1..N', 'exactly one Answer heading per question',
                     'one complete >>...<< answer block per question', 'verified stop cache'],
        'operation': 'Heading markup normalization only; original content/cache/provenance retained.',
        'no_generation_or_partial_batches': True})
    original = cp.parse_samples

    def parse(text, count, labels, provenance):
        key = provenance.get('request')
        try:
            rows, mode = parse_compatible(text, count, labels, provenance, original)
            if mode != 'source':
                path = validate_cached_text(run, key, text)
                # An augmentation request may be reused for multiple draws;
                # each draw has distinct provenance/sample IDs but the same cache.
                usage_key = key + '.' + cp.digest(provenance)
                cp.write(run / 'teacher_sample_parsing' / (usage_key + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance),
                    'expected_count': count, 'parsed_count': len(rows),
                    'cache_sha256': cp.file_hash(path), 'response_sha256': cp.digest(text),
                    'sample_ids': [r['item_id'] for r in rows]})
                print(f'Parsed {len(rows)} samples using Markdown headings. Request: {key}', flush=True)
            return rows
        except ValueError as error:
            raise ValueError(f'Sample parsing failed; request={key}; stage={provenance.get("stage")}; '
                             f'{error}') from None

    cp.parse_samples = parse
    try:
        yield
    finally:
        cp.parse_samples = original
