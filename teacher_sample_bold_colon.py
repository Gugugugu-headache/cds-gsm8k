"""Versioned parser extension for **Question 1:** and boundary horizontal rules.

Normalize heading markup and standalone --- only between complete samples.
Run the existing strict Markdown/summary validators; preserve all sample content.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp
import teacher_sample_format as headings
import teacher_sample_descriptive_footer as previous_version

POLICY_VERSION = 'bold_question_inner_colon_v1'
PARENT_ERROR = 'Markdown fallback requires exactly '
ORIGINAL_CONTEXT = previous_version.sample_format_compatibility
INNER = re.compile(r'^[ \t]{0,3}\*\*Question[ \t]+([1-9][0-9]*)[ \t]*:\*\*[ \t]*(.*)$')
OUTER = re.compile(r'^[ \t]{0,3}\*\*Question[ \t]+([1-9][0-9]*)\*\*:[ \t]*(.*)$')


def header(line):
    for pattern in [INNER, OUTER, headings.MARKDOWN]:
        match = pattern.fullmatch(line)
        if match:
            return match
    return None


def normalize_bold_questions(text, count):
    lines = text.splitlines(keepends=True)
    content = [line.rstrip('\r\n') for line in lines]
    normalized, numbers, changed, separators = [], [], [], []
    inside, opens, closes = False, 0, 0
    for index, (line, body) in enumerate(zip(lines, content)):
        stripped = body.strip()
        if stripped == '>>':
            if inside:
                raise ValueError('Nested answer delimiters.')
            inside, opens = True, opens + 1
        elif stripped == '<<':
            if not inside:
                raise ValueError('Closing answer delimiter without an opening delimiter.')
            inside, closes = False, closes + 1
        elif not inside:
            match = header(body)
            if match:
                number, tail = match.groups()
                numbers.append(int(number))
                if INNER.fullmatch(body):
                    changed.append(index + 1)
                ending = line[len(body):]
                line = f'#### Question {number}:' + (' ' + tail if tail else '') + ending
            elif stripped == '---':
                before, after = index - 1, index + 1
                while before >= 0 and not content[before].strip():
                    before -= 1
                while after < len(content) and not content[after].strip():
                    after += 1
                if (before >= 0 and content[before].strip() == '<<'
                        and after < len(content) and header(content[after])):
                    separators.append(index + 1)
                    continue
        normalized.append(line)
    if inside or opens != count or closes != count:
        raise ValueError('Bold question batches require exactly N complete answer blocks.')
    if not changed or numbers != list(range(1, count + 1)):
        raise ValueError(f'Bold question fallback requires exactly {count} questions numbered 1 through {count}.')
    return ''.join(normalized), {'inner_colon_heading_lines': changed,
                                'removed_boundary_rule_lines': separators,
                                'question_numbers': numbers}


def parse_bold_only(text, count, labels, provenance, parser):
    normalized, edits = normalize_bold_questions(text, count)
    # Markdown form intentionally forces the strict complete-batch validator,
    # rather than falling through to the permissive original regex directly.
    rows, _ = previous_version.parse_compatible(normalized, count, labels, provenance, parser)
    return rows, normalized, edits


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return previous_version.parse_compatible(text, count, labels, provenance, parser)
    except ValueError as error:
        if PARENT_ERROR not in str(error):
            raise
    rows, _, _ = parse_bold_only(text, count, labels, provenance, parser)
    return rows, POLICY_VERSION


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_bold_colon_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__),
        'parent_helper_sha256': cp.file_hash(previous_version.__file__),
        'extension_of': previous_version.POLICY_VERSION,
        'scope': 'Question colon inside bold markup and --- strictly between complete samples.',
        'required': ['exactly 1..N numbered questions', 'exactly N complete answer blocks',
                     'all existing strict answer and summary guards', 'verified stop cache'],
        'operation': 'Heading/boundary metadata only; no question or answer rewriting.'})
    base_parser = cp.parse_samples
    with ORIGINAL_CONTEXT(cfg, run):
        previous = cp.parse_samples

        def parse(text, count, labels, provenance):
            try:
                return previous(text, count, labels, provenance)
            except ValueError as error:
                if PARENT_ERROR not in str(error):
                    raise
            key = provenance.get('request')
            try:
                path = headings.validate_cached_text(run, key, text)
                rows, normalized, edits = parse_bold_only(text, count, labels, provenance, base_parser)
                cp.write(run / 'teacher_sample_bold_colon_parsing' /
                         (key + '.' + cp.digest(provenance) + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance), 'cache_sha256': cp.file_hash(path),
                    'response_sha256': cp.digest(text), 'normalized_view_sha256': cp.digest(normalized),
                    'expected_count': count, 'parsed_count': len(rows), 'edits': edits,
                    'sample_ids': [r['item_id'] for r in rows]})
                print(f'Parsed {len(rows)} samples using bold question colon compatibility. '
                      f'Request: {key}', flush=True)
                return rows
            except ValueError as error:
                raise ValueError(f'Sample parsing failed; request={key}; stage={provenance.get("stage")}; '
                                 f'{error}') from None

        cp.parse_samples = parse
        try:
            yield
        finally:
            cp.parse_samples = previous
