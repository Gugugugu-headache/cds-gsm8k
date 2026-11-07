"""Allow descriptive mathematical verbs in a sample summary, not new tasks.

The previous blanket solve/calculate keyword guard rejected a normal sentence
about the educational purpose of complete samples. This version retains all
structural/content guards and rejects direct instructions at sentence boundaries.
Previous parser modules, policies and accepted records remain unchanged.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp
import teacher_sample_format as headings
import teacher_sample_footer as footer
import teacher_sample_summary_heading as previous_version

POLICY_VERSION = 'descriptive_sample_summary_v1'
PARENT_ERROR = 'Trailing text contains possible extra question/answer structure.'
TASK = r'(?:calculate|solve|find|determine|compute|evaluate|answer|apply|rewrite|replace|revise|correct)'


def separate_descriptive_summary(text):
    closings = list(footer.CLOSE.finditer(text))
    if not closings:
        raise ValueError('No complete answer before descriptive summary.')
    end = closings[-1].end()
    tail = text[end:].strip()
    heading = previous_version.SUMMARY_HEADING.match(tail)
    body = tail[heading.end():].strip() if heading else tail
    if not body or len(body) > 2000 or not footer.SUMMARY_START.match(body):
        raise ValueError('Trailing text is not a recognized descriptive sample summary.')
    structural = (r'(?im)^[ \t]*(?:#{1,6}|\*\*|>>|<<|```|~~~)',
                  r'\b(?:question|answer|problem|exercise|sample)\s*[#:]?\s*\d+\b',
                  r'\b(?:another|additional|extra)\s+(?:question|problem|exercise|sample)\b',
                  r'\b(?:what|how many|how much|correction|incorrect|mistake)\b',
                  r'\b(?:corrected (?:answer|solution)|answer (?:is|should)|should instead|must instead)\b')
    if ('?' in body or any(marker in body for marker in ['\\[', '\\(', '$$', '\\begin{', '='])
            or any(re.search(pattern, body, re.I) for pattern in structural)
            or re.search(r'(?:^|[.!;]\s+|\n)[ \t]*(?:please\s+)?'
                         r'(?:(?:then|next|now|instead)[,;]?\s+)?' + TASK + r'\b', body, re.I)
            or re.search(r'\b(?:now|next|instead|please)[,;]?\s+' + TASK + r'\b', body, re.I)):
        raise ValueError('Descriptive summary contains possible extra task, answer correction or math structure.')
    return text[:end], end, tail


def parse_descriptive_only(text, count, labels, provenance, parser):
    samples, start, tail = separate_descriptive_summary(text)
    rows, _ = headings.parse_compatible(samples, count, labels, provenance, parser)
    return rows, start, tail


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return previous_version.parse_compatible(text, count, labels, provenance, parser)
    except ValueError as error:
        if PARENT_ERROR not in str(error):
            raise
    rows, _, _ = parse_descriptive_only(text, count, labels, provenance, parser)
    return rows, POLICY_VERSION


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_descriptive_footer_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__),
        'parent_helper_sha256': cp.file_hash(previous_version.__file__),
        'extension_of': previous_version.POLICY_VERSION,
        'scope': 'Descriptive solve/calculate/find wording after the final complete answer block.',
        'guards': ['same summary prefixes and 2000-character bound',
                   'reject extra question/answer markers, math, corrections and direct instructions',
                   'all original exact-count/nonempty/numbering/answer-boundary checks remain'],
        'operation': 'Exclude summary metadata only; retain original cache and sample provenance.'})
    base_parser = cp.parse_samples
    with previous_version.sample_format_compatibility(cfg, run):
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
                rows, start, tail = parse_descriptive_only(text, count, labels, provenance, base_parser)
                cp.write(run / 'teacher_sample_descriptive_footer_parsing' /
                         (key + '.' + cp.digest(provenance) + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance), 'cache_sha256': cp.file_hash(path),
                    'response_sha256': cp.digest(text), 'excluded_tail_sha256': cp.digest(tail),
                    'excluded_tail_start_character': start, 'excluded_tail_characters': len(tail),
                    'expected_count': count, 'parsed_count': len(rows),
                    'sample_ids': [r['item_id'] for r in rows]})
                print(f'Parsed {len(rows)} samples; excluded descriptive sample summary. Request: {key}', flush=True)
                return rows
            except ValueError as error:
                raise ValueError(f'Sample parsing failed; request={key}; stage={provenance.get("stage")}; '
                                 f'{error}') from None

        cp.parse_samples = parse
        try:
            yield
        finally:
            cp.parse_samples = previous
