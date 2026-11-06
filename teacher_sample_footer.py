"""Versioned extension for plain summary prose after a complete sample batch.

Retains the original Markdown parser/policy and its prior audit records. Only a
recognized non-question summary after the last << can be omitted for extraction;
the complete original response stays in its verified cache.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp
import teacher_sample_format as legacy

POLICY_VERSION = 'markdown_question_summary_v1'
TRAILING_ERROR = 'Markdown question/answer is empty, incomplete or has unexpected trailing text.'
CLOSE = re.compile(r'^[ \t]*<<[ \t]*\r?$', re.M)
SUMMARY_START = re.compile(r'^(?:These|The above) (?:questions|samples|problems|exercises)\b', re.I)


def separate_summary(text):
    closings = list(CLOSE.finditer(text))
    if not closings:
        raise ValueError('No complete answer block before summary.')
    end = closings[-1].end()
    summary = text[end:].strip()
    if not summary or len(summary) > 2000 or not SUMMARY_START.match(summary):
        raise ValueError('Trailing text is not a recognized short sample summary.')
    # Do not hide extra exercises, math blocks or answer structures as a footer.
    if ('?' in summary or re.search(r'(?im)^[ \t]*(?:#{1,6}|\*\*|>>|<<|```|~~~)', summary)
            or any(marker in summary for marker in ['\\[', '\\(', '$$', '\\begin{', '='])
            or re.search(r'\b(?:question|answer|problem|exercise|sample)\s*[#:]?\s*\d+\b', summary, re.I)
            or re.search(r'\b(?:another|additional|extra)\s+(?:question|problem|exercise|sample)\b', summary, re.I)
            or re.search(r'\b(?:what|how many|how much|calculate|solve|find the|correction|incorrect|mistake)\b', summary, re.I)):
        raise ValueError('Trailing text contains possible extra question/answer structure.')
    return text[:end], summary


def parse_summary_only(text, count, labels, provenance, parser):
    samples, summary = separate_summary(text)
    # Re-run every original count/number/answer-boundary check on the sample text.
    rows, _ = legacy.parse_compatible(samples, count, labels, provenance, parser)
    return rows, summary


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return legacy.parse_compatible(text, count, labels, provenance, parser)
    except ValueError as error:
        if TRAILING_ERROR not in str(error):
            raise
    rows, _ = parse_summary_only(text, count, labels, provenance, parser)
    return rows, POLICY_VERSION


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_footer_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__), 'legacy_helper_sha256': cp.file_hash(legacy.__file__),
        'extension_of': legacy.POLICY_VERSION,
        'scope': 'Recognized plain summary after the last complete answer block only.',
        'summary_prefixes': ['These questions/samples/problems/exercises',
                             'The above questions/samples/problems/exercises'],
        'maximum_summary_characters': 2000,
        'guards': ['original exact count/continuous numbering/nonempty complete answers still required',
                   'reject possible extra questions, answers, code or math structure in footer',
                   'verified stop cache; full original response retained'],
        'no_regeneration_or_question_answer_rewrite': True})
    base_parser = cp.parse_samples
    with legacy.sample_format_compatibility(cfg, run):
        previous = cp.parse_samples

        def parse(text, count, labels, provenance):
            try:
                return previous(text, count, labels, provenance)
            except ValueError as error:
                if TRAILING_ERROR not in str(error):
                    raise
            key = provenance.get('request')
            try:
                path = legacy.validate_cached_text(run, key, text)
                rows, summary = parse_summary_only(text, count, labels, provenance, base_parser)
                usage_key = key + '.' + cp.digest(provenance)
                cp.write(run / 'teacher_sample_footer_parsing' / (usage_key + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance), 'cache_sha256': cp.file_hash(path),
                    'response_sha256': cp.digest(text), 'summary_sha256': cp.digest(summary),
                    'summary_characters': len(summary), 'expected_count': count,
                    'parsed_count': len(rows), 'sample_ids': [r['item_id'] for r in rows]})
                print(f'Parsed {len(rows)} samples; excluded trailing sample summary '
                      f'({len(summary)} characters). Request: {key}', flush=True)
                return rows
            except ValueError as error:
                raise ValueError(f'Sample parsing failed; request={key}; stage={provenance.get("stage")}; '
                                 f'{error}') from None

        cp.parse_samples = parse
        try:
            yield
        finally:
            cp.parse_samples = previous
