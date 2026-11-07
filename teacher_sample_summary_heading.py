"""Accept a Markdown Summary heading before the already-supported sample footer.

Versioned separately: all previous parser code/policies/audits remain immutable.
Only text after the last answer closing delimiter is considered, and the summary
body must still satisfy the prior non-question/footer and complete-batch checks.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp
import teacher_sample_format as headings
import teacher_sample_footer as footer

POLICY_VERSION = 'markdown_sample_summary_heading_v1'
PARENT_ERROR = 'Trailing text is not a recognized short sample summary.'
SUMMARY_HEADING = re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+Summary[ \t]*:?[ \t]*(?:\r?\n|$)', re.I)


def normalize_summary_heading(text):
    closings = list(footer.CLOSE.finditer(text))
    if not closings:
        raise ValueError('No complete answer before Summary heading.')
    end = closings[-1].end()
    tail = text[end:].lstrip()
    heading = SUMMARY_HEADING.match(tail)
    if not heading:
        raise ValueError('Trailing section is not an explicit Markdown Summary heading.')
    body = tail[heading.end():]
    normalized = text[:end] + '\n\n' + body
    # Reject extra questions, corrections, code/math or unrecognized summary
    # prose exactly as before; removing the heading grants no other relaxation.
    footer.separate_summary(normalized)
    return normalized, end, tail


def parse_heading_only(text, count, labels, provenance, parser):
    normalized, start, tail = normalize_summary_heading(text)
    rows, _ = footer.parse_compatible(normalized, count, labels, provenance, parser)
    return rows, start, tail


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return footer.parse_compatible(text, count, labels, provenance, parser)
    except ValueError as error:
        if PARENT_ERROR not in str(error):
            raise
    rows, _, _ = parse_heading_only(text, count, labels, provenance, parser)
    return rows, POLICY_VERSION


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_summary_heading_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__), 'parent_helper_sha256': cp.file_hash(footer.__file__),
        'extension_of': footer.POLICY_VERSION,
        'scope': 'Only Markdown Summary heading after the final complete answer delimiter.',
        'operation': 'Normalize optional heading; reuse all existing summary and sample validation.',
        'no_regeneration_or_question_answer_rewrite': True})
    base_parser = cp.parse_samples
    with footer.sample_format_compatibility(cfg, run):
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
                rows, start, tail = parse_heading_only(text, count, labels, provenance, base_parser)
                cp.write(run / 'teacher_sample_summary_heading_parsing' /
                         (key + '.' + cp.digest(provenance) + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance), 'cache_sha256': cp.file_hash(path),
                    'response_sha256': cp.digest(text), 'excluded_tail_sha256': cp.digest(tail),
                    'excluded_tail_start_character': start, 'excluded_tail_characters': len(tail),
                    'expected_count': count, 'parsed_count': len(rows),
                    'sample_ids': [r['item_id'] for r in rows]})
                print(f'Parsed {len(rows)} samples; excluded Markdown Summary section. Request: {key}', flush=True)
                return rows
            except ValueError as error:
                raise ValueError(f'Sample parsing failed; request={key}; stage={provenance.get("stage")}; '
                                 f'{error}') from None

        cp.parse_samples = parse
        try:
            yield
        finally:
            cp.parse_samples = previous
