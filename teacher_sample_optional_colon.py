"""Versioned support for standalone Markdown Question N headings without ':'.

Only heading metadata is normalized. Reuse the complete-batch validators and
all prior parser versions, retaining raw caches and prior immutable policies.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp
import teacher_sample_bold_colon as previous_version
import teacher_sample_format as headings

POLICY_VERSION = 'markdown_question_missing_colon_v1'
PARENT_ERRORS = ('Bold question fallback requires exactly ',
                 'Markdown fallback requires exactly ')
MISSING_COLON = re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+Question[ \t]+([1-9][0-9]*)[ \t]*$')


def parse_missing_only(text, count, labels, provenance, parser):
    lines, changed, inside = [], [], False
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        body = line.rstrip('\r\n')
        stripped = body.strip()
        if stripped == '>>':
            if inside:
                raise ValueError('Nested answer delimiters.')
            inside = True
        elif stripped == '<<':
            if not inside:
                raise ValueError('Closing answer delimiter without an opening delimiter.')
            inside = False
        elif not inside:
            match = MISSING_COLON.fullmatch(body)
            if match:
                # Intermediate inner-colon form forces the old strict path:
                # every sample must be numbered and have a complete answer.
                line = f'**Question {match.group(1)}:**' + line[len(body):]
                changed.append(number)
        lines.append(line)
    if inside:
        raise ValueError('Unclosed answer delimiter.')
    if not changed:
        raise ValueError('No standalone Markdown question heading without a colon.')
    rows, normalized, edits = previous_version.parse_bold_only(
        ''.join(lines), count, labels, provenance, parser)
    return rows, normalized, {'missing_colon_heading_lines': changed,
                             'intermediate_heading_boundary_edits': edits}


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return previous_version.parse_compatible(text, count, labels, provenance, parser)
    except ValueError as error:
        if not any(reason in str(error) for reason in PARENT_ERRORS):
            raise
    rows, _, _ = parse_missing_only(text, count, labels, provenance, parser)
    return rows, POLICY_VERSION


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_optional_colon_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__),
        'parent_helper_sha256': cp.file_hash(previous_version.__file__),
        'extension_of': previous_version.POLICY_VERSION,
        'scope': 'Standalone Markdown Question N headings without a colon, outside answers only.',
        'required': ['exactly 1..N sequential questions', 'one nonempty complete answer per question',
                     'existing strict boundary/summary checks', 'verified stop cache'],
        'operation': 'Normalize heading markup only; no question/answer/cache rewriting.'})
    base_parser = cp.parse_samples
    with previous_version.sample_format_compatibility(cfg, run):
        previous = cp.parse_samples

        def parse(text, count, labels, provenance):
            try:
                return previous(text, count, labels, provenance)
            except ValueError as error:
                if not any(reason in str(error) for reason in PARENT_ERRORS):
                    raise
            key = provenance.get('request')
            try:
                path = headings.validate_cached_text(run, key, text)
                rows, normalized, edits = parse_missing_only(text, count, labels, provenance, base_parser)
                cp.write(run / 'teacher_sample_optional_colon_parsing' /
                         (key + '.' + cp.digest(provenance) + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance),
                    'cache_sha256': cp.file_hash(path), 'response_sha256': cp.digest(text),
                    'normalized_view_sha256': cp.digest(normalized), 'edits': edits,
                    'expected_count': count, 'parsed_count': len(rows),
                    'sample_ids': [row['item_id'] for row in rows]})
                print(f'Parsed {len(rows)} samples using Markdown headings without colons. '
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
