"""Accept explicit difficulty labels in bold question headings as metadata only.

Use a small vocabulary of observed labels and all previous exact-count and
complete-answer checks. Mathematical contents are retained for quality review.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp
import teacher_sample_format as headings
import teacher_sample_optional_colon as previous_version

POLICY_VERSION = 'bold_question_difficulty_level_v1'
LEVELS = ('Junior High Level', 'Senior High Level', 'Advanced Senior High Level')
QUALIFIED = re.compile(
    r'^[ \t]{0,3}\*\*Question[ \t]+([1-9][0-9]*):[ \t]*('
    + '|'.join(re.escape(level) for level in LEVELS) + r')\*\*[ \t]*$')
PARENT_ERRORS = ('No standalone Markdown question heading without a colon.',
                 'Bold question fallback requires exactly ',
                 'Markdown fallback requires exactly ')


def parse_difficulty_only(text, count, labels, provenance, parser):
    lines, edits, inside = [], [], False
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
            match = QUALIFIED.fullmatch(body)
            if match:
                question, level = match.groups()
                line = f'**Question {question}:**' + line[len(body):]
                edits.append({'line': number, 'question': int(question), 'difficulty': level})
        lines.append(line)
    if inside:
        raise ValueError('Unclosed answer delimiter.')
    if not edits:
        raise ValueError('No recognized difficulty-tagged question headings.')
    normalized = ''.join(lines)
    rows, _ = previous_version.parse_compatible(normalized, count, labels, provenance, parser)
    return rows, normalized, edits


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return previous_version.parse_compatible(text, count, labels, provenance, parser)
    except ValueError as error:
        if not any(reason in str(error) for reason in PARENT_ERRORS):
            raise
    rows, _, _ = parse_difficulty_only(text, count, labels, provenance, parser)
    return rows, POLICY_VERSION


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_difficulty_heading_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__),
        'parent_helper_sha256': cp.file_hash(previous_version.__file__),
        'extension_of': previous_version.POLICY_VERSION,
        'recognized_levels': list(LEVELS),
        'scope': 'Whole bold Question N: <recognized difficulty> headings outside answer blocks.',
        'required': ['numbering exactly 1..N', 'one complete nonempty answer per question',
                     'previous boundary and footer rules', 'verified stop cache'],
        'operation': 'Normalize packaging headings only; retain original mathematics and raw caches.',
        'quality': 'Successful parsing does not certify solvability or correct answers.'})
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
                rows, normalized, edits = parse_difficulty_only(
                    text, count, labels, provenance, base_parser)
                cp.write(run / 'teacher_sample_difficulty_heading_parsing' /
                         (key + '.' + cp.digest(provenance) + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance), 'cache_sha256': cp.file_hash(path),
                    'response_sha256': cp.digest(text), 'normalized_view_sha256': cp.digest(normalized),
                    'difficulty_heading_edits': edits, 'expected_count': count, 'parsed_count': len(rows),
                    'sample_ids': [row['item_id'] for row in rows]})
                print(f'Parsed {len(rows)} samples using difficulty-tagged headings. '
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
