"""Accept bold numbered question headings containing descriptive titles.

Preserve each title as the first line of its question instead of dropping it.
All previous formats, policies, complete-answer checks and mathematics remain.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import cds_pipeline as cp
import teacher_sample_format as headings
import teacher_sample_difficulty_heading as previous_version

POLICY_VERSION = 'bold_question_descriptive_title_v1'
TITLED = re.compile(r'^[ \t]{0,3}\*\*Question[ \t]+([1-9][0-9]*):[ \t]*(.+?)\*\*[ \t]*$')
ANSWER = re.compile(r'^[ \t]*\*\*Answer\*\*:[ \t]*$')
PARENT_ERRORS = (*previous_version.PARENT_ERRORS, 'No recognized difficulty-tagged question headings.')


def parse_titled_only(text, count, labels, provenance, parser):
    source = text.splitlines(keepends=True)
    content = [line.rstrip('\r\n') for line in source]
    lines, edits, inside = [], [], False
    for index, (line, body) in enumerate(zip(source, content)):
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
            match = TITLED.fullmatch(body)
            if match:
                question, title = match.groups()
                title = title.strip()
                if not title or '**' in title or title in ['>>', '<<']:
                    raise ValueError('Empty or ambiguous descriptive question title.')
                answer = next((j for j in range(index + 1, len(content))
                               if ANSWER.fullmatch(content[j])), None)
                if answer is None or not any(
                        part.strip() and part.strip() != '---' for part in content[index + 1:answer]):
                    raise ValueError('A descriptive title must be followed by a nonempty question body.')
                # Old strict parsing treats a same-line heading tail as part
                # of the question. Preserve the title, question and answer.
                line = f'**Question {question}:** {title}' + line[len(body):]
                edits.append({'line': index + 1, 'question': int(question),
                              'title': title, 'retained_as': 'first line of question input'})
        lines.append(line)
    if inside:
        raise ValueError('Unclosed answer delimiter.')
    if not edits:
        raise ValueError('No complete bold numbered headings with descriptive titles.')
    normalized = ''.join(lines)
    rows, _ = previous_version.parse_compatible(normalized, count, labels, provenance, parser)
    return rows, normalized, edits


def parse_compatible(text, count, labels, provenance, parser):
    try:
        return previous_version.parse_compatible(text, count, labels, provenance, parser)
    except ValueError as error:
        if not any(reason in str(error) for reason in PARENT_ERRORS):
            raise
    rows, _, _ = parse_titled_only(text, count, labels, provenance, parser)
    return rows, POLICY_VERSION


@contextmanager
def sample_format_compatibility(cfg, run):
    run = Path(run)
    cp.write(run / 'teacher_sample_titled_heading_policy.json', {
        'version': POLICY_VERSION, 'config_hash': cp.digest(cfg),
        'helper_sha256': cp.file_hash(__file__),
        'parent_helper_sha256': cp.file_hash(previous_version.__file__),
        'extension_of': previous_version.POLICY_VERSION,
        'scope': 'Whole bold Question N: <nonempty single-line title> headings outside answers.',
        'required': ['numbering exactly 1..N', 'nonempty question body after each title',
                     'one complete nonempty answer per question',
                     'no nested bold in titles', 'previous boundary/footer checks', 'verified stop cache'],
        'operation': 'Retain title as first line of input; retain original question body, answer and cache.',
        'quality': 'No mathematical correction, manual score override or automatic quality approval.'})
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
                rows, normalized, edits = parse_titled_only(
                    text, count, labels, provenance, base_parser)
                cp.write(run / 'teacher_sample_titled_heading_parsing' /
                         (key + '.' + cp.digest(provenance) + '.json'), {
                    'policy': POLICY_VERSION, 'request': key, 'stage': provenance.get('stage'),
                    'provenance_sha256': cp.digest(provenance), 'cache_sha256': cp.file_hash(path),
                    'response_sha256': cp.digest(text), 'normalized_view_sha256': cp.digest(normalized),
                    'title_heading_edits': edits, 'expected_count': count, 'parsed_count': len(rows),
                    'sample_ids': [row['item_id'] for row in rows]})
                print(f'Parsed {len(rows)} samples; retained descriptive titles in questions. '
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
