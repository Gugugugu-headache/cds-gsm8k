"""Cached heading compatibility must preserve content, count, IDs and provenance."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import raw_cds18
import teacher_env
import teacher_sample_format as formatting


def batch(markdown=True):
    questions = [
        'Each bus holds 45 students. How many buses are needed for 135 students?',
        'A garden is 12 by 8 meters. Each flower needs 0.5 square meters. How many flowers fit?',
        'A factory produces 250 units/hour, 8 hours/day, 5 days/week. How many units per week?']
    answers = ['135 / 45 = 3 buses.', '12 * 8 / 0.5 = 192 flowers.', '250 * 8 * 5 = 10,000 units.']
    return '### Sample Questions to Reinforce Multiplication\n\n' + '\n\n'.join(
        (f'#### Question {i}:' if markdown else f'**Question {i}**:')
        + f'\n{q}\n\n**Answer**:\n>>\n{a}\n<<'
        for i, (q, a) in enumerate(zip(questions, answers), 1))


def cache(run, text, finish='stop'):
    payload = {'model': 'teacher/model', 'messages': [{'role': 'user', 'content': 'generate 3 samples'}],
               'max_tokens': 4096, 'temperature': 0.5, 'top_p': 0.8, 'stream': False}
    url = 'https://example.com/v1'
    key = cp.digest({'url': url, 'payload': payload})
    path = run / 'teacher_cache' / (key + '.json')
    cp.write(path, {'request': payload, 'url': url, 'content': text, 'finish_reason': finish})
    return key, path


class SampleFormatTests(unittest.TestCase):
    def test_observed_markdown_style_preserves_three_questions_answers_and_ids(self):
        text = batch()
        provenance = {'stage': 'fine', 'request': 'a' * 64}
        expected = cp.parse_samples(batch(False), 3, ['Multiplication'], provenance)
        with self.assertRaises(ValueError):
            cp.parse_samples(text, 3, ['Multiplication'], provenance)
        rows, mode = formatting.parse_compatible(text, 3, ['Multiplication'], provenance, cp.parse_samples)
        self.assertEqual(rows, expected)
        self.assertEqual(mode, formatting.POLICY_VERSION)
        self.assertEqual([r['output'] for r in rows],
                         ['135 / 45 = 3 buses.', '12 * 8 / 0.5 = 192 flowers.', '250 * 8 * 5 = 10,000 units.'])

    def test_original_format_is_returned_unchanged(self):
        text = batch(False)
        expected = cp.parse_samples(text, 3, ['Multiplication'], {'stage': 'fine'})
        rows, mode = formatting.parse_compatible(text, 3, ['Multiplication'], {'stage': 'fine'}, cp.parse_samples)
        self.assertEqual(rows, expected)
        self.assertEqual(mode, 'source')

    def test_partial_duplicate_number_empty_and_unclosed_blocks_rejected(self):
        bad = [batch().replace('#### Question 3:', '#### Question 2:'),
               batch().split('#### Question 3:')[0],
               batch().replace('135 / 45 = 3 buses.', ''),
               batch()[:-2],
               batch() + '\nUnexpected trailing explanation',
               '```markdown\n' + batch() + '\n```']
        for text in bad:
            with self.subTest(text=text[-50:]), self.assertRaises(ValueError):
                formatting.parse_compatible(text, 3, [], {}, cp.parse_samples)

    def test_headers_inside_answers_are_not_extra_questions(self):
        text = batch().replace('135 / 45 = 3 buses.',
                               'Explanation:\n#### Question 99:\n135 / 45 = 3 buses.')
        rows, _ = formatting.parse_compatible(text, 3, [], {}, cp.parse_samples)
        self.assertEqual(len(rows), 3)
        self.assertIn('#### Question 99:', rows[0]['output'])

    def test_crlf_and_question_on_heading_line_supported(self):
        text = batch().replace('#### Question 1:\nEach', '#### Question 1: Each').replace('\n', '\r\n')
        rows, _ = formatting.parse_compatible(text, 3, [], {}, cp.parse_samples)
        self.assertTrue(rows[0]['input'].startswith('Each bus'))
        self.assertEqual(len(rows), 3)

    def test_cached_fallback_is_audited_idempotent_and_does_not_write_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, path = cache(run, batch())
            original = path.read_bytes()
            original_parser = cp.parse_samples
            with formatting.sample_format_compatibility({'seed': 42}, run):
                provenance = {'stage': 'fine', 'request': key}
                rows = cp.parse_samples(batch(), 3, ['Multiplication'], provenance)
                self.assertEqual(cp.parse_samples(batch(), 3, ['Multiplication'], provenance), rows)
                with self.assertRaisesRegex(ValueError, key):
                    cp.parse_samples(batch().replace('Question 3:', 'Question 4:'), 3, [], provenance)
            self.assertIs(cp.parse_samples, original_parser)
            self.assertEqual(path.read_bytes(), original)
            report = cp.read(run / 'teacher_sample_parsing' / (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual(report['parsed_count'], 3)
            self.assertEqual(report['sample_ids'], [r['item_id'] for r in rows])

    def test_same_cached_augmentation_can_be_parsed_for_distinct_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, batch())
            with formatting.sample_format_compatibility({}, run):
                first = cp.parse_samples(batch(), 3, [], {'stage': 'rewrite', 'request': key, 'draw': 1})
                second = cp.parse_samples(batch(), 3, [], {'stage': 'rewrite', 'request': key, 'draw': 2})
            self.assertNotEqual(first[0]['item_id'], second[0]['item_id'])
            self.assertEqual(first[0]['input'], second[0]['input'])
            self.assertEqual(len(list((run / 'teacher_sample_parsing').glob('*.json'))), 2)

    def test_truncated_cache_cannot_be_accepted_by_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, batch(), 'length')
            with formatting.sample_format_compatibility({}, run):
                with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                    cp.parse_samples(batch(), 3, [], {'stage': 'fine', 'request': key})

    def test_parse_cache_preview_does_not_read_env_call_network_or_write_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, batch())
            config = run / 'config.json'
            cp.write(config, {'run_dir': str(run)})
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            output = io.StringIO()
            with patch.object(raw_cds18, 'CONFIG_PATH', config), \
                 patch.object(teacher_env, 'load_teacher_env') as env, \
                 patch.object(teacher_env, 'check_models') as network, \
                 patch.object(sys, 'argv', ['teacher_env.py', 'parse-cache', '--request-key', key, '--count', '3']), \
                 contextlib.redirect_stdout(output):
                teacher_env.main()
                env.assert_not_called()
                network.assert_not_called()
            result = json.loads(output.getvalue())
            self.assertEqual(result['parsed_count'], 3)
            self.assertTrue(result['preview_only'])
            self.assertEqual({p: p.read_bytes() for p in run.rglob('*') if p.is_file()}, before)


if __name__ == '__main__':
    unittest.main()
