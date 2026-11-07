"""Regression: descriptive 'solve equations' is metadata, not an extra task."""
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
import teacher_sample_format as headings
import teacher_sample_footer as footer
import teacher_sample_summary_heading as previous
import teacher_sample_descriptive_footer as descriptive
from test_teacher_sample_format import batch, cache
from test_teacher_sample_footer import SUMMARY

FIXTURE = Path(__file__).with_name('fixtures') / 'descriptive_equations_reply.txt'


class DescriptiveFooterTests(unittest.TestCase):
    def test_actual_reported_output_keeps_every_question_and_answer(self):
        text = FIXTURE.read_text()
        with self.assertRaisesRegex(ValueError, 'possible extra question/answer'):
            previous.parse_compatible(text, 3, [], {}, cp.parse_samples)
        rows, mode = descriptive.parse_compatible(text, 3, [], {}, cp.parse_samples)
        lines = text.splitlines()
        self.assertEqual(mode, descriptive.POLICY_VERSION)
        for row, q, first, last in zip(rows, [4, 36, 62], [8, 40, 66], [32, 58, 83]):
            self.assertEqual(row['input'], lines[q - 1])
            self.assertEqual(row['output'], '\n'.join(lines[first - 1:last]).strip())
        self.assertNotIn('These questions are designed', json.dumps(rows))

    def test_descriptive_solve_calculate_find_wording_supported(self):
        for body in ['These questions help students calculate areas and solve equations.',
                     'These exercises are designed to help learners find the correct approach.']:
            for heading in ['', '### Summary:\n']:
                rows, mode = descriptive.parse_compatible(batch() + '\n\n' + heading + body,
                                                          3, [], {}, cp.parse_samples)
                self.assertEqual(mode, descriptive.POLICY_VERSION)
                self.assertEqual(len(rows), 3)

    def test_direct_tasks_corrections_and_extra_questions_still_rejected(self):
        bodies = ['These questions reinforce equations. Solve for x.',
                  'These questions reinforce equations. Next, calculate the result.',
                  'These questions help learners solve equations. Then apply the previous answer.',
                  'These questions are good. Please find the answer.',
                  'These questions are good. Question 4: What is two plus two?',
                  'These questions need a correction to question 1.',
                  'These questions require learners to solve equations; the answer is 12.',
                  'These questions use x = 3.',
                  'These questions introduce another exercise with five buses.']
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                descriptive.parse_compatible(batch() + '\n\n' + body, 3, [], {}, cp.parse_samples)

    def test_incomplete_batch_not_rescued_by_descriptive_metadata(self):
        tail = '\nThese questions require learners to solve equations.'
        for body in [batch()[:-2], batch().replace('Question 3:', 'Question 4:'),
                     batch().replace('135 / 45 = 3 buses.', '')]:
            with self.assertRaises(ValueError):
                descriptive.parse_compatible(body + tail, 3, [], {}, cp.parse_samples)

    def test_previous_modes_and_sample_ids_are_unchanged(self):
        for text in [batch(False), batch(), batch() + '\n\n' + SUMMARY,
                     batch() + '\n\n### Summary:\n' + SUMMARY]:
            self.assertEqual(descriptive.parse_compatible(text, 3, [], {'stage': 'fine'}, cp.parse_samples),
                             previous.parse_compatible(text, 3, [], {'stage': 'fine'}, cp.parse_samples))

    def test_previous_policy_and_raw_cache_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = Path(tmp), {'seed': 42}
            with previous.sample_format_compatibility(cfg, run):
                pass
            old = {p: p.read_bytes() for p in run.rglob('*.json')}
            text = FIXTURE.read_text()
            key, path = cache(run, text)
            content = path.read_bytes()
            original = cp.parse_samples
            with descriptive.sample_format_compatibility(cfg, run):
                provenance = {'stage': 'fine', 'request': key}
                rows = cp.parse_samples(text, 3, ['Equations'], provenance)
                self.assertEqual(cp.parse_samples(text, 3, ['Equations'], provenance), rows)
            self.assertIs(cp.parse_samples, original)
            self.assertEqual(path.read_bytes(), content)
            for p, data in old.items():
                self.assertEqual(p.read_bytes(), data)
            record = cp.read(run / 'teacher_sample_descriptive_footer_parsing' /
                             (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual(record['parsed_count'], 3)

    def test_length_cache_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            text = FIXTURE.read_text()
            key, _ = cache(run, text, 'length')
            with descriptive.sample_format_compatibility({}, run):
                with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                    cp.parse_samples(text, 3, [], {'stage': 'fine', 'request': key})

    def test_preview_remains_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, FIXTURE.read_text())
            config = run / 'config.json'
            cp.write(config, {'run_dir': str(run)})
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            out = io.StringIO()
            with patch.object(raw_cds18, 'CONFIG_PATH', config), \
                 patch.object(teacher_env, 'load_teacher_env') as env, \
                 patch.object(teacher_env, 'check_models') as network, contextlib.redirect_stdout(out):
                teacher_env.parse_cache(key, 3)
                env.assert_not_called()
                network.assert_not_called()
            self.assertEqual(json.loads(out.getvalue())['parsed_count'], 3)
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
