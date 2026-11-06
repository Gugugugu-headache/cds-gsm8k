"""Plain summary exclusion preserves cached math and prior parser policies."""
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
import teacher_sample_format as legacy
import teacher_sample_footer as footer
from test_teacher_sample_format import batch, cache

SUMMARY = ("These questions are designed to reinforce the student's understanding of multiplication "
           "in various contexts, including division and area calculations, which are common in "
           "junior high school mathematics.")


class SampleFooterTests(unittest.TestCase):
    def test_observed_summary_is_excluded_without_changing_samples(self):
        text = batch() + '\n\n' + SUMMARY
        provenance = {'stage': 'fine', 'request': 'a' * 64}
        with self.assertRaises(ValueError):
            legacy.parse_compatible(text, 3, ['Multiplication'], provenance, cp.parse_samples)
        expected, _ = legacy.parse_compatible(batch(), 3, ['Multiplication'], provenance, cp.parse_samples)
        rows, mode = footer.parse_compatible(text, 3, ['Multiplication'], provenance, cp.parse_samples)
        self.assertEqual(rows, expected)
        self.assertEqual(mode, footer.POLICY_VERSION)
        self.assertNotIn(SUMMARY, json.dumps(rows))

    def test_old_policy_audits_and_cache_bytes_survive_versioned_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = Path(tmp), {'seed': 42}
            old_key, old_path = cache(run, batch())
            old_provenance = {'stage': 'fine', 'request': old_key}
            with legacy.sample_format_compatibility(cfg, run):
                old_rows = cp.parse_samples(batch(), 3, [], old_provenance)
            policy = (run / 'teacher_sample_parser_policy.json').read_bytes()
            audit_path = run / 'teacher_sample_parsing' / (old_key + '.' + cp.digest(old_provenance) + '.json')
            old_audit = audit_path.read_bytes()
            old_bytes = old_path.read_bytes()
            # Same request payload would imply same cache key; create a distinct
            # genuine synthetic cache request for the new response fixture.
            payload = {'model': 'teacher/model', 'messages': [{'role': 'user', 'content': 'new samples'}],
                       'max_tokens': 4096, 'temperature': 0.5, 'top_p': 0.8, 'stream': False}
            url = 'https://example.com/v1'
            key = cp.digest({'url': url, 'payload': payload})
            path = run / 'teacher_cache' / (key + '.json')
            text = batch() + '\n\n' + SUMMARY
            cp.write(path, {'request': payload, 'url': url, 'content': text, 'finish_reason': 'stop'})
            raw = path.read_bytes()
            original_parser = cp.parse_samples
            with footer.sample_format_compatibility(cfg, run):
                self.assertEqual(cp.parse_samples(batch(), 3, [], old_provenance), old_rows)
                provenance = {'stage': 'fine', 'request': key}
                rows = cp.parse_samples(text, 3, [], provenance)
                self.assertEqual(cp.parse_samples(text, 3, [], provenance), rows)
            self.assertIs(cp.parse_samples, original_parser)
            self.assertEqual(policy, (run / 'teacher_sample_parser_policy.json').read_bytes())
            self.assertEqual(old_audit, audit_path.read_bytes())
            self.assertEqual(old_bytes, old_path.read_bytes())
            self.assertEqual(raw, path.read_bytes())
            record = cp.read(run / 'teacher_sample_footer_parsing' / (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual(record['summary_characters'], len(SUMMARY))
            self.assertEqual(record['summary_sha256'], cp.digest(SUMMARY))
            self.assertEqual(record['parsed_count'], 3)

    def test_additional_questions_answers_corrections_and_math_are_not_discarded(self):
        tails = [SUMMARY + '\n#### Question 4: Extra question?',
                 'These questions are good. Question 4: Work out the next example.',
                 'These questions are good. Another problem: there are five buses.',
                 'These questions are good. What is 2 plus 2?',
                 'These questions need correction to the answer in question 1.',
                 'These questions use \\[x=2\\].',
                 'Unrecognized trailing explanation']
        for tail in tails:
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                footer.parse_compatible(batch() + '\n\n' + tail, 3, [], {}, cp.parse_samples)

    def test_summary_does_not_rescue_incomplete_empty_or_misnumbered_batches(self):
        for text in [batch()[:-2], batch().replace('Question 3:', 'Question 4:'),
                     batch().replace('135 / 45 = 3 buses.', ''), batch().split('#### Question 3:')[0]]:
            with self.assertRaises(ValueError):
                footer.parse_compatible(text + '\n\n' + SUMMARY, 3, [], {}, cp.parse_samples)

    def test_preview_is_offline_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, batch() + '\n\n' + SUMMARY)
            config = run / 'config.json'
            cp.write(config, {'run_dir': str(run)})
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            output = io.StringIO()
            with patch.object(raw_cds18, 'CONFIG_PATH', config), \
                 patch.object(teacher_env, 'load_teacher_env') as env, \
                 patch.object(teacher_env, 'check_models') as network, \
                 contextlib.redirect_stdout(output):
                teacher_env.parse_cache(key, 3)
                env.assert_not_called()
                network.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['parser'], footer.POLICY_VERSION)
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})

    def test_length_cache_is_rejected_even_if_three_answers_look_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            text = batch() + '\n\n' + SUMMARY
            key, _ = cache(run, text, 'length')
            with footer.sample_format_compatibility({}, run):
                with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                    cp.parse_samples(text, 3, [], {'stage': 'fine', 'request': key})


if __name__ == '__main__':
    unittest.main()
