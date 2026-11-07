"""Summary headings are metadata only; retain prior immutable parser artifacts."""
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
import teacher_env
import raw_cds18
import teacher_sample_format as headings
import teacher_sample_footer as footer
import teacher_sample_summary_heading as summary
from test_teacher_sample_format import batch, cache
from test_teacher_sample_footer import SUMMARY

TEXT = batch() + '\n\n### Summary:\n' + SUMMARY


class SummaryHeadingTests(unittest.TestCase):
    def test_summary_heading_keeps_questions_answers_and_ids_identical(self):
        provenance = {'stage': 'fine', 'request': 'a' * 64}
        with self.assertRaises(ValueError):
            footer.parse_compatible(TEXT, 3, [], provenance, cp.parse_samples)
        rows, mode = summary.parse_compatible(TEXT, 3, [], provenance, cp.parse_samples)
        expected, _ = footer.parse_compatible(batch(), 3, [], provenance, cp.parse_samples)
        self.assertEqual(rows, expected)
        self.assertEqual(mode, summary.POLICY_VERSION)
        self.assertNotIn('Summary', json.dumps(rows))

    def test_previous_modes_stay_unchanged(self):
        for text in [batch(False), batch(), batch() + '\n\n' + SUMMARY]:
            self.assertEqual(summary.parse_compatible(text, 3, [], {}, cp.parse_samples),
                             footer.parse_compatible(text, 3, [], {}, cp.parse_samples))

    def test_heading_does_not_bypass_missing_questions_or_footer_content_guards(self):
        bad = [TEXT.replace('Question 3:', 'Question 4:'),
               batch()[:-2] + '\n\n### Summary:\n' + SUMMARY,
               TEXT + '\nQuestion 4: An extra question?',
               TEXT + '\nCorrection: answer in question 1 should change.',
               TEXT.replace('### Summary:', '### New Question:'),
               TEXT.replace(SUMMARY, 'These questions need a calculation: x = 3.')]
        for text in bad:
            with self.subTest(text=text[-60:]), self.assertRaises(ValueError):
                summary.parse_compatible(text, 3, [], {}, cp.parse_samples)

    def test_previous_policies_and_cache_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = Path(tmp), {'seed': 42}
            with footer.sample_format_compatibility(cfg, run):
                pass
            before = {p: p.read_bytes() for p in run.rglob('*.json')}
            key, path = cache(run, TEXT)
            original_bytes = path.read_bytes()
            original_parser = cp.parse_samples
            with summary.sample_format_compatibility(cfg, run):
                provenance = {'stage': 'fine', 'request': key}
                rows = cp.parse_samples(TEXT, 3, [], provenance)
                self.assertEqual(cp.parse_samples(TEXT, 3, [], provenance), rows)
            self.assertIs(cp.parse_samples, original_parser)
            self.assertEqual(path.read_bytes(), original_bytes)
            for p, content in before.items():
                self.assertEqual(p.read_bytes(), content)
            audit = cp.read(run / 'teacher_sample_summary_heading_parsing' /
                            (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual(audit['parsed_count'], 3)
            self.assertEqual(audit['excluded_tail_sha256'], cp.digest('### Summary:\n' + SUMMARY))

    def test_truncated_cache_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            key, _ = cache(run, TEXT, 'length')
            with summary.sample_format_compatibility({}, run):
                with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                    cp.parse_samples(TEXT, 3, [], {'stage': 'fine', 'request': key})

    def test_preview_remains_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            key, _ = cache(run, TEXT)
            config=run/'config.json'
            cp.write(config, {'run_dir':str(run)})
            before={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
            output=io.StringIO()
            with patch.object(raw_cds18,'CONFIG_PATH',config), \
                 patch.object(teacher_env,'load_teacher_env') as env, \
                 patch.object(teacher_env,'check_models') as network, contextlib.redirect_stdout(output):
                teacher_env.parse_cache(key,3)
                env.assert_not_called()
                network.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['parser'], summary.POLICY_VERSION)
            self.assertEqual(before,{p:p.read_bytes() for p in run.rglob('*') if p.is_file()})


if __name__=='__main__':
    unittest.main()
