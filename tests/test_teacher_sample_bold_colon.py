"""Reported Ollama output: retain samples while normalizing heading metadata."""
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
import ollama_teacher as ot
import qwen15_cds_ollama as job
import qwen15_cds_ollama_compat as repeat_fix
import qwen15_cds_ollama_samples as entry
import teacher_sample_bold_colon as formatting
import teacher_sample_descriptive_footer as previous
from test_teacher_sample_format import batch, cache
from test_teacher_sample_footer import SUMMARY
import test_qwen15_cds_ollama as local_tests

FIXTURE = Path(__file__).with_name('fixtures') / 'bold_colon_ollama_reply.txt'


class BoldColonTests(unittest.TestCase):
    def test_actual_reply_preserves_every_question_answer_id_and_provenance(self):
        text = FIXTURE.read_text()
        provenance = {'stage': 'fine', 'request': 'a' * 64}
        with self.assertRaisesRegex(ValueError, formatting.PARENT_ERROR):
            previous.parse_compatible(text, 3, [], provenance, cp.parse_samples)
        rows, mode = formatting.parse_compatible(text, 3, ['Mixed Operations'], provenance, cp.parse_samples)
        self.assertEqual(mode, formatting.POLICY_VERSION)
        lines = text.splitlines()
        for row, q, first, last in zip(rows, [4, 34, 84], [8, 38, 88], [28, 78, 125]):
            self.assertEqual(row['input'], lines[q - 1])
            self.assertEqual(row['output'], '\n'.join(lines[first - 1:last]).strip())
            self.assertEqual(row['provenance'], provenance)
        normalized, edits = formatting.normalize_bold_questions(text, 3)
        self.assertEqual(edits['inner_colon_heading_lines'], [3, 33, 83])
        self.assertEqual(edits['removed_boundary_rule_lines'], [31, 81])
        expected = cp.parse_samples(normalized.replace('#### Question ', '**Question ')
                                   .replace('1:', '1**:', 1).replace('2:', '2**:', 1)
                                   .replace('3:', '3**:', 1), 3, ['Mixed Operations'], provenance)
        self.assertEqual(rows, expected)

    def test_previous_formats_and_modes_unchanged(self):
        descriptive = Path(__file__).with_name('fixtures') / 'descriptive_equations_reply.txt'
        for text in [batch(False), batch(), batch() + '\n\n' + SUMMARY,
                     batch() + '\n\n### Summary:\n' + SUMMARY, descriptive.read_text()]:
            self.assertEqual(formatting.parse_compatible(text, 3, [], {}, cp.parse_samples),
                             previous.parse_compatible(text, 3, [], {}, cp.parse_samples))

    def test_mixed_headers_crlf_and_same_line_question_supported(self):
        text = batch().replace('#### Question 1:\nEach', '**Question 1:** Each')
        text = text.replace('#### Question 2:', '**Question 2**:').replace('\n', '\r\n')
        rows, mode = formatting.parse_compatible(text, 3, [], {}, cp.parse_samples)
        self.assertEqual(mode, formatting.POLICY_VERSION)
        self.assertTrue(rows[0]['input'].startswith('Each bus'))
        self.assertEqual(len(rows), 3)

    def test_incomplete_ambiguous_and_additional_content_rejected(self):
        text = FIXTURE.read_text()
        bad = [text.replace('**Question 3:**', '**Question 2:**'),
               text.replace('**Question 2:**', '**Question 4:**'),
               text[:text.index('**Question 3:**')], text[:-3],
               text.replace('>>\n', '>>\n>>\n', 1),
               text.replace('**Answer**:\n>>\n', '**Answer**:\n', 1),
               text[:text.index('>>') + 2] + '\n<<\n' + text[text.index('---'):],
               text + '\n**Question 4:**\nExtra?\n**Answer**:\n>>\n12\n<<',
               text + '\nSolve for x.', text + '\n### Summary\nx = 3',
               '```markdown\n' + text + '\n```',
               text.replace('\n---\n', '\n---\nDiscard this explanation\n', 1)]
        for index, body in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                formatting.parse_compatible(body, 3, [], {}, cp.parse_samples)

    def test_separators_and_header_like_text_inside_answers_preserved(self):
        text = FIXTURE.read_text().replace('The total height of the plant after three weeks is 10 cm.',
                                          '**Question 99:**\n---\nThe total height is 10 cm.')
        rows, _ = formatting.parse_compatible(text, 3, [], {}, cp.parse_samples)
        self.assertIn('**Question 99:**\n---\nThe total height is 10 cm.', rows[0]['output'])
        _, edits = formatting.normalize_bold_questions(text, 3)
        self.assertEqual(len(edits['removed_boundary_rule_lines']), 2)

    def test_original_policies_cache_bytes_and_parser_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = Path(tmp), {'seed': 42}
            with previous.sample_format_compatibility(cfg, run):
                pass
            old = {p: p.read_bytes() for p in run.rglob('*.json')}
            text = FIXTURE.read_text()
            key, path = cache(run, text)
            raw = path.read_bytes()
            original = cp.parse_samples
            provenance = {'stage': 'fine', 'request': key}
            with formatting.sample_format_compatibility(cfg, run):
                rows = cp.parse_samples(text, 3, [], provenance)
                self.assertEqual(cp.parse_samples(text, 3, [], provenance), rows)
            self.assertIs(cp.parse_samples, original)
            self.assertEqual(path.read_bytes(), raw)
            for p, data in old.items():
                self.assertEqual(p.read_bytes(), data)
            record = cp.read(run / 'teacher_sample_bold_colon_parsing' /
                             (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual(record['parsed_count'], 3)
            self.assertEqual(record['sample_ids'], [r['item_id'] for r in rows])

    def test_incomplete_or_changed_cache_is_rejected(self):
        for finish, change in [('length', False), ('stop', True)]:
            with tempfile.TemporaryDirectory() as tmp:
                run, text = Path(tmp), FIXTURE.read_text()
                key, _ = cache(run, text, finish)
                if change:
                    text = text.replace('Georgia', 'Georgina')
                with formatting.sample_format_compatibility({}, run):
                    with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                        cp.parse_samples(text, 3, [], {'stage': 'fine', 'request': key})

    def test_preview_is_offline_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, FIXTURE.read_text())
            config = run / 'config.json'
            cp.write(config, {'run_dir': str(run)})
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            out = io.StringIO()
            with patch.object(job, 'CONFIG_PATH', config), patch.object(job, 'load') as load, \
                 patch.object(ot, 'rpc') as network, contextlib.redirect_stdout(out):
                entry.preview(key, 3)
                load.assert_not_called()
                network.assert_not_called()
            result = json.loads(out.getvalue())
            self.assertEqual((result['parsed_count'], result['preview_only']), (3, True))
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})

    def test_entry_keeps_frozen_workflow_repeat_fix_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = local_tests.LocalCdsTests()
            source = fixture.fixture(root)
            original_teacher, original_parser = cp.Teacher, cp.parse_samples
            original_context = previous.sample_format_compatibility
            with fixture.context(root, source):
                cfg, run = job.prepare()
                key, path = cache(run, FIXTURE.read_text())
                cache_bytes = path.read_bytes()

                def synthesize(cfg, run, args):
                    self.assertIs(cp.Teacher, repeat_fix.ContextRepeatTeacher)
                    rows = cp.parse_samples(FIXTURE.read_text(), 3, [], {'stage': 'fine', 'request': key})
                    self.assertEqual(len(rows), 3)

                with patch.object(cp, 'synthesize', side_effect=synthesize), \
                     patch.object(ot, 'rpc') as network, \
                     patch.object(sys, 'argv', ['qwen15_cds_ollama_samples.py', 'synthesize']):
                    entry.main()
                    network.assert_not_called()
                self.assertEqual(path.read_bytes(), cache_bytes)
                job.load()  # Frozen protocol/source hashes still match after using the new entry.
            self.assertIs(cp.Teacher, original_teacher)
            self.assertIs(cp.parse_samples, original_parser)
            self.assertIs(previous.sample_format_compatibility, original_context)


if __name__ == '__main__':
    unittest.main()
