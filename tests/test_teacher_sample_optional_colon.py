"""Missing-colon headings must not change sample content or weaken validation."""
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
import teacher_sample_descriptive_footer as hook
import teacher_sample_bold_colon as previous
import teacher_sample_optional_colon as formatting
from test_teacher_sample_format import batch, cache
from test_teacher_sample_footer import SUMMARY
import test_qwen15_cds_ollama as local_tests

FIXTURES = Path(__file__).with_name('fixtures')
FIXTURE = FIXTURES / 'missing_colon_probability_reply.txt'


class OptionalColonTests(unittest.TestCase):
    def test_actual_74_line_reply_keeps_all_questions_answers_ids_and_labels(self):
        text = FIXTURE.read_text()
        provenance = {'stage': 'fine', 'request': 'a' * 64, 'source_item': 'diagnosis:517'}
        with self.assertRaisesRegex(ValueError, 'Bold question fallback requires exactly '):
            previous.parse_compatible(text, 3, ['Probability'], provenance, cp.parse_samples)
        rows, mode = formatting.parse_compatible(text, 3, ['Probability'], provenance, cp.parse_samples)
        self.assertEqual(mode, formatting.POLICY_VERSION)
        lines = text.splitlines()
        self.assertEqual(len(lines), 74)
        for row, q, first, last in zip(rows, [4, 24, 50], [8, 28, 54], [20, 46, 73]):
            self.assertEqual(row['input'], lines[q - 1])
            self.assertEqual(row['output'], '\n'.join(lines[first - 1:last]))
            self.assertEqual(row['provenance'], provenance)
            self.assertEqual(row['label'], ['Probability'])
            self.assertNotIn('#### Question', row['input'])
            self.assertNotIn('**Answer**', row['output'])
            self.assertNotIn('>>', row['output'])
        canonical = text
        for i in range(1, 4):
            canonical = canonical.replace(f'#### Question {i}\n', f'**Question {i}**:\n')
        self.assertEqual(rows, cp.parse_samples(canonical, 3, ['Probability'], provenance))

    def test_previously_supported_formats_and_ids_unchanged(self):
        for text in [batch(False), batch(), batch() + '\n\n' + SUMMARY,
                     batch() + '\n\n### Summary:\n' + SUMMARY,
                     (FIXTURES / 'descriptive_equations_reply.txt').read_text(),
                     (FIXTURES / 'bold_colon_ollama_reply.txt').read_text()]:
            self.assertEqual(formatting.parse_compatible(text, 3, [], {}, cp.parse_samples),
                             previous.parse_compatible(text, 3, [], {}, cp.parse_samples))

    def test_mixed_headers_crlf_separators_and_summaries_supported(self):
        text = FIXTURE.read_text().replace('#### Question 2\n', '**Question 2:**\n')
        text = text.replace('#### Question 3\n', '**Question 3**:\n')
        text = text.replace('<<\n\n', '<<\n\n---\n\n')
        for tail in ['', '\n\n### Summary:\n' + SUMMARY]:
            rows, mode = formatting.parse_compatible((text + tail).replace('\n', '\r\n'),
                                                      3, [], {}, cp.parse_samples)
            self.assertEqual(mode, formatting.POLICY_VERSION)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['input'], FIXTURE.read_text().splitlines()[3])

    def test_missing_duplicate_out_of_order_or_extra_questions_rejected(self):
        text = FIXTURE.read_text()
        bad = [text.replace('#### Question 2\n', '#### Question 1\n'),
               text.replace('#### Question 2\n', '#### Question 4\n'),
               text[:text.index('#### Question 3')],
               text.replace('#### Question 2\n', '#### Question 2 extra words\n'),
               text + '\n#### Question 4\nExtra?\n**Answer**:\n>>\n12\n<<',
               text.replace('#### Question 1\n', '#### Question 0\n'),
               text.replace('#### Question 2\n', '#### Question 02\n')]
        for i, body in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                formatting.parse_compatible(body, 3, [], {}, cp.parse_samples)

    def test_answer_completeness_and_trailing_content_guards_remain(self):
        text = FIXTURE.read_text()
        fence = chr(96) * 3
        bad = [text[:-3], text.replace('>>\n', '>>\n>>\n', 1),
               text.replace('**Answer**:\n', '', 1),
               text[:text.index('>>') + 2] + '\n<<\n' + text[text.index('#### Question 2'):],
               text + '\nSolve the following additional problem.',
               text + '\n### Summary:\nx = 3',
               text.replace('<<\n\n', '<<\nUnrecognized interstitial text\n', 1),
               fence + 'markdown\n' + text + '\n' + fence]
        for i, body in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                formatting.parse_compatible(body, 3, [], {}, cp.parse_samples)
        inside = text.replace('we need to follow these steps:', 'we need to follow these steps:\n#### Question 99\n---', 1)
        rows, _ = formatting.parse_compatible(inside, 3, [], {}, cp.parse_samples)
        self.assertIn('#### Question 99\n---', rows[0]['output'])

    def test_old_policies_caches_and_parser_restored_with_idempotent_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = Path(tmp), {'seed': 42}
            with previous.sample_format_compatibility(cfg, run):
                pass
            old = {p: p.read_bytes() for p in run.rglob('*.json')}
            text = FIXTURE.read_text()
            key, path = cache(run, text)
            raw, original = path.read_bytes(), cp.parse_samples
            provenance = {'stage': 'fine', 'request': key}
            with formatting.sample_format_compatibility(cfg, run):
                rows = cp.parse_samples(text, 3, ['Probability'], provenance)
                self.assertEqual(cp.parse_samples(text, 3, ['Probability'], provenance), rows)
            self.assertIs(cp.parse_samples, original)
            self.assertEqual(path.read_bytes(), raw)
            for p, data in old.items():
                self.assertEqual(p.read_bytes(), data)
            record = cp.read(run / 'teacher_sample_optional_colon_parsing' /
                             (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual(record['edits']['missing_colon_heading_lines'], [3, 23, 49])
            self.assertEqual(record['sample_ids'], [row['item_id'] for row in rows])

    def test_length_or_tampered_cache_still_rejected(self):
        for finish, change in [('length', False), ('stop', True)]:
            with tempfile.TemporaryDirectory() as tmp:
                run, text = Path(tmp), FIXTURE.read_text()
                key, _ = cache(run, text, finish)
                if change:
                    text = text.replace('5 red', '7 red')
                with formatting.sample_format_compatibility({}, run):
                    with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                        cp.parse_samples(text, 3, [], {'stage': 'fine', 'request': key})

    def test_same_entry_preview_is_read_only_and_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, FIXTURE.read_text())
            config = run / 'config.json'
            cp.write(config, {'run_dir': str(run)})
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            out = io.StringIO()
            with patch.object(job, 'CONFIG_PATH', config), patch.object(job, 'load') as load, \
                 patch.object(ot, 'rpc') as network, contextlib.redirect_stdout(out), \
                 patch.object(sys, 'argv', ['samples.py', 'parse-cache', '--request-key', key, '--count', '3']):
                entry.main()
                load.assert_not_called()
                network.assert_not_called()
            result = json.loads(out.getvalue())
            self.assertEqual((result['parsed_count'], result['preview_only'], result['parser']),
                             (3, True, formatting.POLICY_VERSION))
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})

    def test_same_entry_retains_native_teacher_and_frozen_run_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = local_tests.LocalCdsTests()
            source = fixture.fixture(root)
            parser, teacher, context = cp.parse_samples, cp.Teacher, hook.sample_format_compatibility
            with fixture.context(root, source):
                _, run = job.prepare()
                key, path = cache(run, FIXTURE.read_text())
                raw = path.read_bytes()

                def synthesize(cfg, run, args):
                    self.assertIs(cp.Teacher, repeat_fix.ContextRepeatTeacher)
                    self.assertEqual(len(cp.parse_samples(FIXTURE.read_text(), 3, [],
                                                          {'stage': 'fine', 'request': key})), 3)

                with patch.object(cp, 'synthesize', side_effect=synthesize), \
                     patch.object(ot, 'rpc') as network, \
                     patch.object(sys, 'argv', ['samples.py', 'synthesize']):
                    entry.main()
                    network.assert_not_called()
                job.load()
                self.assertEqual(path.read_bytes(), raw)
            self.assertIs(cp.parse_samples, parser)
            self.assertIs(cp.Teacher, teacher)
            self.assertIs(hook.sample_format_compatibility, context)


if __name__ == '__main__':
    unittest.main()
