"""Difficulty heading metadata must not alter mathematics or frozen Q10 artifacts."""
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
import qwen15_cds_ollama_samples as entry
import qwen15_cds_quality10 as q10
import teacher_sample_difficulty_heading as formatting
import teacher_sample_optional_colon as previous
from test_teacher_sample_format import batch, cache
from test_teacher_sample_footer import SUMMARY
from test_ollama_teacher import reply, metadata
import test_qwen15_quality10 as q10_tests

FIXTURES = Path(__file__).with_name('fixtures')
FIXTURE = FIXTURES / 'difficulty_heading_fusion_reply.txt'


class DifficultyHeadingTests(unittest.TestCase):
    def test_actual_reply_preserves_questions_answers_ids_and_known_bad_math(self):
        text = FIXTURE.read_text()
        provenance = {'stage': 'fusion', 'request': 'a' * 64, 'draw': 205}
        with self.assertRaisesRegex(ValueError, 'No standalone Markdown'):
            previous.parse_compatible(text, 3, ['Probability'], provenance, cp.parse_samples)
        rows, mode = formatting.parse_compatible(text, 3, ['Probability'], provenance, cp.parse_samples)
        self.assertEqual(mode, formatting.POLICY_VERSION)
        lines = text.splitlines()
        self.assertEqual(len(lines), 117)
        for row, q, first, last in zip(rows, [2, 32, 77], [6, 36, 81], [26, 71, 116]):
            self.assertEqual(row['input'], lines[q - 1])
            self.assertEqual(row['output'], '\n'.join(lines[first - 1:last]))
            self.assertEqual(row['provenance'], provenance)
        self.assertIn('93.6', rows[0]['output'])
        self.assertIn(r'\frac{94}{360}', rows[0]['output'])
        _, normalized, _ = formatting.parse_difficulty_only(
            text, 3, ['Probability'], provenance, cp.parse_samples)
        expected, _ = previous.parse_compatible(normalized, 3, ['Probability'], provenance, cp.parse_samples)
        self.assertEqual(rows, expected)
        self.assertNotIn('Junior High Level', rows[0]['input'])

    def test_all_previous_formats_and_modes_unchanged(self):
        texts = [batch(False), batch(), batch() + '\n\n' + SUMMARY,
                 batch() + '\n\n### Summary:\n' + SUMMARY]
        texts += [(FIXTURES / name).read_text() for name in [
            'descriptive_equations_reply.txt', 'bold_colon_ollama_reply.txt',
            'missing_colon_probability_reply.txt']]
        for text in texts:
            self.assertEqual(formatting.parse_compatible(text, 3, [], {}, cp.parse_samples),
                             previous.parse_compatible(text, 3, [], {}, cp.parse_samples))

    def test_mixed_headings_crlf_and_previous_summary_supported(self):
        text = FIXTURE.read_text().replace('**Question 2: Senior High Level**', '#### Question 2')
        text = text.replace('**Question 3: Advanced Senior High Level**', '#### Question 3:')
        for tail in ['', '\n\n### Summary:\n' + SUMMARY]:
            rows, mode = formatting.parse_compatible((text + tail).replace('\n', '\r\n'),
                                                      3, [], {}, cp.parse_samples)
            self.assertEqual(mode, formatting.POLICY_VERSION)
            self.assertEqual(rows[0]['input'], FIXTURE.read_text().splitlines()[1])
            self.assertEqual(len(rows), 3)

    def test_unknown_or_semantic_heading_tails_not_silently_removed(self):
        text = FIXTURE.read_text()
        bad = [
            text.replace('Junior High Level**', 'Junior High Level: use 94 students**', 1),
            text.replace('Junior High Level**', 'Unspecified Level**', 1),
            text.replace('Junior High Level**', 'Junior High Level** Extra question text', 1),
            text.replace('**Question 2:', '**Question 1:', 1),
            text.replace('**Question 2:', '**Question 4:', 1),
            text[:text.index('**Question 3:')],
            text + '\n**Question 4: Junior High Level**\nExtra?\n**Answer**:\n>>\n12\n<<']
        for i, body in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                formatting.parse_compatible(body, 3, [], {}, cp.parse_samples)

    def test_empty_unclosed_nested_and_trailing_content_still_rejected(self):
        text = FIXTURE.read_text()
        fence = chr(96) * 3
        bad = [text[:-3], text.replace('>>\n', '>>\n>>\n', 1),
               text.replace('**Answer**:\n', '', 1),
               text[:text.index('>>') + 2] + '\n<<\n' + text[text.index('**Question 2:'):],
               text + '\nNow solve for x.', text + '\n### Summary:\nx = 3',
               fence + 'markdown\n' + text + fence]
        for i, body in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                formatting.parse_compatible(body, 3, [], {}, cp.parse_samples)
        inside = text.replace('1. **Calculate the number of students in the junior class**:',
                              '**Question 99: Junior High Level**\n---\n'
                              '1. **Calculate the number of students in the junior class**:', 1)
        rows, _ = formatting.parse_compatible(inside, 3, [], {}, cp.parse_samples)
        self.assertIn('**Question 99: Junior High Level**\n---', rows[0]['output'])

    def test_cached_parsing_preserves_previous_policy_bytes_and_restores_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = Path(tmp), {'seed': 42}
            with previous.sample_format_compatibility(cfg, run):
                pass
            policies = {p: p.read_bytes() for p in run.rglob('*.json')}
            text = FIXTURE.read_text()
            key, path = cache(run, text)
            raw, original = path.read_bytes(), cp.parse_samples
            provenance = {'stage': 'fusion', 'request': key, 'draw': 205}
            with formatting.sample_format_compatibility(cfg, run):
                rows = cp.parse_samples(text, 3, [], provenance)
                self.assertEqual(cp.parse_samples(text, 3, [], provenance), rows)
            self.assertIs(cp.parse_samples, original)
            self.assertEqual(path.read_bytes(), raw)
            for path, data in policies.items():
                self.assertEqual(path.read_bytes(), data)
            audit = cp.read(run / 'teacher_sample_difficulty_heading_parsing' /
                            (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual(audit['parsed_count'], 3)
            self.assertEqual([e['line'] for e in audit['difficulty_heading_edits']], [1, 31, 76])
            self.assertEqual(audit['sample_ids'], [r['item_id'] for r in rows])

    def test_length_or_changed_cache_cannot_be_accepted(self):
        for finish, change in [('length', False), ('stop', True)]:
            with tempfile.TemporaryDirectory() as tmp:
                run, text = Path(tmp), FIXTURE.read_text()
                key, _ = cache(run, text, finish)
                if change:
                    text = text.replace('360 students', '400 students', 1)
                with formatting.sample_format_compatibility({}, run):
                    with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                        cp.parse_samples(text, 3, [], {'stage': 'fusion', 'request': key})

    def test_explicit_q10_preview_is_offline_read_only_and_checks_run_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, FIXTURE.read_text())
            cp.write(run / 'config.frozen.json', {'run_dir': str(run)})
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            out = io.StringIO()
            with patch.object(job, 'load') as load, patch.object(ot, 'rpc') as rpc, \
                 contextlib.redirect_stdout(out), patch.object(sys, 'argv', [
                     'samples.py', 'parse-cache', '--run', str(run),
                     '--request-key', key, '--count', '3']):
                entry.main()
                load.assert_not_called()
                rpc.assert_not_called()
            report = json.loads(out.getvalue())
            self.assertEqual((report['parsed_count'], report['preview_only'], report['parser']),
                             (3, True, formatting.POLICY_VERSION))
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})
            (run / 'config.frozen.json').write_text(json.dumps({'run_dir': str(run / 'other')}))
            with self.assertRaisesRegex(ValueError, 'Preview directory differs'):
                entry.preview(key, 3, run_dir=run)

    def test_frozen_q10_entry_accepts_new_format_and_retains_migration(self):
        with tempfile.TemporaryDirectory() as tmp, q10_tests.fixture(Path(tmp)) as (_, source, _):
            q10.prepare()
            run = cp.resolve(q10.RUN_DIR)
            migration = (run / 'quality10_migration.json').read_bytes()
            original = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            chats = []

            def fake(base, route, payload=None, timeout=30):
                if route != '/api/chat':
                    return metadata(base, route, payload, timeout)
                chats.append(payload)
                return reply(FIXTURE.read_text())

            with patch.object(ot, 'rpc', side_effect=fake), \
                 patch.object(sys, 'argv', ['qwen15_cds_quality10.py', 'augment']):
                q10.main()
            q10.load()
            self.assertEqual(len(chats), 1)
            rows = cp.read(run / 'augmented.json')
            self.assertEqual(len(rows), 3)
            self.assertIn('93.6', rows[0]['output'])
            self.assertEqual((run / 'quality10_migration.json').read_bytes(), migration)
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
