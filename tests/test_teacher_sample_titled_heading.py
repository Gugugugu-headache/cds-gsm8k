"""Retain descriptive titles and body text while preserving frozen Q10 behavior."""
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
import qwen15_cds_ollama_samples as entry
import qwen15_cds_quality10 as q10
import teacher_sample_difficulty_heading as previous
import teacher_sample_titled_heading as formatting
from test_teacher_sample_format import batch, cache
from test_teacher_sample_footer import SUMMARY
from test_ollama_teacher import reply, metadata
import test_qwen15_quality10 as q10_tests

FIXTURES = Path(__file__).with_name('fixtures')
FIXTURE = FIXTURES / 'descriptive_title_fusion_reply.txt'


class TitledHeadingTests(unittest.TestCase):
    def test_actual_191_lines_retain_titles_bodies_answers_ids_and_original_math_error(self):
        text = FIXTURE.read_text()
        provenance = {'stage': 'fusion', 'request': 'a' * 64, 'draw': 315}
        with self.assertRaisesRegex(ValueError, 'No recognized difficulty-tagged'):
            previous.parse_compatible(text, 3, ['Financial Calculations'], provenance, cp.parse_samples)
        rows, mode = formatting.parse_compatible(
            text, 3, ['Financial Calculations'], provenance, cp.parse_samples)
        self.assertEqual(mode, formatting.POLICY_VERSION)
        lines = text.splitlines()
        self.assertEqual(len(lines), 191)
        for row, h, q, first, last in zip(rows, [1, 79, 145], [2, 80, 146],
                                          [6, 84, 150], [74, 140, 190]):
            title = lines[h - 1].split(': ', 1)[1][:-2]
            self.assertEqual(row['input'], title + '\n' + lines[q - 1])
            self.assertEqual(row['output'], '\n'.join(lines[first - 1:last]))
            self.assertEqual(row['provenance'], provenance)
        self.assertIn('180000 - 90000 - 12000 + 2000 = 70000', rows[2]['output'])
        self.assertIn('$70,000', rows[2]['output'])
        _, normalized, _ = formatting.parse_titled_only(
            text, 3, ['Financial Calculations'], provenance, cp.parse_samples)
        expected, _ = previous.parse_compatible(normalized, 3, ['Financial Calculations'],
                                               provenance, cp.parse_samples)
        self.assertEqual(rows, expected)

    def test_previously_supported_formats_and_modes_stay_unchanged(self):
        texts = [batch(False), batch(), batch() + '\n\n' + SUMMARY,
                 batch() + '\n\n### Summary:\n' + SUMMARY]
        texts += [(FIXTURES / name).read_text() for name in [
            'descriptive_equations_reply.txt', 'bold_colon_ollama_reply.txt',
            'missing_colon_probability_reply.txt', 'difficulty_heading_fusion_reply.txt']]
        for text in texts:
            self.assertEqual(formatting.parse_compatible(text, 3, [], {}, cp.parse_samples),
                             previous.parse_compatible(text, 3, [], {}, cp.parse_samples))

    def test_mixed_header_formats_and_semantic_titles_are_retained(self):
        text = FIXTURE.read_text().replace('Advanced Financial Planning', 'Taxes: apply the given 15% rate', 1)
        text = text.replace('**Question 2: Investment and Tax Calculations**', '#### Question 2')
        text = text.replace('**Question 3: Combined Cost and Profit Analysis**', '**Question 3**:')
        rows, _ = formatting.parse_compatible((text + '\n\n### Summary:\n' + SUMMARY).replace('\n', '\r\n'),
                                               3, [], {}, cp.parse_samples)
        self.assertTrue(rows[0]['input'].startswith('Taxes: apply the given 15% rate\r\n'))
        self.assertTrue(rows[1]['input'].startswith('An investor'))
        self.assertEqual(len(rows), 3)

    def test_titles_cannot_replace_absent_questions_or_hide_extra_heading_content(self):
        text = FIXTURE.read_text()
        lines = text.splitlines()
        bad = [text.replace(lines[1], '', 1),
               text.replace('**Question 1: Advanced Financial Planning**', '**Question 1: **'),
               text.replace('Advanced Financial Planning**', 'Advanced **Financial** Planning**', 1),
               text.replace('Advanced Financial Planning**', 'Advanced Financial Planning** extra text', 1),
               text.replace('**Question 2:', '**Question 1:', 1),
               text.replace('**Question 2:', '**Question 4:', 1),
               text[:text.index('**Question 3:')],
               text + '\n**Question 4: Extra Topic**\nAnother question?\n**Answer**:\n>>\n1\n<<']
        for i, body in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                formatting.parse_compatible(body, 3, [], {}, cp.parse_samples)

    def test_answer_and_footer_guards_and_heading_text_inside_answers(self):
        text = FIXTURE.read_text()
        fence = chr(96) * 3
        bad = [text[:-3], text.replace('>>\n', '>>\n>>\n', 1),
               text.replace('**Answer**:\n', '', 1),
               text[:text.index('>>') + 2] + '\n<<\n' + text[text.index('**Question 2:'):],
               text + '\nNow solve an extra task.', text + '\n### Summary:\nx = 3',
               fence + 'markdown\n' + text + fence]
        for i, body in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                formatting.parse_compatible(body, 3, [], {}, cp.parse_samples)
        inside = text.replace('1. Calculate the mortgage amount:',
                              '**Question 99: Additional Details**\n---\n1. Calculate the mortgage amount:', 1)
        rows, _ = formatting.parse_compatible(inside, 3, [], {}, cp.parse_samples)
        self.assertIn('**Question 99: Additional Details**\n---', rows[0]['output'])

    def test_prior_policies_and_cache_bytes_retained_with_idempotent_new_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = Path(tmp), {'seed': 42}
            with previous.sample_format_compatibility(cfg, run):
                pass
            policies = {p: p.read_bytes() for p in run.rglob('*.json')}
            text = FIXTURE.read_text()
            key, path = cache(run, text)
            raw, original = path.read_bytes(), cp.parse_samples
            provenance = {'stage': 'fusion', 'request': key, 'draw': 315}
            with formatting.sample_format_compatibility(cfg, run):
                rows = cp.parse_samples(text, 3, [], provenance)
                self.assertEqual(cp.parse_samples(text, 3, [], provenance), rows)
            self.assertIs(cp.parse_samples, original)
            self.assertEqual(path.read_bytes(), raw)
            for path, data in policies.items():
                self.assertEqual(path.read_bytes(), data)
            audit = cp.read(run / 'teacher_sample_titled_heading_parsing' /
                            (key + '.' + cp.digest(provenance) + '.json'))
            self.assertEqual([e['line'] for e in audit['title_heading_edits']], [1, 79, 145])
            self.assertTrue(all(e['retained_as'] == 'first line of question input'
                                for e in audit['title_heading_edits']))
            self.assertEqual(audit['sample_ids'], [r['item_id'] for r in rows])

    def test_truncated_or_tampered_cache_is_rejected(self):
        for finish, change in [('length', False), ('stop', True)]:
            with tempfile.TemporaryDirectory() as tmp:
                run, text = Path(tmp), FIXTURE.read_text()
                key, _ = cache(run, text, finish)
                if change:
                    text = text.replace('$350,000', '$400,000', 1)
                with formatting.sample_format_compatibility({}, run):
                    with self.assertRaisesRegex(ValueError, 'complete, verified cache'):
                        cp.parse_samples(text, 3, [], {'stage': 'fusion', 'request': key})

    def test_explicit_preview_is_read_only_and_reports_retained_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            key, _ = cache(run, FIXTURE.read_text())
            cp.write(run / 'config.frozen.json', {'run_dir': str(run)})
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            out = io.StringIO()
            with patch.object(ot, 'rpc') as rpc, contextlib.redirect_stdout(out), \
                 patch.object(sys, 'argv', ['samples.py', 'parse-cache', '--run', str(run),
                                           '--request-key', key, '--count', '3']):
                entry.main()
                rpc.assert_not_called()
            report = json.loads(out.getvalue())
            self.assertEqual((report['parsed_count'], report['preview_only'], report['parser']),
                             (3, True, formatting.POLICY_VERSION))
            self.assertTrue(report['questions'][0].startswith('Advanced Financial Planning\n'))
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})

    def test_frozen_q10_augmentation_preserves_migration_and_replays_cached_response(self):
        with tempfile.TemporaryDirectory() as tmp, q10_tests.fixture(Path(tmp)) as (_, source, _):
            q10.prepare()
            run = cp.resolve(q10.RUN_DIR)
            migration = (run / 'quality10_migration.json').read_bytes()
            source_bytes = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            chats = []

            def fake(base, route, payload=None, timeout=30):
                if route != '/api/chat':
                    return metadata(base, route, payload, timeout)
                chats.append(payload)
                return reply(FIXTURE.read_text())

            with patch.object(ot, 'rpc', side_effect=fake), \
                 patch.object(sys, 'argv', ['qwen15_cds_quality10.py', 'augment']):
                q10.main()
                q10.main()
            q10.load()
            self.assertEqual(len(chats), 1)
            rows = cp.read(run / 'augmented.json')
            self.assertEqual(len(rows), 3)
            self.assertIn('= 70000', rows[2]['output'])
            self.assertEqual((run / 'quality10_migration.json').read_bytes(), migration)
            for path, content in source_bytes.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
