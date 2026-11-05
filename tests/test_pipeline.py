import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cds


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cds.read(Path(cds.__file__).with_name('configs') / 'cds_gsm8k.json')

    def test_unparseable_answer_is_wrong_and_diagnosable(self):
        row = {'label': ['Addition'], **cds.grade('I cannot solve this.', '#### 10')}
        self.assertFalse(row['correct'])
        self.assertEqual(row['extraction'], 'no_number')
        profile = cds.profile_rows([row], ['Addition', 'Division'], self.cfg)
        self.assertEqual(profile['Addition']['support'], 1)
        self.assertEqual(profile['Addition']['accuracy'], 0)
        self.assertIsNone(profile['Division']['accuracy'])
        self.cfg['zero_support_policy'] = 'exclude'
        self.assertFalse(cds.profile_rows([row], ['Division'], self.cfg)['Division']['weak'])

    def test_source_answer_precedence(self):
        self.assertTrue(cds.grade('#### 1,000\nI used 5 steps.', '#### 1000')['correct'])
        self.assertTrue(cds.grade('Thus the answer is -0.5', '#### -0.5')['correct'])
        self.assertTrue(cds.grade('We get 12', '#### 12')['correct'])

    def test_synthesis_keeps_alignment_and_rejects_partial_batches(self):
        response = '**Question**: Q1\n**Answer**:\n>>\nA1\n<<\n**Question 2**: Q2\n**Answer**:\n>>\nA2\n<<'
        rows = cds.parse_samples(response, 2, ['Addition'], {'source': 'test'})
        self.assertEqual([r['output'] for r in rows], ['A1', 'A2'])
        self.assertNotEqual(rows[0]['item_id'], rows[1]['item_id'])
        with self.assertRaises(ValueError):
            cds.parse_samples(response, 3, ['Addition'], {})

    def test_score_matches_upstream_function(self):
        # Execute only the pure function AST; no numpy/torch needed for CPU tests.
        import ast
        import math
        source = ast.parse((cds.UPSTREAM / '4_Q_score/Qscore.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'cal_difficulty_weights')
        namespace = {'math': math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<upstream>', 'exec'), namespace)
        rows = [{'label': ['Addition']}, {'label': ['Division']}, {'label': ['Addition', 'Division']}]
        kcs = {'Addition': {'accuracy': 0.8}, 'Division': {'accuracy': 0.2}}
        scored, threshold, weights = cds.score_pool(rows, kcs, self.cfg)
        expected = namespace['cal_difficulty_weights']({'Addition': 0.6667, 'Division': 0.6667},
                                                     {'Addition': 0.8, 'Division': 0.2}, 'max')
        self.assertEqual(weights, expected)
        self.assertGreater(weights['Division'], weights['Addition'])
        self.assertEqual(scored[-1]['cds_score'], sum(expected.values()))

    def test_diagnosis_fallback_is_explicit(self):
        self.assertEqual(cds.parse_unmastered('Malformed', ['Addition'], ['Addition']), (['Addition'], True))
        text = 'Unmastered Knowledge Points: [Division]\nMastered Knowledge Points: [Addition]'
        self.assertEqual(cds.parse_unmastered(text, ['Addition', 'Division'], ['Addition']), (['Division'], False))

    def test_teacher_cache_and_nonstop_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = cds.Teacher(self.cfg, Path(tmp))
            payload = {'model': self.cfg['teacher']['model'], 'messages': [cds.user_message('hello')],
                       'temperature': 0.5, 'top_p': 0.8, 'max_tokens': 4096, 'stream': False}
            key = cds.digest({'url': teacher.url, 'payload': payload})
            cds.write(teacher.cache / (key + '.json'), {'content': 'cached', 'finish_reason': 'stop'})
            with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
                self.assertEqual(teacher.call(payload['messages']), ('cached', key))
            path = teacher.cache / (key + '.json')
            path.write_text(json.dumps({'content': 'partial', 'finish_reason': 'length'}))
            with self.assertRaises(ValueError):
                teacher.call(payload['messages'])

    def test_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'artifact.json'
            cds.write(path, {'x': 1})
            cds.write(path, {'x': 1})
            with self.assertRaises(ValueError):
                cds.write(path, {'x': 2})

    def test_jsonl_unicode_separators_stay_inside_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.jsonl'
            rows = [{'question': 'first\u2028second'}, {'question': 'third\u0085fourth'}]
            path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n')
            self.assertEqual(cds.read(path), rows)

    def test_offline_synthesis_to_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            evaluation = {'records': [{'item_id': 'train:1', 'input': 'Original question',
                'response': 'Incorrect result 2', 'correct': False, 'label': ['Addition']}]}
            cds.write(run / 'eval/baseline.json', evaluation)
            cds.write(run / 'profile.json', {'evaluation': 'baseline',
                'evaluation_sha256': cds.file_hash(run / 'eval/baseline.json'),
                'kcs': {'Addition': {'weak': True, 'accuracy': 0.5}}})
            cds.write(run / 'data/vocab.json', ['Addition'])
            for name in ['test', 'dev', 'warmup']:
                cds.write(run / 'data' / (name + '.json'), [{'input': 'Held-out question'}])
            requests = []

            def fake_call(teacher, messages, max_tokens=4096, scoring=False):
                requests.append(messages)
                key = cds.digest(messages)
                if scoring:
                    return 'Score: 10||Correct.', key
                if max_tokens == 1024:
                    return 'Unmastered Knowledge Points: [Addition]\nMastered Knowledge Points: []', key
                count = int(__import__('re').search(r'generate (\d+) samples', messages[-1]['content']).group(1))
                return '\n'.join(f'**Question**: {key} question {i}\n**Answer**:\n>>\n1+1=2\n<<' for i in range(count)), key

            with patch.object(cds.Teacher, 'call', fake_call):
                cds.synthesize(self.cfg, run, SimpleNamespace())
                cds.quality(self.cfg, run, SimpleNamespace(pool='seed'))
                cds.augment(self.cfg, run, SimpleNamespace())
                cds.quality(self.cfg, run, SimpleNamespace(pool='augmented'))
                cds.select(self.cfg, run, SimpleNamespace())
            self.assertEqual(len(cds.read(run / 'synthesis.json')), 8)
            selected = cds.read(run / 'train_cds.json')
            self.assertTrue(selected)
            self.assertEqual(len(selected), len(cds.read(run / 'train_random.json')))
            # Fine synthesis must carry the teacher diagnosis in conversational context.
            self.assertTrue(any(len(messages) == 3 and messages[1]['role'] == 'assistant' for messages in requests))

    def test_paired_statistics_and_alignment(self):
        from compare import compare
        before_rows = [{'item_id': f'test:{i}', 'input': f'Question {i}', 'output': '#### 1',
                        'response': '#### 0', 'correct': False} for i in range(1319)]
        after_rows = [{**row, 'response': '#### 1' if i < 10 else '#### 0', 'correct': i < 10}
                      for i, row in enumerate(before_rows)]
        before = {'meta': {'split': 'test', 'prompt': 'source_plain'}, 'records': before_rows}
        after = {'meta': {'split': 'test', 'prompt': 'source_plain'}, 'records': after_rows}
        result = compare(before, after, repeats=40)
        self.assertEqual(result['wrong_to_right'], 10)
        self.assertEqual(result['right_to_wrong'], 0)
        self.assertAlmostEqual(result['mcnemar_exact_two_sided_p'], 2 / 1024)
        after['records'][0]['item_id'] = 'different'
        with self.assertRaises(ValueError):
            compare(before, after, repeats=40)


if __name__ == '__main__':
    unittest.main()
