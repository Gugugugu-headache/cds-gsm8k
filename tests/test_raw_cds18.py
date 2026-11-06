"""18-KC migration uses actual annotations and synthetic responses in /tmp only."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import initialization_pilot as pilot
import raw_cds as raw
import raw_cds18 as kc18
from test_raw_cds import fake_evaluation


class Kc18Tests(unittest.TestCase):
    def config(self):
        cfg = cp.read(Path(kc18.__file__).with_name('configs') / 'cds_gsm8k.example.json')
        cfg.update(student_model='/models/Qwen2.5-3B-Instruct',
                   student_revision_note='Qwen/Qwen2.5-3B-Instruct@' + 'a' * 40)
        return cfg

    def test_real_migration_preserves_every_response_split_and_old_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config()
            cfg['run_dir'] = str(root / 'old')
            source = root / 'configs/cds_gsm8k.json'
            cp.write(source, cfg)
            _, old = cp.init_run(source)
            cp.prepare(cfg, old)
            cp.write(old / 'eval/raw_test.json',
                     fake_evaluation(cp.read(old / 'data/test.json'), cfg, 'test'))
            with patch.object(pilot, 'PILOT_RUN', str(root / 'pilot')), \
                 patch.object(raw, 'RUN_DIR', str(root / 'raw28')), \
                 patch.object(raw, 'CONFIG_PATH', root / 'configs/raw28.json'), \
                 patch.object(kc18, 'RUN_DIR', str(root / 'raw18')), \
                 patch.object(kc18, 'CONFIG_PATH', root / 'configs/raw18.json'), \
                 patch.object(cp, 'Teacher') as teacher, patch.object(cp, 'infer') as infer:
                pilot.prepare(source)
                cfg28, run28 = raw.prepare()
                response = fake_evaluation(cp.read(run28 / 'data/diag.json'), cfg28, 'diag')
                cp.write(run28 / 'eval/raw_diag.json', response)
                originals = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
                cfg18, run18 = kc18.prepare()
                kc18.load()
                imported = cp.read(run18 / 'eval/raw_diag.json')
                self.assertEqual(len(imported['records']), 3500)
                self.assertTrue(imported['meta']['responses_reused'])
                self.assertEqual(imported['meta']['generation_config_hash'], cp.digest(cfg28))
                self.assertEqual(imported['meta']['config_hash'], cp.digest(cfg18))
                for a, b in zip(response['records'], imported['records']):
                    self.assertEqual({k: v for k, v in a.items() if k != 'label'},
                                     {k: v for k, v in b.items() if k != 'label'})
                for name in ['diag', 'test', 'dev', 'reserve', 'warmup', 'initialization_holdout']:
                    a = cp.read(run28 / 'data' / (name + '.json'))
                    b = cp.read(run18 / 'data' / (name + '.json'))
                    self.assertEqual([{k: v for k, v in r.items() if k != 'label'} for r in a],
                                     [{k: v for k, v in r.items() if k != 'label'} for r in b])
                    self.assertFalse(any('Angles' in r['label'] for r in b))
                report = kc18.profile()
                self.assertEqual(report['kc_count'], 18)
                self.assertEqual(report['selected_for_synthesis'], 10)
                self.assertEqual(report['reason_counts'], {
                    'low_accuracy': 0, 'low_coverage': 10, 'zero_support': 0})
                self.assertEqual(report['kcs']['Probability']['support'], 13)
                kc18.prepare()  # Can rerun without regeneration or changing provenance.
                for path, content in originals.items():
                    self.assertEqual(path.read_bytes(), content)
                teacher.assert_not_called()
                infer.assert_not_called()
                # Source/import mutation cannot silently continue the experiment.
                destination = run18 / 'eval/raw_diag.json'
                saved = destination.read_bytes()
                destination.write_text('{}')
                with self.assertRaises(ValueError):
                    kc18.load()
                destination.write_bytes(saved)
                (run28 / 'eval/raw_diag.json').write_text('{}')
                with self.assertRaises(ValueError):
                    kc18.load()

    def test_reuse_refuses_prompt_answer_or_model_changes(self):
        cfg = self.config()
        row = {'item_id': 'a', 'instruction': cp.INSTRUCTION, 'input': 'q',
               'output': '#### 2', 'label': ['Addition', 'Angles']}
        after = {**row, 'label': ['Addition']}
        result = fake_evaluation([row], cfg, 'diag')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'response.json'
            cp.write(path, result)
            for field, value in [('input', 'different'), ('instruction', 'different'), ('output', '#### 3')]:
                with self.assertRaises(ValueError):
                    kc18.reuse_answers(result, [row], [{**after, field: value}], cfg, cfg, path)
            for delta in [{'inference_max_tokens': 512}, {'student_model': '/different'},
                          {'student_revision_note': 'different'}, {'seed': 0}]:
                with self.assertRaises(ValueError):
                    kc18.reuse_answers(result, [row], [after], cfg, {**cfg, **delta}, path)

    def test_relabel_uses_authoritative_main_labels_and_checks_old_annotations(self):
        row = {'item_id': 'a', 'input': 'q', 'output': '#### 2', 'label': ['Addition', 'Angles']}
        annotation = {'a': {'question': 'q', 'answer': '#### 2',
                            'kc_labels': ['Addition', 'Angles'], 'diagnosis_kcs': ['Addition']}}
        self.assertEqual(kc18.relabel([row], annotation)[0]['label'], ['Addition'])
        self.assertEqual(row['label'], ['Addition', 'Angles'])
        with self.assertRaises(ValueError):
            kc18.relabel([{**row, 'label': ['Division']}], annotation)

    def test_profile_separates_accuracy_coverage_and_absence(self):
        cfg = self.config()
        stats = {
            'low': {'support': 1000, 'accuracy': 0.4, 'frequency': 0.3, 'weak': True},
            'rare': {'support': 13, 'accuracy': 1.0, 'frequency': 13/3500, 'weak': True},
            'zero': {'support': 0, 'accuracy': None, 'frequency': 0, 'weak': True},
            'both': {'support': 20, 'accuracy': 0.5, 'frequency': 20/3500, 'weak': True},
            'strong': {'support': 1000, 'accuracy': 0.95, 'frequency': 0.3, 'weak': False}}
        report = kc18.explain_profile(stats, cfg)
        self.assertEqual(report['reason_counts'], {'low_accuracy': 2, 'low_coverage': 2, 'zero_support': 1})
        self.assertEqual(report['kcs']['rare']['selection_reasons'], ['low_coverage'])
        self.assertEqual(report['kcs']['zero']['selection_reasons'], ['zero_support'])
        stats['zero']['weak'] = False
        cfg['zero_support_policy'] = 'exclude'
        self.assertEqual(kc18.explain_profile(stats, cfg)['selected_for_synthesis'], 3)
        stats['rare']['weak'] = False
        with self.assertRaises(ValueError):
            kc18.explain_profile(stats, cfg)

    def test_forwarding_uses_18_config_and_restores_original_globals(self):
        old = raw.CONFIG_PATH, raw.RUN_DIR
        with patch.object(kc18, 'load'), patch.object(raw, 'main') as main, \
             patch.object(sys, 'argv', ['raw_cds18.py', 'select']):
            main.side_effect = lambda: self.assertEqual(raw.CONFIG_PATH, kc18.CONFIG_PATH)
            kc18.main()
            main.assert_called_once()
        self.assertEqual((raw.CONFIG_PATH, raw.RUN_DIR), old)
        with patch.object(sys, 'argv', ['raw_cds18.py', 'infer', '--split', 'diag']):
            with self.assertRaises(ValueError):
                kc18.main()


if __name__ == '__main__':
    unittest.main()
