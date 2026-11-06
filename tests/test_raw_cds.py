"""Offline Raw-route tests, including real split export and fake response records."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import initialization_pilot as pilot
import raw_cds as raw


def fake_evaluation(rows, cfg, split, adapter=None):
    records = [{**r, 'response': r['output'], **cp.grade(r['output'], r['output']),
                'finish_reason': 'stop', 'output_tokens': 10} for r in rows]
    return {'meta': {'split': split, 'model': str(Path(cfg['student_model']).resolve()),
                     'adapter': adapter, 'n': len(rows), 'correct': len(rows), 'accuracy': 1.0,
                     'no_number': 0, 'truncated': 0, 'prompt': cfg['inference_prompt'],
                     'config_hash': cp.digest(cfg)}, 'records': records}


class RawRouteTests(unittest.TestCase):
    def config(self):
        cfg = cp.read(Path(raw.__file__).with_name('configs') / 'cds_gsm8k.example.json')
        cfg.update(student_model='/models/Qwen2.5-3B-Instruct',
                   student_revision_note='Qwen/Qwen2.5-3B-Instruct@' + 'a' * 40)
        return cfg

    def test_real_split_export_and_old_artifacts_preserved(self):
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
                 patch.object(raw, 'RUN_DIR', str(root / 'raw')), \
                 patch.object(raw, 'CONFIG_PATH', root / 'configs/cds_gsm8k_raw.json'), \
                 patch.object(cp, 'Teacher') as teacher:
                pilot.prepare(source)
                before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
                new_cfg, run = raw.prepare()
                raw.load()
                self.assertEqual(new_cfg['pre_finetune_size'], 0)
                self.assertEqual(new_cfg['training'], cfg['training'])
                self.assertEqual(cp.read(run / 'data/warmup.json'), [])
                self.assertEqual((run / 'data/initialization_holdout.json').read_bytes(),
                                 (old / 'data/warmup.json').read_bytes())
                for name in ['diag', 'test', 'dev', 'reserve']:
                    self.assertEqual((run / 'data' / (name + '.json')).read_bytes(),
                                     (root / 'pilot/raw/data' / (name + '.json')).read_bytes())
                raw.prepare()  # Idempotent; old files unchanged.
                for p, payload in before.items():
                    self.assertEqual(p.read_bytes(), payload)
                teacher.assert_not_called()
                for name in ['dev', 'vocab']:
                    data = run / 'data' / (name + '.json')
                    saved = data.read_bytes()
                    data.write_text('[]')
                    with self.assertRaises(ValueError):
                        raw.load()
                    data.write_bytes(saved)
                baseline = old / 'eval/raw_test.json'
                baseline.write_text('{}')
                with self.assertRaises(ValueError):
                    raw.load()

    def test_diagnosis_rejects_adapter_subset_wrong_config_and_wrong_labels(self):
        cfg = self.config()
        rows = [{'item_id': str(i), 'input': f'q{i}', 'output': '#### 2',
                 'instruction': cp.INSTRUCTION, 'label': ['kc']} for i in range(3)]
        good = fake_evaluation(rows, cfg, 'diag')
        raw.validate_evaluation(good, rows, cfg, 'diag')
        for delta in [{'adapter': '/old/M0'}, {'split': 'test'}, {'n': 2},
                      {'config_hash': 'old'}, {'correct': 2}]:
            bad = copy.deepcopy(good)
            bad['meta'].update(delta)
            with self.assertRaises(ValueError):
                raw.validate_evaluation(bad, rows, cfg, 'diag')
        bad = copy.deepcopy(good)
        bad['records'][0]['correct'] = False
        with self.assertRaises(ValueError):
            raw.validate_evaluation(bad, rows, cfg, 'diag')

    def test_selection_blocks_all_heldout_text_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            cfg = self.config()
            for name in ['diag', 'test', 'dev', 'initialization_holdout']:
                cp.write(run / 'data' / (name + '.json'), [{'input': f'blocked {name}'}])
            cp.write(run / 'eval/raw_diag.json', {'fake': 'only hash needed in this unit test'})
            cp.write(run / 'profile.json', {
                'evaluation_sha256': cp.file_hash(run / 'eval/raw_diag.json'),
                'kcs': {'kc': {'accuracy': 0.5}}})
            rows = [{'item_id': str(i), 'input': text, 'output': '#### 2', 'label': ['kc']}
                    for i, text in enumerate(['blocked diag', 'blocked test', 'blocked dev',
                                             'blocked initialization_holdout', 'good', 'good', 'other'])]
            cp.write(run / 'seed_quality.json', rows)
            cp.write(run / 'augmented_quality.json', [])
            raw.select_data(cfg, run)
            selected = cp.read(run / 'train_cds.json')
            control = cp.read(run / 'train_random.json')
            self.assertEqual({r['input'] for r in selected}, {'good', 'other'})
            self.assertEqual(len(control), len(selected))
            report = cp.read(run / 'selection.json')
            self.assertEqual(len(report['removed']), 5)
            self.assertEqual(report['training_sha256']['cds'], cp.file_hash(run / 'train_cds.json'))

    def test_training_has_no_parent_adapter_and_keeps_actual_epoch(self):
        import train_lora
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            cfg = self.config()
            cfg['pre_finetune_size'] = 0
            cp.write(run / 'train_cds.json', [{'item_id': str(i)} for i in range(32)])
            cp.write(run / 'selection.json', {'training_sha256': {
                'cds': cp.file_hash(run / 'train_cds.json')}})
            calls = []

            def fake_train():
                calls.append(list(sys.argv))
                checkpoint = run / 'checkpoints/cds'
                cp.write(checkpoint / 'adapter_config.json', {'r': 8})
                cp.write(checkpoint / 'experiment.json', {
                    'parent_adapter': None, 'training_metrics': {'epoch': 0.99}})
                (checkpoint / 'adapter_model.safetensors').write_bytes(b'offline stub')

            with patch.object(train_lora, 'main', side_effect=fake_train), \
                 patch.dict(raw.os.environ, {'WORLD_SIZE': '1'}):
                raw.train(cfg, run, 'cds')
            self.assertNotIn('--adapter', calls[0])
            self.assertEqual(cp.read(run / 'cds_training_completion.json')['actual_epoch'], 0.99)
            self.assertEqual(raw.adapter_path(cfg, run, 'cds'), str(run / 'checkpoints/cds'))
            (run / 'checkpoints/cds/adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                raw.adapter_path(cfg, run, 'cds')
            self.assertIsNone(raw.adapter_path(cfg, run, 'raw'))

    def test_infer_routes_raw_diagnosis_without_adapter(self):
        cfg, run = self.config(), Path('/tmp/fake-raw-run')
        with patch.object(raw, 'load', return_value=(cfg, run, {})), \
             patch.object(cp, 'infer') as infer, \
             patch.object(sys, 'argv', ['raw_cds.py', 'infer', '--split', 'diag']):
            raw.main()
            args = infer.call_args.args[2]
            self.assertEqual(args.tag, 'raw_diag')
            self.assertIsNone(args.adapter)
            self.assertIsNone(args.limit)


if __name__ == '__main__':
    unittest.main()
