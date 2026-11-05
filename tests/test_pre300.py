from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cds
from switch_to_pre300 import update_config, SETTINGS


class Pre300Tests(unittest.TestCase):
    def test_actual_split_is_disjoint_complete_and_repeatable(self):
        cfg = cds.read(Path(cds.__file__).with_name('configs') / 'cds_gsm8k.json')
        self.assertEqual(cfg['pre_finetune_size'], 300)
        self.assertEqual(cfg['development_size'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            cds.prepare(cfg, first)
            cds.prepare(cfg, second)
            manifest = cds.read(first / 'data/manifest.json')
            self.assertEqual(manifest['counts'], {'diag': 3500, 'test': 1319,
                'warmup': 300, 'dev': 0, 'reserve': 3673})
            ids = manifest['ids']
            remaining = cds.read(cds.resolve(cfg['data_root']) / 'D_remaining/data.jsonl')
            self.assertFalse(set(ids['warmup']) & set(ids['reserve']))
            self.assertEqual(set(ids['warmup']) | set(ids['reserve']), {r['item_id'] for r in remaining})
            self.assertFalse(set(ids['warmup']) & (set(ids['diag']) | set(ids['test'])))
            for name, original in [('diag', 'D_diag'), ('test', 'D_test')]:
                rows = cds.read(cds.resolve(cfg['data_root']) / original / 'data.jsonl')
                self.assertEqual(ids[name], [r['item_id'] for r in rows])
            for path in (first / 'data').iterdir():
                self.assertEqual(path.read_bytes(), (second / 'data' / path.name).read_bytes())

    def test_remote_config_migration_preserves_downloaded_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'src/configs/cds_gsm8k.json'
            original = {'run_dir': 'runs/old', 'pre_finetune_size': 292, 'development_size': 500,
                'student_model': '/custom/model_list/model',
                'student_revision_note': 'Qwen/Qwen2.5-3B-Instruct@actual-downloaded-commit',
                'teacher': {'model': 'custom-teacher'}, 'training': {'epochs': 1}}
            cds.write(config, original)
            update_config(config)
            result = cds.read(config)
            self.assertEqual(result, {**original, **SETTINGS})
            backups = list(config.parent.glob('*.before_pre300.*.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(cds.read(backups[0]), original)
            before = config.read_bytes()
            update_config(config)
            self.assertEqual(before, config.read_bytes())

    def test_migration_refuses_conflicting_frozen_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'src/configs/cds_gsm8k.json'
            original = {'pre_finetune_size': 292, 'development_size': 500, 'run_dir': 'runs/old'}
            cds.write(config, original)
            cds.write(root / SETTINGS['run_dir'] / 'config.frozen.json', {'different': True})
            with self.assertRaises(ValueError):
                update_config(config)
            self.assertEqual(cds.read(config), original)

    def test_empty_dev_rejected_before_importing_gpu_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            cds.write(run / 'data/dev.json', [])
            with self.assertRaisesRegex(ValueError, 'No items in dev'):
                cds.infer({}, run, SimpleNamespace(split='dev', limit=None))


if __name__ == '__main__':
    unittest.main()
