from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cds_pipeline import read, write
from configure_torch_adam import update_config, RUN_DIR


class TorchAdamMigrationTests(unittest.TestCase):
    def test_preserves_model_revision_sampling_and_previous_frozen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'src/configs/cds_gsm8k.json'
            original = {'run_dir': 'runs/cds_qwen25_3b_pre300', 'pre_finetune_size': 300,
                'development_size': 0, 'seed': 42, 'training': {'learning_rate': 3e-5},
                'student_model': '/downloaded/model', 'student_revision_note': 'actual HF commit'}
            write(config, original)
            old_frozen = root / original['run_dir'] / 'config.frozen.json'
            write(old_frozen, original)
            before = old_frozen.read_bytes()
            updated = update_config(config)
            self.assertEqual(updated, {**original, 'run_dir': RUN_DIR})
            self.assertEqual(old_frozen.read_bytes(), before)
            self.assertEqual(len(list(config.parent.glob('*.before_torchadam.*.json'))), 1)
            unchanged = config.read_bytes()
            update_config(config)
            self.assertEqual(config.read_bytes(), unchanged)

    def test_rejects_incomplete_pre300_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'src/configs/cds_gsm8k.json'
            original = {'pre_finetune_size': 292, 'development_size': 500}
            write(config, original)
            with self.assertRaises(ValueError):
                update_config(config)
            self.assertEqual(read(config), original)

    def test_rejects_conflicting_target_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'src/configs/cds_gsm8k.json'
            original = {'pre_finetune_size': 300, 'development_size': 0, 'run_dir': 'runs/old'}
            write(config, original)
            write(root / RUN_DIR / 'config.frozen.json', {'different': True})
            with self.assertRaises(ValueError):
                update_config(config)
            self.assertEqual(read(config), original)


if __name__ == '__main__':
    unittest.main()
