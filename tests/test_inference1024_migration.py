from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cds_pipeline import read, write
from configure_inference1024 import PREVIOUS_RUN_DIR, RUN_DIR, SETTINGS, update_config


class Inference1024MigrationTests(unittest.TestCase):
    def original(self):
        return {'run_dir': PREVIOUS_RUN_DIR, 'inference_max_tokens': 512,
                'pre_finetune_size': 300, 'development_size': 0, 'seed': 42,
                'student_model': '/server/downloaded/model', 'student_revision_note': 'actual HF commit',
                'teacher': {'model': 'server-teacher', 'timeout_seconds': 180},
                'inference_prompt': 'chat', 'training': {'max_length': 2048, 'learning_rate': 3e-5}}

    def test_preserves_server_settings_and_old_artifacts_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'src/configs/cds_gsm8k.json'
            original = self.original()
            write(config, original)
            frozen = root / PREVIOUS_RUN_DIR / 'config.frozen.json'
            evaluation = root / PREVIOUS_RUN_DIR / 'eval/raw_smoke_diag.json'
            write(frozen, original)
            write(evaluation, {'meta': {'n': 20, 'truncated': 1}})
            before = {path: path.read_bytes() for path in [frozen, evaluation]}
            updated = update_config(config)
            self.assertEqual(updated, {**original, **SETTINGS})
            self.assertEqual(read(config), updated)
            backups = list(config.parent.glob('*.before_tok1024.*.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(read(backups[0]), original)
            write(root / RUN_DIR / 'config.frozen.json', updated)
            unchanged = config.read_bytes()
            update_config(config)
            self.assertEqual(config.read_bytes(), unchanged)
            self.assertEqual(len(list(config.parent.glob('*.before_tok1024.*.json'))), 1)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_rejects_conflicting_target_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'src/configs/cds_gsm8k.json'
            original = self.original()
            write(config, original)
            write(root / RUN_DIR / 'config.frozen.json', {**original, 'student_revision_note': 'different'})
            before = config.read_bytes()
            with self.assertRaises(ValueError):
                update_config(config)
            self.assertEqual(config.read_bytes(), before)
            self.assertFalse(list(config.parent.glob('*.before_tok1024.*.json')))

    def test_rejects_modified_source_frozen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'src/configs/cds_gsm8k.json'
            original = self.original()
            write(config, original)
            write(root / PREVIOUS_RUN_DIR / 'config.frozen.json', {**original, 'seed': 7})
            with self.assertRaises(ValueError):
                update_config(config)
            self.assertEqual(read(config), original)

    def test_rejects_unexpected_protocol_without_mutation(self):
        changes = [{'pre_finetune_size': 292}, {'development_size': 500},
                   {'run_dir': 'runs/another_model'}, {'inference_max_tokens': 4096}]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / 'src/configs/cds_gsm8k.json'
                original = {**self.original(), **change}
                write(config, original)
                with self.assertRaises(ValueError):
                    update_config(config)
                self.assertEqual(read(config), original)


if __name__ == '__main__':
    unittest.main()
