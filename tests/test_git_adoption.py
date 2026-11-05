import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adopt_git_checkout import adopt


class GitAdoptionTests(unittest.TestCase):
    def prepare(self, root):
        live, checkout = root / 'src', root / 'src_git_checkout'
        for path in [live / 'configs', checkout / 'configs', checkout / '.git']:
            path.mkdir(parents=True)
        cfg = {'run_dir': 'runs/active', 'student_revision_note': 'actual HF commit',
               'student_model': '/server/model', 'inference_max_tokens': 1024}
        (live / 'configs/cds_gsm8k.json').write_text(json.dumps(cfg))
        (live / 'configs/cds_gsm8k.before_tok1024.abc.json').write_text('{"old": true}')
        hashes = {}
        for name in ['cds_pipeline.py', 'train_lora.py']:
            payload = ('# ' + name + '\n').encode()
            (live / name).write_bytes(payload)
            (checkout / name).write_bytes(payload)
            hashes['src/' + name] = hashlib.sha256(payload).hexdigest()
        run = root / cfg['run_dir']
        run.mkdir(parents=True)
        (run / 'config.frozen.json').write_text(json.dumps(cfg))
        (run / 'implementation.frozen.json').write_text(json.dumps(hashes))
        (run / 'raw_test.json').write_text('{"correct":1127,"n":1319}')
        return live, checkout, run

    def test_preserves_config_backups_and_frozen_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live, checkout, run = self.prepare(root)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            backup = adopt(root, checkout)
            self.assertTrue((live / '.git').is_dir())
            self.assertFalse(checkout.exists())
            for relative, payload in before.items():
                if relative.parts[0] == 'src':
                    self.assertEqual((backup / Path(*relative.parts[1:])).read_bytes(), payload)
                    self.assertEqual((live / Path(*relative.parts[1:])).read_bytes(), payload)
                elif relative.parts[0] == 'runs':
                    self.assertEqual((root / relative).read_bytes(), payload)
            with self.assertRaises(ValueError):
                adopt(root, backup)

    def test_refuses_changed_frozen_code_without_moving_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live, checkout, run = self.prepare(root)
            (checkout / 'cds_pipeline.py').write_text('# changed implementation\n')
            with self.assertRaises(ValueError):
                adopt(root, checkout)
            self.assertTrue(live.is_dir())
            self.assertTrue(checkout.is_dir())
            self.assertFalse(list(root.glob('src.before_git_*')))
            self.assertFalse((checkout / 'configs/cds_gsm8k.json').exists())

    def test_refuses_config_mismatch_without_moving_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live, checkout, run = self.prepare(root)
            (run / 'config.frozen.json').write_text('{"different": true}')
            with self.assertRaises(ValueError):
                adopt(root, checkout)
            self.assertTrue(live.is_dir())
            self.assertTrue(checkout.is_dir())
            self.assertFalse(list(root.glob('src.before_git_*')))


if __name__ == '__main__':
    unittest.main()
