import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cds_pipeline import read, write
from configure_qwen2_7b import create_config, REPO_ID, CONFIG_NAME, RUN_DIR
import download_student


class Qwen2CandidateTests(unittest.TestCase):
    def source(self, root):
        path = root / 'src/configs/cds_gsm8k.json'
        cfg = {'run_dir': 'runs/previous_3b', 'student_model': '/server/model_list/Qwen2.5-3B-Instruct',
               'student_revision_note': 'previous actual HF commit', 'seed': 42,
               'pre_finetune_size': 300, 'development_size': 0, 'inference_prompt': 'chat',
               'inference_max_tokens': 1024, 'teacher': {'model': 'server-teacher'},
               'training': {'max_length': 2048, 'learning_rate': 3e-5}}
        write(path, cfg)
        write(root / cfg['run_dir'] / 'config.frozen.json', cfg)
        write(root / cfg['run_dir'] / 'eval/raw_test.json', {'n': 1319, 'correct': 1127})
        return path, cfg

    def test_candidate_preserves_previous_config_and_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, original = self.source(root)
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            candidate = create_config(source)
            changed = {'run_dir', 'student_model', 'student_revision_note', 'student_repo_id'}
            self.assertEqual({k: v for k, v in candidate.items() if k not in changed},
                             {k: v for k, v in original.items() if k not in changed})
            self.assertEqual(candidate['student_repo_id'], REPO_ID)
            self.assertEqual(candidate['student_model'], '/server/model_list/Qwen2-7B-Instruct')
            self.assertEqual(candidate['run_dir'], RUN_DIR)
            for path, payload in before.items():
                self.assertEqual(path.read_bytes(), payload)
            output = source.with_name(CONFIG_NAME)
            candidate['student_revision_note'] = REPO_ID + '@recorded-actual-commit'
            write(root / RUN_DIR / 'config.frozen.json', candidate)
            output.write_text(json.dumps(candidate))
            unchanged = output.read_bytes()
            self.assertEqual(create_config(source), candidate)
            self.assertEqual(output.read_bytes(), unchanged)

    def test_refuses_model_directory_alias_or_original_config_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cfg = self.source(Path(tmp))
            with self.assertRaises(ValueError):
                create_config(source, model_dir=cfg['student_model'])
            with self.assertRaises(ValueError):
                create_config(source, output_path=source)
            self.assertEqual(read(source), cfg)

    def test_refuses_existing_conflicting_config_or_frozen_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, cfg = self.source(root)
            output = source.with_name(CONFIG_NAME)
            write(output, {'different': True})
            with self.assertRaises(ValueError):
                create_config(source)
            self.assertEqual(read(output), {'different': True})
            output.unlink()
            write(root / RUN_DIR / 'config.frozen.json', {'different': True})
            with self.assertRaises(ValueError):
                create_config(source)
            self.assertFalse(output.exists())

    def test_rejects_source_protocol_drift(self):
        for change in [{'inference_max_tokens': 512}, {'pre_finetune_size': 292},
                       {'development_size': 500}, {'inference_prompt': 'source_plain'}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                source, cfg = self.source(Path(tmp))
                source.write_text(json.dumps({**cfg, **change}))
                with self.assertRaises(ValueError):
                    create_config(source)
                self.assertFalse(source.with_name(CONFIG_NAME).exists())

    def test_download_repo_resolution_keeps_legacy_and_rejects_conflicts(self):
        self.assertEqual(download_student.resolve_repo_id({}), 'Qwen/Qwen2.5-3B-Instruct')
        self.assertEqual(download_student.resolve_repo_id({}, REPO_ID), REPO_ID)
        self.assertEqual(download_student.resolve_repo_id({'student_repo_id': REPO_ID}), REPO_ID)
        with self.assertRaises(ValueError):
            download_student.resolve_repo_id({'student_repo_id': REPO_ID}, 'Qwen/Qwen2-7B')

    def test_download_records_configured_repo_and_pins_resolved_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, cfg = self.source(root)
            config = source.with_name(CONFIG_NAME)
            candidate = create_config(source, model_dir=root / 'model_list/Qwen2-7B-Instruct')
            candidate['run_dir'] = str(root / 'unfrozen-run')
            config.write_text(json.dumps(candidate))
            calls = []

            def model_info(repo, revision):
                calls.append(('info', repo, revision))
                return SimpleNamespace(sha='resolved-commit')

            def snapshot_download(**kwargs):
                calls.append(('download', kwargs['repo_id'], kwargs['revision']))
                destination = Path(kwargs['local_dir'])
                destination.mkdir(parents=True)
                (destination / 'config.json').write_text('{}')
                (destination / 'model.safetensors').write_bytes(b'offline stub')

            fake = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=model_info),
                                   snapshot_download=snapshot_download)
            with patch.dict(sys.modules, {'huggingface_hub': fake}), patch.object(
                    sys, 'argv', ['download_student.py', '--config', str(config)]):
                download_student.main()
            self.assertEqual(calls, [('info', REPO_ID, 'main'), ('download', REPO_ID, 'resolved-commit')])
            self.assertEqual(read(config)['student_revision_note'], REPO_ID + '@resolved-commit')
            self.assertEqual(read(Path(candidate['student_model']) / 'download_provenance.json'),
                             {'repo_id': REPO_ID, 'revision': 'resolved-commit'})
            self.assertEqual(read(source), cfg)


if __name__ == '__main__':
    unittest.main()
