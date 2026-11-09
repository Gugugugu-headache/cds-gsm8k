"""Check allocation isolation, downloaded revision binding, and Raw evaluation routing."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import qwen15_raw_eval as q15


class Qwen15EvaluationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'existing_3b'
        cfg = cp.read(Path(q15.__file__).with_name('configs') / 'cds_gsm8k.example.json')
        cfg.update(run_dir=str(source), student_model=str(root / 'models/Qwen2.5-3B-Instruct'),
                   initialization_mode='raw', pre_finetune_size=0, development_size=500,
                   kc_scope='diagnosis_kcs', allocation_entrypoint='raw_cds18.py prepare')
        cp.write(source / 'config.frozen.json', cfg)
        for name, size in q15.COUNTS.items():
            cp.write(source / 'data' / (name + '.json'), [
                {'item_id': f'{name}:{i}', 'instruction': cp.INSTRUCTION, 'input': f'{i} plus one?',
                 'output': f'#### {i + 1}', 'label': ['Addition'], 'label_status': 'confirmed'}
                for i in range(size)])
        cp.write(source / 'data/vocab.json', ['Addition'] + [f'KC{i}' for i in range(17)])
        cp.write(source / 'data/manifest.json', {'counts': q15.COUNTS, 'frozen_ids': True})
        cp.write(source / 'eval/raw_test.json', {'existing': '3B baseline'})
        destination = root / 'models' / q15.MODEL_NAME
        cp.write(destination / 'download_provenance.json',
                 {'repo_id': q15.REPO_ID, 'revision': q15.REVISION})
        weight_map = {f'weight{i}': f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5)}
        cp.write(destination / 'model.safetensors.index.json', {'weight_map': weight_map})
        for name in ['config.json', 'generation_config.json', 'tokenizer.json',
                     'tokenizer_config.json', 'vocab.json']:
            cp.write(destination / name, {})
        (destination / 'merges.txt').write_text('offline tokenizer stub\n')
        for name in weight_map.values():
            (destination / name).write_bytes(b'offline weight stub, not a real model')
        return cfg, source, destination

    def context(self, root, cfg, source):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(q15.raw_cds18, 'load', return_value=(cfg, source, {})))
        stack.enter_context(patch.object(q15, 'CONFIG_PATH', root / 'src/configs/qwen15.json'))
        stack.enter_context(patch.object(q15, 'RUN_DIR', str(root / 'new_qwen15')))
        return stack

    def test_prepare_preserves_old_files_and_every_split_identity_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, source, destination = self.fixture(root)
            original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.context(root, cfg, source), patch.object(cp, 'infer') as infer, \
                 patch.object(cp, 'Teacher') as teacher:
                new_cfg, run = q15.prepare()
                self.assertEqual(new_cfg['student_model'], str(destination))
                changed = {'run_dir', 'student_model', 'student_revision_note',
                           'student_repo_id', 'evaluation_only', 'allocation_entrypoint'}
                self.assertEqual({k: v for k, v in new_cfg.items() if k not in changed},
                                 {k: v for k, v in cfg.items() if k not in changed})
                for name in [*q15.COUNTS, 'vocab', 'manifest']:
                    self.assertEqual(cp.read(run / 'data' / (name + '.json')),
                                     cp.read(source / 'data' / (name + '.json')))
                for path, data in original.items():
                    self.assertEqual(path.read_bytes(), data)
                before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
                q15.prepare()
                self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
                self.assertFalse((run / 'eval/raw_test.json').exists())
                infer.assert_not_called()
                teacher.assert_not_called()

    def test_full_test_and_smoke_route_to_existing_scorer_with_no_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, source, _ = self.fixture(root)
            with self.context(root, cfg, source):
                new_cfg, run = q15.prepare()
                with patch.object(cp, 'infer') as infer:
                    q15.evaluate('test')
                    args = infer.call_args.args
                    self.assertEqual(args[:2], (new_cfg, run))
                    self.assertEqual(vars(args[2]),
                                     {'split': 'test', 'tag': 'raw_test', 'adapter': None,
                                      'tp': 1, 'limit': None})
                    self.assertEqual(new_cfg['inference_max_tokens'], 1024)
                    self.assertEqual(new_cfg['inference_prompt'], 'chat')
                    q15.evaluate('diag', 20)
                    self.assertEqual(infer.call_args.args[2].tag, 'raw_smoke_diag_20')

    def test_wrong_model_revision_and_missing_shard_fail_before_new_artifacts(self):
        for mutation in ['revision', 'missing_shard']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, source, destination = self.fixture(root)
                if mutation == 'revision':
                    path = destination / 'download_provenance.json'
                    path.write_text('{"repo_id": "different", "revision": "unknown"}')
                else:
                    next(destination.glob('*.safetensors')).unlink()
                with self.context(root, cfg, source), self.assertRaises(ValueError):
                    q15.prepare()
                self.assertFalse((root / 'src/configs/qwen15.json').exists())

    def test_conflicting_destination_and_source_protocol_are_rejected(self):
        for mutation in ['conflicting_config', 'same_model', 'source_prompt']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, source, _ = self.fixture(root)
                if mutation == 'source_prompt':
                    cfg = {**cfg, 'inference_prompt': 'source_plain'}
                with self.context(root, cfg, source):
                    if mutation == 'conflicting_config':
                        cp.write(q15.CONFIG_PATH, {'different': True})
                    with self.assertRaises(ValueError):
                        q15.prepare(cfg['student_model'] if mutation == 'same_model' else None)
                    if mutation == 'conflicting_config':
                        self.assertEqual(cp.read(q15.CONFIG_PATH), {'different': True})

    def test_changed_data_or_weight_metadata_prevents_inference(self):
        for mutation in ['target_test', 'source_test', 'model_metadata', 'shard_size']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, source, destination = self.fixture(root)
                with self.context(root, cfg, source):
                    _, run = q15.prepare()
                    if mutation in ['target_test', 'source_test']:
                        path = (run if mutation == 'target_test' else source) / 'data/test.json'
                        rows = copy.deepcopy(cp.read(path))
                        rows[0]['input'] = 'changed question'
                        path.write_text(json.dumps(rows))
                    elif mutation == 'model_metadata':
                        (destination / 'tokenizer_config.json').write_text('{"changed": true}')
                    else:
                        next(destination.glob('*.safetensors')).write_bytes(b'changed size')
                    with patch.object(cp, 'infer') as infer, self.assertRaises(ValueError):
                        q15.evaluate('test')
                    infer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
