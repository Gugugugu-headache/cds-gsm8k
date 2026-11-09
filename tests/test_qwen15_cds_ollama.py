"""Offline local-teacher CDS routing; temporary data and mocked Ollama only."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cds_pipeline as cp
import ollama_teacher as ot
import qwen15_cds_ollama as local
import qwen15_raw_eval as evaluation
import raw_cds as raw
from test_raw_cds import fake_evaluation
from test_ollama_teacher import config as transport_config


class LocalCdsTests(unittest.TestCase):
    def fixture(self, root):
        old = root / 'qwen15_evaluation'
        cfg = cp.read(Path(local.__file__).with_name('configs') / 'cds_gsm8k.example.json')
        cfg.update(run_dir=str(old), student_model=str(root / 'models/Qwen1.5-7B-Chat'),
                   student_revision_note=evaluation.REPO_ID + '@' + evaluation.REVISION,
                   initialization_mode='raw', pre_finetune_size=0, development_size=500,
                   kc_scope='diagnosis_kcs', evaluation_only=True)
        cp.write(old / 'config.frozen.json', cfg)
        cp.write(old / 'evaluation_protocol.json', {'frozen_source': True})
        for name, size in evaluation.COUNTS.items():
            cp.write(old / 'data' / (name + '.json'), [
                {'item_id': f'{name}:{i}', 'instruction': cp.INSTRUCTION, 'input': f'{i} plus one?',
                 'output': f'#### {i + 1}', 'label': ['Addition'], 'label_status': 'confirmed'}
                for i in range(size)])
        cp.write(old / 'data/vocab.json', ['Addition'] + [f'KC{i}' for i in range(17)])
        cp.write(old / 'data/manifest.json', {'counts': evaluation.COUNTS})
        for split, correct in [('diag', 2556), ('test', 815)]:
            rows = cp.read(old / 'data' / (split + '.json'))
            result = fake_evaluation(rows, cfg, split)
            for row in result['records'][correct:]:
                row.update(response='#### -1', **cp.grade('#### -1', row['output']))
            result['meta'].update(correct=correct, accuracy=correct / len(rows))
            cp.write(old / 'eval' / ('raw_' + split + '.json'), result)
        return cfg, old, cp.read(old / 'eval/raw_diag.json')

    def context(self, root, source):
        stack = ExitStack()
        stack.enter_context(patch.object(local, 'source', return_value=source))
        stack.enter_context(patch.object(local, 'CONFIG_PATH', root / 'src/configs/local.json'))
        stack.enter_context(patch.object(local, 'RUN_DIR', str(root / 'new_local_cds')))
        stack.enter_context(patch.object(ot, 'inspect', return_value=transport_config()['teacher']['identity']))
        return stack

    def test_prepare_load_and_profile_reuse_all_answers_without_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            originals = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.context(root, source), patch.object(ot, 'rpc') as network, \
                 patch.object(cp, 'infer') as infer:
                cfg, run = local.prepare()
                self.assertEqual(cfg['quality_min_score'], 8)
                self.assertEqual(cfg['training'], source[0]['training'])
                self.assertEqual(cfg['teacher']['provider'], 'ollama_native')
                imported = cp.read(run / 'eval/raw_diag.json')
                self.assertEqual(imported['records'], source[2]['records'])
                self.assertEqual(imported['meta']['correct'], 2556)
                self.assertEqual(imported['meta']['generation_config_hash'], cp.digest(source[0]))
                self.assertEqual(imported['meta']['config_hash'], cp.digest(cfg))
                for name in [*evaluation.COUNTS, 'vocab', 'manifest']:
                    self.assertEqual(cp.read(run / 'data' / (name + '.json')),
                                     cp.read(source[1] / 'data' / (name + '.json')))
                local.load()
                report = local.profile()
                self.assertEqual(report['correct'], 2556)
                self.assertEqual(report['n'], 3500)
                self.assertEqual(local.prepare(), (cfg, run))
                network.assert_not_called()
                infer.assert_not_called()
                for path, data in originals.items():
                    self.assertEqual(path.read_bytes(), data)

    def test_rebinding_refuses_a_different_student_or_generation_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.fixture(Path(tmp))
            for change in [{'student_model': '/other/model'}, {'inference_max_tokens': 512},
                           {'student_revision_note': 'other revision'}]:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    local.rebound_diagnosis(source[2], source[0], {**source[0], **change},
                                             source[1] / 'eval/raw_diag.json')

    def test_imported_response_or_source_baseline_mutation_blocks_load(self):
        for mutation in ['import', 'baseline']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = self.fixture(root)
                with self.context(root, source):
                    _, run = local.prepare()
                    path = run / 'eval/raw_diag.json' if mutation == 'import' else source[1] / 'eval/raw_test.json'
                    data = cp.read(path)
                    data['records'][0]['response'] = 'changed'
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        local.load()

    def test_teacher_context_is_local_and_restored_and_metadata_drift_blocks_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            with self.context(root, source):
                local.prepare()
                original = cp.Teacher
                seen = []

                def synthesize(cfg, run, args):
                    seen.append(cp.Teacher)
                    self.assertIs(cp.Teacher, ot.OllamaTeacher)

                with patch.object(cp, 'synthesize', side_effect=synthesize):
                    local.teacher_stage('synthesize')
                self.assertEqual(seen, [ot.OllamaTeacher])
                self.assertIs(cp.Teacher, original)
                with patch.object(ot, 'inspect', return_value={'changed': True}), \
                     patch.object(cp, 'quality') as quality, self.assertRaises(ValueError):
                    local.teacher_stage('quality', 'seed')
                quality.assert_not_called()

    def test_route_preserves_other_experiments(self):
        previous = raw.CONFIG_PATH, raw.RUN_DIR
        with local.route():
            self.assertEqual((raw.CONFIG_PATH, raw.RUN_DIR), (local.CONFIG_PATH, local.RUN_DIR))
        self.assertEqual((raw.CONFIG_PATH, raw.RUN_DIR), previous)


if __name__ == '__main__':
    unittest.main()
