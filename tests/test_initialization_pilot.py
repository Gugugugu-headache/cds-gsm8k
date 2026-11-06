"""CPU protocol tests; no model loading, API calls, or GPU training."""
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
from train_lora import make_cds_trainer, make_deepspeed_config


class PilotTests(unittest.TestCase):
    def config(self):
        cfg = cp.read(Path(pilot.__file__).with_name('configs') / 'cds_gsm8k.example.json')
        cfg['student_model'] = '/models/Qwen2.5-3B-Instruct'
        cfg['student_revision_note'] = 'Qwen/Qwen2.5-3B-Instruct@' + 'a' * 40
        return cfg

    def test_real_exports_preserve_previous_data_and_make_dev_from_reserve(self):
        # Uses the actual local source GSM8K manifests; all outputs go to /tmp.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'configs/cds_gsm8k.json'
            cfg = self.config()
            cfg['run_dir'] = str(root / 'previous')
            cp.write(source, cfg)
            _, old = cp.init_run(source)
            cp.prepare(cfg, old)
            cp.write(old / 'eval/raw_test.json', {'historical': 'do not touch'})
            originals = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(pilot, 'PILOT_RUN', str(root / 'pilot')):
                plan = pilot.prepare(source)
                for arm in pilot.ARMS:
                    _, candidate, run = pilot.load_arm(arm)
                    self.assertEqual(candidate['student_revision_note'], cfg['student_revision_note'])
                    for name in ['warmup', 'diag', 'test']:
                        self.assertEqual((run / 'data' / (name + '.json')).read_bytes(),
                                         (old / 'data' / (name + '.json')).read_bytes())
                    reserve = cp.read(old / 'data/reserve.json')
                    self.assertEqual(cp.read(run / 'data/dev.json'), reserve[:500])
                    self.assertEqual(cp.read(run / 'data/reserve.json'), reserve[500:])
                    self.assertEqual(len(cp.read(run / 'data/reserve.json')), 3173)
                    if arm != 'raw':
                        rate = pilot.ARMS[arm]
                        self.assertEqual(candidate['training']['learning_rate'], rate)
                        self.assertEqual(candidate['training']['warmup_min_lr'], rate)
                        self.assertEqual(candidate['training']['warmup_max_lr'], rate)
                self.assertEqual(pilot.prepare(source), plan)  # Safe idempotence.
                for path, content in originals.items():
                    self.assertEqual(path.read_bytes(), content)
                # Config changes cannot be silently adopted by prepare/load.
                arm_path = Path(plan['config_paths']['lr1e5'])
                changed = cp.read(arm_path)
                changed['training']['learning_rate'] = 9e-4
                arm_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    pilot.load_arm('lr1e5')
                with self.assertRaises(ValueError):
                    pilot.prepare(source)
                arm_path.write_text(json.dumps(plan['configs']['lr1e5']))
                data = Path(plan['configs']['raw']['run_dir']) / 'data/dev.json'
                rows = cp.read(data)
                rows[0]['input'] += 'tampered'
                data.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    pilot.load_arm('raw')

    def test_refuses_wrong_model_unrecorded_revision_or_changed_split(self):
        for delta in [{'student_model': '/models/Qwen2-7B-Instruct'},
                      {'student_revision_note': 'Pending download'}, {'development_size': 500},
                      {'seed': 43}, {'inference_max_tokens': 512}]:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                pilot.verify_source({**self.config(), **delta})

    def test_layout_has_fifteen_full_updates_and_rejects_old_tail(self):
        cfg = self.config()
        cfg['training'].update(global_batch_size=20, micro_batch_size=2, epochs=1)
        self.assertEqual(pilot.validate_training_layout(cfg, [None] * 300, 1), 15)
        ds = make_deepspeed_config(cfg['training'], 1)
        self.assertEqual(ds['gradient_accumulation_steps'], 10)
        for samples, world in [(288, 1), (301, 1), (300, 2)]:
            with self.assertRaises(ValueError):
                pilot.validate_training_layout(cfg, [None] * samples, world)
        cfg['training']['global_batch_size'] = 32
        with self.assertRaises(ValueError):
            pilot.validate_training_layout(cfg, [None] * 300, 1)

    def test_equal_schedule_endpoints_keep_constant_lr(self):
        # Exercise the unchanged production scheduler callback; stub DS interpolation.
        class Base:
            lr_scheduler = None
            optimizer = object()
        for rate in [1e-5, 3e-5]:
            t = {'warmup_min_lr': rate, 'warmup_max_lr': rate, 'warmup_num_steps': 2}
            scheduler = make_cds_trainer(Base, lambda optimizer, **kw: kw, t)().create_scheduler(15)
            for gamma in [0, 0.5, 1]:
                value = scheduler['warmup_min_lr'] + gamma * (
                    scheduler['warmup_max_lr'] - scheduler['warmup_min_lr'])
                self.assertEqual(value, rate)

    def test_selection_uses_accuracy_then_format_then_predefined_tie_order(self):
        m = {arm: {'correct': 400, 'hash_count': 450} for arm in pilot.ARMS}
        self.assertEqual(pilot.choose(m), 'raw')
        m['lr1e5']['hash_count'] = 460
        self.assertEqual(pilot.choose(m), 'lr1e5')
        m['lr3e5']['correct'] = 401
        self.assertEqual(pilot.choose(m), 'lr3e5')
        m['raw']['correct'] = 402
        self.assertEqual(pilot.choose(m), 'raw')
        with self.assertRaises(ValueError):
            pilot.choose({'raw': m['raw']})

    def evaluation(self):
        cfg = self.config()
        cfg['development_size'] = 500
        rows = [{'item_id': f'dev-{i}', 'instruction': cp.INSTRUCTION,
                 'input': f'question {i}', 'output': '#### 2'} for i in range(500)]
        records = [{**row, 'response': '#### 2', **cp.grade('#### 2', row['output']),
                    'finish_reason': 'stop', 'output_tokens': 4} for row in rows]
        result = {'meta': {'split': 'dev', 'n': 500, 'config_hash': cp.digest(cfg),
                          'adapter': None, 'model': cfg['student_model'], 'prompt': 'chat',
                          'correct': 500, 'accuracy': 1.0, 'no_number': 0, 'truncated': 0},
                  'records': records}
        return cfg, rows, result

    def test_summary_rejects_test_partial_wrong_model_or_adapter(self):
        cfg, rows, result = self.evaluation()
        self.assertEqual(pilot.summarize(result, rows, cfg, None)['hash_count'], 500)
        for delta in [{'split': 'test'}, {'n': 20}, {'model': '/wrong/model'},
                      {'adapter': '/wrong/adapter'}, {'config_hash': 'wrong'}, {'correct': 499}]:
            with self.subTest(delta=delta):
                wrong = copy.deepcopy(result)
                wrong['meta'].update(delta)
                with self.assertRaises(ValueError):
                    pilot.summarize(wrong, rows, cfg, None)

    def test_summary_rejects_changed_answers_order_or_scores(self):
        cfg, rows, result = self.evaluation()
        for field, value in [('output', '#### 3'), ('correct', False), ('extraction', 'last_number')]:
            wrong = copy.deepcopy(result)
            wrong['records'][0][field] = value
            with self.assertRaises(ValueError):
                pilot.summarize(wrong, rows, cfg, None)
        result['records'].reverse()
        with self.assertRaises(ValueError):
            pilot.summarize(result, rows, cfg, None)

    def test_checkpoint_changed_after_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            checkpoint = run / 'checkpoints/warmup'
            checkpoint.mkdir(parents=True)
            cp.write(checkpoint / 'adapter_config.json', {'r': 8})
            cp.write(checkpoint / 'experiment.json', {'training_metrics': {'epoch': 1.0}})
            weights = checkpoint / 'adapter_model.safetensors'
            weights.write_bytes(b'offline test stub')
            cfg = self.config()
            cp.write(run / 'training_completion.json', {
                'config_hash': cp.digest(cfg), 'actual_epoch': 1.0,
                'checkpoint_sha256': pilot.checkpoint_files(checkpoint)})
            self.assertEqual(pilot.verify_checkpoint('lr1e5', cfg, run), str(checkpoint))
            weights.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                pilot.verify_checkpoint('lr1e5', cfg, run)
            self.assertIsNone(pilot.verify_checkpoint('raw', cfg, run))

    def test_raw_training_is_rejected_before_loading_gpu_dependencies(self):
        with self.assertRaises(ValueError):
            pilot.train('raw')

    def test_training_starts_from_raw_and_checks_completed_epoch(self):
        import train_lora
        for epoch in [0.96, 1.0]:
            with self.subTest(epoch=epoch), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                cfg = self.config()
                cfg['training'].update(global_batch_size=20, micro_batch_size=2)
                cp.write(run / 'data/warmup.json', [None] * 300)
                plan = {'config_paths': {'lr1e5': str(run / 'config.json')}}
                calls = []

                def fake_train():
                    calls.append(list(sys.argv))
                    checkpoint = run / 'checkpoints/warmup'
                    cp.write(checkpoint / 'experiment.json', {'training_metrics': {'epoch': epoch}})
                    cp.write(checkpoint / 'adapter_config.json', {'r': 8})
                    (checkpoint / 'adapter_model.safetensors').write_bytes(b'fake adapter')

                versions = {'transformers': '4.51.3', 'accelerate': '1.6.0', 'deepspeed': '0.16.7'}
                with patch.object(pilot, 'load_arm', return_value=(plan, cfg, run)), \
                     patch.object(pilot.importlib.metadata, 'version', side_effect=versions.__getitem__), \
                     patch.object(train_lora, 'main', side_effect=fake_train), \
                     patch.dict(pilot.os.environ, {'WORLD_SIZE': '1'}):
                    if epoch == 1.0:
                        pilot.train('lr1e5')
                        self.assertEqual(cp.read(run / 'training_completion.json')['actual_epoch'], 1.0)
                    else:
                        with self.assertRaises(ValueError):
                            pilot.train('lr1e5')
                        self.assertFalse((run / 'training_completion.json').exists())
                self.assertNotIn('--adapter', calls[0])
                self.assertIn(str(run / 'data/warmup.json'), calls[0])

    def test_eval_forces_full_dev_without_training_or_test_calls(self):
        cfg = self.config()
        run = Path('/tmp/fake-pilot-run')
        with patch.object(pilot, 'load_arm', return_value=({}, cfg, run)), \
             patch.object(pilot, 'verify_checkpoint', return_value=None), \
             patch.object(cp, 'infer') as infer:
            pilot.evaluate('raw')
            args = infer.call_args.args[2]
            self.assertEqual(args.split, 'dev')
            self.assertIsNone(args.limit)
            self.assertIsNone(args.adapter)
            self.assertEqual(args.tp, 1)


if __name__ == '__main__':
    unittest.main()
