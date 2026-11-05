"""Regression checks for the HF Trainer / DS scheduler callback boundary.

These CPU tests do not establish GPU or actual DeepSpeed integration success.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_lora import make_cds_trainer, make_deepspeed_config


class ScheduleTests(unittest.TestCase):
    def test_callback_uses_real_optimizer_and_preserves_source_schedule(self):
        calls = []

        class Base:
            lr_scheduler = None
            optimizer = object()  # May be the pre-initialization placeholder.

        def scheduler(optimizer, **kwargs):
            value = {'optimizer': optimizer, **kwargs}
            calls.append(value)
            return value

        train = {'learning_rate': 3e-5, 'warmup_min_lr': 1e-4,
                 'warmup_max_lr': 3e-4, 'warmup_num_steps': 300}
        trainer = make_cds_trainer(Base, scheduler, train)()
        actual_optimizer = object()
        result = trainer.create_scheduler(10, optimizer=actual_optimizer)
        self.assertIs(result['optimizer'], actual_optimizer)
        self.assertEqual(result['warmup_max_lr'], 3e-4)
        self.assertEqual(result['warmup_num_steps'], 300)
        self.assertIs(trainer.create_scheduler(10, optimizer=actual_optimizer), result)
        self.assertEqual(len(calls), 1)
        self.assertTrue(trainer._created_lr_scheduler)

    def test_ds_config_does_not_repeat_conflicting_scheduler_values(self):
        train = {'micro_batch_size': 2, 'global_batch_size': 32, 'learning_rate': 3e-5}
        config = make_deepspeed_config(train, 1)
        self.assertIn('optimizer', config)
        self.assertNotIn('scheduler', config)

    def test_torch_adamw_avoids_fused_backend_and_keeps_global_batch(self):
        train = {'micro_batch_size': 2, 'global_batch_size': 32, 'learning_rate': 3e-5}
        for world, accumulation in [(1, 16), (2, 8)]:
            config = make_deepspeed_config(train, world)
            self.assertEqual(config['optimizer']['type'], 'AdamW')
            self.assertIs(config['optimizer']['params']['torch_adam'], True)
            self.assertEqual(config['optimizer']['params']['lr'], 3e-5)
            self.assertEqual(config['gradient_accumulation_steps'], accumulation)
            self.assertEqual(config['train_batch_size'], world * 2 * accumulation)
            self.assertEqual(config['zero_optimization']['stage'], 1)

    def test_invalid_batch_configuration_rejected(self):
        train = {'micro_batch_size': 2, 'global_batch_size': 31, 'learning_rate': 3e-5}
        with self.assertRaises(ValueError):
            make_deepspeed_config(train, 2)


if __name__ == '__main__':
    unittest.main()
