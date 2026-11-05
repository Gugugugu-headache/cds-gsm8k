#!/usr/bin/env python3
"""Single/two-GPU LoRA using upstream CDS tokenization and DeepSpeed schedule."""
import argparse
import json
import os
from pathlib import Path
import sys

from cds_pipeline import UPSTREAM, read, resolve, file_hash


def make_deepspeed_config(train, world):
    """Use native Torch AdamW so ZeRO-1 does not JIT-build FusedAdam."""
    denominator = world * train['micro_batch_size']
    if denominator <= 0 or train['global_batch_size'] % denominator:
        raise ValueError('Global batch size must be divisible by world size * micro batch size.')
    return {
        'bf16': {'enabled': True},
        'optimizer': {'type': 'AdamW', 'params': {'lr': train['learning_rate'],
                      'betas': [0.98, 0.999], 'eps': 1e-9, 'weight_decay': 0.0,
                      'torch_adam': True}},
        'zero_optimization': {'stage': 1},
        'train_micro_batch_size_per_gpu': train['micro_batch_size'],
        'gradient_accumulation_steps': train['global_batch_size'] // denominator,
        'train_batch_size': train['global_batch_size'],
        'gradient_clipping': 0.0
    }


def make_cds_trainer(base_trainer, warmup_scheduler, train):
    """Keep the source WarmupLR without declaring conflicting HF/DS scheduler keys.

    Transformers supports a DS optimizer with a Trainer-created scheduler. The
    callback receives the real optimizer from Accelerate after DS initialization.
    The source optimizer LR and scheduler maximum differ intentionally.
    """
    class CDSTrainer(base_trainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler = warmup_scheduler(
                    optimizer if optimizer is not None else self.optimizer,
                    warmup_min_lr=train['warmup_min_lr'],
                    warmup_max_lr=train['warmup_max_lr'],
                    warmup_num_steps=train['warmup_num_steps'])
                self._created_lr_scheduler = True
            return self.lr_scheduler

    return CDSTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('configs') / 'cds_gsm8k.json'))
    parser.add_argument('--data', required=True, help='JSON relative to repository root, or absolute path')
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapter', help='Continue warmup LoRA; all experimental arms must share it')
    parser.add_argument('--dataset-type', choices=['QwenDataset', 'Llama3Dataset'], default='QwenDataset')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    args = parser.parse_args()
    # Run prepare before torchrun; avoid concurrent writes to frozen manifests.
    cfg = read(args.config)
    run = resolve(cfg['run_dir'])
    if not (run / 'data/manifest.json').exists():
        raise ValueError('Run cds_pipeline.py prepare first.')
    if read(run / 'config.frozen.json') != cfg:
        raise ValueError('Configuration differs from frozen run.')
    hashes = read(run / 'implementation.frozen.json')
    for name, expected in hashes.items():
        path = resolve(name)
        if file_hash(path) != expected:
            raise ValueError(f'Frozen implementation changed: {name}')
    output = resolve(args.output)
    if output.exists():
        raise ValueError('Output directory already exists; choose a new checkpoint directory.')
    data_path = resolve(args.data)
    # Limit training inputs to the explicitly prepared training artifacts.
    allowed = [run / 'data/warmup.json', run / 'train_cds.json', run / 'train_random.json']
    if data_path not in allowed:
        raise ValueError('Use warmup.json, train_cds.json, or train_random.json from this run.')
    rows = read(data_path)
    if not rows:
        raise ValueError('Empty training data')
    if data_path.name != 'warmup.json' and cfg['pre_finetune_size'] and not args.adapter:
        raise ValueError('This protocol requires the common warmup adapter for downstream arms.')
    forbidden = {row['item_id'] for name in ['diag', 'test', 'dev']
                 for row in read(run / 'data' / (name + '.json'))}
    if any(row['item_id'] in forbidden for row in rows):
        raise ValueError('Held-out or diagnostic items in training input')
    import torch
    from deepspeed.runtime.lr_schedules import WarmupLR
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    sys.path.insert(0, str(UPSTREAM / 'finetune/code'))
    from dataset import QwenDataset, Llama3Dataset
    train = cfg['training']
    world = int(os.environ.get('WORLD_SIZE', '1'))
    ds_config = make_deepspeed_config(train, world)
    accumulation = ds_config['gradient_accumulation_steps']
    model_path = Path(cfg['student_model']).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError('Download the model locally before training.')
    set_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_class = {'QwenDataset': QwenDataset, 'Llama3Dataset': Llama3Dataset}[args.dataset_type]
    dataset = dataset_class(tokenizer, train['max_length'], raw_datas_sft=rows)
    if any(not torch.any(labels != -100) for labels in dataset.target_ids):
        raise ValueError('A training item has no supervised tokens')
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                        torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(Path(args.adapter).resolve()), is_trainable=True)
        adapter_cfg = model.peft_config['default']
        if (adapter_cfg.r != train['lora_rank'] or adapter_cfg.lora_alpha != train['lora_alpha']
                or set(adapter_cfg.target_modules) != set(train['target_modules'])):
            raise ValueError('Warmup adapter configuration differs from experiment')
    else:
        model = get_peft_model(model, LoraConfig(r=train['lora_rank'], lora_alpha=train['lora_alpha'],
            target_modules=train['target_modules'], lora_dropout=train['lora_dropout'],
            bias='none', task_type='CAUSAL_LM'))
    training_args = TrainingArguments(output_dir=str(output), num_train_epochs=train['epochs'],
        per_device_train_batch_size=train['micro_batch_size'], gradient_accumulation_steps=accumulation,
        learning_rate=train['learning_rate'], adam_beta1=0.98, adam_beta2=0.999, adam_epsilon=1e-9,
        weight_decay=0.0, max_grad_norm=0.0, bf16=True, deepspeed=ds_config,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='no', logging_steps=1, report_to='none', seed=cfg['seed'],
        data_seed=cfg['seed'], remove_unused_columns=False, ddp_find_unused_parameters=False)

    def collate(batch):
        return {key: torch.stack([row[index] for row in batch])
                for index, key in enumerate(['input_ids', 'labels', 'attention_mask'])}

    trainer_class = make_cds_trainer(Trainer, WarmupLR, train)
    trainer = trainer_class(model=model, args=training_args, train_dataset=dataset, data_collator=collate)
    result = trainer.train()
    trainer.save_model(str(output))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output))
        metadata = {'data_path': str(data_path), 'data_sha256': file_hash(data_path),
                    'model': str(model_path), 'parent_adapter': args.adapter,
                    'global_batch': train['global_batch_size'], 'world_size': world,
                    'gradient_accumulation_steps': accumulation, 'seed': cfg['seed'],
                    'supervised_tokens': int((dataset.target_ids != -100).sum()),
                    'training_metrics': result.metrics, 'deepspeed': ds_config,
                    'optimizer_backend': 'torch.optim.AdamW (DeepSpeed torch_adam=true)',
                    'scheduler': {'type': 'WarmupLR', 'params': {
                        'warmup_min_lr': train['warmup_min_lr'],
                        'warmup_max_lr': train['warmup_max_lr'],
                        'warmup_num_steps': train['warmup_num_steps']}},
                    'dataset_type': args.dataset_type}
        (output / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
