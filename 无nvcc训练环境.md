# 当前服务器：无需FusedAdam编译的训练后端

**后续预算更新：当前学生解题上限1024，新run为 `runs/cds_qwen25_3b_pre300_torchadam_tok1024`，先按 [1024token评测协议.md](1024token评测协议.md) 更新。** 这里旧的torchadam迁移命令保留为历史步骤，已完成者无需重跑。后续训练命令的数据和checkpoint路径使用新run。

用户已完成本文件中的依赖检查、21项测试与300/0/3673的prepare，尚未开始训练。当前先按 [原始模型基线评测.md](原始模型基线评测.md) 采集Raw结果，再回到本文件的训练阶段。该步骤使用独立cdi-infer环境，不改当前已冻结训练实现。

用户已提供的实际环境：Python3.11.18、PyTorch2.6.0+cu124、torch.version.cuda=12.4、CUDA可用、两张RTX A6000、驱动580.142。NumPy尚未安装，`nvcc`不在PATH中。两张卡在提供的15:00状态中利用率均为100%，有正在运行的计算任务。

CUDA运行库已可用，缺少nvcc不等于不能运行GPU PyTorch。原训练入口让DeepSpeed默认选择FusedAdam，可能在运行时JIT编译扩展，需要编译工具。当前适配显式设置 `optimizer.params.torch_adam=true`，由DeepSpeed使用原生 `torch.optim.AdamW`，避开FusedAdam的编译路径。继续使用ZeRO-1、bf16、LoRA、原学习率调度与有效batch32。[DeepSpeed0.16.7优化器选择源码](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.7/deepspeed/runtime/engine.py)

这是优化器执行后端适配，不能宣称与原FusedAdam逐位数值一致。后续CDS、随机对照和主方法均使用相同后端，记录软件版本及实际训练设置。其他DeepSpeed可选功能仍可能涉及编译；本方案只明确避免当前FusedAdam路径。真实GPU训练尚待用户服务器验证。

## 1. 同步补丁

把 `dist/cds_torch_adam_update.tar.gz` 及 `.sha256` 文件上传服务器项目目录，然后：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
sha256sum -c cds_torch_adam_update.tar.gz.sha256
tar -xzf cds_torch_adam_update.tar.gz
```

补丁包含训练入口、候选依赖、配置迁移脚本和测试，不覆盖服务器的默认配置文件。

## 2. 安装剩余训练依赖

保持cdi-train激活。PyTorch已经安装，不需要重新安装它。

```bash
uv pip install --python /media/data/gezhenjiang/venvs/cdi-train/bin/python \
  -r src/requirements-train-torch26.txt
```

候选核心版本为NumPy1.26.4、Transformers4.51.3、PEFT0.15.2、Accelerate1.6.0。Transformers4.51.3要求huggingface-hub<1，故可能会将此前最新版Hub降到兼容版本；不改已下载模型文件或其commit记录。这个文件固定核心组合与必要版本范围，不是经过完整GPU验证的传递依赖锁文件。

然后安装DeepSpeed，安装时关闭扩展预编译，复用上一步已装好的torch等构建依赖：

```bash
DS_BUILD_OPS=0 uv pip install --python /media/data/gezhenjiang/venvs/cdi-train/bin/python --no-build-isolation deepspeed==0.16.7
```

这一整行执行；`DS_BUILD_OPS=0`不能单独拆成未export的变量赋值。它只控制安装期预编译，不能单独阻止默认FusedAdam在运行时JIT，故需要第1步的原生AdamW补丁。[DeepSpeed安装脚本](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.7/setup.py)

如安装失败，保留完整错误，不使用 `DS_SKIP_CUDA_CHECK` 强行绕过CUDA编译检查。

## 3. 检查依赖与导入

```bash
uv pip check --python /media/data/gezhenjiang/venvs/cdi-train/bin/python
python - <<'PY'
import numpy, torch, transformers, peft, accelerate, deepspeed
for module in [numpy, torch, transformers, peft, accelerate, deepspeed]:
    print(module.__name__, module.__version__)
print('CUDA:', torch.version.cuda, torch.cuda.is_available())
PY
```

应无NumPy缺失警告，并显示以上核心版本和DeepSpeed0.16.7。导入通过不等于GPU训练已通过。DeepSpeed探测未安装的可选扩展可能打印提示，是否影响运行要以完整错误和实际所需功能判断。

## 4. 为后端适配创建新的运行目录

```bash
python src/configure_torch_adam.py
python -m unittest discover -s src/tests -v
python src/cds_pipeline.py prepare
```

预期21项CPU测试通过。脚本备份当前配置，只把run_dir改为 `runs/cds_qwen25_3b_pre300_torchadam`；样本量、seed、模型路径、HF revision、教师与训练超参数均保留。此前冻结的pre300目录保留，新的prepare重新记录训练实现哈希。同seed与同原始数据确保300题成员、诊断集、测试集及3673题保留池与此前一致，无需再下载权重。

预期prepare输出仍为：

```json
{"diag": 3500, "test": 1319, "warmup": 300, "dev": 0, "reserve": 3673}
```

## 5. 有空闲GPU时再训练

确认资源安排并重新查看 `nvidia-smi`。即使显存还有余量，100%利用率也表示已有计算任务在使用GPU，不能由空闲显存直接判断计算卡空闲。

若0号卡已可用，在训练环境中执行：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 src/train_lora.py \
  --data runs/cds_qwen25_3b_pre300_torchadam_tok1024/data/warmup.json \
  --output runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/warmup
```

如果可用的是1号卡，将CUDA_VISIBLE_DEVICES改成1。入口仍自动保持有效batch32。首次运行如报错，保留完整堆栈及输出目录状态，先排障；不把CPU测试或模型成功下载当作训练成功。

实际训练通过后，记录 `experiment.json`、adapter和tokenizer，保存环境freeze与驱动信息，再进入学生模型答题与诊断阶段。
