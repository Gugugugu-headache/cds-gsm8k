# Qwen2-7B-Instruct目标候选

2026-11-05，用户选择 `Qwen/Qwen2-7B-Instruct` 作为提升目标候选。现有Qwen2.5-3B-Instruct的1024预算Raw test与默认配置保留。候选检查先使用诊断题；尚未启动候选下载、真实推理、预微调或教师API。

这是Instruct版，使用自身聊天模板。已有数学指令、CDS兼容评分、greedy、1024新token预算、3500/1319诊断/测试划分、从remaining中固定seed42抽300题的规则均保留。模型之间共享题目身份，不共享权重；每个目标的CDS与主方法在该目标内共用同一个M0。参考模型不要求进行300题微调。

| 项目 | 候选设置 |
|---|---|
| HF仓库 | Qwen/Qwen2-7B-Instruct |
| 服务器实际配置 | src/configs/cds_gsm8k_qwen2_7b.json |
| 版本控制中的模板 | src/configs/cds_gsm8k_qwen2_7b.example.json |
| 默认权重目录 | 原3B权重同级的Qwen2-7B-Instruct目录 |
| run_dir | runs/cds_qwen2_7b_pre300_torchadam_tok1024 |

实际配置由服务器现有配置生成，保留服务器的教师设置和训练超参数；模型路径换到独立目录、HF版本注记重新等待下载记录。默认 `src/configs/cds_gsm8k.json` 继续指向已有3B，**候选每条命令都显式传入 --config**。

## 1. 拉取更新并创建候选配置

在服务器项目根目录、已有cdi-infer环境执行：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
source /media/data/gezhenjiang/venvs/cdi-infer/bin/activate
git -C src pull --ff-only
python -m unittest discover -s src/tests -v
python src/configure_qwen2_7b.py
```

配置脚本只新建 `src/configs/cds_gsm8k_qwen2_7b.json`。同设置重复调用会保留已下载HF commit；若已有不同配置或冲突冻结run，脚本拒绝覆盖。

如模型目录需要放到其他位置，可在首次创建时加 `--model-dir /absolute/path/to/Qwen2-7B-Instruct`，不要使用3B目录。实际配置默认不提交Git，模板提交Git；pull不会替换服务器实际模型路径或commit。

## 2. 下载并记录精确HF版本

先下载，成功记录commit后再prepare，避免提前冻结待下载注记。服务器访问huggingface.co曾失败，因此沿用此前成功的匿名镜像方式：

```bash
export HF_HOME=/media/data/gezhenjiang/model_list/.hf_cache

env HF_ENDPOINT=https://hf-mirror.com HF_TOKEN= \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 \
  python src/download_student.py \
  --config src/configs/cds_gsm8k_qwen2_7b.json \
  --repo-id Qwen/Qwen2-7B-Instruct
```

镜像属于第三方，这条命令不会向其传入现有HF token。下载器先解析commit，再按相同commit下载，将仓库与版本写入权重目录的download_provenance.json和候选配置的student_revision_note。此次只下载候选，不改3B权重或3B配置，不调用教师。

student_repo_id声明了候选仓库；如果--repo-id指向其他模型会直接报错，避免把基础版或旧3B权重下载进候选目录。不传--repo-id时，下载器也会读取候选配置中的student_repo_id；旧3B配置仍保持原行为。

## 3. CPU导出与验证

下载成功后执行：

```bash
python src/cds_pipeline.py \
  --config src/configs/cds_gsm8k_qwen2_7b.json prepare
```

预期数量保持：

```json
{"diag": 3500, "test": 1319, "warmup": 300, "dev": 0, "reserve": 3673}
```

此时才冻结候选配置与已有训练/推理实现。300题与3B当前协议的成员和顺序一致，无需重新划分源数据。

## 4. 候选Raw诊断题检查

先确认分配给你的GPU及显存。7B比3B需要更多显存，现有vLLM入口按可见单卡总显存的85%规划；共享卡上其他任务仍可能导致初始化失败。以下假定0号卡可用：

```bash
nvidia-smi

CUDA_VISIBLE_DEVICES=0 python src/cds_pipeline.py \
  --config src/configs/cds_gsm8k_qwen2_7b.json infer \
  --split diag --tag raw_smoke_diag --limit 20
```

不传adapter，此时使用Raw候选。检查输出n=20、adapter=null、truncated和抽取方式，查看少量回答原文。20题只确认加载与生成流程，不能据此确认模型能力、诊断稳定性或提升空间。

结果文件： `runs/cds_qwen2_7b_pre300_torchadam_tok1024/eval/raw_smoke_diag.json`。

如需采集候选完整Raw诊断响应，在该检查成功后执行：

```bash
CUDA_VISIBLE_DEVICES=0 python src/cds_pipeline.py \
  --config src/configs/cds_gsm8k_qwen2_7b.json infer \
  --split diag --tag raw_diag
```

候选判断和开发使用训练来源的诊断题，不在test上反复筛选模型或预算。Raw响应可以描述预微调前的模型；后续M0用于CDS/主方法时仍需采集M0自身的诊断响应，不能直接复用Raw错题画像。

## 5. 确定目标后的Raw test与训练

目标与共同协议确定后，才用同一配置采集完整Raw test：

```bash
mkdir -p runs/cds_qwen2_7b_pre300_torchadam_tok1024/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python src/cds_pipeline.py \
  --config src/configs/cds_gsm8k_qwen2_7b.json infer \
  --split test --tag raw_test \
  2>&1 | tee runs/cds_qwen2_7b_pre300_torchadam_tok1024/logs/raw_test.log
```

之后在训练环境里运行300题初始化时，也需在train_lora.py命令中显式加 `--config src/configs/cds_gsm8k_qwen2_7b.json`，使用该候选run下的data/warmup.json与checkpoints/warmup。此文档不自动启动训练。7B的真实训练显存和运行情况尚待服务器验证。

CDS与主方法的主评分均保留三层数值抽取与容差，补充报告抽取方式、格式与截断统计。选择较低起点不保证增益，最终证据是同一目标和预算下相对CDS、随机对照等方法的收益。
