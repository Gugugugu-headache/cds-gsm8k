# CDS 在现有 GSM8K 划分上的复现

结论：可以做 **CDS 方法的适配复现**。现有源码足以恢复主流程，但不能直接运行得到论文数字；本项目更换了数据划分和教师模型，且原发布代码存在未公开参数和论文/代码差异。当前提供数据导出、目标模型评测、全局统计、教师诊断与合成、增强、两阶段筛选、LoRA 训练及配对结果比较入口。CPU 逻辑可本地检查，GPU/API 端到端结果必须在服务器实际运行后才能确认。

详细实验判断见 [复现审查.md](复现审查.md)。原目录 `cds-04D1` 和原数据划分保持不变。

本代码目录单独使用GitHub同步，仓库地址为 https://github.com/Gugugugu-headache/cds-gsm8k 。现有服务器src接入与日常pull步骤见 [GitHub同步.md](GitHub同步.md)。仓库仅提供配置模板；首次独立checkout需复制 `configs/cds_gsm8k.example.json` 为 `configs/cds_gsm8k.json` 并填写本机模型路径与实际版本，已有服务器接入时会保留原实际配置。

## 当前协议

**当前目标候选：Qwen/Qwen2-7B-Instruct。** 独立配置、下载与诊断题检查步骤见 [Qwen2-7B候选部署.md](Qwen2-7B候选部署.md)。已有3B默认配置与结果保留；候选所有命令显式使用 `--config src/configs/cds_gsm8k_qwen2_7b.json`。下方默认配置命令仍是已有3B流程参考，不会自动切换候选。

用户已在完整Raw test前确认学生解题上限改为1024新token。服务器使用 [1024token评测协议.md](1024token评测协议.md) 的增量更新，仅修改预算与run_dir，保留512历史run及模型commit。参考模型直接采集响应；少数目标模型的CDS与主方法才共用各自的300题M0。

| 项目 | 配置 |
|---|---|
| 新增提升目标候选 | Qwen/Qwen2-7B-Instruct（独立配置和run，先检查诊断题） |
| 已完成Raw基线的模型 | Qwen/Qwen2.5-3B-Instruct（原默认配置与85.44%兼容评分保留） |
| 教师 | 在线 Qwen/Qwen2.5-72B-Instruct，需确认账户可用的精确 model ID |
| 诊断 | 原 D_diag，3500 题，仅官方 train |
| 最终测试 | 原 D_test，1319 题，官方 test |
| 剩余训练题 | 原 D_remaining 3973 题，固定抽300题预微调，其余3673题暂不分配；不额外划开发集 |
| KC | 默认原始 28 维 `kc_labels`；18 维 `diagnosis_kcs` 可作单独的适配实验 |
| CDS 数值规则 | 默认跟随发布代码：细粒度每次 3 题、质量 10 分、平滑 0.01、max 归一化、≥μ−σ |
| 全局薄弱阈值 | 准确率≤0.5 或频率≤0.05；这是新增的显式实验选择，不是作者已确认参数 |
| 零支持 KC | 默认模拟源码零准确率处理用于合成/打分；统计文件仍用 null 表示无法估计掌握率 |
| 学生评测 | Qwen chat template、greedy、1024新tokens、源码数值抽取/容差；不同于原文512预算，不等同于严格官方 EM |
| LoRA | r=8、alpha=16、q_proj/v_proj、dropout=0.05、bf16、2048、1 epoch |
| 有效 batch | 32；两卡 micro batch=2 时 gradient accumulation=8 |

300/0/3673 是当前对 **剩余集合** 的用途分配，seed=42；诊断集和测试集成员保留。dev.json仅为空的兼容文件。原论文报告292题预微调，本实验将共同初始化预算设为300题，属于统一条件下的适配，不声称重建原292题身份或证明300最优。

当前300题协议、服务器迁移命令和可直接参考的论文表述见 [300题预微调与论文表述.md](300题预微调与论文表述.md)。本次修正使用Trainer回调创建源码WarmupLR，避免原适配入口中HF/DeepSpeed调度参数冲突；没有改变配置中原有的三个调度数值。仍需服务器GPU实际验证。

后续主实验与CDS共用300题训练得到的同一个M0。改变初始化、划分或共同评测条件后，需要补跑对应条件的基线。此后若要调参，应从3673题未分配池中明确安排独立开发用途，不使用test选择配置。

主实验各baseline与本文方法统一采用发布代码的三层抽取（####数值、the answer is数值、最后数字）及数值容差作为主评分，同时报告抽取方式和截断统计。仅####规则的匹配比例是补充指标，不替代CDS兼容Acc。

服务器已完成1024预算的Raw test：用户报告1127/1319（85.4435%）、无数字0、截断0；hash抽取1028条其中886条判对，last_number抽取291条其中241条判对。该记录来自服务器输出，逐题文件仍保存在服务器runs内。尚未执行300题预微调。

## 环境与文件

已确认服务器PyTorch2.6.0+cu124可识别GPU，用户已完成训练依赖检查与旧21项测试，nvcc不在PATH。训练安装步骤以 [无nvcc训练环境.md](无nvcc训练环境.md) 为准：核心依赖组合使用requirements-train-torch26.txt，DeepSpeed0.16.7关闭安装期预编译，训练显式选原生Torch AdamW避免FusedAdam JIT。当前1024协议使用新目录重新记录配置，300题成员不变。GPU训练仍待实际验证。

服务器首次部署按 [服务器部署.md](服务器部署.md) 执行。上传包中代码目录命名为 `src`；本说明中的 `CDI（CDS复现）/` 命令前缀在服务器应替换为 `src/`。

在项目根目录运行命令。默认服务器模型路径是 `/media/data/gezhenjiang/model_list/Qwen2.5-3B-Instruct`。下载由用户在服务器完成；此项目不自动下载模型。

训练和推理分开虚拟环境：训练环境按上述部署文档使用 `requirements-train-torch26.txt`，推理环境使用 `requirements-infer-torch26.txt`。用户已确认cdi-infer的torch2.6.0+cu124、vLLM0.8.5.post1导入及1024预算的完整Raw test通过；GPU训练需在服务器运行验证。保存各自 `pip freeze`、CUDA/驱动信息及HF模型revision；候选requirements不是完整锁定环境。

```bash
python -m pip install -r CDI（CDS复现）/requirements-cpu.txt
python -m unittest discover -s CDI（CDS复现）/tests -v
python CDI（CDS复现）/cds_pipeline.py prepare
```

运行前编辑 `CDI（CDS复现）/configs/cds_gsm8k.json` 的目标模型路径、模型 revision 注记和 `run_dir`。`prepare` 会校验原始文件及划分清单的 SHA256、题目身份和跨集合规范化题文重叠。冻结配置与关键实现哈希；之后改参数/实现必须使用新的 run_dir，避免混入已有结果。导出文件默认在被 git 忽略的 `runs/cds_qwen25_3b_pre300_torchadam_tok1024`。

## 运行顺序

原始模型基线已在服务器完成，结果应单独保留；后续无需重跑已有Raw test。首次部署者可按 [原始模型基线评测.md](原始模型基线评测.md) 检查推理环境并采集Raw，然后回训练环境执行下面第1步。Raw、M0、CDS的评测分别保存，test分数不用于调参。

以下均以默认 run_dir 为例；更改 run_dir 时同步修改命令中的数据/检查点路径。可先用少量诊断题检查基础设施是否正常，再进行正式完整推理；当前不单设开发集。训练入口只接受本次 run 的 warmup/CDS/random 文件，不接受诊断集或测试集。

1. 格式预微调。所有下游组从这个共同检查点出发。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 CDI（CDS复现）/train_lora.py \
  --data runs/cds_qwen25_3b_pre300_torchadam_tok1024/data/warmup.json \
  --output runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/warmup
```

2. 目标模型作答与诊断。以下在 vLLM 环境运行。`--tp 1` 使用一张卡；确需张量并行时设置可见卡并改为 `--tp 2`。

```bash
CUDA_VISIBLE_DEVICES=0 python CDI（CDS复现）/cds_pipeline.py infer --split diag --tag smoke_diag --limit 20 \
  --adapter runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/warmup
CUDA_VISIBLE_DEVICES=0 python CDI（CDS复现）/cds_pipeline.py infer --split diag --tag diag_before \
  --adapter runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/warmup
python CDI（CDS复现）/cds_pipeline.py profile --tag diag_before
```

`profile` 拒绝测试集、开发集和不完整的诊断集。没有数字的回答记错并进入细粒度诊断。检查 `eval/*.json` 中 `no_number` 和 `truncated`，如需比较不同输出预算，应先从未分配训练池中安排独立开发用途，不用test选择；最终比较必须保持统一提示词与预算。针对新选的 Qwen2.5-3B-Instruct，默认使用模型官方聊天模板，训练/评测保持一致；这是相对 CDS 源码 plain prompt 的明确适配。若需要源码提示词对照，在独立 run 中把 `inference_prompt` 改为 `source_plain`。

3. 教师 API。通过进程环境变量传入凭据；程序不会自动执行或输出根目录 `.env`。变量名称与现有文件一致。可通过交互输入避免在命令中写密钥：

```bash
read -rs -p 'SiliconFlow API key: ' Siliconflow_API_KEY
export Siliconflow_API_KEY
export Siliconflow_BASE_URL=https://api.siliconflow.cn/v1
python CDI（CDS复现）/cds_pipeline.py synthesize
python CDI（CDS复现）/cds_pipeline.py quality --pool seed
python CDI（CDS复现）/cds_pipeline.py augment
python CDI（CDS复现）/cds_pipeline.py quality --pool augmented
python CDI（CDS复现）/cds_pipeline.py select
```

这些命令会产生实际 API 用量。教师可见诊断题、学生错答、诊断分析及合成题；测试题不会发送给教师。所有请求按 endpoint/model/messages/采样参数生成缓存键，保存请求、原始输出、结束原因及 usage，不保存鉴权头。成功请求可复用，限流/部分服务端错误会重试。进程在响应返回但写缓存前退出仍可能重复计费，不能保证恰好一次调用。

解析失败或被截断的合成/诊断输出会停止阶段，缓存保留以供检查；不静默丢弃整批题。重新运行会复用缓存，故格式错误需要检查原因并在新 run 中调整协议，不能仅靠重跑期待不同输出。质量打分无法解析时进入 rejected 文件。

SiliconFlow 的公开参数表没有列出 `repetition_penalty`，这里省略原源码的 1.05，不拿含义不同的 `frequency_penalty` 替代。在线模型是否可调用及精确模型 ID 仍须用用户账户验证；没有自动降级到别的模型。

4. 分别训练 CDS 与随机筛选对照。两组都继续同一个 warmup adapter；不能让随机组继续 CDS 训练结果。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 CDI（CDS复现）/train_lora.py \
  --data runs/cds_qwen25_3b_pre300_torchadam_tok1024/train_cds.json \
  --adapter runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/warmup \
  --output runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/cds
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 CDI（CDS复现）/train_lora.py \
  --data runs/cds_qwen25_3b_pre300_torchadam_tok1024/train_random.json \
  --adapter runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/warmup \
  --output runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/random
```

训练复用原 `QwenDataset` 的聊天模板、截断及仅答案 loss；训练器改为 HF Trainer + DeepSpeed ZeRO-1，保留源码 WarmupLR，使用 SDPA、梯度检查点降低部署依赖/显存需求。完整训练器轨迹不应宣称与原脚本逐步一致。训练输出包含监督 token 数、数据哈希及有效 batch。当前训练入口没有中断续训功能；输出目录非空时需用新路径重跑。

Llama-3-8B-Instruct 实验需要更换模型路径、run_dir，并在每个训练命令中加 `--dataset-type Llama3Dataset`。Qwen2.5-7B-Instruct 使用 QwenDataset，但属于替换学生模型的实验。

5. 冻结配置后，统一最终测试与比较。

```bash
CUDA_VISIBLE_DEVICES=0 python CDI（CDS复现）/cds_pipeline.py infer --split test --tag raw_test
CUDA_VISIBLE_DEVICES=0 python CDI（CDS复现）/cds_pipeline.py infer --split test --tag before_test \
  --adapter runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/warmup
CUDA_VISIBLE_DEVICES=0 python CDI（CDS复现）/cds_pipeline.py infer --split test --tag cds_test \
  --adapter runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/cds
CUDA_VISIBLE_DEVICES=0 python CDI（CDS复现）/cds_pipeline.py infer --split test --tag random_test \
  --adapter runs/cds_qwen25_3b_pre300_torchadam_tok1024/checkpoints/random
python CDI（CDS复现）/compare.py \
  --before runs/cds_qwen25_3b_pre300_torchadam_tok1024/eval/before_test.json \
  --after runs/cds_qwen25_3b_pre300_torchadam_tok1024/eval/cds_test.json \
  --output runs/cds_qwen25_3b_pre300_torchadam_tok1024/comparison_before_cds.json
python CDI（CDS复现）/compare.py \
  --before runs/cds_qwen25_3b_pre300_torchadam_tok1024/eval/random_test.json \
  --after runs/cds_qwen25_3b_pre300_torchadam_tok1024/eval/cds_test.json \
  --output runs/cds_qwen25_3b_pre300_torchadam_tok1024/comparison_random_cds.json
```

输出准确率、百分点变化、错→对/对→错、按题目配对 bootstrap 区间和 exact McNemar p。区间不涵盖教师合成或训练随机性；正式论文仍需多 seed。随机对照仅控制 **CDS_score 筛选**，因为它也使用 CDS 诊断合成的候选池，不能拿它证明整个 CDS 诊断策略优于通用合成。

## 当前验证边界

离线测试覆盖答案抽取、无数字错题、零支持规则、与源码分数函数数值一致、格式解析、缓存不联网、截断输出拒绝、模拟教师的合成到筛选串联和配对统计。真实数据校验/导出可在 CPU 完成。

当前本地没有 torch/transformers/peft/vLLM/DeepSpeed，也没有目标模型权重，因此尚无真实学生推理、API 成功调用、GPU 训练或提升结果。语义近重复、教师诊断正确性和合成数学答案的独立正确性也未得到验证；教师打满分只是筛选信号。
