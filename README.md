# CDS 在现有 GSM8K 划分上的复现

结论：可以做 **CDS 方法的适配复现**。现有源码足以恢复主流程，但不能直接运行得到论文数字；本项目更换了数据划分和教师模型，且原发布代码存在未公开参数和论文/代码差异。当前提供数据导出、目标模型评测、全局统计、教师诊断与合成、增强、两阶段筛选、LoRA 训练及配对结果比较入口。CPU 逻辑可本地检查，GPU/API 端到端结果必须在服务器实际运行后才能确认。

详细实验判断见 [复现审查.md](复现审查.md)。原目录 `cds-04D1` 和原数据划分保持不变。

本代码目录单独使用GitHub同步，仓库地址为 https://github.com/Gugugugu-headache/cds-gsm8k 。现有服务器src接入与日常pull步骤见 [GitHub同步.md](GitHub同步.md)。仓库仅提供配置模板；首次独立checkout需复制 `configs/cds_gsm8k.example.json` 为 `configs/cds_gsm8k.json` 并填写本机模型路径与实际版本，已有服务器接入时会保留原实际配置。

## 当前协议

Qwen1.5已采集的Raw回答通过 `qwen15_cds_ollama_samples.py` 接入服务器现有的Qwen2.5-72B Q4_K_M教师，复现CDS baseline。该入口保留repeat_last_n=32768兼容修复，并支持缓存中 `**Question 1:**`、无冒号的 `#### Question 1` 标题和完整样本之间的分隔线；原冻结代码与JSON不改。独立目录、本轮8分质量门槛及完整命令见 [Ollama教师CDS.md](Ollama教师CDS.md)。check/prepare/profile不加载72B模型；教师生成需安排GPU资源。

Qwen1.5-7B-Chat的独立Raw评测使用 `qwen15_raw_eval.py`，完整命令见 [Qwen1.5原始模型评测.md](Qwen1.5原始模型评测.md)。用户已报告test为815/1319（61.79%）、diag为2556/3500（73.03%），均无抽取失败和截断；完整回答保留在原评测目录。其后续CDS使用上面的Ollama入口，已准备代码但尚待服务器实际运行。

### 3B运行记录（历史）

**Qwen/Qwen2.5-3B-Instruct已从Raw完成CDS试跑。** 用户为后续多模型认知诊断与少数目标提升选择统一Raw起点；完整历史步骤见 [Raw起点CDS流程.md](Raw起点CDS流程.md)。该路线使用18个主诊断KC，入口为 `raw_cds18.py`；迁移与画像命令见 [18KC迁移与画像.md](18KC迁移与画像.md)。原 `raw_cds.py` 的28维run也保留。

| 项目 | 3B运行设置 |
|---|---|
| Raw路线配置 / run | `configs/cds_gsm8k_raw_kc18.json` / `runs/cds_qwen25_3b_raw_kc18_dev500_tok1024` |
| 诊断 / 最终测试 | 原3500 / 1319题，成员和顺序不变 |
| KC范围 | 固定18维diagnosis_kcs；每题仅重绑标签，保留回答与生成来源 |
| 开发集 | 初始化pilot固定的500题，不因取消预微调而重划 |
| 原300题 | 单独保留为initialization_holdout；Raw路线warmup为空 |
| reserve | 3173题 |
| 起点 | Raw；CDS和随机对照分别从同一原始权重新建LoRA |
| 下游训练 | 沿用原CDS主配置的batch32与WarmupLR；不是初始化pilot的batch20/恒定LR |
| 评测 | 同一chat数学指令、greedy、1024新token、CDS三层数值抽取及容差 |
| 教师 | 在线Qwen/Qwen2.5-72B-Instruct；仅显式运行教师阶段时调用 |

已有Raw test（1127/1319，85.44%）由新协议引用和校验，无需重跑。3500题Raw诊断已报告3203题正确、5条截断；旧28维weak_kcs=20包括低覆盖/零支持，不代表20项实际能力不足。18维画像分别列出低准确率、低覆盖、零支持原因。新入口拒绝用M0/子集回答建立Raw诊断；生成训练题排除与diag、dev、test及原300题的规范化文本精确重合，并保留相同题数的随机对照。相同题数不等于相同token或API预算，需额外记录。

用户已报告该轮CDS组test为1102/1319（83.55%），随机组为1098/1319（83.24%）；均低于Raw。两组各354题、11次优化器更新，实际epoch=0.9943502824858758。这些结果及10分质量门槛的运行完整保留，不因新模型/本地教师接入而替换。本地没有逐题真实生成结果。

教师配置可从服务器项目根目录 `.env` 显式读取，使用 `teacher_env.py check` 检查，再用 `teacher_env.py synthesize` 运行教师阶段。步骤见 [教师env调用.md](教师env调用.md)。仅加载教师变量，以文件值覆盖旧终端值；请求经curl传输，实验配置和缓存语义保留。

教师错题诊断若达到1024输出上限，现支持保留原缓存，按2048/4096上限作有界重试；成功请求直接复用。规则、缓存映射及全部尝试的usage单独记录，见 [教师诊断截断恢复.md](教师诊断截断恢复.md)。这是教师预算适配，学生1024评测上限保持原设置。

题目合成/增强若被4096输出预算截断，现增加独立4096→8192重试，原成功缓存及诊断恢复策略保留，详见 [教师合成截断恢复.md](教师合成截断恢复.md)。`teacher_env.py inspect --request-key <缓存键>` 可在不调用API的情况下确认失败请求的预算、阶段线索与用量。

教师题目标题若使用 `#### Question 1:`，新增保守Markdown标题兼容层；仍要求完整N题及连续编号，原缓存和正文保留。离线预检用 `teacher_env.py parse-cache --request-key <缓存键> --count 3`，详见 [合成Markdown标题解析.md](合成Markdown标题解析.md)。

完整Markdown样本后附带的可识别总结句，现通过独立版本排除出训练样本并记录哈希，保留原标题策略和审计；见 [合成尾部说明解析.md](合成尾部说明解析.md)。不删除原缓存、不改题文答案、不接受缺题或截断样本。

尾部带 `### Summary:` 标题时，使用独立兼容扩展并保留旧两个解析版本；仍对总结正文和完整样本执行原检查，详见 [合成尾部说明解析.md](合成尾部说明解析.md)。

总结中的描述性数学动词（如“requiring them to ... solve equations”）由独立版本兼容；不再只按单词判断新增任务。新分支保留完整样本及结构检查、直接指令/纠错检测，旧解析行为和策略文件保留。详见 [合成尾部说明解析.md](合成尾部说明解析.md)。

### 初始化预实验保留

[开发集初始化验证.md](开发集初始化验证.md) 记录原定三组开发集比较：Raw、1e-5、3e-5均468/500，hash覆盖分别413/408/440，因此原规则选择3e-5。所选新M0 test为1126/1319（85.37%），旧高LR M0为1089/1319（82.56%），Qwen2-7B Raw候选为1137/1319（86.20%）。这些统计由用户在服务器运行后提供，本地不持有逐题真实响应。

当前选择Raw是统一实验条件的设计决策，**不能写成开发集选中了Raw**。旧配置、检查点、结果和选择记录均保留。此前已观察test的探索历史也应保留；后续不根据test继续选模型、prompt或阈值。论文应明确标注为“CDS（统一Raw初始化）”适配版本，而非原论文数值的严格复现。

原初始化pilot用global20令300题恰好15次更新；这不是通用Trainer尾批修复。3B下游354题已观察到旧Trainer尾批省略；Qwen1.5合成训练数据大小尚未知，同样保存实际epoch，不将其误报为完整遍历所有样本。任何后续训练协议修改应在训练前明确冻结，并同步用于所有对照。

## 环境与文件

已确认服务器PyTorch2.6.0+cu124可识别GPU，用户已完成训练依赖检查与旧21项测试，nvcc不在PATH。训练安装步骤以 [无nvcc训练环境.md](无nvcc训练环境.md) 为准：核心依赖组合使用requirements-train-torch26.txt，DeepSpeed0.16.7关闭安装期预编译，训练显式选原生Torch AdamW避免FusedAdam JIT。旧300题GPU训练和M0评测、开发集初始化验证、3B Raw路线CDS/随机组训练及评测已完成；Qwen1.5本地教师路线尚待服务器实际验证。

服务器首次部署按 [服务器部署.md](服务器部署.md) 执行。上传包中代码目录命名为 `src`；本说明中的 `CDI（CDS复现）/` 命令前缀在服务器应替换为 `src/`。

在项目根目录运行命令。默认服务器模型路径是 `/media/data/gezhenjiang/model_list/Qwen2.5-3B-Instruct`。下载由用户在服务器完成；此项目不自动下载模型。

训练和推理分开虚拟环境：训练环境按上述部署文档使用 `requirements-train-torch26.txt`，推理环境使用 `requirements-infer-torch26.txt`。用户已确认cdi-infer的torch2.6.0+cu124、vLLM0.8.5.post1导入及1024预算的完整Raw test通过。保存各自 `pip freeze`、CUDA/驱动信息及HF模型revision；候选requirements不是完整锁定环境。

```bash
python -m pip install -r CDI（CDS复现）/requirements-cpu.txt
python -m unittest discover -s CDI（CDS复现）/tests -v
python CDI（CDS复现）/cds_pipeline.py prepare
```

运行前编辑 `CDI（CDS复现）/configs/cds_gsm8k.json` 的目标模型路径、模型 revision 注记和 `run_dir`。`prepare` 会校验原始文件及划分清单的 SHA256、题目身份和跨集合规范化题文重叠。冻结配置与关键实现哈希；之后改参数/实现必须使用新的 run_dir，避免混入已有结果。导出文件默认在被 git 忽略的 `runs/cds_qwen25_3b_pre300_torchadam_tok1024`。

## 历史M0路线参考（当前Raw路线请用上方专用入口）

以下是旧默认配置的M0流程说明，保留用于解释已有实验；当前Raw路线执行清单在《Raw起点CDS流程.md》。不要把本节的旧warmup路径用于新Raw实验，也不要使用旧prepare重新划分Raw路线的开发集。

原始模型基线已在服务器完成，结果应单独保留；后续无需重跑已有Raw test。首次部署者可按 [原始模型基线评测.md](原始模型基线评测.md) 检查推理环境并采集Raw，然后回训练环境执行下面第1步。Raw、M0、CDS的评测分别保存，test分数不用于调参。

以下均以默认 run_dir 为例；更改 run_dir 时同步修改命令中的数据/检查点路径。可先用少量诊断题检查基础设施是否正常，再进行正式完整推理；旧默认配置不单设开发集，新验证协议已独立留出500题。训练入口只接受本次 run 的 warmup/CDS/random 文件，不接受诊断集或测试集。

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
