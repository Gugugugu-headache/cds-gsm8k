# 从 Raw 开始运行 CDS

用户明确选择 Qwen2.5-3B-Instruct 的原始发布检查点作为本阶段共同起点，以便后续多模型认知诊断与目标提升采用一致的模型状态。这是 **CDS（统一 Raw 初始化）适配版本**，省略原始方法的初始化微调。

开发集初始化实验已选择 lr3e5（准确率与Raw同为468/500，hash覆盖440/500优于Raw的413/500）；其test为1126/1319。这个选择记录保留，不能改写成开发集选中了Raw。当前Raw路线是后续研究设计决策。旧M0、新M0及Qwen2-7B候选结果均不覆盖。

## 独立目录与固定数据

* 配置：`src/configs/cds_gsm8k_raw.json`，由服务器已有配置生成，不提交Git。
* 实验目录：`runs/cds_qwen25_3b_raw_dev500_tok1024/`。
* diag=3500、test=1319、dev=500：继承已冻结数据，成员、内容、顺序不变。
* `warmup.json` 为空，`pre_finetune_size=0`；本路线不执行预微调。
* 原300题另存 `data/initialization_holdout.json`；其余reserve=3173。
* 继续固定chat数学指令、greedy、1024新token及CDS兼容评分。

**所有步骤使用 `raw_cds.py`。不要对Raw配置直接运行旧 `cds_pipeline.py prepare`**：旧导出器会按照预微调题数重新切开发集，不能表达此保留划分。新入口独立冻结数据与代码，不修改旧训练、推理核心文件或初始化验证入口，旧run的实现哈希保持有效。

新入口引用已有Raw test文件并校验模型、HF revision对应的配置、题目身份、评分和文件哈希，不重复采集Raw test，不把原记录的config_hash改写成新配置。之前的Raw test为1127/1319（85.4435%）。本地测试使用临时模拟响应验证引用逻辑；真实逐题结果在服务器。

## 1. 同步、验证、导出

本地先提交推送，再在服务器项目根目录执行：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
git -C src pull --ff-only
python -m unittest discover -s src/tests -v
python src/raw_cds.py prepare
```

预期输出包含：

```json
{"diag":3500,"test":1319,"dev":500,"reserve":3173,"warmup":0,"initialization_holdout":300}
```

不会下载模型、运行GPU任务或调用教师。若原Raw test、pilot冻结文件或配置不一致，脚本停止；不要修改旧记录绕过检查。

## 2. Raw 在3500题上作答并建立全局诊断统计

确认分配的GPU可用，下例使用0号卡。仍使用vLLM单卡显存比例0.85。

```bash
source /media/data/gezhenjiang/venvs/cdi-infer/bin/activate
nvidia-smi
mkdir -p runs/cds_qwen25_3b_raw_dev500_tok1024/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python src/raw_cds.py infer --split diag \
  2>&1 | tee runs/cds_qwen25_3b_raw_dev500_tok1024/logs/raw_diag.log
```

结果在 `eval/raw_diag.json`，必须为diag、n=3500、adapter=null。完成后：

```bash
python src/raw_cds.py profile
```

生成 `profile.json`，并打印错题数与全局薄弱KC数。profile是按KC统计的全局画像；后续教师还会逐错题做细粒度诊断。该入口拒绝M0回答、子集或其他配置的回答。

## 3. 教师诊断、合成、质量过滤与增强

下面四个阶段会实际调用在线教师并产生API用量。教师仅接收诊断错题及候选合成样本，不接收dev/test答案。沿用现有Qwen2.5-72B-Instruct教师设置。

在当前服务器终端交互输入自己的API密钥；不要将密钥写入Git或粘贴到聊天中：

```bash
read -rs -p 'SiliconFlow API key: ' Siliconflow_API_KEY
printf '\n'
export Siliconflow_API_KEY
export Siliconflow_BASE_URL=https://api.siliconflow.cn/v1
```

依次执行，每条成功后再执行下一条：

```bash
python src/raw_cds.py synthesize
```

```bash
python src/raw_cds.py quality --pool seed
```

```bash
python src/raw_cds.py augment
```

```bash
python src/raw_cds.py quality --pool augmented
```

若有E道错题、W个薄弱KC，初始合成阶段约W+2E次调用（不含重试），请求5W+3E道初始题；质量过滤每道题一次调用，增强另计。先用profile输出估算规模。响应按请求缓存，中途失败可保留缓存重跑同一阶段；缓存不能保证网络中断时恰好一次计费。格式解析错误需要检查具体缓存，不通过静默少收题绕过。

## 4. CDS筛选及随机对照

```bash
python src/raw_cds.py select
```

沿用原全局准确率/频率加权、max归一化及均值减标准差筛选。额外对diag、dev、test、原300题做规范化题文精确去重，禁止把它们原样收入合成训练数据；这是本适配版本的数据隔离规则，不声称能够检测语义近重复。

输出 `train_cds.json`、`train_random.json`、`selection.json`；随机对照来自同一质量过滤后的候选池，与CDS题数相同，未匹配token数或教师成本。若选中题数等于池大小，两组可能相同，应如实报告，不能称其为有区分力的筛选对照。

## 5. 从Raw开始针对性微调

下游训练沿用原3B主配置的CDS训练参数，**不是**初始化pilot的global20与恒定LR方案：global32、micro2、单卡累积16、LoRA r8/alpha16/q_proj+v_proj、名义1 epoch、原WarmupLR(min1e-4,max3e-4,steps300)。配置中的optimizer lr3e-5并非实际恒定学习率。

本阶段仍使用旧训练器。对于不可整除的合成数据量，旧Trainer可能舍去最后不足一个累积组的样本；必须保留实际epoch与日志，不宣称完整遍历所有题。两组题数、优化器与更新规则相同。此行为与原300题pilot通过batch20避免尾批的办法不同。筛选后不足一个有效batch时入口会拒绝训练。

```bash
source /media/data/gezhenjiang/venvs/cdi-train/bin/activate
set -o pipefail
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  src/raw_cds.py train --arm cds \
  2>&1 | tee runs/cds_qwen25_3b_raw_dev500_tok1024/logs/cds_train.log
```

CDS完成后，若要跑随机对照：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  src/raw_cds.py train --arm random \
  2>&1 | tee runs/cds_qwen25_3b_raw_dev500_tok1024/logs/random_train.log
```

两组都不加载任何初始化/已有CDS适配器，分别从同一个Raw检查点新建LoRA。训练完成保存检查点哈希、训练数据哈希、实际epoch及起点声明；评测会检查这些记录。旧训练入口不支持中断恢复，失败时保留输出和日志，不删除重跑。

## 6. 固定协议评测与比较

CDS训练完成后：

```bash
source /media/data/gezhenjiang/venvs/cdi-infer/bin/activate
CUDA_VISIBLE_DEVICES=0 python src/raw_cds.py infer --split test --arm cds
python src/raw_cds.py compare --arm cds
```

随机组训练完成后：

```bash
CUDA_VISIBLE_DEVICES=0 python src/raw_cds.py infer --split test --arm random
python src/raw_cds.py compare --arm random
```

比较脚本报告相对既有Raw基线的准确率差、错→对/对→错、配对bootstrap区间和McNemar检验。区间只反映test题目抽样，不覆盖训练/合成seed变异。当前先查看单次CDS流程效果，不把一次结果当作稳定方法收益；正式主实验还需共同预算下的多次验证。test不用来继续选择训练轮数、prompt或数据筛选阈值。
