# CDS预微调源码证据核查（2026-11-05）

结论：本地这份 `cds-04D1` 不仅有通用LoRA接口，还保存了**初始1000题微调的数据、日志、adapter，以及后续合成数据训练显式加载这个adapter的记录**。但尚未找到论文表10所写的292题版本的完整实现，不能将1000题产物说成292题实验。

本次检查包含被`.gitignore`默认排除的训练日志和checkpoint。此前只检查主要脚本的做法不够完整，这些新增证据纠正了“只有通用接口可供参考”的笼统描述。数值与路径核查记录在 `audit/cds_prefinetune_evidence.json`。

## 1. 有构造1000题初始训练集的代码和数据

`cds-04D1/work_on_math/data/traindata/merge.py:34` 实际执行：

```python
datas = readJson(".../work_on_math/data/indomain/gsm8k_dev_3792.json")[:1000]
saveJson(f".../work_on_math/data/traindata/dev_{len(datas)}.json", datas)
```

实际存在 `work_on_math/data/traindata/dev_1000.json`，共1000条。逐条比较完整JSON记录，确认它恰好等于当前 `gsm8k_dev_3792.json` 的前1000条。

这证明公开目录中的该脚本/数据版本使用1000题，不是 `3792-3500=292` 的划分。

## 2. 有初始微调的真实运行日志和adapter文件

`work_on_math/ckpt/qwen1.5-7b-chat-dev_1000/train.log:6` 记录的启动参数包含：

```text
--data_path_cit .../data/gsm8k/dev_1000.json
--model_path .../llm/qwen1.5-7b-chat
--max_epoches 1
--max_length 2048
--save_name qwen1.5-7b-chat-dev_1000
```

该次启动没有 `--lora_path`；日志第71行出现 `No LoRA path provided, training from scratch.`。结合 `finetune/code/finetune_lora.py:97`，这里是加载已有Qwen权重后**新建LoRA**，并非从随机权重训练整个语言模型。日志显示处理1000条IFT数据。

还存在相应checkpoint目录：

```text
work_on_math/ckpt/qwen1.5-7b-chat-dev_1000/
  qwen1.5-7b-chat-dev_1000_lora_epoch0/
    adapter_config.json
    adapter_model.safetensors
    ...
```

日志时间为2024-07-10；只能作为该目录版本的历史运行证据，不证明它与最终论文的所有实验设置一致。日志中的训练输入原始绝对路径和当前保存文件路径不同，不能验证历史文件当时的字节内容与当前文件完全一致。

## 3. 后续合成训练明确继续加载dev_1000 adapter

`work_on_math/ckpt/qwen1.5-7b-chat-cog_single_filter_2145/train.log:3` 的启动参数同时包含：

```text
--lora_path .../qwen1.5-7b-chat-dev_1000/qwen1.5-7b-chat-dev_1000_lora_epoch0
--data_path_cit .../data/traindata/cog_single_filter_2145.json
```

第86行记录 `LoRA Weights loaded.`。这与 `finetune/code/finetune_lora.py:102` 的可训练adapter加载分支一致。

同样的加载关系也出现在以下训练日志：

| 追加训练数据/实验目录 | 初始化adapter |
|---|---|
| cog_single_filter_2145 | dev_1000 |
| mathpool_2680 | dev_1000 |
| mathpool_2680_Q0.5 | dev_1000 |
| mathpool_2680_Q0.75 | dev_1000 |
| mathpool_2680_sigma1_2002 | dev_1000 |
| mathpool_2680_sigma2_2680 | dev_1000 |

这六次运行在2024-12月记录。它们直接支持“先有初始微调adapter，再在其上做合成数据微调”，而不是从原始Qwen每组重新创建LoRA。

## 4. 诊断入口有dev_1000线索，但证据链不完整

`2_synthesis/cognitive_math.py:72` 诊断部分读取的错题路径是：

```text
work_on_math/output/gsm8k/Llama3-8B-Instruct-dev_1000_0.5838/
  Llama3-8B-Instruct-dev_1000_W.json
```

这个名称指向 `dev_1000` 版本学生的作答记录，支持“初始微调后的模型用于诊断”的解释。但该具体文件在当前目录中不存在，并且它是Llama路径；上面完整的训练日志链是Qwen。不能把不同模型的材料拼接成已证实的同一次完整实验。

当前确实有 `Meta-Llama-3-8B-Instruct-dev_1000_lora_epoch0_0.6202` 的输出，但检查得它是5000条测试题作答，**不是**被上面路径引用的诊断错题集。`infer_math.py:116`支持加载LoRA作答，也不能单独证明某次诊断具体用了哪个checkpoint。

因此，已有资料对“初始微调→合成数据微调”有直接运行证据；对“292题初始微调→3500题诊断”的精确运行链路仍缺证据。

## 5. 292题版本还缺什么

论文表10脚注写额外292题进行pre-fine-tuning以便模型按指定格式回答。[论文表10](https://aclanthology.org/2025.findings-acl.439.pdf)

对当前目录递归检查、包含忽略文件，解析了456个JSON数组文件，没有发现292条或3500条的独立数组文件；Python/Shell脚本中也没有找到导出292题并训练的专门入口。这个范围内还不能恢复：

- 292题的题目身份及划分规则；
- 292题单独使用的超参数与checkpoint；
- 3500题诊断记录的完整生成命令与模型初始化；
- 该292题阶段与最终论文各组实验之间的完整关联。

这不证明作者没有执行论文所说的292题实验，只说明现有本地材料不能完整还原它。1000题版本可能是早期实验或未同步到最终稿的代码产物，但“早期版本”仅是根据数量/时间差异作出的解释，尚未得到作者确认。

## 6. Prompt(vanilla)是否已经初始微调

论文§4.3将Prompt定义为直接用提示词回答，没有在这一定义中逐项说明checkpoint是否已经接受格式微调。[论文表1及§4.3](https://aclanthology.org/2025.findings-acl.439.pdf)

新查到强对应证据：论文表1中Llama3-8B-Instruct的Prompt(vanilla)是GSM8K **62.02%**、GSM-Plus **42.50%**；当前目录中恰好有这两份结果：

```text
work_on_math/output/gsm8k/
  Meta-Llama-3-8B-Instruct-dev_1000_lora_epoch0_0.6202/
    Meta-Llama-3-8B-Instruct-dev_1000_lora_epoch0.json
work_on_math/output/gsmplus/
  Meta-Llama-3-8B-Instruct-dev_1000_lora_epoch0_0.4250/
    Meta-Llama-3-8B-Instruct-dev_1000_lora_epoch0.json
```

不仅是文件名相同：重新按源码数值抽取规则评分，分别得到 **3101/5000=62.02%** 和 **2125/5000=42.50%**；两份文件的题文、标准答案和顺序也分别与作者两个5000题评测文件完全一致。对应sha256、计数和对齐结果已加入JSON证据记录。

因此，现有产物**强烈支持Llama的Prompt(vanilla)行来自dev_1000初始微调版本**，而不是把vanilla一词直接解释成绝对未进行过任何数学微调的原始checkpoint。但是仅凭匹配的产物名称与准确率，还没有完全恢复当时所有加载命令；不能把它扩展成“两个模型均按292题协议预微调后报告vanilla”。论文Qwen的54.00%/33.92%对应基线产物尚未找到。

对自己的实验应显式报告两种基线：`Raw + Prompt`（原始模型直接解题），`Format-SFT + Prompt`（292题适配后的M0直接解题）。CDS和新方法都从M0开始，以M0的直接解题结果作为共同比较起点；同时保留Raw结果，单列292题训练带来的变化。Prompt描述回答方式，实验表格仍应额外说明模型接受过哪些训练。

## 对当前复现的影响

保留“先初始微调、再诊断与定向训练”的当前适配协议，有源码线索和历史训练记录作为依据，但必须明确292题规模来自论文，新的292题身份来自自己的固定划分，不能声称逐项重现作者完整执行流程。现阶段不因发现1000题产物就把自己的292改成1000，也不直接训练作者dev_1000数据，因为原始CDS划分与当前官方train/test协议不同。

当前已下载模型、已完成prepare，没有真实292题GPU训练结果。本次只更新核查文档与证据记录，没有更改冻结配置、数据或训练代码。
