# 当前学生解题预算：1024新token

2026-11-05，用户在完整Raw test开始前确认，将学生解题的最大生成长度由512改为1024。原因是512下的诊断题小样本出现截断，旨在减少生成长度限制对诊断的干扰；不是根据test准确率选择预算，不声称1024最优或能消除理解错误。

CDS论文表9的Model Evaluation是512；1024属于本项目统一实验条件下的适配。目标模型的Raw、M0、CDS及同类方法的普通解题诊断/最终评测均使用1024。为认知诊断器采集参考模型的普通解题响应时也采用这一预设上限和相同任务指令，使用各模型适用的聊天模板。参考模型无需统一进行300题预微调；少数目标模型各自的CDS与主方法从相同M0出发。

当前只改变 `inference_max_tokens` 与 `run_dir`，不改变prompt、评分器、教师各阶段生成上限、训练max_length=2048、预微调300题、seed=42或数据划分。教师诊断/合成/质量打分预算分别由已有实现控制，并不因这次学生解题预算修改而一起变成1024。

| 项目 | 512历史记录 | 1024当前协议 |
|---|---|---|
| run_dir | runs/cds_qwen25_3b_pre300_torchadam | runs/cds_qwen25_3b_pre300_torchadam_tok1024 |
| 最大生成新token | 512 | 1024 |
| Raw小样本 | 已完成20道诊断题 | 更新后重新采集相同20题 |
| Raw完整test | 用户尚未报告执行 | 检查小样本后采集1319题 |

512小样本的用户报告：宽松评分20/20，hash抽取14条，last_number回退6条，截断1条。截断题错误引入箱重W，生成6W+16后尚未完成作答，评分器从中取16而判对。因此20/20不是已确认的逐题完成解题率。1024仍保留该CDS兼容评分，同时需报告hash抽取匹配比例、回退比例及截断比例；这些不能冒充人工数学推理正确率或GSM8K官方严格EM。增加预算不修复评分回退本身。

## 服务器更新

将本地 `dist/cds_inference1024_update.tar.gz` 与同名 `.sha256` 文件上传服务器项目根目录。在已有cdi-infer环境执行：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
source /media/data/gezhenjiang/venvs/cdi-infer/bin/activate
sha256sum -c cds_inference1024_update.tar.gz.sha256
tar -xzf cds_inference1024_update.tar.gz
python src/configure_inference1024.py
python -m unittest discover -s src/tests -v
python src/cds_pipeline.py prepare
```

增量包不包含默认配置、cds_pipeline.py或train_lora.py，不会替换服务器模型路径、HF commit或已有冻结实现。迁移脚本读取服务器当前配置，备份后仅修改两个字段；若目标run存在冲突冻结配置，会拒绝覆盖。不要重新运行旧的pre300/torchadam迁移脚本，它们对应此前目录。

prepare只做CPU校验与数据导出，不下载权重、不调用教师、不使用GPU。预期25项测试通过；prepare仍输出：

```json
{"diag": 3500, "test": 1319, "warmup": 300, "dev": 0, "reserve": 3673}
```

同一seed与原数据使新旧run的题目身份、顺序和prompt保持一致。旧512 run保留，不复制其中的eval或checkpoint到新run。新结果通过同目录config.frozen.json与meta.config_hash追溯1024预算。

## 先检查相同20道诊断题

确认可用GPU，下面假设0号卡：

```bash
CUDA_VISIBLE_DEVICES=0 python src/cds_pipeline.py infer \
  --split diag --tag raw_smoke_diag --limit 20
```

在新目录中使用同名tag不会覆盖旧512记录。查看统计和截断输出：

```bash
python - <<'PY'
import json
from pathlib import Path
from collections import Counter
cfg = json.loads(Path('src/configs/cds_gsm8k.json').read_text())
result = json.loads((Path(cfg['run_dir']) / 'eval/raw_smoke_diag.json').read_text())
print('最大生成token:', cfg['inference_max_tokens'])
print('运行目录:', cfg['run_dir'])
print(result['meta'])
print('抽取方式:', dict(Counter(r['extraction'] for r in result['records'])))
for i, row in enumerate(result['records'], 1):
    if i == 5 or row['finish_reason'] == 'length':
        print('\n记录:', i, '结束原因:', row['finish_reason'])
        print('抽取方式:', row['extraction'], '预测值:', row['predicted'], '标准值:', row['gold'])
        print('回答:', row['response'])
PY
```

20题用于检查明显截断、格式与运行情况，不据此证明1024最优，也不把它作为泛化成绩。达到长度上限不一定表示最终答案未生成，应查看原文。

## Raw完整test

基础设施检查通过后，保持1024设置采集，不传adapter、不加limit：

```bash
mkdir -p runs/cds_qwen25_3b_pre300_torchadam_tok1024/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python src/cds_pipeline.py infer --split test --tag raw_test \
  2>&1 | tee runs/cds_qwen25_3b_pre300_torchadam_tok1024/logs/raw_test.log
```

结果为 `runs/cds_qwen25_3b_pre300_torchadam_tok1024/eval/raw_test.json`。应为n=1319、adapter=null；accuracy是0到1的小数。后续评测继续使用同一预算，不能根据test成绩选择长度或prompt。

论文表述示例：

> 为统一各方法的推理条件，我们将学生模型解题时的最大生成长度设为1024 token，采用相同的解码参数和评分规则。该设置在正式测试前确定，与原CDS论文的512 token上限不同，因此本文报告统一实验协议下的CDS适配复现结果。

1024增加允许的最大生成长度；模型提前结束时不会强制生成满1024。真实运行时间与显存占用需以服务器结果为准。
