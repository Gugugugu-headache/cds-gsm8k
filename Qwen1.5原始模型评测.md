# Qwen1.5-7B-Chat Raw baseline

本入口只评测已下载的原始模型，不执行预微调、教师调用、数据合成或训练，也不冻结下一轮完整CDS实验的预算选择。

模型固定为 `Qwen/Qwen1.5-7B-Chat@5f4f5e69ac7f1d508f8369e977de208b4803444b`，下载目录需包含 `download_provenance.json`、索引、四个权重分片和分词器文件。

`qwen15_raw_eval.py prepare` 从已冻结的3B Raw/18KC运行复用所有数据分配，包括3500诊断、1319测试、500开发、3173 reserve、300初始化保留题和空warmup。每个成员、顺序、题干、标准答案、标签与提示词保持一致；不重新调用 `cds_pipeline.py prepare` 分配remaining。旧3B配置、结果和检查点保留。

新配置为 `src/configs/cds_gsm8k_qwen15_7b_raw_eval.json`，新目录为 `runs/qwen15_7b_raw_eval_kc18_dev500_tok1024`。`evaluation_protocol.json` 保存数据来源及内容哈希、模型版本与元数据哈希、权重分片大小和评测入口哈希。分片大小检查不是权重全文件校验；真实模型加载由推理阶段验证。

共用原有 `cds_pipeline.infer`：单卡、bf16、各模型自身聊天模板、同一数学指令、贪心temperature=0/top_p=1/top_k=1、最大输出1024，兼容数值评分器不变，adapter始终为None。输出含正确数、准确率、抽取方式、结束原因和token数；不是原论文512输出token/plain prompt下的数值复刻。

服务器中先同步代码，再执行：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
source /media/data/gezhenjiang/venvs/cdi-infer/bin/activate
python src/qwen15_raw_eval.py prepare
nvidia-smi
```

若0号卡可用，运行完整1319题：

```bash
mkdir -p runs/qwen15_7b_raw_eval_kc18_dev500_tok1024/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u src/qwen15_raw_eval.py infer --split test \
  2>&1 | tee runs/qwen15_7b_raw_eval_kc18_dev500_tok1024/logs/raw_test.log
```

若使用1号卡，修改CUDA_VISIBLE_DEVICES。日志应报告split=test、n=1319、adapter=null，结果文件为新run的 `eval/raw_test.json`。入口禁止覆盖已有评测tag。

需要单独的加载检查时，可选择诊断集的20题：

```bash
CUDA_VISIBLE_DEVICES=0 python src/qwen15_raw_eval.py infer --split diag --limit 20
```

该检查保存为独立 `eval/raw_smoke_diag_20.json`，不作为完整诊断集指标，也不替代1319题基线。入口同样支持固定500道开发题的Raw评测。

本地CPU测试只使用临时合成模型文件并拦截推理调用；本地未下载真实模型、未读取.env、未调用教师或运行GPU推理。实际准确率须由服务器运行后报告。
