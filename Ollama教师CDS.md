# Qwen1.5 Raw + 已部署的Ollama Q4教师

本入口只复现CDS baseline：全局Acc/Freq OR诊断、逐错题教师诊断与合成、改写/融合、质量过滤、CDSscore筛选、Raw起点LoRA训练及固定test评测。没有加入新的诊断策略。

用户已经报告Qwen1.5-7B-Chat的Raw test为815/1319、Raw diag为2556/3500（944道判错）。本入口复用这些完整回答，不重新推理学生，不把3B合成数据作为Qwen1.5的定向数据。

冻结入口为 `qwen15_cds_ollama.py`，实际操作使用新增入口 `qwen15_cds_ollama_samples.py`，同时接入既有repeat_last_n修复和下述标题格式兼容。run仍为 `runs/cds_qwen15_7b_raw_kc18_ollamaq4_q8_tok1024`，本机配置为 `src/configs/cds_gsm8k_qwen15_7b_ollama.json`。原3B、Qwen1.5 Raw评测目录及所有既有冻结代码保持原样。

## 教师与参数来源

* 使用服务器已经安装的 `qwen2.5:72b`，Q4_K_M、72.7B，用户报告短digest为424bad2cc13f。prepare通过元数据接口获取并冻结完整digest、实际Ollama服务端版本、模板/系统提示/参数/模型信息哈希。与在线API不混用缓存。
* 学生沿用同一3500诊断、1319测试、500开发、3173 reserve、300初始化保留题及空warmup，18个KC、Raw起点、聊天评测、1024输出token及兼容数值评分器。
* 全局OR规则保留，准确率0.5/频率0.05为继承的显式适配阈值，不称为恢复了作者未公开的阈值。
* **本轮质量分数门槛为8，依据论文表9。**原3B运行的10分门槛及结果保留；源码增强脚本的10分做法与论文存在差异。
* 全局每KC5题、细粒度每错题3题、int(0.5×种子池大小)次增强且每次3题，跟随已发布数学源码。按用户已报告画像为10个全局KC、944错题，初始预计50+2832=2882道题；增强请求数取决于种子质量过滤结果。
* 使用原数学提示词、原CDSscore代码方向/0.01平滑/max归一化/μ−σ筛选和原下游训练设置。细粒度未解析出1至6个合法KC时，原题标签回退仍在diagnostics/provenance中记录。

这是一份在统一数据划分、Raw起点、18KC及本地量化教师下的CDS方法重实现，不声称数值复刻原论文。

## 元数据检查和准备：不加载模型

在服务器同步代码后：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
source /media/data/gezhenjiang/venvs/cdi-infer/bin/activate
python src/qwen15_cds_ollama_samples.py check
python src/qwen15_cds_ollama_samples.py prepare
python src/qwen15_cds_ollama_samples.py profile
```

默认服务地址为 `http://127.0.0.1:11434`。若Ollama CLI实际连接其他已安排的服务器，可在check和首次prepare时用 `--base-url http://实际地址:端口`。地址需要为服务根地址，不添加/v1。

check只请求 `/api/version`、`/api/tags`、`/api/show`、`/api/ps`；prepare只查询模型元数据并读写本次运行的配置/数据文件；profile为纯CPU统计。这三条均不调用/api/chat、不预加载72B、不读取.env、不使用在线API密钥、不拉取模型、不更改共享服务。

prepare导出带generation_config_hash和reuse_provenance的诊断副本，记录新CDS配置绑定；逐条回答、原正确性、token数和结束原因保持不变。Raw test作为带文件哈希的外部基线引用，不伪装成新生成结果。

## 可用GPU时段内运行教师阶段

用户2026-11-09 15:11报告两张A6000分别99%/93%利用率，Ollama尚无模型驻留。约63GiB的剩余显存不代表计算卡空闲。教师原生API由已有Ollama服务决定GPU分布；客户端终端的CUDA_VISIBLE_DEVICES不能重新配置该共享服务。本接入不停止其他任务、不重启服务，也不自动卸载共享模型。

元数据检查和准备通过后，在已安排可用GPU时段运行：

```bash
mkdir -p runs/cds_qwen15_7b_raw_kc18_ollamaq4_q8_tok1024/logs
set -o pipefail
python -u src/qwen15_cds_ollama_samples.py synthesize \
  2>&1 | tee -a runs/cds_qwen15_7b_raw_kc18_ollamaq4_q8_tok1024/logs/synthesize.log
```

每次新生成请求会先打印请求hash和输出上限，随后显示Fine synthesis i/944。初始全局生成先处理10个KC。原模型未驻留时第一次请求可能包含加载时间；不预先宣称能达到特定吞吐率或一定全GPU加载。

每阶段正常完成后再执行下一条：

```bash
python -u src/qwen15_cds_ollama_samples.py quality --pool seed
python -u src/qwen15_cds_ollama_samples.py augment
python -u src/qwen15_cds_ollama_samples.py quality --pool augmented
python src/qwen15_cds_ollama_samples.py select
```

## 第453道错题的标题格式兼容

2026-11-10用户提供请求 `db02b7d9a0bc0712194b6c92b67c5835a72444c904093ec7e1697b4810a2b0bc` 的完整缓存回答：finish_reason=stop，输出1119/4096 token，3道题均有完整答案。暂停原因是标题写成 `**Question 1:**`（冒号在粗体内）且样本之间有独立的 `---`，旧解析器未支持该组合。

新增 `teacher_sample_bold_colon.py` 与入口 `qwen15_cds_ollama_samples.py`。先尝试已有解析链，仅在无法识别所需编号时启用新兼容：将标题转换为已有严格解析器接受的视图，只移除前一非空行为完整答案结束符、后一非空行为下一题标题的独立分隔线。题干和答案保持原样，原缓存不修改；仍要求编号恰为1至N、每题一个非空完整答案、无额外题目或无法识别的尾文。分隔线或类似标题若位于答案内则保留。

已冻结的原入口、repeat修复及旧解析辅助代码不修改；新增独立的 `teacher_sample_bold_colon_policy.json` 和 `teacher_sample_bold_colon_parsing/` 审计记录。后者记录响应/缓存/规范视图哈希、行号变动、样本ID及provenance哈希。

同步后可先离线检查这条服务器缓存，不调用教师、不写实验文件：

```bash
python src/qwen15_cds_ollama_samples.py parse-cache \
  --request-key db02b7d9a0bc0712194b6c92b67c5835a72444c904093ec7e1697b4810a2b0bc \
  --count 3
```

预期parsed_count=3、parser=bold_question_inner_colon_v1、preview_only=true。随后重新执行上面的synthesize命令，不需要重新prepare/profile或删除缓存。程序会重放进度并复用此前成功请求；这条已有的第453题回答也可直接解析，无需再次生成。整阶段完成后才写出synthesis.json和diagnostics.json。

针对用户原文及边界条件新增9项离线测试，全套137项通过。解析成功只确认格式和完整性；题意歧义、数学正确性、多问答案仍交给后续质量环节，解析器不替教师改题。

## 第517道错题：无冒号Markdown标题兼容

2026-11-11用户报告前516道错题完成后，请求 `ee49e6ef26489ba678f7512b078246aedc62ac895396d606399b61c9d7be68e1` 暂停。该回答正常stop，输出982/4096 token，含3道完整概率题；标题为独立一行的 `#### Question 1`、`#### Question 2`、`#### Question 3`，均没有冒号。

新增 `teacher_sample_optional_colon.py`；继续使用同一个 `qwen15_cds_ollama_samples.py` 入口，只更新其解析模块引用。所有既有解析辅助代码保持原字节和哈希，以便重放旧缓存及其解析策略记录。新增策略文件为 `teacher_sample_optional_colon_policy.json`，审计目录为 `teacher_sample_optional_colon_parsing/`。

新兼容只识别答案块外、独立成行、无尾随题文的Markdown编号标题。仅在原解析链未识别出所需编号时补充标题冒号，并继续使用既有严格完整性校验；不补题、不重编号、不重写题干或答案。题数、编号顺序、非空答案、边界和尾文规则保留。标题或分隔线若位于答案中则不改动。

同步后先在服务器离线预览：

```bash
python src/qwen15_cds_ollama_samples.py parse-cache \
  --request-key ee49e6ef26489ba678f7512b078246aedc62ac895396d606399b61c9d7be68e1 \
  --count 3
```

预期parsed_count=3、parser=markdown_question_missing_colon_v1、preview_only=true。预览不调用教师、不写实验文件。随后按上文原命令恢复synthesize；无需重新prepare/profile、清除缓存或重新生成这条回答。前516道及第517道已有响应继续复用。

本次新增9项离线测试，全套146项通过，包括用户74行完整回答的逐题逐答案核对、旧格式兼容、缓存和旧策略原字节保留、拒绝缺题及不完整答案、离线预览与冻结流程加载。解析包装标题与训练答案正文分离；答案正文仍保留教师原文，未自动补成 `#### 数值`，格式兼容不等于数学质量验证。

## repeat_last_n兼容修复

用户实际服务0.40.8在首个请求 `1a63b8f78eebcbfa9ff2ce3735701d7e464464800283136f76320ef32fdcacaa` 返回HTTP 400，明确拒绝repeat_last_n=-1，要求0至2147483647。该请求的其他参数、完整提示词和seed=2103518819可以在本地准确重建。

新增 `qwen15_cds_ollama_compat.py` 在运行时把repeat_last_n设置为num_ctx的明确正整数32768，保持原来的上下文范围意图。该兼容操作不修改 `ollama_teacher.py`、原入口或已冻结JSON，不需要重新prepare/profile。后续所有教师阶段统一用兼容入口。

原raw_protocol里的-1保留为历史原始配置；`teacher_repeat_scope_policy.json` 单独记录原值、实际值、配置及新旧代码哈希。实际native请求始终写入原响应缓存，另在 `teacher_repeat_scope_requests/` 记录每个完整/length响应的缓存哈希和实际重复窗口。旧的不同native请求缓存会被拒绝，不被覆盖；用户当前首请求即失败，未报告任何成功教师缓存。

兼容层还会把HTTP错误正文打印出来，并在 `teacher_http_errors/` 按内容保存错误与实际请求。提示词、温度/top_p/top_k/min_p、seed、repeat_penalty、上下文和输出上限、质量门槛均保持原样。首个请求的离线回归测试验证只有repeat_last_n字段变化；这不宣称不同推理框架的重复惩罚数值等价。

## 传输与缓存

使用原生 `/api/chat`、stream=false，显式设置num_ctx=32768、num_predict为请求输出上限、temperature=0.5/top_p=0.8/top_k=0（生成），评分temperature=0/top_p=1/top_k=1、repeat_penalty=1.05、min_p=0。兼容入口实际发送repeat_last_n=32768；原入口保留的-1不能通过用户服务的参数检查。seed由实验seed和请求稳定派生。保存实际native_request、完整native_response、token计数和时间信息；不把不同推理引擎的重复惩罚实现称为数值等价。

原有重试辅助代码仍负责诊断1024→2048→4096、合成4096→8192的length重试；评分512输出token不变。每次尝试分别保留，未正常完成的输出不会作为完整样本接受。已有保守格式解析辅助代码继续使用。

缓存request字段为兼容既有重试/解析审计的规范化CDS请求，url为Ollama服务根地址，native_endpoint明确保存实际/api/chat地址。缓存key按既有规则对url+规范化请求取hash；实际native请求、完整模型digest及配置hash也须匹配才能复用，因此不能跨教师/配置混用。本地不读取代理或云端鉴权环境变量。

teacher使用keep_alive=5m。教师阶段完成后，可用 `ollama ps` 查看驻留情况；学生训练需安排足够可用显存。不要把自动卸载、其他用户服务配置或模型停止操作混入本实验脚本。

## 学生训练和固定评测

数据筛选完成并核对样本量后，使用已验证的训练环境和可用GPU分别运行两组：

```bash
source /media/data/gezhenjiang/venvs/cdi-train/bin/activate
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  src/qwen15_cds_ollama_samples.py train --arm cds
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  src/qwen15_cds_ollama_samples.py train --arm random
```

保留实际epoch、学习率轨迹和监督token数。两组各从Raw新建LoRA，随机组不加载CDS适配器。训练完成后：

```bash
source /media/data/gezhenjiang/venvs/cdi-infer/bin/activate
CUDA_VISIBLE_DEVICES=0 python src/qwen15_cds_ollama_samples.py infer --split test --arm cds
python src/qwen15_cds_ollama_samples.py compare --arm cds
CUDA_VISIBLE_DEVICES=0 python src/qwen15_cds_ollama_samples.py infer --split test --arm random
python src/qwen15_cds_ollama_samples.py compare --arm random
```

这些命令只是完整运行入口；本地验证未运行真实教师、GPU训练或推理。

参考：[Ollama原生chat](https://docs.ollama.com/api/chat)、[模型元数据](https://docs.ollama.com/api-reference/show-model-details)、[运行参数](https://docs.ollama.com/modelfile)、[CDS论文](https://aclanthology.org/2025.findings-acl.439.pdf)。
