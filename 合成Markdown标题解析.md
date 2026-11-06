# 兼容教师回答的Markdown题目标题

用户提供缓存 `479b9d74a9fd5633d3bba5b657b1c46dd26a461a88e9c28bb57307579b7939aa.json`：4096输出预算，实际输出701 token，包含3道完整题及答案，标题是 `#### Question 1:`，而原解析器仅识别 `**Question**:` / `**Question 1**:`，因此得到parsed 0。本次为格式不匹配，不能通过提高token预算或删除缓存解决。

新增 `teacher_sample_format.py` 辅助层，由 `teacher_env.py synthesize/augment` 自动启用。先调用原解析器；原格式成功时直接返回原结果。只有样本数量不匹配时，才尝试将行首Markdown标题（1至6个#，Question及连续编号）转换为原解析器的题目标题写法。

兼容分支要求：

* 必须恰好N题，编号为1到N，不能重复、缺号或多收题。
* 每题恰好一个 `**Answer**:` 标题及一对完整 `>>`、`<<` 答案边界。
* 题干、答案均非空，拒绝未闭合答案、代码围栏和模糊/多余结构。
* 对应缓存必须正常stop、请求hash正确、content与解析输入完全一致。

原始缓存不变，模型prompt、请求预算及采样参数不变。兼容层只修改送给原解析器的标题标记，正文仍按原规则提取；不生成新题，不接受少于N道的批次。样本ID与provenance仍按原请求生成，不伪造API结果。run内新增：

* `teacher_sample_parser_policy.json`：规则、配置hash和兼容代码hash。
* `teacher_sample_parsing/<请求hash>.<使用来源hash>.json`：实际使用的缓存hash、解析数和样本ID。允许增强阶段同一缓存用于不同draw，同时保留各次来源。

未来再有解析失败，错误中会直接包含request hash和stage，不必凭最后修改时间猜测缓存。

## 服务器执行

本地推送后，在服务器cdi-infer环境执行：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
git -C src pull --ff-only
python -m unittest discover -s src/tests -v
python src/teacher_env.py parse-cache \
  --request-key 479b9d74a9fd5633d3bba5b657b1c46dd26a461a88e9c28bb57307579b7939aa \
  --count 3
```

预期parsed_count=3、parser=markdown_question_headings_v1、preview_only=true，并列出3个题干。预检不读.env、不调用API、不写实验产物。

随后续跑：

```bash
set -o pipefail
python -u src/teacher_env.py synthesize \
  2>&1 | tee -a runs/cds_qwen25_3b_raw_kc18_dev500_tok1024/logs/synthesize_from_env.log
```

预期缓存解析处出现 `Parsed 3 samples using Markdown headings`，随后Fine synthesis 37/297。前36条进度可再次打印，但已完成的请求读取缓存；第37条本身也不需重新调用教师。后续尚未完成的请求会继续正常调用。

格式成功只表示结构完整，数学正确性与KC相关性仍需后续质量评分和检查。合成完成后运行 `python src/teacher_env.py quality --pool seed`。
