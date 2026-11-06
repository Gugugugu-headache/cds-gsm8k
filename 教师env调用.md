# 从服务器项目根目录.env调用教师

用户已将自己的.env复制到 `/media/data/gezhenjiang/CDI（CDS复现）/.env`。新入口显式读取该文件，默认位置由脚本所在项目目录确定，不自动搜索其他目录。使用python-dotenv解析，不把.env当shell脚本执行。

仅加载冻结教师配置所指定的两个变量：`Siliconflow_API_KEY`、`Siliconflow_BASE_URL`。文件中的教师密钥优先于旧终端变量；API地址未填写时使用原默认地址。文件中的其他服务密钥不导入进程环境，不打印、不记录.env内容。

该入口使用curl HTTP/1.1完成模型列表检查和教师API传输，密钥、请求体由标准输入传递，不进入curl命令行参数。响应错误会遮盖密钥。使用curl是针对此前Python请求返回通用400、curl能返回明确401的诊断结果，并不保证解决无效密钥；真实状态必须检查。

原冻结脚本、18维数据与配置不改，正常请求仍使用原模型、采样参数、token上限和缓存键。若synthesize中的1024-token教师诊断被length截断，则按新增固定策略依次尝试2048、4096，记录所有尝试和成本；详情见 [教师诊断截断恢复.md](教师诊断截断恢复.md)。生成阶段在run内追加 `teacher_transport.jsonl` 记录入口哈希和传输方式，不包含密钥或.env内容。`check`仅GET模型列表，不调用生成接口；其他阶段显式执行。

## 服务器命令

代码推送后，在已有cdi-infer环境执行：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
git -C src pull --ff-only
uv pip install --python /media/data/gezhenjiang/venvs/cdi-infer/bin/python 'python-dotenv>=1.0,<2'
chmod 600 .env
python src/teacher_env.py check
```

预期输出包括“模型列表 HTTP: 200”“配置模型出现在列表中: True”。不会输出密钥。如果仍401，表示文件中读取的密钥也未被接口接受；复制文件本身不能使无效密钥生效。如果配置模型未出现，程序停止，不自行更换教师。

检查成功后，启动原18 KC合成流程：

```bash
set -o pipefail
python src/teacher_env.py synthesize \
  2>&1 | tee runs/cds_qwen25_3b_raw_kc18_dev500_tok1024/logs/synthesize_from_env.log
```

每次新进程都自动读取指定.env，无需手动export；加载只影响本进程，不会给当前交互式终端设置变量。后续教师阶段同样使用此入口：

```bash
python src/teacher_env.py quality --pool seed
python src/teacher_env.py augment
python src/teacher_env.py quality --pool augmented
```

以上每个生成/评分阶段会产生实际API用量，依次成功后再运行下一条。原缓存仍可复用。筛选、训练和评测继续使用 `raw_cds18.py`，它们不需要教师密钥。

需要指定其他位置时：

```bash
python src/teacher_env.py --env-file /absolute/path/to/teacher.env check
```

无需再次prepare、profile或采集3500题。不要将.env提交到Git；仓库已忽略.env类文件。项目根目录.env位于服务器src仓库之外。
