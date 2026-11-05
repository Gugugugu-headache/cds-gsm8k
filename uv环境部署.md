# 用 uv 建立主实验共用环境

**GPU环境检查已完成：PyTorch2.6.0+cu124、CUDA可用、2张GPU；NumPy未安装，nvcc不在PATH。** 当前继续操作见 [无nvcc训练环境.md](无nvcc训练环境.md)，采用原生Torch AdamW后端及独立torchadam运行目录，300题数据保持一致。

**当前数据协议已切换为300题预微调，余下3673题不额外分开发集。** 你已完成的模型下载、uv环境及旧prepare可保留；按 [300题预微调与论文表述.md](300题预微调与论文表述.md) 的补丁步骤生成新的pre300运行目录。本文件后面旧292/500/3181等是此前步骤记录。

已确认的服务器进度：uv实际安装在 `/home/gezhenjiang/.local/bin`，`venvs/cdi-train`已创建并激活，Python为3.11.18。新终端若找不到uv，执行 `export PATH="/home/gezhenjiang/.local/bin:$PATH"`。接下来进入第3步下载模型；执行prepare前先同步 [292题预微调说明](292题预微调.md) 中的训练入口修正。

当前采用 uv。它可管理独立Python和虚拟环境，后续CDS基线、改进方法、消融实验复用同一套训练环境和同一套推理环境。项目代码和数据不需要因环境管理器变化而重新上传。

新环境使用 `/media/data/gezhenjiang/venvs/cdi-train`，推理环境使用 `/media/data/gezhenjiang/venvs/cdi-infer`。此前conda创建失败的 `/media/data/gezhenjiang/envs/cdi-train` 留在原处即可；这里另用新路径。

## 1. 检查或安装 uv

在服务器终端执行：

```bash
command -v uv
uv --version
```

若服务器找不到uv，使用官方安装器安装到本人目录：

```bash
mkdir -p /media/data/gezhenjiang/tools/bin
curl -LsSf https://astral.sh/uv/install.sh -o /media/data/gezhenjiang/tools/uv-install.sh
UV_INSTALL_DIR=/media/data/gezhenjiang/tools/bin UV_NO_MODIFY_PATH=1 sh /media/data/gezhenjiang/tools/uv-install.sh
export PATH="/media/data/gezhenjiang/tools/bin:$PATH"
uv --version
```

上述安装器不需要sudo，`UV_NO_MODIFY_PATH=1`使它不自动改shell启动文件。新终端使用自装uv时需再次设置PATH，或用 `/media/data/gezhenjiang/tools/bin/uv` 绝对路径。[uv官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)、[安装路径变量](https://docs.astral.sh/uv/reference/environment/)

## 2. 创建 Python 3.11 训练环境

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
export UV_CACHE_DIR=/media/data/gezhenjiang/uv_cache
export UV_PYTHON_INSTALL_DIR=/media/data/gezhenjiang/uv_python
uv venv --python 3.11 --managed-python --seed /media/data/gezhenjiang/venvs/cdi-train
source /media/data/gezhenjiang/venvs/cdi-train/bin/activate
python -c "import sys; print(sys.executable); print(sys.version)"
```

`--managed-python`只使用uv管理的Python；若没有，uv会下载对应解释器。`--seed`为环境安装pip，确保已有说明里的 `python -m pip` 命令也可使用。缓存和解释器都放在本人数据盘目录。[uv环境说明](https://docs.astral.sh/uv/pip/environments/)、[Python管理说明](https://docs.astral.sh/uv/guides/install-python/)

预期解释器路径：`/media/data/gezhenjiang/venvs/cdi-train/bin/python`，版本为3.11.x。

## 3. 下载模型与校验数据

```bash
uv pip install --python /media/data/gezhenjiang/venvs/cdi-train/bin/python \
  huggingface_hub -r src/requirements-cpu.txt
export HF_HOME=/media/data/gezhenjiang/model_list/.hf_cache
python src/download_student.py
python -m unittest discover -s src/tests -v
python src/cds_pipeline.py prepare
```

先完成模型下载并记录revision，再执行prepare冻结配置。模型仍放在 `model_list/Qwen2.5-3B-Instruct`，与虚拟环境分开。目标数据规模是diag3500、test1319、warmup292、dev500、reserve3181。

### 服务器连接 Hugging Face 超时（已遇到）

用户已确认HF_TOKEN在Python进程中非空，但 `hf auth whoami` 超时，`curl -I --connect-timeout 10 --max-time 20 https://huggingface.co` 报连接443端口超时。说明当前路径上的HTTPS连接尚未建立，不能据此判断token有效或无效。重新输入token不能修复网络连通性。

可使用实验室提供的可用代理访问官方站，或先检查第三方HF-Mirror能否连接：

```bash
curl -I --connect-timeout 10 --max-time 20 https://hf-mirror.com
```

若返回HTTP响应且站点可用，可为本次公开Qwen模型下载临时切换端点。下面整行作为一条命令执行：

```bash
env HF_ENDPOINT=https://hf-mirror.com HF_TOKEN= HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 python src/download_student.py
```

这只修改该下载进程的环境。HF-Mirror是第三方站点；此公开模型的下载禁用token自动附带，并将本进程HF_TOKEN置空，避免发送用户Hugging Face凭据。`HF_HUB_DISABLE_XET=1`采用常规HTTP传输，避免镜像的Xet认证/重定向兼容问题；没有降低HTTPS证书校验。镜像连通不保证其所有大文件地址可达，若下载阶段仍超时，应保留报错中不含凭据的主机/状态信息继续排查。[镜像站使用说明](https://hf-mirror.com/)、[HF环境变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

下载脚本仍会记录实际取得的HF commit并写回配置。若镜像也无法访问，可在能连接官方HF的本人电脑上下载同一repo的固定revision，再把完整模型目录与provenance记录传到服务器；需要同步配置中的revision后再prepare。不要只拷贝权重而遗漏tokenizer和config。

## 4. 后续训练依赖

已激活此环境时，可以用uv安装训练依赖；以下是服务器驱动580.142下的初始候选组合，尚需实际训练验证：

```bash
uv pip install --python /media/data/gezhenjiang/venvs/cdi-train/bin/python \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python /media/data/gezhenjiang/venvs/cdi-train/bin/python \
  -r src/requirements-train.txt
uv pip check --python /media/data/gezhenjiang/venvs/cdi-train/bin/python
```

vLLM后续安装在单独的 `venvs/cdi-infer` 中。两个环境分别验证后固定版本，所有方法对应阶段使用同一环境。环境管理器本身不改变数学数据或CDS算法；比较时仍要固定模型、软件版本、提示词、预算与随机种子方案。

## 5. 保存可复用的环境记录

首次小规模训练/推理跑通后再记录完整依赖；以下在项目根目录执行，推理环境相应使用infer文件名：

```bash
mkdir -p environment_records
uv pip freeze --python /media/data/gezhenjiang/venvs/cdi-train/bin/python > environment_records/train.requirements.lock.txt
uv --version > environment_records/uv.version.txt
/media/data/gezhenjiang/venvs/cdi-train/bin/python --version > environment_records/train.python.version.txt
nvidia-smi > environment_records/nvidia-smi.txt
```

freeze是已安装Python包版本清单，不包含Python解释器、驱动以及软件包索引来源，故单独记录Python/驱动版本，重建PyTorch时保留cu124安装源。它也不等同于带制品哈希的 `uv.lock`。正式比较期间固定这些记录；核心依赖变化时重新验证并重跑比较基线。

以后打开新终端，运行：

```bash
export UV_CACHE_DIR=/media/data/gezhenjiang/uv_cache
export UV_PYTHON_INSTALL_DIR=/media/data/gezhenjiang/uv_python
source /media/data/gezhenjiang/venvs/cdi-train/bin/activate
cd '/media/data/gezhenjiang/CDI（CDS复现）'
```

若uv装在自定义tools目录，还需设置第1步的PATH。当前两张A6000有正在满负载运行的任务，可以先完成安装和下载，确认有空闲卡再开始训练/推理。
