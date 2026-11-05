# 只同步CDI代码目录

仓库地址： https://github.com/Gugugugu-headache/cds-gsm8k

Git仓库根目录是本地 `CDI（CDS复现）`，部署到已有服务器项目根目录下的 `src`。父目录的论文、GSM8K、DP-CDI、cds-04D1、模型、根目录.env和实验runs不纳入此仓库。当前代码仍依赖项目父目录里已部署的数据及原CDS文件子集；只clone代码到任意空目录不足以独立运行整个实验。

## 配置规则

版本控制只包含无密钥的 `configs/*.example.json` 模板。实际 `configs/*.json`（包括新增Qwen2-7B候选配置）和迁移备份被忽略，保留各机器的模型路径、实际HF commit与run设置。后续pull更新模板，不会自动替换实际配置；实验参数变更通过迁移脚本或另建配置与run进行。

模型权重、checkpoint、缓存、runs、dist和.env类文件被忽略。GitHub只用于代码与说明版本同步。公开仓库不包含原CDS源码或数据副本。

## 服务器首次接入已有项目

在未运行实验任务时执行，当前环境Python3.11即可。先确认服务器能连接GitHub，然后clone到现有src旁边：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
git clone https://github.com/Gugugugu-headache/cds-gsm8k.git src_git_checkout
```

clone成功后再执行：

```bash
python src_git_checkout/adopt_git_checkout.py
git -C src status --short
python -m unittest discover -s src/tests -v
```

接入脚本核对当前两份冻结代码、活动run配置及实现哈希，保留实际配置与JSON配置备份，将原src整个移到 `src.before_git_时间戳`，再将新checkout改名为src。原始数据、模型和runs不移动、不删除。如果冻结代码不一致，脚本拒绝接入，先核对差异；不使用强制覆盖。此接入只做一次，之后直接pull。

预期接入后Git工作区干净，实际配置虽然存在却不会被提交。旧src备份在项目根目录，不在新仓库内。无需重新prepare已经冻结的1024 run，也无需重新下载模型或重做Raw test。

## 日常更新

本地在这个子目录提交和推送：

```bash
cd '/home/gzj/桌面/dual-perspective-cd/CDI（CDS复现）'
git status --short
git add .
git commit -m "Describe the experiment code change"
git push origin main
```

服务器在没有运行中的实验任务时拉取：

```bash
cd '/media/data/gezhenjiang/CDI（CDS复现）'
git -C src pull --ff-only
```

服务器通常只拉取，不直接改受版本控制的代码。pull若报告本地代码修改，先查看差异，不用reset --hard清除。如果更新涉及cds_pipeline.py或train_lora.py，既有run的实现哈希会阻止继续混用，应明确迁移到新run；仅更新说明或新增独立工具不会改动这两个文件的字节。

记录每次实验的代码版本可执行 `git -C src rev-parse HEAD`，在已有run的日志里另存新文件，保留freeze与环境记录。

GitHub连接是否可用取决于服务器网络。已知服务器访问huggingface.co曾超时，不能据此判断GitHub也能或不能连接；以本次clone结果为准。公共仓库HTTPS拉取通常不需要token；本地推送需要GitHub写入认证。[GitHub官方说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
