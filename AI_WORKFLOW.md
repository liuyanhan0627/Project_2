# AI Workflow Reference

本文件用于让后续接手的 AI 明确当前实验工作流。优先阅读本文件，再阅读实验方案与代码。

## 1. 项目定位

当前实验实现名为 **ASPS (Asynchronous Speculative Path Switching)**。

- `ASPS/`: 当前实验代码目录，由 CNTP 官方源码改造而来。
- `CNTP_reference/`: CNTP 官方 GitHub 仓库的干净对照副本，来源为 `https://github.com/wyzjack/CNTP.git`，只用于本地 diff / 对照。
- `Llama大小模型协作推理实验方案.md`: 实验设计、组别定义、运行计划与结果分析的主文档。

不要再把当前实验代码称为 `CNTP/`。`CNTP` 只作为论文方法或上游参考源码出现。

## 2. 固定工作流

当前采用“本地开发，服务器跑实验，结果回传本地分析”的循环：

```text
本地完成代码、配置、脚本和分析准备
-> 本地 commit / push 到 GitHub main
-> 服务器 git pull / clone main
-> 服务器运行实验
-> 服务器将完整可分析结果导出到 experiments/result_exports/<bundle>/
-> 服务器把 result_exports 结果分支 commit / push 到 GitHub
-> 本地 fetch / checkout 结果分支
-> 本地读取 result_exports 并重新汇总 / 分析
-> 根据分析结果进入下一轮修改与调参
```

这个循环是当前默认流程。不要默认在本地跑大模型实验；本地主要负责代码修改、结果解析、报告和下一轮参数设计。

明确要求：

- 直接在本地做好各种准备工作，包括代码修改、配置文件、服务器运行脚本和结果分析脚本。
- 本地代码和实验数据通过 GitHub 上传到服务器，服务器 `clone/pull` 后应可直接运行。
- 服务器跑完实验后，不把原始 `outputs/`、`checkpoints/` 直接提交到 GitHub。
- 服务器使用导出脚本生成 `experiments/result_exports/<bundle>/`，再通过 GitHub 结果分支传回本地。
- 本地拉取结果分支后，在本地完成完整分析。

## 3. 本地开发约定

本地仓库路径：

```bash
/Users/liuyanhan/Documents/实验2
```

本地修改代码后，通常执行：

```bash
cd "/Users/liuyanhan/Documents/实验2"
git status
git add <changed_files>
git commit -m "<message>"
git push origin main
```

注意：

- 提交前检查 `git status`，不要误提交模型缓存、环境目录或未压缩的大量输出。
- 实验数据目录 `ASPS/experiments/SelfEval-Guided-Decoding/data/` 需要随代码提交，便于服务器直接运行。
- `outputs/`、`checkpoints/`、`server_logs/` 与 `experiments/registry.csv` 默认被 `.gitignore` 忽略。
- 需要通过 GitHub 回传的结果，只提交 `experiments/result_exports/<bundle>/`。
- `experiments/result_exports/` 已明确允许被 Git 跟踪，用于上传服务器跑完后的完整可分析结果。
- 不要提交 HuggingFace token、GitHub token 或服务器私密信息。

## 4. 服务器运行约定

服务器拉取代码后，系统化实验入口是仓库根目录：

```bash
/root/autodl-tmp/Project_2
```

当前 `train.py`、`configs/`、`scripts/` 都位于仓库根目录。旧的单脚本源码位置是：

```bash
/root/autodl-tmp/Project_2/ASPS/experiments/SelfEval-Guided-Decoding
```

该路径主要用于查看原始推理脚本、安装 custom transformers 或做底层调试，不再作为第一选择的运行入口。

旧路径：

```bash
/root/autodl-tmp/Project_2/CNTP/experiments/SelfEval-Guided-Decoding
```

已经废弃，不要继续使用。

服务器常用准备命令：

```bash
cd /root/autodl-tmp/Project_2
git pull origin main

source /root/miniconda3/etc/profile.d/conda.sh
conda activate gsm8k_strategyqa

export HF_TOKEN=...
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
```

如果 GitHub 直连很慢或超时，服务器可临时使用平台提供的 GitHub 镜像源，例如：

```bash
git remote set-url origin https://ghfast.top/https://github.com/liuyanhan0627/Project_2.git
git pull origin main
```

拉取成功后可以再按需要切回原始 GitHub 地址。

## 4.1 服务器环境激活与依赖检查

每次重新登录服务器后，先激活 conda 环境和 HuggingFace 缓存路径：

```bash
cd /root/autodl-tmp/Project_2

source /root/miniconda3/etc/profile.d/conda.sh
conda activate gsm8k_strategyqa

export HF_TOKEN=...
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
```

如果 `conda activate gsm8k_strategyqa` 提示找不到环境，先查看已有环境：

```bash
conda info --envs
```

如果环境显示在数据盘路径，可直接用完整路径激活：

```bash
conda activate /root/autodl-tmp/conda/envs/gsm8k_strategyqa
```

激活后检查关键依赖：

```bash
which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

注意：当前实验目录已从 `CNTP/` 改名为 `ASPS/`。服务器 `git pull` 后，旧的 editable install 可能仍指向旧路径。因此第一次切换到 `ASPS/` 后，需要重新安装本项目的 custom transformers：

```bash
cd /root/autodl-tmp/Project_2/ASPS/custom_transformers_packages/gsm8k_strategyqa
pip install -e .

cd /root/autodl-tmp/Project_2
```

实验数据默认应随代码存在于：

```text
ASPS/experiments/SelfEval-Guided-Decoding/data/
```

默认第一轮配置会查找：

```text
ASPS/experiments/SelfEval-Guided-Decoding/data/gsm8k_test.jsonl
ASPS/experiments/SelfEval-Guided-Decoding/data/strategyqa_test.jsonl
ASPS/experiments/SelfEval-Guided-Decoding/data/MATH/test.jsonl
ASPS/experiments/SelfEval-Guided-Decoding/data/truthfulqa/TruthfulQA.csv
```

服务器正式跑之前先检查：

```bash
ls ASPS/experiments/SelfEval-Guided-Decoding/data/gsm8k_test.jsonl
ls ASPS/experiments/SelfEval-Guided-Decoding/data/strategyqa_test.jsonl
ls ASPS/experiments/SelfEval-Guided-Decoding/data/MATH/test.jsonl
ls ASPS/experiments/SelfEval-Guided-Decoding/data/truthfulqa/TruthfulQA.csv
```

如果服务器使用了其他数据位置，可以再覆盖：

```bash
export DATAHOME=/path/to/SelfEval-Guided-Decoding/data
export MATH_FILE=/path/to/MATH/test.jsonl
export TRUTHFULQA_FILE=/path/to/truthfulqa/TruthfulQA.csv
```

## 5. 实验运行约定

当前推荐使用根目录系统化调参入口：

```bash
cd /root/autodl-tmp/Project_2
python3 train.py --config configs/smoke.yaml --dry-run
python3 train.py --config configs/smoke.yaml
```

长时间实验必须用 `nohup` 后台运行，避免本地电脑关机、断网或 SSH 终端关闭后服务器进程被中断。`smoke` 可以在前台验证；`first100`、完整 Group A/C/B、结果导出等长流程默认使用 `nohup`。

推荐启动方式。先在普通 SSH 终端里准备环境、设置缓存 / token / 数据路径：

```bash
cd /root/autodl-tmp/Project_2
source /root/miniconda3/etc/profile.d/conda.sh
conda activate gsm8k_strategyqa

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export DATAHOME=/root/autodl-tmp/Project_2/ASPS/experiments/SelfEval-Guided-Decoding/data
read -s HF_TOKEN
export HF_TOKEN

export EXPORT_NAME="$(date +%Y%m%d-%H%M%S)_all_first100"
```

如需丢弃中断过的半截结果，先清理旧输出：

```bash
rm -rf outputs server_logs experiments/registry.csv server_run.log
```

然后用 `nohup` 后台启动完整实验：

```bash
nohup bash scripts/run_group_ac_experiments.sh > server_run.log 2>&1 &
```

确认后台进程仍在：

```bash
ps -ef | grep run_group_ac_experiments | grep -v grep
ps -ef | grep "python3 train.py" | grep -v grep
```

查看日志：

```bash
tail -f server_run.log
```

查看日志时按 `Ctrl-C` 只会退出 `tail -f`，不会停止 `nohup` 后台实验。只要进程是用上面的 `nohup ... &` 启动的，之后可以关闭 SSH 或本地电脑。

如果需要停止后台实验，先查 PID，再手动 kill：

```bash
ps -ef | grep run_group_ac_experiments | grep -v grep
kill <PID>
```

Group A / Group C 第一轮调参配置位于：

```text
configs/group_ac/
```

包含：

- `baseline_groupb.yaml`: Baseline 与 Group B 共同参照。
- `a_safe.yaml`: Group A 稳健起点。
- `a_fast.yaml`: Group A 低延迟候选。
- `a_accuracy.yaml`: Group A 准确率候选。
- `c_low.yaml`: Group C 小长度权重。
- `c_mid.yaml`: Group C 中等长度权重。
- `c_fast.yaml`: Group C 低延迟候选。

服务器一键顺序运行：

```bash
bash scripts/run_group_ac_experiments.sh
```

可选控制：

```bash
DRY_RUN=1 bash scripts/run_group_ac_experiments.sh
RUN_SMOKE=0 bash scripts/run_group_ac_experiments.sh
RUN_CONFIGS="a_safe c_low c_mid" RUN_SMOKE=0 bash scripts/run_group_ac_experiments.sh
```

服务器检查是否在跑：

```bash
nvidia-smi
```

如果看到 `python` 进程占用 A800 显存和 GPU util，说明实验正在运行。

## 6. 结果回传约定

服务器实验完成后，不直接提交整个 `outputs/` 目录。先汇总与导出：

```bash
cd /root/autodl-tmp/Project_2
python3 scripts/collect_results.py --outputs outputs --registry experiments/registry.csv
python3 scripts/export_results_for_github.py \
  --outputs outputs \
  --registry experiments/registry.csv \
  --export-root experiments/result_exports
```

导出目录格式：

```text
experiments/result_exports/<bundle_name>/
```

该目录包含：

- 每次实验保存的 `config.yaml`
- `metadata.json`
- `jobs.json`
- `metrics.json`
- `best_metric.json`
- 训练 / 实验日志
- summary 文件
- JSONL generations
- 轻量 `checkpoints/experiment_state.json`
- `registry.csv`

该目录不包含模型权重类 checkpoint 文件，但保留轻量实验状态文件。然后把导出目录推到单独结果分支：

```bash
git switch -c results/group-ac-first100
git add experiments/result_exports/<bundle_name>
git commit -m "Add Group A/C first100 results"
git push origin results/group-ac-first100
```

如果同名结果分支已经存在，使用新的分支名，例如 `results/group-ac-first100-v2`。

不要 `git add outputs/ checkpoints/ server_logs/ experiments/registry.csv`。数据应在本地准备好后随主代码分支提交，不要放进结果分支重复提交。

## 7. 本地分析约定

本地拿结果：

```bash
cd "/Users/liuyanhan/Documents/实验2"
git fetch origin results/group-ac-first100
git switch -c analysis/group-ac-first100 origin/results/group-ac-first100
```

重新生成本地 registry：

```bash
python3 scripts/collect_results.py \
  --outputs experiments/result_exports/<bundle_name>/outputs \
  --registry experiments/result_exports/<bundle_name>/registry_local.csv
```

分析优先看：

- `experiments/result_exports/<bundle_name>/registry.csv`: 服务器端汇总。
- `experiments/result_exports/<bundle_name>/registry_local.csv`: 本地重新汇总。
- `experiments/result_exports/<bundle_name>/outputs/**/config.yaml`: 每个实验实际使用的配置。
- `experiments/result_exports/<bundle_name>/outputs/**/logs/*.log`: 训练 / 推理日志。
- `experiments/result_exports/<bundle_name>/outputs/**/results/**/*.jsonl`: 每个样本的完整输出与 `groupa_metrics` / `groupc_metrics` / `groupb_metrics`。

Group A 重点指标：

- `entropy_triggers`
- `draft_calls`
- `verify_calls`
- `small_draft_wins`
- `fallback_wins`
- `switches`
- `late_draft_drops`
- `wasted_big_tokens`
- `verify_modes`
- `accepted_tokens_per_verify`
- wall-clock time

Group C 重点指标：

- `length_weight_overrides`
- `candidate_base_scores`
- `candidate_length_weights`
- `candidate_weighted_scores`
- `accepted_tokens_per_verify`
- `switches`
- `late_draft_drops`
- wall-clock time compared with Group A

Group B 重点指标：

- `restart_rate`
- `first_solution_tokens`
- `reflection_tokens`
- `total_output_tokens`
- `hit_reflection_limit`
- wall-clock time

## 8. 下一轮修改原则

每一轮分析后，需要明确回答：

1. Baseline、Group A、Group C、Group B 的准确率、时间、token 消耗分别如何？
2. Group A 是否真的发生了在线切换？
3. Group A 慢在哪里：小模型 draft、verify 次数、late draft、fallback 浪费，还是大模型重算？
4. Group C 是否通过长度权重提高 `accepted_tokens_per_verify` 并降低延迟？是否牺牲准确率？
5. Group B 是否被反思 token 拖慢，是否真的发生有效 restart？
6. 下一轮要调哪些参数，并说明为什么。

优先做小规模 smoke / first100 验证，再决定是否扩大数据量或增加模型组合。
