# ASPS Experiment Workspace

这个仓库用于维护基于 CNTP 源码改造的大小模型协作推理实验。

## Directory Layout

- `ASPS/`: 当前实验代码。该目录从 CNTP 官方源码改造而来，用于实现 Asynchronous Speculative Path Switching (ASPS)。
- `CNTP_reference/`: CNTP 官方仓库的干净对照副本，来源为 `https://github.com/wyzjack/CNTP.git`。这个目录只用于本地 diff / 对照，不作为本实验代码入口。
- `Llama大小模型协作推理实验方案.md`: 当前实验方案、运行设置和结果记录入口。

## Server Path Update

旧运行路径：

```bash
/root/autodl-tmp/Project_2/CNTP/experiments/SelfEval-Guided-Decoding
```

新运行路径：

```bash
/root/autodl-tmp/Project_2/ASPS/experiments/SelfEval-Guided-Decoding
```

服务器 `git pull` 后，后续实验命令都应从 `ASPS/experiments/SelfEval-Guided-Decoding` 进入。

## NSCC Running

NSCC / PBS 服务器运行说明见 `docs/NSCC_RUNBOOK.md`。常用入口：

```bash
PROJECT=personal-e1547010 bash scripts/nscc_interactive.sh

PROJECT=personal-e1547010 \
WALLTIME=02:00:00 \
RUN_SCRIPT=scripts/run_group_ac_daytime_k2c_sweep.sh \
bash scripts/nscc_submit.sh
```

Group A/C 默认使用两张 GPU：大模型在 `cuda:0`，小模型在 `cuda:1`。如果只租一张卡，优先跑 baseline / Group B，或另建单卡配置。

## Systematic Experiments

现在可以用根目录的 `train.py` 统一调度实验：

```bash
python3 train.py --config configs/smoke.yaml --dry-run
python3 train.py --config configs/smoke.yaml
```

约定：

- 所有可调参数放在 `configs/*.yaml`。
- 每次运行自动创建独立的 `outputs/<timestamp>_<name>/`。
- 每个运行目录保存 `config.yaml`、`metadata.json`、`logs/`、`metrics.json`、`best_metric.json` 和 `checkpoints/experiment_state.json`。
- 结果汇总使用：

```bash
python3 scripts/collect_results.py --outputs outputs --registry experiments/registry.csv
```

实验数据放在 `ASPS/experiments/SelfEval-Guided-Decoding/data/` 并随代码提交，服务器 `git clone/pull` 后应可直接运行。`outputs/`、`checkpoints/` 和生成的 `experiments/registry.csv` 仍被 `.gitignore` 排除。

## Server Group A/C Tuning

第一轮 Group A/C 调参配置在 `configs/group_ac/`：

- `baseline_groupb.yaml`: Baseline 与 Group B 共同参照。
- `a_safe.yaml`: Group A 稳健起点。
- `a_fast.yaml`: Group A 低延迟候选。
- `a_accuracy.yaml`: Group A 准确率候选。
- `c_low.yaml`: Group C 小长度权重。
- `c_mid.yaml`: Group C 中等长度权重。
- `c_fast.yaml`: Group C 低延迟候选。

服务器上先设置环境变量：

```bash
export HF_TOKEN=...
export DATAHOME=/path/to/SelfEval-Guided-Decoding/data
export MATH_FILE=/path/to/MATH/test.jsonl
export TRUTHFULQA_FILE=/path/to/truthfulqa/TruthfulQA.csv
export BIG_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export SMALL_MODEL=meta-llama/Llama-3.2-1B-Instruct
```

然后顺序跑完整第一轮：

```bash
bash scripts/run_group_ac_experiments.sh
```

可选控制：

```bash
DRY_RUN=1 bash scripts/run_group_ac_experiments.sh
RUN_SMOKE=0 bash scripts/run_group_ac_experiments.sh
RUN_CONFIGS="a_safe c_low" bash scripts/run_group_ac_experiments.sh
```

脚本会自动生成 `experiments/registry.csv`，并把可通过 GitHub 传回本地分析的完整结果复制到 `experiments/result_exports/<bundle>/`。`experiments/result_exports/` 已明确允许被 Git 跟踪。该导出包含配置、日志、summary、metrics、轻量 experiment state 和 JSONL generations，不包含 outputs 原目录或模型权重类 checkpoint 文件。数据已随主代码分支同步，不需要在结果分支重复提交。

服务器跑完后，把导出的结果推到单独分支：

```bash
git switch -c results/group-ac-first100
git add experiments/result_exports/<bundle_name>
git commit -m "Add Group A/C first100 results"
git push origin results/group-ac-first100
```

本地拉回分析：

```bash
git fetch origin results/group-ac-first100
git switch results/group-ac-first100
python3 scripts/collect_results.py \
  --outputs experiments/result_exports/<bundle_name>/outputs \
  --registry experiments/result_exports/<bundle_name>/registry_local.csv
```
