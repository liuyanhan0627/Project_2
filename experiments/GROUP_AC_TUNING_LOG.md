# Group A/C Tuning Log

Last updated: 2026-06-12

本文件专门记录 Group A 和 Group C 的调参过程。后续每一轮服务器实验、结果回传、本地分析和下一轮参数选择，都继续追加到这里。

## Planned Run 20260612: Group A/C Nightly Sweep + Group D Paper Parameters

### Goal

本轮方案已确认。调参主线保持不变：

- 小模型继续使用 `meta-llama/Llama-3.2-1B-Instruct`
- 不切换到 0.5B 小模型
- 调参阶段不对小模型所在 A100 做限速
- Baseline 和 Group B 本轮不重跑，继续使用已有对照结果
- 数据面继续使用 GSM8K/StrategyQA/MATH/TruthfulQA first100，加 RULER/NIAH small first20

本轮包含两部分：

1. 重新跑 Group D/CNTP，使用论文推荐参数，修正前一轮 `temperature=0.0, top_p=1.0` 与论文设置不一致的问题。
2. 继续跑 10 组 Group A/C 参数，其中 A 6 组围绕当前最好区域细调，C 4 组保持 `k=2` 并继续探索小模型多路径采纳策略。

### Group D Paper-Parameter Rerun

Group D 使用 CNTP/Cautious Next Token Prediction 论文口径：

| Dataset | Config | Temperature | Top-p | Hmin | Hmax | Max Trials | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| GSM8K | `d_paper_gsm8k_first100` | 1.2 | 0.9 | 0.01 | 1.5 | 10 | 论文 Llama CNTP GSM8K 温度 |
| StrategyQA | `d_paper_strategyqa_first100` | 0.8 | 0.9 | 0.01 | 1.5 | 10 | 论文 Llama CNTP StrategyQA 温度 |
| MATH | `d_paper_math_first100` | 0.6 | 0.9 | 0.01 | 1.5 | 10 | 论文 Llama CNTP MATH 温度 |
| TruthfulQA | `d_paper_truthfulqa_first100` | 0.8 | 0.9 | 0.01 | 1.5 | 10 | 论文 TruthfulQA CNTP 温度 |
| RULER/NIAH | `d_paper_ruler_first20` | 0.8 | 0.9 | 0.01 | 1.5 | 10 | 补充 sanity check，论文未报告 RULER |

拆成单数据集 config 的原因：当前 `train.py` 会把 `groups.group_d.args.temperature` 作为 group 级参数应用到整个 config；若放在同一个 config 里，无法按 dataset 切换论文温度。

### Group A/C Configs To Run

| Config | Group | k | Entropy | Draft | Fallback | Alpha | Margin | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `a_ruler_k2_h158_d20_margin001` | A | 2 | 1.58 | 20 | 32 | - | 0.001 | 在当前最优 A 附近稍降阈值，看能否提升准确率 |
| `a_ruler_k2_h162_d20_margin001` | A | 2 | 1.62 | 20 | 32 | - | 0.001 | 在当前最优 A 附近稍升阈值，测试更保守触发 |
| `a_ruler_k2_h16_d18_margin001` | A | 2 | 1.60 | 18 | 32 | - | 0.001 | 保持阈值，缩短 draft，主攻延迟 |
| `a_ruler_k2_h158_d18_margin0015` | A | 2 | 1.58 | 18 | 32 | - | 0.0015 | 降阈值加短 draft，同时略增 margin 抑制误切 |
| `a_ruler_k2_h155_d18_margin002` | A | 2 | 1.55 | 18 | 32 | - | 0.002 | 更积极触发但更严格 switch，验证准确率收益 |
| `a_ruler_k2_h16_d20_f28_margin001` | A | 2 | 1.60 | 20 | 28 | - | 0.001 | 减小 fallback window，测试延迟恢复空间 |
| `c_ruler_k2_h145_a003_d14_margin005` | C | 2 | 1.45 | 14 | 32 | 0.030 | 0.005 | C 主线缩短 draft，保留原 alpha |
| `c_ruler_k2_h145_a0035_d14_margin005` | C | 2 | 1.45 | 14 | 32 | 0.035 | 0.005 | 增强长度偏置，看是否改善多路径采纳 |
| `c_ruler_k2_h1425_a003_d14_margin005` | C | 2 | 1.425 | 14 | 32 | 0.030 | 0.005 | 更积极触发，观察是否提升准确率 |
| `c_ruler_k2_h145_a0025_d12_margin005` | C | 2 | 1.45 | 12 | 32 | 0.025 | 0.005 | 更短 draft 和更弱长度偏置，主攻延迟 |

### Local Preparation

本地已准备：

```text
scripts/run_group_ac_paperd_nightly_sweep.sh
configs/group_ac/a_ruler_k2_h158_d20_margin001.yaml
configs/group_ac/a_ruler_k2_h162_d20_margin001.yaml
configs/group_ac/a_ruler_k2_h16_d18_margin001.yaml
configs/group_ac/a_ruler_k2_h158_d18_margin0015.yaml
configs/group_ac/a_ruler_k2_h155_d18_margin002.yaml
configs/group_ac/a_ruler_k2_h16_d20_f28_margin001.yaml
configs/group_ac/c_ruler_k2_h145_a003_d14_margin005.yaml
configs/group_ac/c_ruler_k2_h145_a0035_d14_margin005.yaml
configs/group_ac/c_ruler_k2_h1425_a003_d14_margin005.yaml
configs/group_ac/c_ruler_k2_h145_a0025_d12_margin005.yaml
configs/group_ac/d_paper_gsm8k_first100.yaml
configs/group_ac/d_paper_strategyqa_first100.yaml
configs/group_ac/d_paper_math_first100.yaml
configs/group_ac/d_paper_truthfulqa_first100.yaml
configs/group_ac/d_paper_ruler_first20.yaml
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_paperd_nightly_sweep.sh > server_run_paperd_nightly.log 2>&1 &
```

### Decision Rules

1. A/C 主指标继续看 GSM8K、StrategyQA、MATH 的自动 accuracy 与 latency；TruthfulQA 暂不纳入 accuracy 结论。
2. RULER/NIAH small 如果继续全 1.0，主要作为长上下文 latency 和稳定性参考。
3. A 若 `draft=18` 不掉准确率，则下一轮优先继续压 draft 或 fallback；若准确率掉，回到 `draft=20` 周围微调。
4. C 若 `alpha=0.035` 明显提升准确率但延迟升高可接受，则说明长度偏置仍有空间；若不提升，下一轮回到 `0.025-0.030` 区间。
5. GroupD 若使用论文参数后仍明显慢且准确率无优势，后续只保留为论文对照，不继续大量调参。
6. 若 GroupD 论文参数在 GSM8K/MATH 有明显改善，需要单独分析 CNTP sampling 与我们 A/C 异步小模型路径搜索的差异。

## 0. Control Groups

### 0.1 Group D: CNTP / Cautious Next Token Prediction

新增 Group D 作为论文方法对照，不作为当前 Group A/C 优化目标的一部分。

代码来源：项目内 `ASPS/` 保留的 Cautious Next Token Prediction / CNTP 源码与 patched `transformers` 生成逻辑。

接入方式：

- 生成脚本：`ASPS/experiments/SelfEval-Guided-Decoding/src/generate_code_groupd_cautious_llama.py`
- 训练入口：`train.py` 的 `group_d`
- 汇总字段：`scripts/collect_results.py` 读取 `groupd_metrics`
- 默认配置：`configs/group_ac/d_cautious_first100.yaml`
- 运行脚本：`scripts/run_group_d_cautious_first100.sh`

默认 first100 参数：

| Config | Method | Low Entropy | High Entropy | Max Trials | Sampling |
|---|---|---:|---:|---:|---|
| `d_cautious_first100` | Group D / CNTP perplexity | 0.01 | 1.5 | 10 | greedy, `n_samples=1`, `mini_n_samples=1` |

注意：Group D 依赖 ASPS 的 custom transformers。服务器运行前需要确认当前环境已经重新安装项目内 patched 包；否则 CNTP flag 可能不会真正进入论文的生成逻辑。

本地检查记录（2026-06-08）：

| Check | Result |
|---|---|
| Python AST syntax check | 11314 个 `.py` 文件通过 |
| YAML parse check | 315 个 YAML/YML 文件通过 |
| Shell syntax check | 所有 `.sh` 文件 `bash -n` 通过 |
| Train dry-run | `configs/smoke.yaml` + `configs/group_ac/*.yaml` 共 63 个配置通过 |
| Group D dry-run | 生成 `gsm8k/strategyqa/math/truthfulqa` 4 个 first100 job，参数包含 `--cntp_mode perplexity --max_trials 10` |
| Result collect check | 临时 Group D registry 正确汇总 `groupd_metrics.wall_time` |
| Export check | 临时 Group D 结果可被 `export_results_for_github.py` 打包 |
| CNTP source check | `gsm8k_strategyqa` 与 `math_truthfulqa` 两套 patched transformers 均包含 CNTP 分发与 `_sample_reflect_perplexity*` 方法 |
| Whitespace check | `git diff --check` 通过 |

## Planned Run 20260608: RULER Small Overnight Sweep

### Goal

本轮按“对照 + 已知强参数 + GroupD + 新探索”的结构加入一个小规模
RULER/NIAH 长上下文检索数据集，同时保留完整的旧文本数据面：
GSM8K/StrategyQA/MATH/TruthfulQA first100 + RULER small。

RULER 接入口径：

- 任务：`ruler_niah`
- 样本数：20 条，`start=0, end=19`
- 默认文件：`ASPS/experiments/SelfEval-Guided-Decoding/data/ruler/ruler_niah_words_2k_small.jsonl`
- 生成脚本：`scripts/prepare_ruler_small.py`
- schema：兼容 RULER 常见 JSONL 字段 `input` / `outputs` / `length`
- 单独生成长度：约 2K words context，`max_tokens=64`
- 评分：expected outputs 在模型输出中做规范化字符串匹配

选择依据：

- 对照组：Baseline、Group B 必须和 A/C 在同一批 RULER 样本上重跑，避免只拿旧 first100 结论外推。
- Group C finalist：`c_k2_h145_a003_d16_margin005`
- Group A finalist：`a_k2_h16_d20`
- Group A 低延迟备选：`a_k2_h155_d20_margin002`
- Group D/CNTP：作为论文原方法对照，和 A/C 在同一数据面上比较。
- RULER 先只做小规模加测，不把它直接作为最终排名主指标；主要观察长上下文检索下各组是否出现明显崩溃、超慢或无效 switch。

### Configs To Run

| Config | Group | k | entropy | draft | fallback | alpha | margin | Type | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `ruler_baseline_groupb` | Baseline + B | - | - | - | - | - | - | control | 对照组重跑旧四数据集 + RULER，给 A/C/D 同题比较基线 |
| `a_ruler_k2_h16_d20_margin001` | A | 2 | 1.60 | 20 | 32 | - | 0.001 | finalist | A 主线加极轻 margin，测试 RULER 是否减少误 switch |
| `a_ruler_k2_h16_d20` | A | 2 | 1.60 | 20 | 32 | - | 0 | finalist | A 当前均衡主线，加 RULER 小规模检索压力测试 |
| `a_ruler_k2_h155_d20_margin002` | A | 2 | 1.55 | 20 | 32 | - | 0.002 | finalist | A 低延迟备选，检查 RULER 稳定性 |
| `c_ruler_k2_h145_a003_d16_margin005` | C | 2 | 1.45 | 16 | 32 | 0.03 | 0.005 | finalist | C 当前主线，加 RULER 小规模检索压力测试 |
| `d_ruler_cautious_first100` | D/CNTP | - | high=1.50 | - | - | - | - | control | CNTP 论文方法同数据面对照，检查是否值得作为额外 baseline |
| `a_ruler_k2_h165_d18_margin001` | A | 2 | 1.65 | 18 | 32 | - | 0.001 | explore | 提高阈值并缩短 draft，尝试在长上下文下压 A 延迟 |
| `a_ruler_k2_h16_d16_margin001` | A | 2 | 1.60 | 16 | 32 | - | 0.001 | explore | 保持触发阈值，缩短 draft，观察准确率是否还能稳定 |
| `c_ruler_k2_h1475_a0025_d16_margin005` | C | 2 | 1.475 | 16 | 32 | 0.025 | 0.005 | explore | 提高阈值并降低长度权重，减少长上下文中的过长路径偏好 |
| `c_ruler_k2_h145_a0025_d14_margin005` | C | 2 | 1.45 | 14 | 32 | 0.025 | 0.005 | explore | 缩短 C draft 并降低长度权重，主攻延迟 |

### Local Preparation

已在本地准备：

```text
scripts/prepare_ruler_small.py
scripts/run_group_ac_ruler_small_overnight.sh
configs/group_ac/ruler_baseline_groupb.yaml
configs/group_ac/a_ruler_k2_h16_d20.yaml
configs/group_ac/a_ruler_k2_h16_d20_margin001.yaml
configs/group_ac/a_ruler_k2_h155_d20_margin002.yaml
configs/group_ac/c_ruler_k2_h145_a003_d16_margin005.yaml
configs/group_ac/d_ruler_cautious_first100.yaml
configs/group_ac/a_ruler_k2_h165_d18_margin001.yaml
configs/group_ac/a_ruler_k2_h16_d16_margin001.yaml
configs/group_ac/c_ruler_k2_h1475_a0025_d16_margin005.yaml
configs/group_ac/c_ruler_k2_h145_a0025_d14_margin005.yaml
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_ruler_small_overnight.sh > server_run_ruler_small.log 2>&1 &
```

### Decision Rules

1. RULER small 只做长上下文稳定性压力测试，不替代 GSM8K/StrategyQA/MATH 的主调参目标。
2. 若某配置在 RULER 上准确率明显低于其他 finalist，同时 `switches` 高或 `length_weight_overrides` 多，优先怀疑长上下文下误切换。
3. 若 `a_ruler_k2_h165_d18_margin001` 比 A finalist 更快且不掉主数据集准确率，下一轮 A 可继续提高阈值或缩短 draft。
4. 若 C 的 `alpha=0.025` 比 `0.03` 更稳，说明 RULER 检索不适合过强长度偏置。
5. 若 GroupD 明显慢但准确率没有优势，则后续只保留为论文对照，不纳入主优化线。
6. 若 RULER 全部接近满分但延迟显著变慢，下一步再提高 RULER context 长度或换 RULER multi-key / variable tracking。
7. TruthfulQA 仍只作为 latency / generation-shape 参考，不纳入最终 accuracy 结论。

## 1. Optimization Goal

核心目标：优化 **Group A** 和 **Group C**，让它们在准确率和延迟上都优于 Baseline 和 Group B。

判定优先级：

1. 对有自动判分的数据集，Group A/C 的 accuracy 应高于或至少不低于 `max(Baseline, Group B)`。
2. Group A/C 的平均延迟和 P90 延迟应低于 Baseline，并显著低于 Group B。
3. Group A/C 必须保持真实在线切换：`entropy_triggers > 0`、`verify_calls > 0`、`switches > 0`。
4. Group C 的长度权重必须带来明确收益：更高 `accepted_tokens_per_verify`、更低延迟，且不牺牲 accuracy。
5. TruthfulQA 必须先修复输出终止和接入 judge 后，再纳入 accuracy 目标。

## 2. Current Experiment Bundle

当前已分析的服务器导出包：

```text
experiments/result_exports/20260602-192356_all_first100
```

本地重新汇总文件：

```text
experiments/result_exports/20260602-192356_all_first100/registry_local.csv
```

实际导出的数据集：

| Dataset | Sample Count | Auto Accuracy | Notes |
|---|---:|---:|---|
| GSM8K | 100 | yes | 数学代码生成判分 |
| StrategyQA | 100 | yes | yes/no 推理判分 |
| MATH | 100 | yes | boxed answer 判分 |
| TruthfulQA | 100 | no | 当前只保存生成结果，未接 judge |

未在当前导出包中发现 MMVet / MathVista / VLM 结果。

## 3. Baseline References

| Dataset | Baseline Accuracy | Baseline Avg Time | Group B Accuracy | Group B Avg Time | Group B P90 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 0.77 | 3.42s | 0.80 | 11.73s | 13.94s |
| StrategyQA | 0.79 | 1.61s | 0.73 | 7.40s | 8.44s |
| MATH | 0.67 | 3.37s | 0.67 | 9.32s | 12.33s |
| TruthfulQA | n/a | 18.03s | n/a | 20.62s | 31.70s |

Group B 反思重启诊断：

| Dataset | Restart Count | Hit Reflection Limit | Wrong to Right | Right to Wrong | Net |
|---|---:|---:|---:|---:|---:|
| GSM8K | 85/100 | 100/100 | 5 | 2 | +3 |
| StrategyQA | 34/100 | 97/100 | 3 | 9 | -6 |
| MATH | 22/100 | 88/100 | 1 | 1 | 0 |
| TruthfulQA | 49/100 | 100/100 | n/a | n/a | n/a |

结论：Group B 很慢，且反思 prompt 过于激进。GSM8K 有轻微收益，但 StrategyQA 明显变差。

## 4. Tried Group A/C Parameters

| Config | Method | Entropy Threshold | k | Draft Tokens | Fallback Tokens | Small Temp | Length Weight |
|---|---|---:|---:|---:|---:|---:|---|
| `a_safe` | Group A | 1.5 | 3 | 64 | 64 | 0.5 | none |
| `a_fast` | Group A | 1.3 | 3 | 32 | 32 | 0.5 | none |
| `a_accuracy` | Group A | 1.8 | 5 | 64 | 64 | 0.7 | none |
| `c_low` | Group C | 1.5 | 3 | 64 | 64 | 0.5 | `alpha=0.01`, `log_longer` |
| `c_mid` | Group C | 1.5 | 3 | 64 | 64 | 0.5 | `alpha=0.02`, `log_longer` |
| `c_fast` | Group C | 1.3 | 3 | 32 | 32 | 0.5 | `alpha=0.05`, `longer` |

## 5. Current Results

### 5.1 Accuracy

| Dataset | Baseline | Group B | A Safe | A Fast | A Accuracy | C Low | C Mid | C Fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 0.77 | 0.80 | 0.77 | 0.78 | 0.77 | 0.77 | 0.77 | 0.78 |
| StrategyQA | 0.79 | 0.73 | 0.78 | 0.78 | 0.76 | 0.78 | 0.78 | 0.79 |
| MATH | 0.67 | 0.67 | 0.64 | 0.67 | 0.67 | 0.67 | 0.67 | 0.65 |
| TruthfulQA | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

当前没有任何 Group A/C 同时满足准确率优于 Baseline 和 Group B。

### 5.2 Latency

| Dataset | Baseline Avg | Group B Avg | A Safe | A Fast | A Accuracy | C Low | C Mid | C Fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 3.42s | 11.73s | 4.01s | 4.15s | 3.49s | 3.95s | 4.00s | 4.17s |
| StrategyQA | 1.61s | 7.40s | 2.33s | 2.32s | 2.49s | 2.34s | 2.36s | 2.36s |
| MATH | 3.37s | 9.32s | 4.76s | 5.25s | 4.68s | 4.62s | 4.91s | 5.64s |
| TruthfulQA | 18.03s | 20.62s | 18.57s | 18.92s | 18.61s | 18.45s | 18.25s | 18.62s |

注意：这轮 A/C 的 prefix-cache verify 基本没有跑通，几乎都回退到 `full_batch`。因此旧延迟不能代表修复后的真实性能。

### 5.3 Group A/C Internal Metrics

| Dataset | Config | Triggers / Q | Verify / Trigger | Late Drop / Trigger | Switch / Trigger | Small Win / Verify | Overrides |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | A safe | 2.38 | 51.7% | 48.3% | 15.5% | 30.1% | 0 |
| GSM8K | A fast | 2.84 | 51.4% | 48.6% | 16.9% | 32.9% | 0 |
| GSM8K | A accuracy | 1.74 | 16.1% | 83.9% | 1.7% | 10.7% | 0 |
| GSM8K | C low | 2.38 | 50.4% | 49.6% | 16.0% | 31.7% | 0 |
| GSM8K | C mid | 2.41 | 51.0% | 49.0% | 13.3% | 26.0% | 0 |
| GSM8K | C fast | 2.82 | 52.1% | 47.9% | 18.1% | 34.7% | 0 |
| StrategyQA | A safe | 2.59 | 80.3% | 19.7% | 22.0% | 27.4% | 0 |
| StrategyQA | A fast | 2.98 | 83.6% | 16.4% | 28.5% | 34.1% | 0 |
| StrategyQA | A accuracy | 2.47 | 56.3% | 43.7% | 15.4% | 27.3% | 0 |
| StrategyQA | C low | 2.59 | 80.3% | 19.7% | 22.0% | 27.4% | 0 |
| StrategyQA | C mid | 2.61 | 80.1% | 19.9% | 22.6% | 28.2% | 0 |
| StrategyQA | C fast | 3.00 | 83.7% | 16.3% | 30.3% | 36.3% | 3 |
| MATH | A safe | 4.46 | 47.1% | 52.9% | 13.9% | 29.5% | 0 |
| MATH | A fast | 5.48 | 59.5% | 40.5% | 23.2% | 39.0% | 0 |
| MATH | A accuracy | 3.83 | 31.6% | 68.4% | 5.7% | 18.2% | 0 |
| MATH | C low | 4.34 | 47.5% | 52.5% | 13.6% | 28.6% | 0 |
| MATH | C mid | 4.48 | 47.3% | 52.7% | 12.7% | 26.9% | 1 |
| MATH | C fast | 5.83 | 57.5% | 42.5% | 20.1% | 34.9% | 7 |
| TruthfulQA | A safe | 23.85 | 95.2% | 4.8% | 49.6% | 52.1% | 0 |
| TruthfulQA | A fast | 25.12 | 94.9% | 5.1% | 50.1% | 52.7% | 0 |
| TruthfulQA | A accuracy | 22.88 | 76.3% | 23.7% | 43.1% | 56.5% | 0 |
| TruthfulQA | C low | 23.74 | 94.9% | 5.1% | 48.5% | 51.1% | 1 |
| TruthfulQA | C mid | 23.19 | 94.4% | 5.6% | 49.6% | 52.5% | 1 |
| TruthfulQA | C fast | 24.56 | 95.0% | 5.0% | 51.6% | 54.3% | 20 |

## 6. Diagnostics From Current Runs

### 6.1 Prefix Cache Verification Failed In Old Runs

旧结果日志中，A/C 几乎全部使用 `full_batch` verify。原因是 prefix-cache path 报错：

```text
'tuple' object has no attribute 'get_seq_length'
```

已在本地修复：

```text
ASPS/experiments/SelfEval-Guided-Decoding/src/groupa_async_decoding.py
```

修复方向：

- legacy tuple cache 转成 `DynamicCache`
- suffix scoring forward 使用 `use_cache=True`

下一轮必须检查：

```text
verify_modes.prefix_cache / verify_calls
```

目标：prefix-cache 占比接近 100%。如果仍大量 `full_batch`，优先修实现，不急着扩大调参。

### 6.2 Late Drop Is Too High On Math Tasks

GSM8K 和 MATH 的 late drop 仍偏高：

- GSM8K 常见 late drop 约 48% 到 84%
- MATH 常见 late drop 约 40% 到 68%

含义：小模型 draft 经常没有赶上一个 fallback punctuation span。

可能调参方向：

- 降低 `max_draft_tokens`
- 降低 `draft_candidates`
- 降低 small temperature，让 draft 更快遇到标点
- 允许 fallback span 稍长或改为 token-level late accept
- 为小模型侧加更明确 stop 标点

### 6.3 A Accuracy Config Is Not Actually Accuracy-Oriented

`a_accuracy` 使用 `entropy_threshold=1.8, k=5, draft64, small_temp=0.7`，但结果是 verify 少、late drop 高、switch 少。

结论：提高阈值和增加 k 没带来准确率收益，反而让小模型更晚返回。后续不建议继续以当前 `a_accuracy` 为基础扩展。

### 6.4 Group C Length Weight Is Too Weak Or Too Risky

`c_low` 和 `c_mid` override 几乎为 0，说明长度权重基本没参与决策。

`c_fast` override 变多：

- StrategyQA accuracy 达到 0.79，打平 baseline
- MATH accuracy 降到 0.65

结论：长度权重能改变路径，但对 MATH 有风险。后续需要任务分层或更保守的长度权重门控。

### 6.5 TruthfulQA Stop Condition Is Broken

TruthfulQA 大量输出在回答后继续生成新的 `Q:` / `A:` 示例。

| Method | Avg Chars | Contains Follow-up Q |
|---|---:|---:|
| Baseline | 2429.8 | 98/100 |
| Group B | 1572.7 | 62/100 |
| A safe | 2471.2 | 97/100 |
| A fast | 2477.1 | 97/100 |
| A accuracy | 2473.7 | 98/100 |
| C low | 2466.3 | 96/100 |
| C mid | 2459.6 | 96/100 |
| C fast | 2473.0 | 97/100 |

TruthfulQA 在修 stop 和 judge 前，不纳入最终准确率结论。

## 7. Current Best Takeaways

### GSM8K

- Baseline 0.77，Group B 0.80。
- A/C 最高 0.78，尚未超过 Group B。
- 当前最接近目标的是 `a_fast` / `c_fast`，但延迟仍慢于 baseline。

### StrategyQA

- Baseline 0.79，Group B 0.73。
- `c_fast` 达到 0.79，打平 baseline，优于 Group B。
- 但延迟仍慢于 baseline。

### MATH

- Baseline 0.67，Group B 0.67。
- `a_fast` / `a_accuracy` / `c_low` / `c_mid` 达到 0.67，打平 baseline。
- `c_fast` 掉到 0.65，说明强长度权重不适合 MATH。

### TruthfulQA

- A/C 触发和切换很活跃。
- 输出停止坏了，不能用于质量判断。

## 8. Next Tuning Plan

### Step 1: Rerun After Prefix-Cache Fix

先不扩大搜索，重跑小规模验证 prefix-cache：

```bash
RUN_SMOKE=0 RUN_CONFIGS="a_fast c_fast c_low" bash scripts/run_group_ac_experiments.sh
```

检查标准：

| Metric | Target |
|---|---:|
| `verify_modes.prefix_cache / verify_calls` | >= 95% |
| `full_batch` fallback count | close to 0 |
| latency vs old A/C | should drop |
| accuracy | no obvious regression |

### Step 2: Fix TruthfulQA Stop

TruthfulQA stop 条件建议：

```text
stop_strings = ("\n\nQ:", "\nQ:", "\n\nQuestion:", "\nQuestion:")
```

输出保存时也可以做后处理截断：

```text
truncate before follow-up Q/Question marker
```

之后接 TruthfulQA judge，再将其纳入 accuracy 表。

### Step 3: New Candidate Configs

#### Candidate A1: lower late drop

目标：减少 GSM8K/MATH late draft drop。

| Param | Value |
|---|---:|
| entropy_threshold | 1.3 |
| draft_candidates | 2 |
| max_draft_tokens | 24 |
| max_fallback_tokens | 48 |
| small_temperature | 0.3 |
| small_top_p | 0.9 |

理由：减少候选数和 draft 长度，让小模型更快返回；fallback 给到 48，降低过短 span 导致频繁触发的问题。

#### Candidate A2: conservative switch

目标：避免小模型路径扰动正确推理。

| Param | Value |
|---|---:|
| entropy_threshold | 1.5 |
| draft_candidates | 3 |
| max_draft_tokens | 32 |
| max_fallback_tokens | 64 |
| small_temperature | 0.3 |
| small_top_p | 0.8 |

选择策略建议新增门控：

```text
only switch if best_small_avg_logprob >= fallback_avg_logprob + margin
margin = 0.02 or 0.05
```

#### Candidate C1: gated length weight

目标：保留 C 在 StrategyQA 的收益，避免 MATH 下降。

| Param | Value |
|---|---:|
| entropy_threshold | 1.3 |
| draft_candidates | 3 |
| max_draft_tokens | 32 |
| max_fallback_tokens | 32 |
| length_weight_alpha | 0.02 |
| length_weight_mode | longer |

新增门控：

```text
apply length weight only if abs(best_base_score - fallback_score) <= 0.03
```

#### Candidate C2: accepted-token latency variant

目标：让 accepted tokens per verify 上升，同时不明显损害 accuracy。

| Param | Value |
|---|---:|
| entropy_threshold | 1.5 |
| draft_candidates | 2 |
| max_draft_tokens | 48 |
| max_fallback_tokens | 64 |
| length_weight_alpha | 0.01 |
| length_weight_mode | log_longer |

理由：保守长度权重，优先试 latency 改善。

## 9. Future Run Record Template

每轮追加一个小节。

```markdown
## Run YYYYMMDD-HHMM: <short name>

### Goal

- 

### Code Version

- Branch:
- Commit:
- Important changes:

### Configs

| Config | Method | Dataset | Key Params |
|---|---|---|---|
| | | | |

### Results

| Dataset | Method/Config | Accuracy | Avg Time | P90 Time | Triggers/Q | Verify/Trigger | Late Drop | Switch Rate | Prefix Cache % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | |

### Analysis

- Accuracy:
- Latency:
- Switch behavior:
- Failure cases:

### Decision

- Keep:
- Drop:
- Modify:
- Next run:
```

## 10. Current Action Items

1. Commit and push prefix-cache fix.
2. Rerun `a_fast`, `c_fast`, `c_low` after prefix-cache fix.
3. Confirm `prefix_cache` verify dominates.
4. Fix TruthfulQA stop condition.
5. Add TruthfulQA judge before reporting truthfulness accuracy.
6. Add explicit token accounting for Group A/C:
   - big input tokens
   - big output tokens
   - big verify tokens
   - small input tokens
   - small output tokens
   - wasted fallback tokens
7. Add switch margin / gated length weight before broader search.

## Planned Run 20260603: Candidate Count k Sweep

### Goal

本轮只调候选数 `draft_candidates`，取 `k=1,2,3`。目标是隔离候选数对 accuracy、latency、late drop、switch rate 和 prefix-cache verify 的影响。

Baseline 和 Group B 已作为对照组跑过，本轮不重复跑，避免浪费服务器时间。

### Base Selection

| Group | Base Config | Reason |
|---|---|---|
| Group A | `a_fast` | 旧结果中 A 组综合最好：GSM8K 0.78，StrategyQA 0.78，MATH 0.67，且 draft/fallback 窗口更短。 |
| Group C | `c_fast` | 旧结果中 C 组最接近目标：GSM8K 0.78，StrategyQA 0.79，但 MATH 曾降到 0.65，需要重点观察。 |

### Prepared Configs

| Config | Method | Base | k | Entropy | Draft Tokens | Fallback Tokens | Length Weight |
|---|---|---|---:|---:|---:|---:|---|
| `a_fast_k1` | Group A | `a_fast` | 1 | 1.3 | 32 | 32 | none |
| `a_fast_k2` | Group A | `a_fast` | 2 | 1.3 | 32 | 32 | none |
| `a_fast_k3` | Group A | `a_fast` | 3 | 1.3 | 32 | 32 | none |
| `c_fast_k1` | Group C | `c_fast` | 1 | 1.3 | 32 | 32 | `alpha=0.05`, `longer` |
| `c_fast_k2` | Group C | `c_fast` | 2 | 1.3 | 32 | 32 | `alpha=0.05`, `longer` |
| `c_fast_k3` | Group C | `c_fast` | 3 | 1.3 | 32 | 32 | `alpha=0.05`, `longer` |

### Server Command

```bash
RUN_SMOKE=0 bash scripts/run_group_ac_k_sweep.sh
```

后台跑：

```bash
nohup bash scripts/run_group_ac_k_sweep.sh > server_run_k_sweep.log 2>&1 &
```

### Analysis Checklist

| Metric | Target / Decision Use |
|---|---|
| `verify_modes.prefix_cache / verify_calls` | >= 95%，否则先修 verify，不扩大调参 |
| accuracy | 每个数据集分别对比 baseline、Group B、同组 k=3 |
| average latency / P90 latency | 必须优于 Group B，并争取低于 baseline |
| late drop / trigger | k 降低后应该下降，尤其 GSM8K 和 MATH |
| switch / trigger | 不能因为 k 太小而塌掉 |
| small win / verify | 判断小模型候选质量是否足够 |

### Decision Rules

1. 若 `k=1` accuracy 不降且 latency / late drop 明显更好，优先保留 `k=1` 作为下一轮基底。
2. 若 `k=1` switch rate 过低或 accuracy 下降超过 1 个百分点，比较 `k=2` 与 `k=3`。
3. Group C 要单独看 MATH；如果 `c_fast_k*` 仍压低 MATH accuracy，下一轮改为更保守的长度权重或加入 score-margin gate。
4. TruthfulQA 在 stop / judge 修复前仍只看机制指标，不纳入 accuracy 结论。

## Run 20260603-110746: Candidate Count k Sweep Results

### Export Bundle

```text
experiments/result_exports/20260603-110746_group_ac_k_sweep_first100
```

该导出包包含本次 k sweep 的 6 个新 run，也包含之前导出的 baseline / GroupB / 旧 A/C run。下面只把本轮新 run 与必要对照放在一起分析。

### Key Finding

prefix-cache verify 修复已生效。本轮 6 个新配置在 GSM8K、StrategyQA、MATH、TruthfulQA 上的 `verify_modes.prefix_cache / verify_calls` 都是 100%。

同一组参数的 `k=3` 与旧 `a_fast` / `c_fast` 相比，延迟明显下降，说明旧结果的 A/C latency 主要被 full-batch verify 拖慢。

### Auto-Accuracy Results

| Dataset | Baseline Acc | GroupB Acc | A k1 | A k2 | A k3 | C k1 | C k2 | C k3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 0.77 | 0.80 | 0.77 | 0.77 | 0.78 | 0.78 | 0.78 | 0.77 |
| StrategyQA | 0.79 | 0.73 | 0.78 | 0.76 | 0.78 | 0.78 | 0.76 | 0.77 |
| MATH | 0.67 | 0.67 | 0.67 | 0.70 | 0.66 | 0.67 | 0.67 | 0.62 |

本轮仍没有任何 Group A/C 配置在三个自动判分数据集上同时超过 `max(Baseline, GroupB)`。

### Latency Results

Baseline 的 avg time 用 `job_duration_sec / sample_count` 估算；A/C 和 GroupB 使用 registry 中的 sample wall time。

| Dataset | Baseline Avg | GroupB Avg | A k1 | A k2 | A k3 | C k1 | C k2 | C k3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 3.42s | 11.73s | 3.44s | 3.45s | 3.56s | 3.57s | 3.47s | 3.41s |
| StrategyQA | 1.61s | 7.40s | 1.95s | 2.06s | 2.01s | 1.92s | 2.24s | 2.17s |
| MATH | 3.37s | 9.32s | 4.25s | 4.18s | 4.39s | 4.81s | 4.30s | 4.52s |
| TruthfulQA | 18.03s | 20.62s | 15.20s | 15.76s | 16.23s | 15.61s | 15.12s | 15.41s |

结论：新 A/C 已经显著快于 GroupB；GSM8K 已经接近 baseline，C k3 略低于 baseline；StrategyQA 和 MATH 仍慢于 baseline。TruthfulQA 延迟低于 baseline，但因 stop / judge 未修复，不能当质量结论。

### Mechanism Summary

| Config | Macro Acc | Avg Time | P90 Mean | Prefix Cache | Late Drop / Trigger | Switch / Trigger |
|---|---:|---:|---:|---:|---:|---:|
| `a_fast_k1` | 0.740 | 3.21s | 4.96s | 100% | 0.12 | 0.21 |
| `a_fast_k2` | 0.743 | 3.23s | 4.51s | 100% | 0.24 | 0.23 |
| `a_fast_k3` | 0.740 | 3.32s | 5.42s | 100% | 0.36 | 0.23 |
| `c_fast_k1` | 0.743 | 3.43s | 6.49s | 100% | 0.14 | 0.23 |
| `c_fast_k2` | 0.737 | 3.34s | 5.24s | 100% | 0.26 | 0.22 |
| `c_fast_k3` | 0.720 | 3.37s | 5.35s | 100% | 0.36 | 0.22 |

宏平均只计算 GSM8K / StrategyQA / MATH。

### TruthfulQA Stop Check

| Config | Avg Chars | Follow-up Q Count |
|---|---:|---:|
| `a_fast_k1` | 2481.2 | 98/100 |
| `a_fast_k2` | 2481.0 | 98/100 |
| `a_fast_k3` | 2468.0 | 97/100 |
| `c_fast_k1` | 2476.4 | 98/100 |
| `c_fast_k2` | 2476.0 | 97/100 |
| `c_fast_k3` | 2474.4 | 98/100 |

TruthfulQA 仍然大量生成后续 `Q:` / `A:` 示例。速度改善不代表质量改善，下一步仍要先修 stop 条件和 judge。

### Decision

Keep:

- Group A：优先保留 `a_fast_k2`。它是本轮 A 组宏平均最高，MATH 达到 0.70，超过 baseline / GroupB。
- Group A：保留 `a_fast_k1` 作为低延迟备选。它 late drop 最低，StrategyQA 延迟最好，但 MATH 只打平 baseline。
- Group C：保留 `c_fast_k1` 作为 C 组下一步基底。它宏平均最高，late drop 较低，MATH 没有掉到 0.62/0.65。

Drop:

- `c_fast_k3`：MATH 0.62，且 late drop 高，不适合继续扩展。
- `a_fast_k3`：不是完全不能用，但在本轮没有带来宏平均收益，MATH 低于 baseline。
- `c_fast_k2`：GSM8K/MATH 可用，但 StrategyQA 掉到 0.76，暂不作为主线。

Next:

1. 以 `a_fast_k2` 为主线，试更低延迟版本：`entropy_threshold=1.4/1.5`、`draft_candidates=2`、`max_draft_tokens=24/32`。
2. 以 `c_fast_k1` 为 C 组主线，降低长度权重风险：`length_weight_alpha=0.02/0.03` 或加入 score-margin gate。
3. 在扩大调参前修 TruthfulQA stop 条件，否则 TruthfulQA 的 accuracy 仍不可用。

## Diagnosis 20260603: Why `c_fast_k2` Is Slower Than `a_fast_k2`

### Observation

直觉上，Group C 的长度权重应该偏向更长候选，从而减少后续触发次数并降低延迟。但本轮 `c_fast_k2` 在 GSM8K / StrategyQA / MATH 上没有稳定快于 `a_fast_k2`。

| Dataset | A k2 Avg | C k2 Avg | Difference | A k2 Acc | C k2 Acc |
|---|---:|---:|---:|---:|---:|
| GSM8K | 3.45s | 3.47s | +0.02s | 0.77 | 0.78 |
| StrategyQA | 2.06s | 2.24s | +0.19s | 0.76 | 0.76 |
| MATH | 4.18s | 4.30s | +0.12s | 0.70 | 0.67 |
| TruthfulQA | 15.76s | 15.12s | -0.64s | n/a | n/a |

TruthfulQA 虽然 C k2 更快，但 stop / judge 未修复，不能当质量结论。

### Mechanism Comparison

| Dataset | Metric | A k2 | C k2 | Interpretation |
|---|---|---:|---:|---|
| StrategyQA | triggers | 330 | 345 | C 触发更多 |
| StrategyQA | verify calls | 286 | 291 | C verify 更多 |
| StrategyQA | late drops | 44 | 54 | C late drop 更多 |
| StrategyQA | avg output chars | 299.3 | 324.1 | C 输出更长 |
| MATH | triggers | 538 | 543 | C 触发更多 |
| MATH | verify calls | 371 | 364 | C verify 略少 |
| MATH | late drops | 167 | 179 | C late drop 更多 |
| MATH | avg output chars | 376.8 | 390.6 | C 输出更长 |
| MATH | P90 | 6.58s | 8.42s | C 长尾更差 |

### Length Weight Participation

| Dataset | C k2 Length Overrides |
|---|---:|
| GSM8K | 0 |
| StrategyQA | 2 |
| MATH | 3 |
| TruthfulQA | 13 |

结论：`c_fast_k2` 大多数时候并没有真正改变候选选择；少数 override 又可能把路径推向更长、更不稳定的生成。因此它没有稳定减少 trigger / verify，反而在 StrategyQA 和 MATH 上增加了输出长度和 late drop。

### Decision

- `c_fast_k2` 不作为下一轮主线。
- Group C 继续以 `c_fast_k1` 为主线，因为它宏平均更好，late drop 更低，MATH 没有明显崩掉。
- 下一轮 Group C 不再直接使用 `alpha=0.05 longer` 做主线，应降低长度权重或加入 gate。

## Planned Next Tuning 20260603: Latency Recovery With Accuracy Guard

### Goal

在 prefix-cache 已修复的基础上，继续优化 Group A 和 Group C，使它们更接近目标：

1. accuracy 至少不低于 `max(Baseline, GroupB)`。
2. avg / P90 latency 继续向 baseline 靠近，同时保持显著优于 GroupB。
3. 保持 `prefix_cache >= 95%`。
4. 避免 StrategyQA 和 MATH 因小模型切换或长度权重掉分。

### Recommended Group A Search

以 `a_fast_k2` 为主线。它是目前 A 组最接近目标的配置：MATH 0.70 超过 baseline / GroupB，宏平均打平 baseline。

| Candidate | k | Entropy | Draft Tokens | Fallback Tokens | Small Temp | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `a_k2_h14_d32` | 2 | 1.4 | 32 | 32 | 0.5 | 减少触发，观察延迟是否低于当前 A k2，同时保留 MATH 收益 |
| `a_k2_h15_d32` | 2 | 1.5 | 32 | 32 | 0.5 | 更强 latency recovery，风险是切换减少后 MATH 收益消失 |
| `a_k2_h13_d24` | 2 | 1.3 | 24 | 32 | 0.5 | 保持当前触发策略，但让 draft 更快返回，目标是降低 late drop / P90 |
| `a_k2_h14_d24` | 2 | 1.4 | 24 | 32 | 0.5 | 同时减少触发和 late drop，作为低延迟候选 |

优先级：先跑 `a_k2_h14_d32` 和 `a_k2_h14_d24`。如果准确率不掉，再跑 `h15`。

### Recommended Group C Search

以 `c_fast_k1` 为主线。当前 `alpha=0.05 longer` 对 MATH 有风险，下一轮应更保守。

| Candidate | k | Entropy | Draft Tokens | Fallback Tokens | Length Alpha | Mode | Purpose |
|---|---:|---:|---:|---:|---:|---|---|
| `c_k1_a002` | 1 | 1.3 | 32 | 32 | 0.02 | longer | 降低长度权重，避免 MATH 掉分 |
| `c_k1_a003` | 1 | 1.3 | 32 | 32 | 0.03 | longer | 在保守和收益之间折中 |
| `c_k1_h14_a002` | 1 | 1.4 | 32 | 32 | 0.02 | longer | 减少触发，压低延迟 |
| `c_k1_h14_a003` | 1 | 1.4 | 32 | 32 | 0.03 | longer | 观察 C 是否能在少触发下保留 StrategyQA/GSM8K 收益 |

如果愿意改代码，优先新增 score-margin gate：

```text
只在 abs(best_base_score - fallback_score) <= 0.03 时应用 length weight。
```

这样可以避免长度权重在大模型明显偏好 fallback 时强行 override。

### Stop Before Broader Search

TruthfulQA 的输出停止仍坏：

```text
97-98/100 samples contain follow-up Q/A generations
```

在下一轮大规模搜索前，应单独修：

```text
stop_strings = ("\n\nQ:", "\nQ:", "\n\nQuestion:", "\nQuestion:")
```

并在保存结果前截断 follow-up Q/Question marker。否则 TruthfulQA 只能看机制指标，不能进入最终 accuracy 表。

### Next Run Recommendation

建议下一轮先跑 6 个配置：

```text
Group A:
- a_k2_h14_d32
- a_k2_h14_d24
- a_k2_h15_d32

Group C:
- c_k1_a002
- c_k1_a003
- c_k1_h14_a002
```

如果服务器时间有限，先跑 4 个：

```text
a_k2_h14_d32
a_k2_h14_d24
c_k1_a002
c_k1_h14_a002
```

判定规则：

1. GSM8K：需要接近 0.80，至少不低于 0.78。
2. StrategyQA：至少 0.79 才算超过/打平 baseline 目标。
3. MATH：优先保留 >= 0.70 的 A 配置；C 至少不能低于 0.67。
4. 延迟：GSM8K 目标 <= 3.42s；StrategyQA 目标 <= 1.61s 较难，但应继续压到 <= 1.9s；MATH 目标先压到 <= 4.0s。
5. 若 prefix-cache 低于 95%，停止调参，先修实现。

### Local Preparation

已在本地准备 6 个配置和专用脚本：

```text
configs/group_ac/a_k2_h14_d32.yaml
configs/group_ac/a_k2_h14_d24.yaml
configs/group_ac/a_k2_h15_d32.yaml
configs/group_ac/c_k1_a002.yaml
configs/group_ac/c_k1_a003.yaml
configs/group_ac/c_k1_h14_a002.yaml
scripts/run_group_ac_latency_recovery.sh
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_latency_recovery.sh > server_run_latency_recovery.log 2>&1 &
```

## Run 20260603-181452: Latency Recovery Results

### Export Bundle

结果在远端分支：

```text
origin/results/latency-recovery-first100-20260603
```

导出包路径：

```text
experiments/result_exports/20260603-181452_group_ac_latency_recovery_first100
```

本轮只分析新 6 个配置，并与 baseline、GroupB、上一轮最佳 `a_fast_k2` / `c_fast_k1` 对比。

### Key Finding

prefix-cache verify 仍然保持 100%，实现层面没有回退。

本轮出现两个新的强候选：

- Group A: `a_k2_h15_d32`
- Group C: `c_k1_h14_a002`

它们在 GSM8K / StrategyQA / MATH 的宏准确率都是 0.753，高于 baseline 的 0.743 和 GroupB 的 0.733。但平均延迟仍高于 baseline，尚未完全达到最终优化目标。

### Auto-Accuracy Results

| Config | GSM8K | StrategyQA | MATH | Macro Acc |
|---|---:|---:|---:|---:|
| Baseline | 0.77 | 0.79 | 0.67 | 0.743 |
| GroupB | 0.80 | 0.73 | 0.67 | 0.733 |
| `a_fast_k2` prev | 0.77 | 0.76 | 0.70 | 0.743 |
| `a_k2_h14_d32` | 0.77 | 0.76 | 0.68 | 0.737 |
| `a_k2_h14_d24` | 0.78 | 0.76 | 0.67 | 0.737 |
| `a_k2_h15_d32` | 0.78 | 0.76 | 0.72 | 0.753 |
| `c_fast_k1` prev | 0.78 | 0.78 | 0.67 | 0.743 |
| `c_k1_a002` | 0.77 | 0.78 | 0.67 | 0.740 |
| `c_k1_a003` | 0.77 | 0.78 | 0.66 | 0.737 |
| `c_k1_h14_a002` | 0.78 | 0.79 | 0.69 | 0.753 |

### Latency Results

Baseline avg 仍使用 `job_duration_sec / sample_count` 估算；A/C 和 GroupB 使用 sample wall time。

| Config | GSM8K Avg | StrategyQA Avg | MATH Avg | Macro Avg | Macro P90 |
|---|---:|---:|---:|---:|---:|
| Baseline | 3.42s | 1.61s | 3.37s | 2.80s | n/a |
| GroupB | 11.73s | 7.40s | 9.32s | 9.48s | 11.57s |
| `a_fast_k2` prev | 3.45s | 2.06s | 4.18s | 3.23s | 4.51s |
| `a_k2_h14_d32` | 3.53s | 1.98s | 4.05s | 3.19s | 4.69s |
| `a_k2_h14_d24` | 3.37s | 2.12s | 4.23s | 3.24s | 4.84s |
| `a_k2_h15_d32` | 3.47s | 2.10s | 3.94s | 3.17s | 4.21s |
| `c_fast_k1` prev | 3.57s | 1.92s | 4.81s | 3.43s | 6.49s |
| `c_k1_a002` | 3.45s | 1.95s | 4.45s | 3.28s | 6.24s |
| `c_k1_a003` | 3.41s | 1.85s | 4.50s | 3.25s | 6.22s |
| `c_k1_h14_a002` | 3.37s | 2.11s | 4.36s | 3.28s | 5.26s |

### Mechanism Summary

| Config | Prefix Cache | Late Drop / Trigger | Switch / Trigger |
|---|---:|---:|---:|
| `a_fast_k2` prev | 100% | 0.24 | 0.23 |
| `a_k2_h14_d32` | 100% | 0.27 | 0.22 |
| `a_k2_h14_d24` | 100% | 0.24 | 0.21 |
| `a_k2_h15_d32` | 100% | 0.25 | 0.20 |
| `c_fast_k1` prev | 100% | 0.14 | 0.23 |
| `c_k1_a002` | 100% | 0.14 | 0.21 |
| `c_k1_a003` | 100% | 0.14 | 0.22 |
| `c_k1_h14_a002` | 100% | 0.13 | 0.19 |

### TruthfulQA Stop Check

| Config | Avg Chars | Follow-up Q Count |
|---|---:|---:|
| `a_k2_h14_d32` | 2492.3 | 98/100 |
| `a_k2_h14_d24` | 2480.0 | 98/100 |
| `a_k2_h15_d32` | 2454.5 | 97/100 |
| `c_k1_a002` | 2478.8 | 97/100 |
| `c_k1_a003` | 2479.5 | 98/100 |
| `c_k1_h14_a002` | 2456.9 | 96/100 |

TruthfulQA 仍未修复，不能进入最终 accuracy 结论。

### Decision

Keep:

- Group A 主线改为 `a_k2_h15_d32`。它 MATH 0.72，是目前 MATH 最强；宏准确率 0.753，macro P90 也是 A 组最好。
- Group C 主线改为 `c_k1_h14_a002`。它 StrategyQA 0.79，打平 baseline，MATH 0.69，GSM8K 0.78，是目前 C 组最均衡。
- `a_k2_h14_d24` 可作为 GSM8K 低延迟参考：GSM8K avg 3.37s，低于 baseline 3.42s，但 MATH 回落到 0.67。
- `c_k1_a003` 可作为 StrategyQA 低延迟参考：StrategyQA avg 1.85s，但 MATH 0.66，不作为主线。

Drop:

- `a_k2_h14_d32`：accuracy 没有收益。
- `c_k1_a002`：相比 `c_k1_h14_a002` 没有优势。
- `c_k1_a003`：MATH 降到 0.66，除非只追 StrategyQA latency，否则不继续。

Next:

1. Group A 围绕 `a_k2_h15_d32` 微调，目标是保持 MATH >= 0.70，同时把 StrategyQA 从 0.76 拉回 0.78/0.79。
2. Group C 围绕 `c_k1_h14_a002` 微调，目标是保持 StrategyQA 0.79，并把 MATH 从 0.69 推到 0.70+。
3. 下一轮应优先尝试 switch margin / length-weight gate，而不是继续只调 k / entropy / alpha。
4. TruthfulQA stop 条件必须单独修，否则它仍只能作为 latency / mechanism 数据。

## Planned Run 20260604: Overnight Margin Sweep

### Goal

在上一轮 `a_k2_h15_d32` 和 `c_k1_h14_a002` 的基础上做 10 组夜跑。目标不是重新探索候选数，而是围绕当前最强点压延迟、恢复 StrategyQA/GSM8K，并验证 `switch_score_margin=0.02` 是否能减少无收益切换。

本轮仍只跑 Group A / Group C，不重跑 baseline 和 GroupB。对照继续使用已记录的 first100 baseline / GroupB。

### Code Change

新增 `switch_score_margin`：

```text
only accept a small-model path if:
small_path_avg_logprob >= fallback_avg_logprob + switch_score_margin
```

默认值为 `0.0`，因此旧配置行为不变。margin 配置使用 `0.02`。registry 新增 `switch_margin_rejections`，用于回收结果后判断 margin 是否实际减少了切换。夜跑脚本默认使用独立 `OUTPUT_ROOT=outputs/${EXPORT_NAME}`，避免导出包混入服务器上的旧输出。

### Configs To Run

| Config | Group | k | entropy | draft | fallback | alpha | margin | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `a_k2_h15_d24` | A | 2 | 1.5 | 24 | 32 | - | 0 | 从 `a_k2_h15_d32` 压 draft，目标降低 MATH/GSM 延迟 |
| `a_k2_h16_d24` | A | 2 | 1.6 | 24 | 32 | - | 0 | 更高触发率 + 短 draft，尝试补 StrategyQA/GSM 准确率 |
| `a_k2_h15_d20` | A | 2 | 1.5 | 20 | 32 | - | 0 | 激进低延迟组，验证 draft 进一步缩短是否崩准确率 |
| `a_k2_h16_d32` | A | 2 | 1.6 | 32 | 32 | - | 0 | 准确率上限组，看 h16+d32 是否超过 `a_k2_h15_d32` |
| `a_k2_h15_d24_margin` | A | 2 | 1.5 | 24 | 32 | - | 0.02 | 延迟恢复组，验证 margin 是否减少无收益 switch |
| `c_k1_h15_a002` | C | 1 | 1.5 | 32 | 32 | 0.02 | 0 | 在 C 最优 alpha 上提高 entropy，尝试补 MATH/GSM |
| `c_k1_h14_a0015` | C | 1 | 1.4 | 32 | 32 | 0.015 | 0 | 降低 length weight，目标减少 C 的额外延迟 |
| `c_k1_h14_a002_margin` | C | 1 | 1.4 | 32 | 32 | 0.02 | 0.02 | 以当前 C 主线加 margin，重点看延迟尾部 |
| `c_k1_h15_a0015` | C | 1 | 1.5 | 32 | 32 | 0.015 | 0 | h15 + 低 alpha 平衡组 |
| `c_k1_h15_a002_margin` | C | 1 | 1.5 | 32 | 32 | 0.02 | 0.02 | h15 + 当前 alpha + margin，验证准确率/延迟平衡 |

### Local Preparation

已在本地准备：

```text
configs/group_ac/a_k2_h15_d24.yaml
configs/group_ac/a_k2_h16_d24.yaml
configs/group_ac/a_k2_h15_d20.yaml
configs/group_ac/a_k2_h16_d32.yaml
configs/group_ac/a_k2_h15_d24_margin.yaml
configs/group_ac/c_k1_h15_a002.yaml
configs/group_ac/c_k1_h14_a0015.yaml
configs/group_ac/c_k1_h14_a002_margin.yaml
configs/group_ac/c_k1_h15_a0015.yaml
configs/group_ac/c_k1_h15_a002_margin.yaml
scripts/run_group_ac_overnight_sweep.sh
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_overnight_sweep.sh > server_run_overnight_margin.log 2>&1 &
```

### Decision Rules

1. A 组优先看 `a_k2_h15_d24_margin` 能否在不丢 MATH 的情况下压 macro avg / P90。
2. A 组若 `a_k2_h16_d32` 准确率明显提升，但延迟高于 baseline，后续再试 h16+d24+margin。
3. C 组优先看 `c_k1_h14_a002_margin` 是否保持 StrategyQA 0.79 且压低 C 的 P90。
4. 若 `switch_margin_rejections` 很低，说明 margin=0.02 太小；下一轮可以试 0.05。
5. 若 `switch_margin_rejections` 很高但准确率下降，说明 margin 过保守，应回退到 0.01 或改成 dataset-specific margin。
6. TruthfulQA 仍只作为 latency / generation-shape 参考，不纳入最终 accuracy 结论。

## Planned Run 20260604: Daytime C-k2 Sweep

### Goal

白天只跑 6 组，保留最高信息量。Group A 继续围绕 `a_k2_h15_d20` 压延迟/补 StrategyQA；Group C 本轮全部设为 `k=2`，以更符合 Group C “小模型多路径探索 + 大模型验证选择” 的设计初衷。

旧 `c_fast_k2` 的问题不是 `k=2` 本身，而是 `draft=32 + alpha=0.05 + no margin` 太重，导致 StrategyQA 掉到 0.76、MATH 只有 0.67、late drop/trigger 约 0.26、P90 偏高。本轮 C 采用轻量 k=2：

```text
k=2 + draft=16 + alpha=0.02 + small margin
```

### Configs To Run

| Config | Group | k | entropy | draft | fallback | alpha | margin | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `a_k2_h15_d16` | A | 2 | 1.5 | 16 | 32 | - | 0 | 继续压 A 延迟，观察 MATH 0.72 是否能保住 |
| `a_k2_h15_d20_margin005` | A | 2 | 1.5 | 20 | 32 | - | 0.005 | A 的轻量 margin，避免 0.02 过强导致准确率下降 |
| `a_k2_h155_d20` | A | 2 | 1.55 | 20 | 32 | - | 0 | 介于 h1.5/h1.6，目标补 StrategyQA 且少伤 MATH |
| `c_k2_h14_a002_d16_margin005` | C | 2 | 1.4 | 16 | 32 | 0.02 | 0.005 | C k=2 最轻 margin，目标恢复 GSM8K 并保留多路径探索 |
| `c_k2_h14_a002_d16_margin01` | C | 2 | 1.4 | 16 | 32 | 0.02 | 0.01 | C k=2 主测组，平衡 GSM8K/StrategyQA/MATH |
| `c_k2_h145_a002_d16_margin01` | C | 2 | 1.45 | 16 | 32 | 0.02 | 0.01 | 略提高阈值减少触发，观察延迟和 MATH 是否更稳 |

### Local Preparation

已在本地准备：

```text
configs/group_ac/a_k2_h15_d16.yaml
configs/group_ac/a_k2_h15_d20_margin005.yaml
configs/group_ac/a_k2_h155_d20.yaml
configs/group_ac/c_k2_h14_a002_d16_margin005.yaml
configs/group_ac/c_k2_h14_a002_d16_margin01.yaml
configs/group_ac/c_k2_h145_a002_d16_margin01.yaml
scripts/run_group_ac_daytime_k2c_sweep.sh
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_daytime_k2c_sweep.sh > server_run_daytime_k2c.log 2>&1 &
```

### Decision Rules

1. A 组：优先看 `a_k2_h15_d16` 是否保持 MATH >= 0.70 且降低 macro avg；若掉分，保留 `a_k2_h15_d20` 主线。
2. A 组：`margin005` 只要不明显掉 MATH/StrategyQA，就可继续往 margin `0.01` 扩。
3. C 组：三组都必须检查 late drop/trigger，若 `k=2,d16` 仍 late drop 很高，说明 C 的多路径探索主要瓶颈在 draft 返回时间。
4. C 组：优先保留同时满足 StrategyQA >= 0.79、MATH >= 0.70、GSM8K >= 0.77 的配置。
5. 若 C k=2 的 accuracy 无收益但 latency 更差，报告里可以说明“符合初衷的多路径探索已验证，但当前小模型/窗口设置下不占优”。
6. TruthfulQA 仍只作为 latency / generation-shape 参考。

## Planned Run 20260604: K2 Refine Sweep

### Goal

继续固定 `k=2`。Group A 不再回到 k=1，而是在当前主线 `a_k2_h15_d20` 周围试更细的 entropy / draft / small temperature。Group C 也继续固定 `k=2`，围绕当前最好 k=2 点 `c_k2_h145_a002_d16_margin01` 微调，目标是把 StrategyQA 拉到 0.79+、MATH 拉到 0.70+，同时保留 k=2 多路径探索的论文一致性。

### Configs To Run

| Config | Group | k | entropy | draft | fallback | alpha | margin | small temp | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `a_k2_h145_d20` | A | 2 | 1.45 | 20 | 32 | - | 0 | 0.5 | 比 h1.5 更保守，目标减少触发/延迟并保住 MATH |
| `a_k2_h15_d18` | A | 2 | 1.5 | 18 | 32 | - | 0 | 0.5 | 介于 d16/d20，找 A 的 draft 长度临界点 |
| `a_k2_h15_d20_t04` | A | 2 | 1.5 | 20 | 32 | - | 0 | 0.4 | 降低小模型温度，目标减少 late drop 并稳定 draft |
| `c_k2_h145_a002_d20_margin01` | C | 2 | 1.45 | 20 | 32 | 0.02 | 0.01 | 0.5 | 放宽 draft 到 20，目标把 MATH 从 0.68 拉到 0.70 |
| `c_k2_h145_a002_d16_margin005` | C | 2 | 1.45 | 16 | 32 | 0.02 | 0.005 | 0.5 | 降低 margin，让 small path 多切换，目标补 StrategyQA/MATH |
| `c_k2_h145_a003_d16_margin01` | C | 2 | 1.45 | 16 | 32 | 0.03 | 0.01 | 0.5 | 提高 alpha，验证 length weight 是否能更频繁参与选择 |

### Local Preparation

已在本地准备：

```text
configs/group_ac/a_k2_h145_d20.yaml
configs/group_ac/a_k2_h15_d18.yaml
configs/group_ac/a_k2_h15_d20_t04.yaml
configs/group_ac/c_k2_h145_a002_d20_margin01.yaml
configs/group_ac/c_k2_h145_a002_d16_margin005.yaml
configs/group_ac/c_k2_h145_a003_d16_margin01.yaml
scripts/run_group_ac_k2_refine_sweep.sh
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_k2_refine_sweep.sh > server_run_k2_refine.log 2>&1 &
```

### Decision Rules

1. A 组：若 `a_k2_h15_d18` 保持 MATH >= 0.70 且 macro avg 低于 `a_k2_h15_d20`，下一轮继续在 d18 附近细调。
2. A 组：若 `small_temperature=0.4` 降低 late drop 且不掉准确率，则后续把 temp 作为 A 主线参数。
3. C 组：`c_k2_h145_a002_d20_margin01` 若 MATH >= 0.70 且延迟不明显恶化，保留 d20。
4. C 组：`alpha=0.03` 需要重点看 `length_weight_overrides`，若仍接近 1%，说明 length weight 本身不是当前瓶颈。
5. 本轮所有 C 配置都固定 `k=2`；若仍无法超过 k=1 主线，需要在报告中明确 k=2 已按轻量化多路径方案验证。
6. TruthfulQA 仍只作为 latency / generation-shape 参考。

## Result 20260605: K2 Refine Sweep

### Export

```text
experiments/result_exports/20260604-210303_group_ac_k2_refine_first100
```

完整性检查：

- `registry.csv`: 24 rows
- `metrics.json`: 6 files
- `jsonl`: 24 files
- `summary.txt`: 24 files

### Main Result

| Config | GSM8K | StrategyQA | MATH | Macro Acc | Macro Avg | Macro P90 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.77 | 0.79 | 0.67 | 0.743 | 2.80s | n/a |
| GroupB | 0.80 | 0.73 | 0.67 | 0.733 | 9.48s | 11.57s |
| `groupa_k2_h145_d20` | 0.78 | 0.77 | 0.68 | 0.743 | 3.07s | 4.24s |
| `groupa_k2_h15_d18` | 0.77 | 0.76 | 0.70 | 0.743 | 3.20s | 4.61s |
| `groupa_k2_h15_d20_t04` | 0.78 | 0.74 | 0.70 | 0.740 | 3.21s | 4.61s |
| `groupc_k2_h145_a002_d20_margin01` | 0.77 | 0.78 | 0.68 | 0.743 | 3.07s | 4.46s |
| `groupc_k2_h145_a002_d16_margin005` | 0.78 | 0.77 | 0.67 | 0.740 | 3.06s | 4.49s |
| `groupc_k2_h145_a003_d16_margin01` | 0.77 | 0.79 | 0.69 | 0.750 | 3.00s | 4.39s |

### Interpretation

- 本轮最佳新配置是 `groupc_k2_h145_a003_d16_margin01`，也是当前最好的 C k=2 主线。
- 它比 GroupB 明显更好：macro accuracy 更高，macro average latency 低约 6.49s。
- 相比 baseline，它 macro accuracy 高 0.007，但 macro average latency 仍高约 0.20s；最终目标还没有完全达成。
- A 组本轮没有刷新最优。旧 `groupa_k2_h15_d20` 仍是 A 主线：GSM8K 0.78 / StrategyQA 0.76 / MATH 0.72 / Macro Acc 0.753 / Macro Avg 3.10s。
- `small_temperature=0.4` 不继续：`groupa_k2_h15_d20_t04` 的 StrategyQA 掉到 0.74。
- C 组 `alpha=0.03` 优于 `alpha=0.02`；`length_weight_overrides` 仍然低，但综合准确率/延迟最优。

### Mechanism Notes

| Config | Late Drop / Trigger | Switch / Trigger | Margin / Trigger | Length / Verify |
|---|---:|---:|---:|---:|
| `groupa_k2_h145_d20` | 0.218 | 0.210 | 0.000 | 0.000 |
| `groupa_k2_h15_d18` | 0.216 | 0.201 | 0.000 | 0.000 |
| `groupa_k2_h15_d20_t04` | 0.213 | 0.224 | 0.000 | 0.000 |
| `groupc_k2_h145_a002_d20_margin01` | 0.223 | 0.065 | 0.133 | 0.009 |
| `groupc_k2_h145_a002_d16_margin005` | 0.197 | 0.071 | 0.120 | 0.019 |
| `groupc_k2_h145_a003_d16_margin01` | 0.194 | 0.066 | 0.121 | 0.012 |

TruthfulQA 仍有 96-97/100 个样本出现 follow-up Q/A 形态，继续不纳入自动 accuracy 结论。

## Planned Run 20260605: K2 Overnight Mix Sweep

### Goal

今晚跑 10 组，全部固定 `k=2`。6 组围绕当前最优点继续优化，4 组相对发散探索。

当前主线：

- Group A: `a_k2_h15_d20`，优势是 MATH 0.72 / Macro Acc 0.753，问题是 StrategyQA 0.76。
- Group C: `c_k2_h145_a003_d16_margin01`，优势是 StrategyQA 0.79 / Macro Avg 3.00s，问题是 MATH 0.69。

### Configs To Run

| Config | Group | k | entropy | draft | fallback | alpha | margin | Type | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `a_k2_h153_d20` | A | 2 | 1.53 | 20 | 32 | - | 0 | refine | 在 h1.5/h1.55 中间找点，尝试补 StrategyQA 且保 MATH |
| `a_k2_h155_d18` | A | 2 | 1.55 | 18 | 32 | - | 0 | refine | 用更短 draft 抵消 h1.55 的延迟，争取 StrategyQA 提升 |
| `a_k2_h155_d20_margin002` | A | 2 | 1.55 | 20 | 32 | - | 0.002 | refine | 轻微 margin，过滤低置信 switch，避免 0.005 太重 |
| `c_k2_h145_a003_d18_margin01` | C | 2 | 1.45 | 18 | 32 | 0.03 | 0.01 | refine | 当前 C 最优基础上加 draft，冲 MATH 0.70 |
| `c_k2_h145_a003_d16_margin005` | C | 2 | 1.45 | 16 | 32 | 0.03 | 0.005 | refine | 降低 margin，让小模型多一点探索机会 |
| `c_k2_h145_a004_d16_margin01` | C | 2 | 1.45 | 16 | 32 | 0.04 | 0.01 | refine | 继续验证 alpha 增大是否改善 C 的选择质量 |
| `a_k2_h16_d20` | A | 2 | 1.60 | 20 | 32 | - | 0 | explore | 冲 StrategyQA，上轮 h16 对 StrategyQA 有帮助 |
| `a_k2_h15_d20_f24` | A | 2 | 1.50 | 20 | 24 | - | 0 | explore | 缩短 fallback，专门看能不能压 MATH/StrategyQA 延迟 |
| `c_k2_h145_a003_d14_margin01` | C | 2 | 1.45 | 14 | 32 | 0.03 | 0.01 | explore | 更激进压 C 延迟，看准确率底线 |
| `c_k2_h15_a003_d16_margin01` | C | 2 | 1.50 | 16 | 32 | 0.03 | 0.01 | explore | 减少触发，验证 C 能否更快且保持 StrategyQA |

### Local Preparation

已在本地准备：

```text
configs/group_ac/a_k2_h153_d20.yaml
configs/group_ac/a_k2_h155_d18.yaml
configs/group_ac/a_k2_h155_d20_margin002.yaml
configs/group_ac/c_k2_h145_a003_d18_margin01.yaml
configs/group_ac/c_k2_h145_a003_d16_margin005.yaml
configs/group_ac/c_k2_h145_a004_d16_margin01.yaml
configs/group_ac/a_k2_h16_d20.yaml
configs/group_ac/a_k2_h15_d20_f24.yaml
configs/group_ac/c_k2_h145_a003_d14_margin01.yaml
configs/group_ac/c_k2_h15_a003_d16_margin01.yaml
scripts/run_group_ac_k2_overnight_mix_sweep.sh
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_k2_overnight_mix_sweep.sh > server_run_k2_overnight_mix.log 2>&1 &
```

### Decision Rules

1. C 组优先看 `c_k2_h145_a003_d18_margin01` 是否把 MATH 推到 0.70，且 Macro Avg 不超过 3.10s。
2. C 组若 `c_k2_h145_a004_d16_margin01` 继续提升 MATH/StrategyQA，下一轮把 alpha 0.04 设为 C k=2 主线。
3. A 组优先看 `a_k2_h153_d20` 是否把 StrategyQA 提到 0.77/0.78，同时保住 MATH 0.71/0.72。
4. A 组若 `a_k2_h15_d20_f24` 准确率不崩，则继续沿 fallback 24 压延迟。
5. 若 C d14 accuracy 明显下降，则不再继续压 draft；说明 C k=2 的 draft 下限约在 16。
6. TruthfulQA 仍只作为 latency / generation-shape 参考，不纳入最终 accuracy 结论。

## Result 20260605: K2 Overnight Mix Sweep

### Export

```text
experiments/result_exports/20260605-020438_group_ac_k2_overnight_mix_first100
```

完整性检查：

- `registry.csv`: 40 rows
- `metrics.json`: 10 files
- `jsonl`: 40 files
- `summary.txt`: 40 files

### Main Result

| Config | GSM8K | StrategyQA | MATH | Macro Acc | Macro Avg | Macro P90 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.77 | 0.79 | 0.67 | 0.743 | 2.80s | n/a |
| GroupB | 0.80 | 0.73 | 0.67 | 0.733 | 9.48s | 11.57s |
| `groupa_k2_h153_d20` | 0.78 | 0.77 | 0.70 | 0.750 | 3.03s | 4.14s |
| `groupa_k2_h155_d18` | 0.78 | 0.77 | 0.72 | 0.757 | 3.18s | 4.65s |
| `groupa_k2_h155_d20_margin002` | 0.77 | 0.78 | 0.70 | 0.750 | 3.01s | 4.19s |
| `groupa_k2_h15_d20_f24` | 0.78 | 0.76 | 0.65 | 0.730 | 3.15s | 5.06s |
| `groupa_k2_h16_d20` | 0.78 | 0.78 | 0.71 | 0.757 | 3.16s | 4.60s |
| `groupc_k2_h145_a003_d14_margin01` | 0.77 | 0.78 | 0.70 | 0.750 | 3.14s | 4.87s |
| `groupc_k2_h145_a003_d16_margin005` | 0.78 | 0.79 | 0.71 | 0.760 | 2.99s | 4.19s |
| `groupc_k2_h145_a003_d18_margin01` | 0.78 | 0.79 | 0.68 | 0.750 | 3.05s | 4.37s |
| `groupc_k2_h145_a004_d16_margin01` | 0.77 | 0.79 | 0.68 | 0.747 | 3.06s | 4.19s |
| `groupc_k2_h15_a003_d16_margin01` | 0.78 | 0.79 | 0.68 | 0.750 | 3.05s | 4.43s |

### Interpretation

- 本轮最强配置是 `groupc_k2_h145_a003_d16_margin005`，也是目前整体最强候选。
- 它比 GroupB 明显更好：Macro Acc 0.760 > 0.733，Macro Avg 2.99s << 9.48s。
- 它比 baseline 准确率更高：Macro Acc 0.760 > 0.743；但延迟仍略慢：2.99s > 2.80s。
- `c_k2_h145_a003_d16_margin005` 相比上一轮 C k=2 主线 `c_k2_h145_a003_d16_margin01`，MATH 从 0.69 提到 0.71，Macro P90 从 4.39s 降到 4.19s。
- A 组也有进展：`a_k2_h16_d20` 和 `a_k2_h155_d18` 都达到 Macro Acc 0.757；其中 `a_k2_h16_d20` 更均衡，StrategyQA 0.78，MATH 0.71。
- A 的 `a_k2_h155_d20_margin002` 是低延迟备选，Macro Avg 3.01s，但 MATH 只有 0.70。

### Mechanism Notes

| Config | Late Drop / Trigger | Switch / Trigger | Margin / Trigger | Length / Verify |
|---|---:|---:|---:|---:|
| `groupa_k2_h153_d20` | 0.207 | 0.220 | 0.000 | 0.000 |
| `groupa_k2_h155_d18` | 0.218 | 0.214 | 0.000 | 0.000 |
| `groupa_k2_h155_d20_margin002` | 0.223 | 0.077 | 0.135 | 0.000 |
| `groupa_k2_h15_d20_f24` | 0.203 | 0.216 | 0.000 | 0.000 |
| `groupa_k2_h16_d20` | 0.228 | 0.194 | 0.000 | 0.000 |
| `groupc_k2_h145_a003_d14_margin01` | 0.175 | 0.066 | 0.123 | 0.018 |
| `groupc_k2_h145_a003_d16_margin005` | 0.198 | 0.069 | 0.120 | 0.012 |
| `groupc_k2_h145_a003_d18_margin01` | 0.183 | 0.058 | 0.139 | 0.008 |
| `groupc_k2_h145_a004_d16_margin01` | 0.189 | 0.066 | 0.125 | 0.018 |
| `groupc_k2_h15_a003_d16_margin01` | 0.187 | 0.072 | 0.114 | 0.021 |

TruthfulQA 仍有 95-98/100 个样本出现 follow-up Q/A 形态，继续不纳入自动 accuracy 结论。

### Decision

Keep:

- Group C 主线改为 `c_k2_h145_a003_d16_margin005`。
- Group A 主线改为 `a_k2_h16_d20`。
- Group A 低延迟备选保留 `a_k2_h155_d20_margin002`。

Drop:

- `a_k2_h15_d20_f24`：fallback 24 导致 MATH 掉到 0.65。
- `c_k2_h145_a004_d16_margin01`：alpha 0.04 没有收益。
- `c_k2_h145_a003_d18_margin01`：draft 18 没把 MATH 拉起来。
- `c_k2_h145_a003_d14_margin01`：MATH 到 0.70，但延迟/P90 变差，draft 14 不划算。

## Planned Run 20260605: Second100 Validation + First100 Explore

### Goal

本轮给 10 个参数空间：

1. 3 个配置跑 `start=100, end=199`，验证当前最好的三个参数在 second100 上是否稳定。
2. 7 个配置继续跑 first100，在当前最好边界附近探索更强参数。

注意：second100 验证结果不能和 first100 探索结果直接混比；second100 主要看 finalist 泛化是否崩。

### Configs To Run

| Config | Group | Range | k | entropy | draft | fallback | alpha | margin | Type | Purpose |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `c_k2_h145_a003_d16_margin005_second100` | C | 100-199 | 2 | 1.45 | 16 | 32 | 0.03 | 0.005 | validate | 验证当前最强 C k=2 是否在新 100 题上稳定 |
| `a_k2_h16_d20_second100` | A | 100-199 | 2 | 1.60 | 20 | 32 | - | 0 | validate | 验证当前最均衡 A 是否稳定 |
| `a_k2_h155_d20_margin002_second100` | A | 100-199 | 2 | 1.55 | 20 | 32 | - | 0.002 | validate | 验证 A 低延迟备选是否稳定 |
| `c_k2_h145_a003_d16_margin002` | C | 0-99 | 2 | 1.45 | 16 | 32 | 0.03 | 0.002 | explore | 继续降低 margin，让小模型多探索，测试是否还能补 MATH |
| `c_k2_h145_a003_d16_margin0075` | C | 0-99 | 2 | 1.45 | 16 | 32 | 0.03 | 0.0075 | explore | 在 0.005/0.01 中间找稳定点 |
| `c_k2_h1425_a003_d16_margin005` | C | 0-99 | 2 | 1.425 | 16 | 32 | 0.03 | 0.005 | explore | 略降低阈值，增加触发，测试准确率上限 |
| `c_k2_h1475_a003_d16_margin005` | C | 0-99 | 2 | 1.475 | 16 | 32 | 0.03 | 0.005 | explore | 略提高阈值，减少触发，测试能否压延迟 |
| `c_k2_h145_a0035_d16_margin005` | C | 0-99 | 2 | 1.45 | 16 | 32 | 0.035 | 0.005 | explore | 在 alpha 0.03/0.04 中间找更优选择质量 |
| `a_k2_h16_d18` | A | 0-99 | 2 | 1.60 | 18 | 32 | - | 0 | explore | 在 A h16 主线上缩短 draft，争取压延迟并保 MATH |
| `a_k2_h16_d20_margin001` | A | 0-99 | 2 | 1.60 | 20 | 32 | - | 0.001 | explore | 给 A h16 主线加极轻 margin，测试能否保准确率并减少无效 switch |

### Local Preparation

需要在本地准备：

```text
configs/group_ac/c_k2_h145_a003_d16_margin005_second100.yaml
configs/group_ac/a_k2_h16_d20_second100.yaml
configs/group_ac/a_k2_h155_d20_margin002_second100.yaml
configs/group_ac/c_k2_h145_a003_d16_margin002.yaml
configs/group_ac/c_k2_h145_a003_d16_margin0075.yaml
configs/group_ac/c_k2_h1425_a003_d16_margin005.yaml
configs/group_ac/c_k2_h1475_a003_d16_margin005.yaml
configs/group_ac/c_k2_h145_a0035_d16_margin005.yaml
configs/group_ac/a_k2_h16_d18.yaml
configs/group_ac/a_k2_h16_d20_margin001.yaml
scripts/run_group_ac_k2_validate_explore_sweep.sh
```

服务器运行命令：

```bash
nohup bash scripts/run_group_ac_k2_validate_explore_sweep.sh > server_run_k2_validate_explore.log 2>&1 &
```

### Decision Rules

1. second100 上若 `c_k2_h145_a003_d16_margin005_second100` 仍保持 C 组优势，下一步将它作为 finalist。
2. second100 上若 A 两个候选都掉分，A 组回看 first100 排名，不急着定 finalist。
3. first100 探索中若 C `margin002` 提升准确率但延迟变差，下一轮考虑 dataset-specific margin。
4. first100 探索中若 C `h1475` 保持 0.76 macro 且更快，优先用它压 baseline latency。
5. first100 探索中若 A `h16_d18` 保住 MATH >= 0.70 且 Macro Avg 下降，下一轮继续缩 draft / 加轻 margin。
6. TruthfulQA 仍只作为 latency / generation-shape 参考，不纳入最终 accuracy 结论。
