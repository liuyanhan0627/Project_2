# Group A/C Tuning Log

Last updated: 2026-06-03

本文件专门记录 Group A 和 Group C 的调参过程。后续每一轮服务器实验、结果回传、本地分析和下一轮参数选择，都继续追加到这里。

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
