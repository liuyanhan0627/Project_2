# Llama 大小模型协作推理实验方案

## 1. 实验目标

本实验参考两篇论文的思想：

- Cautious Next Token Prediction (CNTP)：当模型不确定时，探索多个短路径，并以标点符号作为短路径终止信号。
- SpecEdge：小模型负责 draft，大模型负责 verify，大小模型之间只交换 token / 文本，不交换隐藏状态。

实验目标是比较以下两种推理增强方案在相同任务上的表现：

1. 大模型犹豫时，让小模型异步生成短路径 draft；大模型不中断 decoding，继续生成 fallback path，并在 draft 返回后用自身 logprob / PPL 切换到最优路径。
2. 大模型在思维链中加入反思，允许大模型重新开始。

核心比较指标：

- token 消耗
- 推理时间
- 准确率
- 方案 A 中小模型 draft 的接受率与在线切换收益
- 方案 B 中反思重启的有效性

## 1.1 代码目录约定

当前实验实现命名为 **ASPS (Asynchronous Speculative Path Switching)**。

| 目录 | 用途 |
|---|---|
| `ASPS/` | 本实验代码目录，由 CNTP 官方源码改造而来 |
| `CNTP_reference/` | CNTP 官方 GitHub 仓库的干净对照副本：`https://github.com/wyzjack/CNTP.git` |

服务器运行路径需要从旧路径：

```bash
/root/autodl-tmp/Project_2/CNTP/experiments/SelfEval-Guided-Decoding
```

改为新路径：

```bash
/root/autodl-tmp/Project_2/ASPS/experiments/SelfEval-Guided-Decoding
```

后续所有 smoke / first-round / full-run 命令都以 `ASPS/experiments/SelfEval-Guided-Decoding` 作为实验入口。`CNTP_reference/` 只用于和原论文代码做 diff / 对照，不参与实验运行。

## 2. 模型与硬件设置

### 2.1 模型选择

第一轮实验建议使用 Llama 系列模型：

| 角色 | 模型 | 说明 |
|---|---|---|
| 大模型 Target | Llama-3.1-8B-Instruct | 负责主推理、验证、最终答案 |
| 小模型 Draft | Llama-3.2-1B-Instruct 或 TinyLlama-1.1B-Chat | 负责生成候选短路径 |

如果算力充足，可增加一组更强模型：

| 角色 | 模型 | 说明 |
|---|---|---|
| 大模型 Target | Llama-3.1-70B-Instruct | 更强 target model |
| 小模型 Draft | Llama-3.1-8B-Instruct | 更强 draft model |

### 2.2 硬件设置

理想设置：

| 角色 | 硬件 |
|---|---|
| 大模型 Server | A100 |
| 小模型 Edge | RTX 4090 / RTX 3090 / A10 / L4 |

如果只有两张 A100：

- 大模型运行在一张 A100 上。
- 小模型运行在另一张 A100 上。
- 小模型侧需要进行性能限制或延迟模拟，避免高估 edge draft 的性能收益。

推荐的限速方式：

```text
target_draft_latency = generated_tokens / target_edge_tokens_per_second
sleep_time = max(0, target_draft_latency - actual_draft_latency)
```

可额外模拟网络 RTT：

| RTT 场景 | 数值 |
|---|---:|
| 低延迟 | 15 ms |
| 中等延迟 | 40 ms |
| 高延迟 | 65 ms |

## 3. 数据集

第一轮实验统一使用 CNTP 主实验涉及的数据集，每个数据集先取 100 条，先观察跨任务趋势：

| 数据集 | 数量 | 目的 |
|---|---:|---|
| GSM8K | 100 | 数学文字题，便于自动判分 |
| StrategyQA | 100 | 常识推理，便于观察推理路径差异 |
| MATH | 100 | 更难数学推理，观察反思重启是否更有优势 |
| TruthfulQA | 100 | 开放问答与幻觉控制，保存生成结果，按 CNTP 口径使用外部 judge |
| MMVet | 100 | 多模态开放问答，验证视觉输入下的推理切换趋势 |
| MathVista_MINI | 100 | 多模态数学推理，验证图文数学题上的稳定性 |

实现口径：

- GSM8K、StrategyQA、MATH、TruthfulQA 走 `SelfEval-Guided-Decoding` 文本 runner。
- MMVet、MathVista_MINI 走 CNTP 原来的 `VLMEvalKit` VLM runner。
- MATH 参考 CNTP 的 `problem` / `solution` 字段处理和 boxed answer 抽取；本实验为保证四组同题，使用 `start 0/end 99` 固定截取 100 条。
- TruthfulQA 参考 CNTP 的 `TruthfulQA.csv` 读取和 QA prompt；本实验为保证四组同题，使用 `start 0/end 99` 固定截取 100 条。当前 Group A/C/B 文本 runner 只保存生成和运行指标，不直接产出 TruthfulQA judge 分数。
- MMVet、MathVista_MINI 通过 `VLMEvalKit/run.py --limit 100` 限制样本数，结果文件名带 `_n100`，避免和全量结果混淆。

## 4. 实验组设计

### 4.1 Group 0：Baseline，大模型普通 CoT

大模型单独完成推理：

```text
Question -> Big Model CoT -> Final Answer
```

用途：

- 得到基础准确率。
- 得到基础 token 消耗和时间。
- 作为方案 A 和方案 B 的对照。

建议参数：

| 参数 | 值 |
|---|---|
| temperature | 0 |
| max output tokens | 根据数据集设置 |
| final answer format | 所有实验组保持一致 |

### 4.2 Group A：异步小模型短路径 draft，大模型在线 PPL verify

该组结合 CNTP 和 SpecEdge 的思想：

- 参考 CNTP：只在大模型不确定时探索多个短路径，短路径遇到标点符号停止。
- 参考 SpecEdge：小模型负责 draft，大模型负责 verify，二者可以并行工作。
- 短路径不使用固定长度作为主要终止条件，而是遇到标点符号停止。
- 大模型 verify 不使用自然语言结构化评分，而是在 decoding loop 中计算候选路径的条件 log probability / perplexity。
- 小模型 draft 时，大模型不等待，而是继续 decoding 生成 fallback path，避免空转。

整体流程：

```text
Question
-> 大模型正常 decoding，并维护当前 prefix 与 KV cache
-> 每一步计算大模型 next-token entropy
-> 如果 entropy 低：大模型正常生成下一个 token
-> 如果 entropy 高：触发小模型异步生成 k 条 draft path
-> 小模型 draft 时，大模型继续 decoding，生成 fallback path
-> 小模型每条 draft path 遇到标点符号停止
-> 如果 draft 在回滚窗口内返回，大模型计算每条 draft path 在触发 prefix 下的条件 PPL
-> 同时计算大模型 fallback path 在同一触发 prefix 下的条件 PPL
-> 选择 PPL 最低且通过基本约束的 path
-> 如果最优 path 不是当前 fallback path，则切换 prefix 与 KV cache
-> 如果 draft 超过回滚窗口仍未返回，则丢弃该轮 draft，沿 fallback 继续
-> 继续 decoding 直到最终答案
```

#### 4.2.1 犹豫判断

第一轮优先使用 CNTP 风格的 entropy 触发：

```text
H(prefix) = - sum_w P_big(w | prefix) log P_big(w | prefix)
```

触发规则：

```text
如果 H(prefix) > H_threshold，则触发小模型 draft
```

阈值建议：

- 使用前 20 道题作为 calibration set。
- 记录大模型正常 decoding 时每一步的 entropy。
- 取 entropy 的 P75 或 P80 作为 `H_threshold`。

本实验不使用大模型自然语言自评 confidence 作为触发机制，因为自评需要中断 decoding 过程，无法体现在线路径选择的核心思想。若推理框架无法暴露 logits / logprobs，则该框架不适合作为 Group A 主实验实现。

#### 4.2.2 小模型 draft

小模型在触发时刻的 `trigger_prefix` 后异步生成多个候选短路径。

建议参数：

| 参数 | 值 |
|---|---:|
| draft candidates k | 3 |
| max draft tokens | 128 |
| temperature | 0.7 |
| top_p | 0.9 |

标点终止集合参考 CNTP：

```text
{ ".", "?", "!", ":", ";", ")", "]", "}", "\n" }
```

如果后续使用中文任务，可额外加入：

```text
{ "。", "？", "！", "：", "；" }
```

`max draft tokens = 128` 仅作为兜底条件，防止小模型长时间不生成标点。

#### 4.2.3 大模型并行 fallback decoding

小模型 draft 期间，大模型不等待，继续从 `trigger_prefix` 自己生成 fallback path。

fallback path 的终止条件与小模型一致：

```text
遇到标点符号停止，或达到 max fallback tokens
```

建议参数：

| 参数 | 值 |
|---|---:|
| max fallback tokens | 128 |
| temperature | 0 |

该 fallback path 既是不中断输出的保障，也是候选路径之一。

#### 4.2.4 大模型在线 PPL verify

当小模型 draft 返回后，大模型不生成自然语言评分，而是对所有候选路径进行目标模型打分。

大模型 verify 必须采用 batched candidate verification，而不是逐条路径串行评分。这一点是方案 A 的核心系统假设：所有候选路径的 token 应当被组织成一个 batch / 矩阵，通过一次大模型并行 forward 完成打分。

候选集合：

```text
Candidates = {small_draft_1, ..., small_draft_k, big_fallback}
```

矩阵化输入形式：

```text
[
  trigger_prefix + small_draft_1,
  trigger_prefix + small_draft_2,
  ...,
  trigger_prefix + small_draft_k,
  trigger_prefix + big_fallback
]
```

由于所有候选共享同一个 `trigger_prefix`，理想实现应复用大模型在 `trigger_prefix` 处的 KV cache，只对候选 suffix tokens 做 batched forward。不同候选路径长度不同，因此需要 padding 与 attention mask：

```text
candidate_suffix_tokens:
[
  [a1, a2, a3, PAD, PAD],
  [b1, b2, b3, b4, PAD],
  [c1, c2, PAD, PAD, PAD],
  [d1, d2, d3, d4, d5]
]
```

attention / loss 计算要求：

```text
- PAD token 不参与 logprob / PPL 计算
- 每条候选只能 attend 到 trigger_prefix 和自己路径内的前序 token
- 不同候选路径之间不能互相 attend
- PPL 只在候选 suffix tokens 上计算，不重复计算 trigger_prefix
```

对每条候选路径 `path_i = (x_1, ..., x_m)`，计算：

```text
score(path_i) = (1 / m) * sum_t log P_big(x_t | trigger_prefix, x_<t)
PPL(path_i) = exp(-score(path_i))
```

选择规则：

```text
best_path = argmin_i PPL(path_i)
```

如果 `best_path` 是小模型 draft，则将当前 decoding 状态切换到：

```text
trigger_prefix + best_path
```

如果 `best_path` 是大模型 fallback，则保持当前 decoding 状态继续。

串行 path scoring 只能作为 debug baseline 或 ablation，不能作为主实验实现。

#### 4.2.5 No-wait 切换策略

为了避免大模型等待小模型，同时控制回滚复杂度，第一轮采用“标点片段级 no-wait 切换”：

1. 大模型触发小模型 draft 时记录 `trigger_prefix`。
2. 大模型立即继续生成 fallback path，不等待小模型。
3. 小模型也从 `trigger_prefix` 开始生成 `k` 条 draft path，每条到标点停止。
4. 大模型遇到第一个标点时，得到 fallback punctuation span。
5. 如果此时小模型 draft 已返回，则立刻进行 PPL verify。
6. 如果此时小模型 draft 未返回，大模型继续 decoding，不阻塞。
7. 小模型 draft 在回滚窗口内返回时，大模型对 draft path 与 fallback punctuation span 计算 PPL。
8. 如果小模型路径更优，则回滚到 `trigger_prefix + best_path`，丢弃从 `trigger_prefix` 后已经生成但未正式接受的 fallback tokens。
9. 如果 fallback 更优，则保持当前状态继续。
10. 如果小模型 draft 超过回滚窗口仍未返回，则丢弃该轮 draft，不再切换。

回滚窗口建议第一轮设置为：

```text
rollback_window = 1 punctuation span
```

也就是只允许在当前触发点后的第一个标点片段内切换。这样既保留“在线切换”的思想，又避免大模型已经生成很远后再大规模回滚。

该策略的关键是：大模型始终不等待小模型。小模型 draft 只是在后台提供可能更优的候选路径；如果它来得足够及时，就参与 PPL verify；如果来得太晚，就被丢弃。

第二轮可实现更激进的 token-level switch：

- 小模型 draft 一返回，大模型立即打分并切换。
- 如果大模型已经生成超过一个片段，则需要更复杂的 KV cache 回滚和重放。

#### 4.2.6 基本约束

PPL 最低不一定等于逻辑正确，因此第一轮加入轻量约束，但不打断 decoding：

```text
- path 不得为空
- path 必须以允许的标点集合之一结束
- path 不得包含 final answer 标记，除非大模型已经进入最终答案阶段
- path 长度不得超过 max draft tokens
```

不使用自然语言评分作为主方案，避免中断 decoding。

### 4.3 Group C：异步小模型 draft + 长度加权路径选择

Group C 与 Group A 使用相同的大小模型工作模式：

```text
大模型继续生成 fallback path
小模型异步生成 k 条短路径 draft
大模型对 fallback + small drafts 做 batched PPL verify
在 avg logprob 选择分数上加入路径长度权重
选择 weighted score 最高的路径
```

Group C 的目标不是改变候选生成方式，而是改变候选选择偏好：在大模型条件 PPL 接近时，更偏向一次接受更多 token 的路径，从而减少后续 verify / fallback 轮次，降低延迟。

第一轮采用长度越长权重越高：

```text
base_score_i = avg_logprob_i
length_weight_i = len(path_i) / max_j len(path_j)
weighted_score_i = base_score_i + alpha * length_weight_i
selected_path = argmax_i weighted_score_i
```

其中 `alpha = 0` 时 Group C 退化为 Group A。第一轮推荐 `alpha = 0.05`，`length_weight_mode = longer`。

该组仍然遵守 Group A 的约束：

- 小模型只提供候选路径，不直接决定最终答案。
- 大模型 verify 只使用自身 logits / logprobs / PPL，不使用自然语言评分，也不能看到标准答案。
- 所有候选必须基于相同的 `trigger_prefix` 进行 batched verification。
- 长度权重只参与路径选择，不改变候选 PPL 的记录。

### 4.4 Group B：大模型反思并允许重启

该组只使用大模型，不调用小模型。

整体流程：

```text
Question
-> 大模型完整 CoT 解答
-> 大模型短 JSON 反思检查，只判断是否需要重启
-> 如果发现明确错误，从头重启一次
-> Final Answer
```

Group B 中反思阶段只作为 verifier / control signal，不生成、改写或覆盖最终答案。这样可以避免 reflection 文本污染 Python 执行结果或最终答案解析。

为了公平，第一轮限制：

| 参数 | 值 |
|---|---:|
| max restart | 1 |
| temperature | 0 |
| reflection format | short JSON verifier only |

反思输出建议：

```json
{
  "has_error": true,
  "error_type": "calculation | logic | assumption | format | none",
  "should_restart": true,
  "reflection": "brief reason"
}
```

处理规则：

- 如果 `should_restart = false`，最终答案直接使用第一轮完整解答。
- 如果 `should_restart = true`，大模型丢弃第一轮推理，从原题 prompt 重新解一次，最多重启一次。
- reflection JSON 不包含 `final_answer`，也不允许作为最终答案或最终解法参与评测。

## 5. 记录指标

### 5.1 主指标

| Method | Accuracy | Big Tokens | Small Tokens | Total Tokens | Avg Time | P90 Time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | | | | | | |
| Group A | | | | | | |
| Group C | | | | | | |
| Group B | | | | | | |

### 5.2 成本指标

如果使用 API 或需要估算成本，记录：

| Method | Big Token Cost | Small Token Cost | Total Estimated Cost |
|---|---:|---:|---:|
| Baseline | | | |
| Group A | | | |
| Group C | | | |
| Group B | | | |

注意：Group A 的总 token 可能高于 Baseline，但由于小模型 token 更便宜，总成本可能仍然更低。

### 5.3 Group A 额外指标

| 指标 | 含义 |
|---|---|
| entropy trigger rate | 触发小模型 draft 的题目比例 |
| avg draft calls | 每题平均 draft 次数 |
| avg draft path length | 每条 draft path 的平均 token 数 |
| fallback path length | 大模型并行 fallback path 的平均 token 数 |
| small draft win rate | PPL 最低路径来自小模型的比例 |
| fallback win rate | PPL 最低路径来自大模型 fallback 的比例 |
| switch rate | 最终从 fallback 切换到小模型 draft 的比例 |
| late draft drop rate | 小模型 draft 超过回滚窗口而被丢弃的比例 |
| avg candidate PPL | 小模型候选与 fallback 的平均 PPL |
| accepted tokens per verify | 每次 PPL verify 后正式接受的 token 数 |
| wasted big tokens | 因选择小模型路径而丢弃的大模型 fallback token 数 |
| accuracy on triggered cases | 触发犹豫样本上的准确率 |

### 5.4 Group C 额外指标

| 指标 | 含义 |
|---|---|
| length weight alpha | 长度权重强度 |
| length weight mode | 长度权重模式，第一轮为 `longer` |
| candidate base scores | 未加权的大模型条件 avg logprob |
| candidate length weights | 每条候选路径的长度权重 |
| candidate weighted scores | 加权后的最终选择分数 |
| length weight overrides | 加权后选择结果不同于纯 PPL 选择的次数 |
| accepted tokens per verify | 每次 verify 后正式接受的 token 数 |
| latency vs Group A | 与 Group A 相比的平均时间 / P90 时间变化 |

### 5.5 Group B 额外指标

| 指标 | 含义 |
|---|---|
| restart rate | 触发重启的题目比例 |
| wrong-to-right | 重启后由错误变正确的比例 |
| right-to-wrong | 重启后由正确变错误的比例 |
| reflection overhead | 反思额外消耗的 token 和时间 |
| first solution accuracy | 第一轮完整解答的准确率 |
| final accuracy | Group B 最终答案准确率 |
| reflection parse failed | 反思 JSON 解析失败的比例 |
| hit reflection limit | 反思输出达到 max reflection tokens 的比例 |
| restart tokens | 实际重启解答消耗的 token |

## 6. 公平性控制

为保证实验比较公平：

1. 四组使用完全相同的题目集合。
2. 四组最终答案格式保持一致。
3. 四组使用相同的最终答案抽取和正确性判断函数。
4. Group A / Group C 中小模型只生成候选，不直接决定最终答案。
5. Group A / Group C 中大模型 verify 只使用自身 logits / logprobs / PPL，不能使用自然语言评分，也不能看到标准答案。
6. Group C 的长度权重只能使用候选路径长度，不能使用标准答案、自然语言自评或任务标签。
7. Group C 必须同时记录未加权 PPL 分数和加权分数，保证可还原为 Group A 选择。
8. Group B 最多只能重启一次。
9. Group B 的 reflection 只输出 JSON 控制信号，不能输出或覆盖最终答案。
10. Group B 中 `should_restart = false` 时必须保留第一轮解答；只有 `should_restart = true` 时才允许从原题 prompt 重启。
11. 所有 token 分开记录：
   - 大模型 input tokens
   - 大模型 output tokens
   - 小模型 input tokens
   - 小模型 output tokens
12. Group A / Group C 需要额外记录大模型 fallback token，其中被切换丢弃的 token 单独统计。
13. Group A / Group C 的 PPL verify 必须基于相同的 `trigger_prefix`，保证小模型 draft 与大模型 fallback 比较公平。
14. Group A / Group C 的主实验必须使用 batched candidate verification：所有候选路径一次并行 forward 评分；逐条串行评分只能作为 debug 或 ablation。
15. Group A / Group C 的 batched verify 必须复用 `trigger_prefix` 的 KV cache，并只在候选 suffix tokens 上统计 PPL。
16. 如果用 A100 模拟 edge GPU，小模型侧必须加入限速或 RTT。
17. 所有随机方案需要固定 random seed。

## 7. 第一轮推荐配置

第一轮以可落地、低成本为主：

| 项目 | 配置 |
|---|---|
| 大模型 | Llama-3.1-8B-Instruct |
| 小模型 | Llama-3.2-1B-Instruct |
| 文本数据集 | GSM8K 100 + StrategyQA 100 + MATH 100 + TruthfulQA 100 |
| 多模态数据集 | MMVet 100 + MathVista_MINI 100 |
| Group A draft candidates | k = 3 |
| Group A draft stop | 标点符号终止 |
| Group A max draft tokens | 128 |
| Group A fallback stop | 标点符号终止 |
| Group A max fallback tokens | 128 |
| Group A 触发机制 | next-token entropy > H_threshold |
| Group A 路径选择 | 大模型条件 PPL 最低 |
| Group A verify 实现 | batched candidate verification |
| Group A rollback window | 1 punctuation span |
| Group C path choice | 大模型条件 avg logprob + 长度权重 |
| Group C length weight alpha | 0.05 |
| Group C length weight mode | longer |
| Group B max restart | 1 |
| Group B reflection format | short JSON verifier only |
| Group B max reflection tokens | 256 |
| 实验重复次数 | 第一轮 1 次，关键结果 3 次 |

## 8. 预期观察

实验主要回答三个问题：

1. 小模型短路径 draft 是否能提升准确率？
2. 相比大模型反思重启，小模型 draft + 大模型 verify 是否更省 token 和时间？
3. 使用标点符号终止的短路径 draft 是否比固定长度 draft 更稳定？
4. 小模型 draft 时大模型继续 decoding，是否能减少等待时间？
5. 大模型 PPL 在线选择是否足以找到更优推理片段？
6. 小模型 draft 的迟到丢弃率是否会抵消其收益？
7. Group C 的长度权重是否能提高 accepted tokens per verify 并降低平均延迟？

预期结果：

- Group A 的 token 成本可能低于 Group B。
- Group A 的时间优势来自小模型 draft 与大模型 fallback decoding 的并行；即使小模型较慢，也不应让大模型空等。
- Group B 在数学题上可能提升准确率，但 token 和时间开销较大。
- Group B 可能出现 right-to-wrong，即反思把原本正确的答案改错。
- Group A 的关键成功指标是 small draft win rate、switch rate、wasted big tokens 和 accuracy on triggered cases。
- Group C 可能比 Group A 更快，但如果 alpha 过大，可能选择 PPL 较差的长路径导致准确率下降。

## 9. 后续扩展

第一轮完成后，可以继续做以下扩展：

1. 比较不同 entropy 阈值：P70、P75、P80、P90。
2. 比较不同小模型：Llama-3.2-1B、TinyLlama-1.1B、Llama-3.2-3B。
3. 比较不同 draft candidates：k = 1, 3, 5。
4. 比较不同 RTT：15 ms, 40 ms, 65 ms。
5. 比较真实 edge GPU 与限速 A100 的差异。
6. 将六个数据集从 first100 扩到 CNTP 原始规模：GSM8K 1319、StrategyQA 2290、MATH 200、TruthfulQA 817、MMVet 218、MathVista_MINI 1000。
7. 比较路径选择策略：大模型 PPL 最低、avg logprob + 长度权重、随机选择、小模型 PPL 最低、自然语言结构化评分。
8. 比较切换粒度：标点片段级切换与 token-level switch。

## 10. 双 A800 执行策略

当前建议先只租一台双 A800 机器，不急于扩卡。原因是第一阶段最不确定的是 Group A 原型行为，而不是纯算力。

### 10.1 Smoke Test

先在双 A800 上跑小规模验证：

| 项目 | 设置 |
|---|---:|
| GSM8K | 20 题 |
| StrategyQA | 20 题 |
| MATH | 20 题 |
| TruthfulQA | 20 题 |
| MMVet | 可选 20 题，设置 `RUN_VLM_SMOKE=1` |
| MathVista_MINI | 可选 20 题，设置 `RUN_VLM_SMOKE=1` |
| 实验组 | Baseline + Group A + Group C + Group B |
| 预估时间 | 1-3 小时 |

Smoke test 重点检查：

- Group A 的 `entropy_triggers` 是否非零。
- Group A 的 `verify_calls` 是否正常。
- Group A 的 `switches` 是否出现。
- Group A 的 `late_draft_drops` 是否过高。
- Group C 的 `length_weight_overrides` 是否非零。
- Group C 的 `accepted_tokens_per_verify` 是否高于 Group A。
- Group C 的平均时间 / P90 时间是否低于 Group A。
- batched PPL verify 是否稳定。
- 输出格式是否能被 GSM8K / StrategyQA / MATH 评测脚本正确解析。
- TruthfulQA 是否能正常保存开放式回答，后续再接 CNTP 的 judge 评测。
- MMVet / MathVista_MINI 是否能通过 `VLMEvalKit --limit` 正确截断到小样本。

### 10.2 第一轮正式实验

Smoke test 通过后，在同一台双 A800 机器上跑第一轮：

| 项目 | 设置 |
|---|---:|
| GSM8K | 100 题 |
| StrategyQA | 100 题 |
| MATH | 100 题 |
| TruthfulQA | 100 题 |
| MMVet | 100 题 |
| MathVista_MINI | 100 题 |
| 实验组 | Baseline + Group A + Group C + Group B |
| seed | 1 个 |
| 预估纯运行时间 | 6-16 小时 |

建议直接过夜运行。

运行前需要确认数据路径：

```bash
export HF_TOKEN=...
export DATAHOME=/path/to/SelfEval-Guided-Decoding/data
export MATH_FILE=/path/to/MATH/test.jsonl
export TRUTHFULQA_FILE=/path/to/truthfulqa/TruthfulQA.csv
export RUN_VLM=1
```

如果只想先跑文本四个数据集，设置 `RUN_VLM=0`。

### 10.3 何时扩卡

只有当第一轮结果出现明确信号后再扩卡：

| 指标 | 建议阈值 |
|---|---:|
| Group A switch rate | 大于 10%-20% |
| late draft drop rate | 不应明显压倒 switch rate |
| accuracy | 接近或优于 Baseline，最好接近 Group B |
| batched verify | 稳定运行，无频繁 fallback / crash |

如果满足上述条件，再租 4-8 张卡做多 seed / 多阈值 / 多 k 值实验。

扩卡方式：

| 卡数 | 推荐用法 |
|---:|---|
| 2 卡 | 跑一个 Group A 配置，大模型一张，小模型一张 |
| 4 卡 | 同时跑两个 Group A 配置 |
| 8 卡 | 同时跑四个 Group A 配置 |

多卡主要用于并行跑多个 seed、entropy threshold、dataset shard 或 `k` 值，而不是把单个 Group A 配置拆到更多卡上。

### 10.4 推荐执行顺序

1. 双 A800 跑 `GSM8K 20 + StrategyQA 20 + MATH 20 + TruthfulQA 20` smoke test；如需验证 VLM，额外设置 `RUN_VLM_SMOKE=1`。
2. 检查 Group A 日志和指标。
3. 双 A800 跑 `GSM8K 100 + StrategyQA 100 + MATH 100 + TruthfulQA 100 + MMVet 100 + MathVista_MINI 100` 第一轮。
4. 若第一轮有信号，再扩到 4/8 卡跑 3 seeds 与消融实验。
