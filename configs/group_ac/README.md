# Group A/C First Tuning Sweep

Run order:

1. `baseline_groupb.yaml`: common Baseline and Group B references.
2. `a_safe.yaml`: stable Group A starting point.
3. `a_fast.yaml`: shorter draft/fallback windows for latency.
4. `a_accuracy.yaml`: more candidates and higher trigger threshold.
5. `c_low.yaml`: Group C based on A safe with small log length weight.
6. `c_mid.yaml`: Group C based on A safe with medium log length weight.
7. `c_fast.yaml`: Group C based on A fast with stronger linear length weight.

All configs use the same four text datasets and `start=0, end=99` for a first-100 run.

## Candidate-count sweep

The k sweep keeps the best current Group A and Group C bases fixed, and only changes
`draft_candidates`.

Run command:

```bash
bash scripts/run_group_ac_k_sweep.sh
```

Default configs:

1. `a_fast_k1.yaml`: Group A, based on `a_fast`, `k=1`.
2. `a_fast_k2.yaml`: Group A, based on `a_fast`, `k=2`.
3. `a_fast_k3.yaml`: Group A, based on `a_fast`, `k=3`.
4. `c_fast_k1.yaml`: Group C, based on `c_fast`, `k=1`.
5. `c_fast_k2.yaml`: Group C, based on `c_fast`, `k=2`.
6. `c_fast_k3.yaml`: Group C, based on `c_fast`, `k=3`.

This script sets `RUN_SMOKE=0` and excludes Baseline / Group B by default.

## K2 overnight mix sweep

This sweep keeps `k=2` for both Group A and Group C. Six configs refine the
current best lines (`a_k2_h15_d20` and `c_k2_h145_a003_d16_margin01`), while
four configs explore more aggressive latency or trigger changes. By default it
writes to `outputs/${EXPORT_NAME}` so the export bundle only contains this
sweep.

Run command:

```bash
bash scripts/run_group_ac_k2_overnight_mix_sweep.sh
```

Default configs:

1. `a_k2_h153_d20.yaml`: Group A, `k=2`, entropy `1.53`, draft `20`.
2. `a_k2_h155_d18.yaml`: Group A, `k=2`, entropy `1.55`, draft `18`.
3. `a_k2_h155_d20_margin002.yaml`: Group A, `k=2`, entropy `1.55`, draft `20`, margin `0.002`.
4. `c_k2_h145_a003_d18_margin01.yaml`: Group C, `k=2`, entropy `1.45`, draft `18`, alpha `0.03`, margin `0.01`.
5. `c_k2_h145_a003_d16_margin005.yaml`: Group C, `k=2`, entropy `1.45`, draft `16`, alpha `0.03`, margin `0.005`.
6. `c_k2_h145_a004_d16_margin01.yaml`: Group C, `k=2`, entropy `1.45`, draft `16`, alpha `0.04`, margin `0.01`.
7. `a_k2_h16_d20.yaml`: Group A, `k=2`, entropy `1.6`, draft `20`.
8. `a_k2_h15_d20_f24.yaml`: Group A, `k=2`, entropy `1.5`, draft `20`, fallback `24`.
9. `c_k2_h145_a003_d14_margin01.yaml`: Group C, `k=2`, entropy `1.45`, draft `14`, alpha `0.03`, margin `0.01`.
10. `c_k2_h15_a003_d16_margin01.yaml`: Group C, `k=2`, entropy `1.5`, draft `16`, alpha `0.03`, margin `0.01`.

This script sets `RUN_SMOKE=0` and excludes Baseline / Group B by default.

## K2 validation and explore sweep

This mixed sweep uses ten configs. Three configs validate the best current
finalists on `start=100, end=199`; seven configs continue first-100 exploration
around the strongest Group A and Group C lines. Treat the second-100 configs as
stability checks, not direct rows in the first-100 leaderboard.

Run command:

```bash
bash scripts/run_group_ac_k2_validate_explore_sweep.sh
```

Default configs:

1. `c_k2_h145_a003_d16_margin005_second100.yaml`: Group C, second100, `k=2`, entropy `1.45`, draft `16`, alpha `0.03`, margin `0.005`.
2. `a_k2_h16_d20_second100.yaml`: Group A, second100, `k=2`, entropy `1.6`, draft `20`.
3. `a_k2_h155_d20_margin002_second100.yaml`: Group A, second100, `k=2`, entropy `1.55`, draft `20`, margin `0.002`.
4. `c_k2_h145_a003_d16_margin002.yaml`: Group C, first100, `k=2`, entropy `1.45`, draft `16`, alpha `0.03`, margin `0.002`.
5. `c_k2_h145_a003_d16_margin0075.yaml`: Group C, first100, `k=2`, entropy `1.45`, draft `16`, alpha `0.03`, margin `0.0075`.
6. `c_k2_h1425_a003_d16_margin005.yaml`: Group C, first100, `k=2`, entropy `1.425`, draft `16`, alpha `0.03`, margin `0.005`.
7. `c_k2_h1475_a003_d16_margin005.yaml`: Group C, first100, `k=2`, entropy `1.475`, draft `16`, alpha `0.03`, margin `0.005`.
8. `c_k2_h145_a0035_d16_margin005.yaml`: Group C, first100, `k=2`, entropy `1.45`, draft `16`, alpha `0.035`, margin `0.005`.
9. `a_k2_h16_d18.yaml`: Group A, first100, `k=2`, entropy `1.6`, draft `18`.
10. `a_k2_h16_d20_margin001.yaml`: Group A, first100, `k=2`, entropy `1.6`, draft `20`, margin `0.001`.

This script sets `RUN_SMOKE=0` and excludes Baseline / Group B by default.

## Daytime C-k2 sweep

This sweep keeps Group A on the best low-latency k=2 line, and sets every
Group C config to `k=2` to test lightweight multi-path exploration. Compared
with the earlier `c_fast_k2`, these C configs use shorter drafts, lower
length-weight alpha, and small switch margins.

Run command:

```bash
bash scripts/run_group_ac_daytime_k2c_sweep.sh
```

Default configs:

1. `a_k2_h15_d16.yaml`: Group A, `k=2`, entropy `1.5`, draft `16`.
2. `a_k2_h15_d20_margin005.yaml`: Group A, `k=2`, entropy `1.5`, draft `20`, margin `0.005`.
3. `a_k2_h155_d20.yaml`: Group A, `k=2`, entropy `1.55`, draft `20`.
4. `c_k2_h14_a002_d16_margin005.yaml`: Group C, `k=2`, entropy `1.4`, draft `16`, alpha `0.02`, margin `0.005`.
5. `c_k2_h14_a002_d16_margin01.yaml`: Group C, `k=2`, entropy `1.4`, draft `16`, alpha `0.02`, margin `0.01`.
6. `c_k2_h145_a002_d16_margin01.yaml`: Group C, `k=2`, entropy `1.45`, draft `16`, alpha `0.02`, margin `0.01`.

This script sets `RUN_SMOKE=0`, excludes Baseline / Group B by default, and
writes to `outputs/${EXPORT_NAME}`.

## K2 refine sweep

This sweep keeps `k=2` for both Group A and Group C. Group A refines the current
`a_k2_h15_d20` line with entropy, draft length, and small-model temperature.
Group C continues from `c_k2_h145_a002_d16_margin01` and tests whether longer
drafts, smaller margin, or stronger length weight can recover MATH/StrategyQA.

Run command:

```bash
bash scripts/run_group_ac_k2_refine_sweep.sh
```

Default configs:

1. `a_k2_h145_d20.yaml`: Group A, `k=2`, entropy `1.45`, draft `20`.
2. `a_k2_h15_d18.yaml`: Group A, `k=2`, entropy `1.5`, draft `18`.
3. `a_k2_h15_d20_t04.yaml`: Group A, `k=2`, entropy `1.5`, draft `20`, small temperature `0.4`.
4. `c_k2_h145_a002_d20_margin01.yaml`: Group C, `k=2`, entropy `1.45`, draft `20`, alpha `0.02`, margin `0.01`.
5. `c_k2_h145_a002_d16_margin005.yaml`: Group C, `k=2`, entropy `1.45`, draft `16`, alpha `0.02`, margin `0.005`.
6. `c_k2_h145_a003_d16_margin01.yaml`: Group C, `k=2`, entropy `1.45`, draft `16`, alpha `0.03`, margin `0.01`.

This script sets `RUN_SMOKE=0`, excludes Baseline / Group B by default, and
writes to `outputs/${EXPORT_NAME}`.

## Latency-recovery sweep

This sweep follows the k-sweep result. Group A continues from `a_fast_k2`, and
Group C continues from `c_fast_k1` with safer length weights.

Run command:

```bash
bash scripts/run_group_ac_latency_recovery.sh
```

Default configs:

1. `a_k2_h14_d32.yaml`: Group A, `k=2`, entropy `1.4`, draft `32`.
2. `a_k2_h14_d24.yaml`: Group A, `k=2`, entropy `1.4`, draft `24`.
3. `a_k2_h15_d32.yaml`: Group A, `k=2`, entropy `1.5`, draft `32`.
4. `c_k1_a002.yaml`: Group C, `k=1`, entropy `1.3`, alpha `0.02`.
5. `c_k1_a003.yaml`: Group C, `k=1`, entropy `1.3`, alpha `0.03`.
6. `c_k1_h14_a002.yaml`: Group C, `k=1`, entropy `1.4`, alpha `0.02`.

This script sets `RUN_SMOKE=0` and excludes Baseline / Group B by default.

## Overnight margin sweep

This sweep follows the latency-recovery result. Group A continues from
`a_k2_h15_d32`, Group C continues from `c_k1_h14_a002`, and the margin configs
use `switch_score_margin=0.02`. By default it writes to
`outputs/${EXPORT_NAME}` so the export bundle only contains this sweep.

Run command:

```bash
bash scripts/run_group_ac_overnight_sweep.sh
```

Default configs:

1. `a_k2_h15_d24.yaml`: Group A, `k=2`, entropy `1.5`, draft `24`.
2. `a_k2_h16_d24.yaml`: Group A, `k=2`, entropy `1.6`, draft `24`.
3. `a_k2_h15_d20.yaml`: Group A, `k=2`, entropy `1.5`, draft `20`.
4. `a_k2_h16_d32.yaml`: Group A, `k=2`, entropy `1.6`, draft `32`.
5. `a_k2_h15_d24_margin.yaml`: Group A, `k=2`, entropy `1.5`, draft `24`, margin `0.02`.
6. `c_k1_h15_a002.yaml`: Group C, `k=1`, entropy `1.5`, alpha `0.02`.
7. `c_k1_h14_a0015.yaml`: Group C, `k=1`, entropy `1.4`, alpha `0.015`.
8. `c_k1_h14_a002_margin.yaml`: Group C, `k=1`, entropy `1.4`, alpha `0.02`, margin `0.02`.
9. `c_k1_h15_a0015.yaml`: Group C, `k=1`, entropy `1.5`, alpha `0.015`.
10. `c_k1_h15_a002_margin.yaml`: Group C, `k=1`, entropy `1.5`, alpha `0.02`, margin `0.02`.

This script sets `RUN_SMOKE=0` and excludes Baseline / Group B by default.
