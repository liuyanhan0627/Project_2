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
