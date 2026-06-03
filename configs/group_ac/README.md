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
