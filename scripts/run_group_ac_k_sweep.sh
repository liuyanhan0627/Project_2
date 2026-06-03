#!/usr/bin/env bash
set -euo pipefail

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-a_fast_k1 a_fast_k2 a_fast_k3 c_fast_k1 c_fast_k2 c_fast_k3}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_k_sweep_first100}"

bash scripts/run_group_ac_experiments.sh
