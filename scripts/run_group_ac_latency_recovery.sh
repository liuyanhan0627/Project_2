#!/usr/bin/env bash
set -euo pipefail

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-a_k2_h14_d32 a_k2_h14_d24 a_k2_h15_d32 c_k1_a002 c_k1_a003 c_k1_h14_a002}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_latency_recovery_first100}"

bash scripts/run_group_ac_experiments.sh
