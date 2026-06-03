#!/usr/bin/env bash
set -euo pipefail

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-a_k2_h15_d24 a_k2_h16_d24 a_k2_h15_d20 a_k2_h16_d32 a_k2_h15_d24_margin c_k1_h15_a002 c_k1_h14_a0015 c_k1_h14_a002_margin c_k1_h15_a0015 c_k1_h15_a002_margin}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_overnight_margin_first100}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

bash scripts/run_group_ac_experiments.sh
