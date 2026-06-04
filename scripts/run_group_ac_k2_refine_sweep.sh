#!/usr/bin/env bash
set -euo pipefail

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-a_k2_h145_d20 a_k2_h15_d18 a_k2_h15_d20_t04 c_k2_h145_a002_d20_margin01 c_k2_h145_a002_d16_margin005 c_k2_h145_a003_d16_margin01}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_k2_refine_first100}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

bash scripts/run_group_ac_experiments.sh
