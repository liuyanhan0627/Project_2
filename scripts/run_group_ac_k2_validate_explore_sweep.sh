#!/usr/bin/env bash
set -euo pipefail

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-c_k2_h145_a003_d16_margin005_second100 a_k2_h16_d20_second100 a_k2_h155_d20_margin002_second100 c_k2_h145_a003_d16_margin002 c_k2_h145_a003_d16_margin0075 c_k2_h1425_a003_d16_margin005 c_k2_h1475_a003_d16_margin005 c_k2_h145_a0035_d16_margin005 a_k2_h16_d18 a_k2_h16_d20_margin001}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_k2_validate_explore_mix}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

bash scripts/run_group_ac_experiments.sh
