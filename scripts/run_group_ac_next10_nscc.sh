#!/usr/bin/env bash
set -euo pipefail

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-a_k2_h16_d20_margin0015 a_k2_h16_d18_margin0015 a_k2_h165_d20_margin001 a_k2_h155_d20_margin0015 c_k2_h145_a003_d16_margin006 c_k2_h1475_a003_d16_margin0075 c_k2_h145_a0035_d16_margin0075 c_k2_h145_a0025_d18_margin0075 c_k3_h145_a003_d16_margin0075 c_k2_h15_a003_d16_margin0075}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_next10_nscc_first100}"

bash scripts/run_group_ac_experiments.sh
