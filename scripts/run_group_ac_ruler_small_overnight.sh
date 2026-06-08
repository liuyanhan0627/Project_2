#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RULER_NIAH_FILE:-}" ]]; then
  python3 scripts/prepare_ruler_small.py
fi

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-ruler_baseline_groupb a_ruler_k2_h16_d20_margin001 a_ruler_k2_h16_d20 a_ruler_k2_h155_d20_margin002 c_ruler_k2_h145_a003_d16_margin005 d_ruler_cautious_first100 a_ruler_k2_h165_d18_margin001 a_ruler_k2_h16_d16_margin001 c_ruler_k2_h1475_a0025_d16_margin005 c_ruler_k2_h145_a0025_d14_margin005}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_ruler_mixed_overnight}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

bash scripts/run_group_ac_experiments.sh
