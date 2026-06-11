#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RULER_NIAH_FILE:-}" ]]; then
  python3 scripts/prepare_ruler_small.py
fi

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-a_ruler_k2_h158_d20_margin001 a_ruler_k2_h162_d20_margin001 a_ruler_k2_h16_d18_margin001 a_ruler_k2_h158_d18_margin0015 a_ruler_k2_h155_d18_margin002 a_ruler_k2_h16_d20_f28_margin001 c_ruler_k2_h145_a003_d14_margin005 c_ruler_k2_h145_a0035_d14_margin005 c_ruler_k2_h1425_a003_d14_margin005 c_ruler_k2_h145_a0025_d12_margin005 d_paper_gsm8k_first100 d_paper_strategyqa_first100 d_paper_math_first100 d_paper_truthfulqa_first100 d_paper_ruler_first20}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_paperd_nightly}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

bash scripts/run_group_ac_experiments.sh
