#!/usr/bin/env bash
set -euo pipefail

export CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
export RUN_SMOKE="${RUN_SMOKE:-0}"
export RUN_CONFIGS="${RUN_CONFIGS:-e_tot_bfs_first100}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_e_tot_bfs_first100}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

bash scripts/run_group_ac_experiments.sh
