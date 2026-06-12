#!/usr/bin/env bash
set -euo pipefail

export PBS_SCRIPT="${PBS_SCRIPT:-scripts/nscc_run_group_d.pbs}"
export RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_group_d_cautious_first100.sh}"
export JOB_NAME="${JOB_NAME:-asps_group_d}"
export NGPUS="${NGPUS:-1}"
export NCPUS="${NCPUS:-16}"
export MEM="${MEM:-64gb}"
export WALLTIME="${WALLTIME:-02:00:00}"

bash scripts/nscc_submit.sh
