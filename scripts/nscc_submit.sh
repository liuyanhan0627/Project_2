#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-personal-e1547010}"
QUEUE="${QUEUE:-normal}"
NGPUS="${NGPUS:-2}"
NCPUS="${NCPUS:-32}"
MEM="${MEM:-110gb}"
WALLTIME="${WALLTIME:-02:00:00}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_group_ac_daytime_k2c_sweep.sh}"
PBS_SCRIPT="${PBS_SCRIPT:-scripts/nscc_run.pbs}"
JOB_NAME="${JOB_NAME:-asps_nscc}"

run_label="$(basename "${RUN_SCRIPT%.sh}")"
EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_nscc_${run_label}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

if [[ -z "${PROJECT}" ]]; then
  echo "Set PROJECT to your NSCC project first, for example:" >&2
  echo "  PROJECT=personal-e1547010 bash scripts/nscc_submit.sh" >&2
  exit 64
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub was not found. Run this script on the NSCC login node." >&2
  exit 69
fi

if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "PBS script not found: ${PBS_SCRIPT}" >&2
  exit 66
fi

mkdir -p nscc_logs

qsub -V \
  -P "${PROJECT}" \
  -q "${QUEUE}" \
  -N "${JOB_NAME}" \
  -l "select=1:ngpus=${NGPUS}:ncpus=${NCPUS}:mem=${MEM}" \
  -l "walltime=${WALLTIME}" \
  -v "RUN_SCRIPT=${RUN_SCRIPT},EXPORT_NAME=${EXPORT_NAME},OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "${PBS_SCRIPT}"
