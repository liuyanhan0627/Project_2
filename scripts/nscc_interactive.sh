#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-personal-e1547010}"
QUEUE="${QUEUE:-normal}"
NGPUS="${NGPUS:-2}"
NCPUS="${NCPUS:-32}"
MEM="${MEM:-110gb}"
WALLTIME="${WALLTIME:-01:00:00}"

if [[ -z "${PROJECT}" ]]; then
  echo "Set PROJECT to your NSCC project first, for example:" >&2
  echo "  PROJECT=personal-e1547010 bash scripts/nscc_interactive.sh" >&2
  exit 64
fi

qsub -I -V \
  -P "${PROJECT}" \
  -q "${QUEUE}" \
  -l "select=1:ngpus=${NGPUS}:ncpus=${NCPUS}:mem=${MEM}" \
  -l "walltime=${WALLTIME}"
