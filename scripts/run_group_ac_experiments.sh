#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${CONFIG_DIR:-configs/group_ac}"
LOG_DIR="${LOG_DIR:-server_logs}"
RUN_SMOKE="${RUN_SMOKE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXPORT_RESULTS="${EXPORT_RESULTS:-1}"
EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_ac_first100}"

if [[ -n "${RUN_CONFIGS:-}" ]]; then
  read -r -a CONFIG_NAMES <<< "${RUN_CONFIGS}"
else
  CONFIG_NAMES=(
    baseline_groupb
    a_safe
    a_fast
    a_accuracy
    c_low
    c_mid
    c_fast
  )
fi

mkdir -p "${LOG_DIR}"

run_train() {
  local config_path="$1"
  local log_name="$2"
  local extra_args=()
  if [[ "${DRY_RUN}" == "1" ]]; then
    extra_args+=(--dry-run)
  fi
  echo "=== Running ${config_path} ==="
  python3 train.py --config "${config_path}" "${extra_args[@]}" 2>&1 | tee "${LOG_DIR}/${log_name}.log"
}

if [[ "${RUN_SMOKE}" == "1" ]]; then
  run_train "configs/smoke.yaml" "smoke"
fi

for name in "${CONFIG_NAMES[@]}"; do
  run_train "${CONFIG_DIR}/${name}.yaml" "${name}"
done

python3 scripts/collect_results.py --outputs outputs --registry experiments/registry.csv

if [[ "${EXPORT_RESULTS}" == "1" ]]; then
  python3 scripts/export_results_for_github.py \
    --outputs outputs \
    --registry experiments/registry.csv \
    --export-root experiments/result_exports \
    --name "${EXPORT_NAME}"
fi

echo "=== Done ==="
echo "Registry: experiments/registry.csv"
if [[ "${EXPORT_RESULTS}" == "1" ]]; then
  echo "Export: experiments/result_exports/${EXPORT_NAME}"
fi
