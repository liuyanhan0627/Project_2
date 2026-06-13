#!/usr/bin/env bash
set -euo pipefail

export RUN_SMOKE="${RUN_SMOKE:-0}"
export EXPORT_RESULTS="${EXPORT_RESULTS:-1}"
export EXPORT_NAME="${EXPORT_NAME:-$(date +%Y%m%d-%H%M%S)_group_f1_offline_distill_gsm8k}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${EXPORT_NAME}}"

LOG_DIR="${LOG_DIR:-server_logs}"
TRAINER="ASPS/experiments/SelfEval-Guided-Decoding/src/distill/trainer.py"
F1_EVAL_CONFIG="${F1_EVAL_CONFIG:-configs/group_ac/f1_offline_distill_eval_gsm8k_heldout30.yaml}"
F1_TRAIN_OUTPUT_DIR="${F1_TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/checkpoints/group_f1_lora}"
ANALYZE_DISTILL="${ANALYZE_DISTILL:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${F1_TRAIN_OUTPUT_DIR}"

latest_f0_records() {
  find outputs experiments/result_exports -path "*/group_f0/distill_logs/trigger_records.jsonl" -type f 2>/dev/null | sort | tail -n 1
}

if [[ -z "${F0_RECORDS_PATH:-}" && -n "${F0_DISTILL_LOG_DIR:-}" ]]; then
  F0_RECORDS_PATH="${F0_DISTILL_LOG_DIR%/}/trigger_records.jsonl"
fi

if [[ -z "${F0_RECORDS_PATH:-}" ]]; then
  F0_RECORDS_PATH="$(latest_f0_records || true)"
  if [[ -n "${F0_RECORDS_PATH}" ]]; then
    echo "Using latest discovered F0 records: ${F0_RECORDS_PATH}"
  fi
fi

if [[ -z "${F0_RECORDS_PATH:-}" || ! -f "${F0_RECORDS_PATH}" ]]; then
  echo "Set F0_RECORDS_PATH to a Group F0 trigger_records.jsonl file, for example:" >&2
  echo "  F0_RECORDS_PATH=outputs/<f0_run>/results/gsm8k/group_f0/distill_logs/trigger_records.jsonl bash ${0}" >&2
  exit 64
fi

TRAIN_ARGS=(
  --records_path "${F0_RECORDS_PATH}"
  --output_dir "${F1_TRAIN_OUTPUT_DIR}"
  --small_model_name "${SMALL_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
  --auth_token "${HF_TOKEN:-YOUR_HF_TOKEN}"
  --device "${F1_TRAIN_DEVICE:-cuda:1}"
  --top_k "${F1_TOP_K:-20}"
  --train_qid_max "${F1_TRAIN_QID_MAX:-69}"
  --eval_qid_min "${F1_EVAL_QID_MIN:-70}"
  --batch_size "${F1_BATCH_SIZE:-4}"
  --epochs "${F1_EPOCHS:-2}"
  --lr "${F1_LR:-2e-5}"
  --tau "${F1_TAU:-1.0}"
  --lambda_anchor "${F1_LAMBDA_ANCHOR:-0.1}"
  --lora_r "${F1_LORA_R:-8}"
  --lora_alpha "${F1_LORA_ALPHA:-16}"
  --lora_dropout "${F1_LORA_DROPOUT:-0.0}"
  --seed "${F1_SEED:-0}"
)

if [[ "${DRY_RUN}" == "1" || "${F1_TRAIN_DRY_RUN:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dry_run)
fi

echo "=== Group F1 offline LoRA distillation ==="
echo "records: ${F0_RECORDS_PATH}"
echo "train_output: ${F1_TRAIN_OUTPUT_DIR}"
python3 "${TRAINER}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/group_f1_train.log"

if [[ -z "${F1_LORA_PATH:-}" ]]; then
  F1_LORA_PATH="${F1_TRAIN_OUTPUT_DIR}/adapter"
fi
if [[ "${F1_LORA_PATH}" != /* ]]; then
  F1_LORA_PATH="$(pwd)/${F1_LORA_PATH}"
fi
export F1_LORA_PATH

EVAL_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EVAL_ARGS+=(--dry-run)
fi

echo "=== Group F1 ASPS evaluation ==="
echo "eval_config: ${F1_EVAL_CONFIG}"
echo "adapter: ${F1_LORA_PATH}"
python3 train.py --config "${F1_EVAL_CONFIG}" "${EVAL_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/group_f1_eval.log"

if [[ "${ANALYZE_DISTILL}" == "1" && "${DRY_RUN}" != "1" ]]; then
  while IFS= read -r log_dir; do
    echo "=== Analyzing F1 distill logs: ${log_dir} ==="
    python3 ASPS/experiments/SelfEval-Guided-Decoding/src/distill/analyze_divergence.py "${log_dir}"
  done < <(find "${OUTPUT_ROOT}" -path "*/group_f1/distill_logs" -type d 2>/dev/null | sort)
fi

python3 scripts/collect_results.py --outputs "${OUTPUT_ROOT}" --registry experiments/registry.csv

if [[ "${EXPORT_RESULTS}" == "1" ]]; then
  python3 scripts/export_results_for_github.py \
    --outputs "${OUTPUT_ROOT}" \
    --registry experiments/registry.csv \
    --export-root experiments/result_exports \
    --name "${EXPORT_NAME}"
fi

echo "=== Done ==="
echo "Outputs: ${OUTPUT_ROOT}"
echo "F1 adapter: ${F1_LORA_PATH}"
echo "Registry: experiments/registry.csv"
if [[ "${EXPORT_RESULTS}" == "1" ]]; then
  echo "Export: experiments/result_exports/${EXPORT_NAME}"
fi
