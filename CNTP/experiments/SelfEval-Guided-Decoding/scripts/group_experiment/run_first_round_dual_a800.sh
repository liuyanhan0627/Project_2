#!/usr/bin/env bash
set -euo pipefail

HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
DATAHOME="${DATAHOME:-../data}"
OUTROOT="${OUTROOT:-../outputs}"
BIG_MODEL="${BIG_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
SMALL_MODEL="${SMALL_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../src" && pwd)"
cd "${SRC_DIR}"

run_one_dataset() {
  local dtname="$1"
  local count="$2"
  local max_tokens="$3"
  local end_idx=$((count - 1))
  local input_file="${DATAHOME}/${dtname}_test.jsonl"
  local output_dir="${OUTROOT}/${dtname}/test_outputs"
  mkdir -p "${output_dir}"

  echo "=== First-round Baseline ${dtname} (${count}) ==="
  CUDA_VISIBLE_DEVICES=0 python generate_code_baseline_llama3.1.py \
    --dt_name "${dtname}" \
    --input_file "${input_file}" \
    --output_dir "${output_dir}" \
    --max_tokens "${max_tokens}" \
    --temperature 0.0 \
    --top_p 1.0 \
    --mini_n_samples 1 --n_samples 1 \
    --batch_size 1 \
    --model_name "${BIG_MODEL}" \
    --auth_token "${HF_TOKEN}" \
    --start 0 --end "${end_idx}" \
    --seed 0

  echo "=== First-round Group A ${dtname} (${count}) ==="
  CUDA_VISIBLE_DEVICES=0,1 python generate_code_groupa_llama.py \
    --dt_name "${dtname}" \
    --input_file "${input_file}" \
    --output_dir "${output_dir}" \
    --max_tokens "${max_tokens}" \
    --big_model_name "${BIG_MODEL}" \
    --small_model_name "${SMALL_MODEL}" \
    --big_device cuda:0 \
    --small_device cuda:1 \
    --entropy_threshold 1.5 \
    --draft_candidates 3 \
    --max_draft_tokens 128 \
    --max_fallback_tokens 128 \
    --big_temperature 0.0 \
    --small_temperature 0.7 \
    --small_top_p 0.9 \
    --start 0 --end "${end_idx}" \
    --seed 0

  echo "=== First-round Group B ${dtname} (${count}) ==="
  CUDA_VISIBLE_DEVICES=0 python generate_code_groupb_reflect_llama.py \
    --dt_name "${dtname}" \
    --input_file "${input_file}" \
    --output_dir "${output_dir}" \
    --max_tokens "${max_tokens}" \
    --max_reflection_tokens 900 \
    --temperature 0.0 \
    --top_p 1.0 \
    --model_name "${BIG_MODEL}" \
    --auth_token "${HF_TOKEN}" \
    --device cuda:0 \
    --start 0 --end "${end_idx}" \
    --seed 0
}

run_one_dataset gsm8k 100 512
run_one_dataset strategyqa 100 256
