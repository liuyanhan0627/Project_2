# Group A: ASPS Small Draft Model Prototype

This prototype implements **Asynchronous Speculative Path Switching (ASPS)** on top of CNTP's GSM8K / StrategyQA experiment layout:

```text
target Llama entropy trigger
-> async small-model draft paths until punctuation
-> target Llama keeps decoding fallback path
-> target Llama batched-PPL-verifies small drafts + fallback
-> switch to the lowest-PPL path within one punctuation-span rollback window
```

## Files

- `src/groupa_async_decoding.py`: core Group A decoder.
- `src/generate_code_groupa_llama.py`: GSM8K / StrategyQA runner adapted from CNTP's Llama scripts.

## Example

Run from `ASPS/experiments/SelfEval-Guided-Decoding/src`:

```bash
python generate_code_groupa_llama.py --verbal \
  --dt_name gsm8k \
  --input_file /path/to/gsm8k_test.jsonl \
  --output_dir ../outputs/gsm8k/test_outputs \
  --max_tokens 512 \
  --big_model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --small_model_name meta-llama/Llama-3.2-1B-Instruct \
  --big_device cuda:0 \
  --small_device cuda:1 \
  --entropy_threshold 1.5 \
  --draft_candidates 3 \
  --max_draft_tokens 128 \
  --max_fallback_tokens 128 \
  --big_temperature 0.0 \
  --small_temperature 0.7 \
  --small_top_p 0.9 \
  --seed 0
```

## Notes

- The main verification path uses batched candidate scoring.
- By default the verifier first tries to reuse the target model's trigger-prefix KV cache. If cache verification fails for a model/backend, it falls back to full-sequence batched scoring and prints a warning.
- The fallback path is included as one candidate. If the small draft is late after one punctuation span, it is dropped and counted in `late_draft_drops`.
- Output JSONL records `groupa_metrics`, including trigger count, switch count, fallback wins, small draft wins, late drops, candidate PPLs, and wasted fallback tokens.
