# Llama Big/Small Reasoning Experiment

This directory now contains code for the three groups in the proposed experiment:

| Group | File |
|---|---|
| Baseline | `src/generate_code_baseline_llama3.1.py` |
| Group A: async small draft + target PPL verify | `src/generate_code_groupa_llama.py` |
| Group B: target reflection + one restart | `src/generate_code_groupb_reflect_llama.py` |

## Dual A800 Plan

Start with one dual-A800 machine:

1. Run smoke test: `GSM8K 20 + StrategyQA 20`.
2. Inspect Group A metrics, especially `entropy_triggers`, `verify_calls`, `switches`, `late_draft_drops`, and `wasted_big_tokens`.
3. If smoke test is healthy, run first round: `GSM8K 100 + StrategyQA 100`.
4. Only rent more GPUs after first-round metrics show signal.

## Environment Variables

Set these before running scripts:

```bash
export HF_TOKEN="your_huggingface_token"
export DATAHOME="/path/to/SelfEval-Guided-Decoding/data"
export OUTROOT="/path/to/SelfEval-Guided-Decoding/outputs"
export BIG_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export SMALL_MODEL="meta-llama/Llama-3.2-1B-Instruct"
```

Expected input files:

```text
${DATAHOME}/gsm8k_test.jsonl
${DATAHOME}/strategyqa_test.jsonl
```

## Run

From `experiments/SelfEval-Guided-Decoding`:

```bash
bash scripts/group_experiment/run_smoke_dual_a800.sh
```

Then:

```bash
bash scripts/group_experiment/run_first_round_dual_a800.sh
```

## Notes

- Group A uses both GPUs: target model on `cuda:0`, draft model on `cuda:1`.
- Baseline and Group B use only `cuda:0`.
- Baseline and Group B must use standard greedy/sample decoding; CNTP multi-trial decoding is only enabled by explicit `cntp_perplexity`, `cntp_same_num_trials`, or `cntp_negatively_correlated` flags.
- Group A includes batched candidate PPL verification. Serial path scoring should only be used for debugging or ablation.
- Group B records `should_restart`, first solution tokens, reflection tokens, and wall-clock time in `groupb_metrics`.
