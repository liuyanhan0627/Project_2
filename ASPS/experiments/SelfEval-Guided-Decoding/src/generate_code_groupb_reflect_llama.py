import argparse
import json
import os
import time as _time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from utils.dataset import jsonlines_load, load_dataset_examples
from utils.prompt import get_prompt_inputs, get_prompts
from utils.tool import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--auth_token", default="YOUR_HF_TOKEN", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=600, type=int)
    parser.add_argument("--max_reflection_tokens", default=256, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--chatgpt", default=False, action="store_true")
    parser.add_argument(
        "--dt_name",
        required=True,
        type=str,
        choices=[
            "gsm8k",
            "aqua",
            "svamp",
            "asdiv",
            "mawps",
            "tabmwp",
            "finqa",
            "object_counting",
            "repeat_copy",
            "colored_object",
            "penguin",
            "date_understanding",
            "sports",
            "csqa",
            "saycan",
            "strategyqa",
            "gsm8k_cot",
            "math",
            "truthfulqa",
        ],
    )
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--reverse", default=False, action="store_true")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--verbal", default=False, action="store_true")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_dt_string", default="", type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    args.prompts = get_prompts(args.dt_name, return_eval=False, use_chatgpt=args.chatgpt)
    return args


def load_model_and_tokenizer(model_name, auth_token, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
        use_auth_token=auth_token,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=auth_token,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def assert_standard_decoding(generation_config):
    cntp_flags = ("cntp_perplexity", "cntp_same_num_trials", "cntp_negatively_correlated")
    enabled = [flag for flag in cntp_flags if getattr(generation_config, flag, False)]
    if enabled:
        raise RuntimeError(f"Group B must use standard decoding; enabled CNTP flags: {enabled}")


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p, stop_strings="\n\n\n"):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if stop_strings:
        generation_kwargs["stop_strings"] = stop_strings
    generation_config = GenerationConfig(**generation_kwargs)
    assert_standard_decoding(generation_config)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=generation_config,
        tokenizer=tokenizer,
    )
    gen_ids = outputs[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), int(gen_ids.numel())


def reflection_prompt(question_text, first_solution, dt_name, reasoning_type):
    if reasoning_type in ["arithmetic", "symbolic", "algorithmic"]:
        return (
            "You are a strict verifier. Do not rewrite or continue the Python solution.\n"
            "Return exactly one JSON object and no extra text.\n\n"
            f"Question:\n{question_text}\n\n"
            f"Attempted Python solution:\n{first_solution}\n\n"
            "Check whether the attempted solution would execute and return the correct answer. "
            "Set should_restart to true only for a clear calculation, logic, execution, or answer-format error. "
            "Keep reflection under 30 words. Do not include final_answer or solution text.\n"
            'JSON schema: {"has_error": false, "error_type": "none", '
            '"should_restart": false, "reflection": "brief reason"}\n'
        )
    if reasoning_type == "math":
        return (
            "You are a strict verifier. Do not rewrite or continue the solution.\n"
            "Return exactly one JSON object and no extra text.\n\n"
            f"Question:\n{question_text}\n\n"
            f"Attempted solution:\n{first_solution}\n\n"
            "Check whether the reasoning and boxed final answer are mathematically valid. "
            "Set should_restart to true only for a clear calculation, logic, or answer-format error. "
            "Keep reflection under 30 words. Do not include final_answer or solution text.\n"
            'JSON schema: {"has_error": false, "error_type": "none", '
            '"should_restart": false, "reflection": "brief reason"}\n'
        )
    if reasoning_type == "open_qa":
        return (
            "You are a strict verifier. Do not rewrite or continue the answer.\n"
            "Return exactly one JSON object and no extra text.\n\n"
            f"Question:\n{question_text}\n\n"
            f"Attempted answer:\n{first_solution}\n\n"
            "Check whether the answer is factual, literal, and not misleading. "
            "Set should_restart to true only for a clear factual or answer-format error. "
            "Keep reflection under 30 words. Do not include final_answer or replacement text.\n"
            'JSON schema: {"has_error": false, "error_type": "none", '
            '"should_restart": false, "reflection": "brief reason"}\n'
        )
    return (
        "You are a strict verifier. Do not rewrite or continue the reasoning.\n"
        "Return exactly one JSON object and no extra text.\n\n"
        f"Question:\n{question_text}\n\n"
        f"Attempted reasoning:\n{first_solution}\n\n"
        "Check whether the attempted reasoning supports the final yes/no answer. "
        "Set should_restart to true only for a clear factual, logical, or answer-format error. "
        "Keep reflection under 30 words. Do not include final_answer or solution text.\n"
        'JSON schema: {"has_error": false, "error_type": "none", '
        '"should_restart": false, "reflection": "brief reason"}\n'
    )


def strip_code_fence(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _boolish(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return default


def _extract_json_object(text):
    cleaned = strip_code_fence(text)
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{", start + 1)
    return None


def parse_reflection_output(text):
    parsed = _extract_json_object(text)
    if parsed is not None:
        parsed.pop("final_answer", None)
        return _boolish(parsed.get("should_restart", parsed.get("has_error"))), parsed

    match = regex.search(r"SHOULD_RESTART\s*:\s*(yes|no|true|false)", text, flags=regex.I)
    if match:
        return _boolish(match.group(1)), None
    return False, None


def build_restart_prompt(full_prompt, reflection_info, reasoning_type):
    reflection = ""
    error_type = "unknown"
    if isinstance(reflection_info, dict):
        reflection = str(reflection_info.get("reflection", "")).strip()
        error_type = str(reflection_info.get("error_type", error_type)).strip() or error_type

    if reasoning_type in ["arithmetic", "symbolic", "algorithmic"]:
        note = (
            f"# Previous attempt was flagged for {error_type}: {reflection}\n"
            "# Restart from scratch and provide only the corrected Python solution.\n\n"
        )
    elif reasoning_type == "math":
        note = (
            f"Previous attempt was flagged for {error_type}: {reflection}\n"
            "Restart from scratch and put the final answer within \\boxed{}.\n"
        )
    elif reasoning_type == "open_qa":
        note = (
            f"Previous answer was flagged for {error_type}: {reflection}\n"
            "Restart from scratch and answer truthfully and concisely.\n"
        )
    else:
        note = (
            f"Previous attempt was flagged for {error_type}: {reflection}\n"
            "Restart from scratch. End with exactly: So the answer is yes. or So the answer is no.\n"
        )
    return f"{full_prompt.rstrip()}\n{note}"


def extract_prediction(dt_name, generated_text):
    return extract_prediction_from_generation(dt_name, generated_text)


def model_tag(name):
    return name.rstrip("/").split("/")[-1].replace("Meta-", "")


def make_output_filename(args, dt_string):
    tag = model_tag(args.model_name)
    filename = (
        f"{args.output_dir}/{args.dt_name}_groupB_reflect_{tag}"
        f"_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}.jsonl"
    )
    if args.reverse:
        filename = filename.replace(".jsonl", "_reverse.jsonl")
    return filename


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data_test = load_dataset_examples(args.dt_name, args.input_file)
    for i, _ in enumerate(data_test):
        data_test[i]["index"] = i

    dt_string = args.resume_dt_string if args.resume else datetime.now().strftime("%m_%d_%H_%M")
    args.end = len(data_test) if args.end == -1 else args.end + 1
    data_test = data_test[args.start : args.end]
    print("number of examples: ", len(data_test))

    filename = make_output_filename(args, dt_string)
    if os.path.exists(filename):
        prev = jsonlines_load(filename)
        indexes = [x["index"] for x in prev if "index" in x]
    else:
        indexes = []
        with jsonlines.open(filename, mode="w") as writer:
            writer.write(args.prompts)

    inputs = []
    for example in tqdm(data_test):
        index = example["index"]
        if index in indexes:
            continue
        rst = {"index": index}
        rst.update(example)
        inputs.append(rst)
    if args.reverse:
        inputs = inputs[::-1]

    print(f"Loading model on {args.device}: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.auth_token, args.device)

    correct, wrong = 0, 0
    restart_count = 0
    wrong_to_right = 0
    right_to_wrong = 0
    for batch_idx in tqdm(range((len(inputs) + args.batch_size - 1) // args.batch_size)):
        batch = inputs[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        for exp in batch:
            full_prompt, _ = get_prompt_inputs(args.dt_name, args.prompts, exp, use_chatgpt=args.chatgpt)
            if args.verbal:
                print("======================")
                print(f'Index: {exp["index"]}\nQuestion: {exp.get("question", "")}')

            start_time = _time.time()
            first_solution, first_tokens = generate_text(
                model, tokenizer, full_prompt, args.max_tokens, args.temperature, args.top_p
            )
            first_prediction = extract_prediction(args.dt_name, first_solution)
            gt_ans = exp.get("answer", None)
            first_correct = score_prediction(args.dt_name, first_prediction, gt_ans)

            review_prompt = reflection_prompt(
                exp.get("question", ""), first_solution, args.dt_name, args.prompts["type"]
            )
            reflection_output, reflection_tokens = generate_text(
                model,
                tokenizer,
                review_prompt,
                args.max_reflection_tokens,
                args.temperature,
                args.top_p,
            )
            should_restart, reflection_info = parse_reflection_output(reflection_output)
            final_solution = first_solution
            restart_solution = ""
            restart_tokens = 0
            if should_restart:
                restart_count += 1
                restart_prompt = build_restart_prompt(full_prompt, reflection_info, args.prompts["type"])
                restart_solution, restart_tokens = generate_text(
                    model, tokenizer, restart_prompt, args.max_tokens, args.temperature, args.top_p
                )
                if restart_solution.strip():
                    final_solution = restart_solution

            prediction = extract_prediction(args.dt_name, final_solution)
            is_correct = score_prediction(args.dt_name, prediction, gt_ans)
            if is_correct is True:
                correct += 1
            elif is_correct is False:
                wrong += 1
            if should_restart and first_correct is False and is_correct is True:
                wrong_to_right += 1
            if should_restart and first_correct is True and is_correct is False:
                right_to_wrong += 1

            exp.update(
                {
                    "executed": prediction,
                    "first_executed": first_prediction,
                    "generated": [final_solution],
                    "first_solution": first_solution,
                    "reflection_output": reflection_output,
                    "reflection_info": reflection_info,
                    "restart_solution": restart_solution,
                    "groupb_metrics": {
                        "should_restart": should_restart,
                        "first_solution_tokens": first_tokens,
                        "reflection_tokens": reflection_tokens,
                        "restart_tokens": restart_tokens,
                        "total_output_tokens": first_tokens + reflection_tokens + restart_tokens,
                        "hit_reflection_limit": reflection_tokens >= args.max_reflection_tokens,
                        "reflection_parse_failed": reflection_info is None,
                        "first_correct": first_correct,
                        "final_correct": is_correct,
                        "wall_time": _time.time() - start_time,
                    },
                }
            )
            with jsonlines.open(filename, mode="a") as writer:
                writer.write(exp)
            torch.cuda.empty_cache()

    total = correct + wrong
    accuracy = correct / total if total else None
    print("======================")
    if accuracy is None:
        print("Accuracy: N/A (no automatic metric)")
    else:
        print(accuracy, "(", correct, "/", total, ")")
    summary_filename = filename.replace(".jsonl", "_summary.txt")
    with open(summary_filename, "w") as f:
        if accuracy is None:
            f.write("Accuracy: N/A (no automatic metric)\n")
        else:
            f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
        f.write(f"Restart rate: {restart_count / total if total else 0.0:.4f} ({restart_count}/{total})\n")
        f.write(f"Wrong-to-right: {wrong_to_right}\n")
        f.write(f"Right-to-wrong: {right_to_wrong}\n")
