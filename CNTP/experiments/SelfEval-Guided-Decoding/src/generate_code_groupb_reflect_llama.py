import argparse
import os
import time as _time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from execute_and_evaluate.interpret_and_evaluate import check_eq
from utils.dataset import jsonlines_load
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
    parser.add_argument("--max_reflection_tokens", default=900, type=int)
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
def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings="\n\n\n",
    )
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
            f"Question:\n{question_text}\n\n"
            f"Attempted Python solution:\n{first_solution}\n\n"
            "# Review the attempted Python solution above. If it is correct, keep it. "
            "If it has any calculation, logic, or formatting mistake, discard it and restart once.\n"
            "# Return exactly in this format:\n"
            "# SHOULD_RESTART: yes or no\n"
            "# FINAL_SOLUTION:\n"
            "# <final Python solution only>\n"
        )
    return (
        f"Question:\n{question_text}\n\n"
        f"Attempted reasoning:\n{first_solution}\n\n"
        "Review the attempted reasoning above. If it is correct, keep it. "
        "If it has a factual, logical, or answer-format mistake, discard it and restart once.\n"
        "Return exactly in this format:\n"
        "SHOULD_RESTART: yes or no\n"
        "FINAL_SOLUTION:\n"
        "<final step-by-step reasoning ending with 'So the answer is yes.' or 'So the answer is no.'>\n"
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


def parse_reflection_output(text):
    restart = False
    match = regex.search(r"SHOULD_RESTART\s*:\s*(yes|no)", text, flags=regex.I)
    if match:
        restart = match.group(1).lower() == "yes"

    final = text
    marker = regex.search(r"FINAL_SOLUTION\s*:", text, flags=regex.I)
    if marker:
        final = text[marker.end() :]
    final = strip_code_fence(final)
    return restart, final


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

    data_test = jsonlines_load(args.input_file)
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
            review_prompt = reflection_prompt(
                exp.get("question", ""), first_solution, args.dt_name, args.prompts["type"]
            )
            reflection_output, reflection_tokens = generate_text(
                model, tokenizer, review_prompt, args.max_reflection_tokens, args.temperature, args.top_p
            )
            should_restart, final_solution = parse_reflection_output(reflection_output)
            if not final_solution.strip():
                final_solution = first_solution
            if should_restart:
                restart_count += 1

            prediction = extract_prediction(args.dt_name, final_solution)
            gt_ans = exp.get("answer", None)
            is_correct = check_eq(prediction, gt_ans, percent_check=exp.get("question", ""), dtname=args.dt_name)
            if is_correct:
                correct += 1
            else:
                wrong += 1

            exp.update(
                {
                    "executed": prediction,
                    "generated": [final_solution],
                    "first_solution": first_solution,
                    "reflection_output": reflection_output,
                    "groupb_metrics": {
                        "should_restart": should_restart,
                        "first_solution_tokens": first_tokens,
                        "reflection_tokens": reflection_tokens,
                        "wall_time": _time.time() - start_time,
                    },
                }
            )
            with jsonlines.open(filename, mode="a") as writer:
                writer.write(exp)
            torch.cuda.empty_cache()

    total = correct + wrong
    accuracy = correct / total if total else 0.0
    print("======================")
    print(accuracy, "(", correct, "/", total, ")")
    summary_filename = filename.replace(".jsonl", "_summary.txt")
    with open(summary_filename, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
        f.write(f"Restart rate: {restart_count / total if total else 0.0:.4f} ({restart_count}/{total})\n")
