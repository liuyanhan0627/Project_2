import argparse
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tot_bfs_decoding import ToTBFSConfig, ToTBFSDecoder
from utils.dataset import jsonlines_load, load_dataset_examples
from utils.prompt import get_prompt_inputs, get_prompts
from utils.tool import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--auth_token", default="YOUR_HF_TOKEN", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--max_tokens", default=600, type=int)
    parser.add_argument("--max_depth", default=3, type=int)
    parser.add_argument("--thought_branching", default=3, type=int)
    parser.add_argument("--beam_width", default=2, type=int)
    parser.add_argument("--n_evaluate_sample", default=1, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--value_temperature", default=0.0, type=float)
    parser.add_argument("--value_top_p", default=1.0, type=float)
    parser.add_argument("--final_temperature", default=0.0, type=float)
    parser.add_argument("--final_top_p", default=1.0, type=float)
    parser.add_argument("--max_thought_tokens", default=96, type=int)
    parser.add_argument("--max_value_tokens", default=32, type=int)
    parser.add_argument("--max_final_tokens", default=256, type=int)
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
            "ruler_niah",
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


def model_tag(name):
    return name.rstrip("/").split("/")[-1].replace("Meta-", "")


def make_output_filename(args, dt_string):
    tag = model_tag(args.model_name)
    filename = (
        f"{args.output_dir}/{args.dt_name}_groupE_tot_bfs_{tag}"
        f"_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}"
        f"_d{args.max_depth}_b{args.thought_branching}_beam{args.beam_width}.jsonl"
    )
    if args.reverse:
        filename = filename.replace(".jsonl", "_reverse.jsonl")
    return filename


def groupe_generation_result(decoder, full_prompt, question, dt_name):
    result = decoder.generate(full_prompt, question, dt_name)
    return {
        "choices": [
            {
                "text": result["text"],
                "logprobs": {
                    "tokens": result["tokens"],
                    "token_logprobs": [],
                    "top_logprobs": [],
                },
                "groupe_metrics": result["metrics"],
                "tot_path": result["tot_path"],
            }
        ]
    }


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
    decoder_config = ToTBFSConfig(
        max_depth=args.max_depth,
        thought_branching=args.thought_branching,
        beam_width=args.beam_width,
        n_evaluate_sample=args.n_evaluate_sample,
        thought_temperature=args.temperature,
        thought_top_p=args.top_p,
        value_temperature=args.value_temperature,
        value_top_p=args.value_top_p,
        final_temperature=args.final_temperature,
        final_top_p=args.final_top_p,
        max_thought_tokens=args.max_thought_tokens,
        max_value_tokens=args.max_value_tokens,
        max_final_tokens=args.max_final_tokens or args.max_tokens,
    )
    decoder = ToTBFSDecoder(model, tokenizer, decoder_config)

    correct, wrong = 0, 0
    for batch_idx in tqdm(range((len(inputs) + args.batch_size - 1) // args.batch_size)):
        batch = inputs[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        for exp in batch:
            full_prompt, _ = get_prompt_inputs(args.dt_name, args.prompts, exp, use_chatgpt=args.chatgpt)
            question = exp.get("question", "")
            if args.verbal:
                print("======================")
                print(f'Index: {exp["index"]}\nQuestion: {question}')

            raw_results = groupe_generation_result(decoder, full_prompt, question, args.dt_name)
            results = parse_api_result(raw_results, llama=True, return_prob=False)

            result_counter = Counter()
            for code in results:
                ans = extract_prediction_from_generation(args.dt_name, code)
                if ans is not None:
                    result_counter.update([ans])

            prediction = None
            if len(result_counter) > 0:
                prediction = result_counter.most_common(1)[0][0]
            gt_ans = exp.get("answer", None)
            score = score_prediction(args.dt_name, prediction, gt_ans)
            if score is True:
                correct += 1
            elif score is False:
                wrong += 1

            exp.update(
                {
                    "executed": prediction,
                    "generated": results,
                    "is_correct": score,
                    "raw_generation": raw_results["choices"][0]["text"],
                    "tot_path": raw_results["choices"][0]["tot_path"],
                    "groupe_metrics": raw_results["choices"][0]["groupe_metrics"],
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
