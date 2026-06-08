import argparse
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from groupa_async_decoding import GroupAAsyncDecoder, GroupADecodingConfig
from utils.dataset import jsonlines_load, load_dataset_examples
from utils.prompt import get_prompt_inputs, get_prompts
from utils.tool import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big_model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--small_model_name", default="meta-llama/Llama-3.2-1B-Instruct", type=str)
    parser.add_argument("--auth_token", default="YOUR_HF_TOKEN", type=str)
    parser.add_argument("--big_device", default="cuda:0", type=str)
    parser.add_argument("--small_device", default="cuda:1", type=str)
    parser.add_argument("--max_tokens", default=600, type=int)
    parser.add_argument("--entropy_threshold", default=1.5, type=float)
    parser.add_argument("--draft_candidates", default=3, type=int)
    parser.add_argument("--max_draft_tokens", default=128, type=int)
    parser.add_argument("--max_fallback_tokens", default=128, type=int)
    parser.add_argument("--big_temperature", default=0.0, type=float)
    parser.add_argument("--big_top_p", default=1.0, type=float)
    parser.add_argument("--small_temperature", default=0.7, type=float)
    parser.add_argument("--small_top_p", default=0.9, type=float)
    parser.add_argument("--length_weight_alpha", default=0.05, type=float)
    parser.add_argument(
        "--length_weight_mode",
        default="longer",
        choices=["longer", "log_longer", "shorter", "none"],
        type=str,
    )
    parser.add_argument("--switch_score_margin", default=0.0, type=float)
    parser.add_argument("--disable_prefix_cache_verify", default=False, action="store_true")
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
    tail = name.rstrip("/").split("/")[-1]
    return tail.replace("Meta-", "").replace("/", "_")


def _float_tag(value):
    return str(value).replace("-", "m").replace(".", "p")


def make_output_filename(args, dt_string):
    big = model_tag(args.big_model_name)
    small = model_tag(args.small_model_name)
    filename = (
        f"{args.output_dir}/{args.dt_name}_groupC_big{big}_small{small}"
        f"_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}"
        f"_entropy{args.entropy_threshold}_k{args.draft_candidates}"
        f"_draft{args.max_draft_tokens}_fallback{args.max_fallback_tokens}"
        f"_lw{_float_tag(args.length_weight_alpha)}_{args.length_weight_mode}.jsonl"
    )
    if args.switch_score_margin > 0:
        filename = filename.replace(".jsonl", f"_margin{_float_tag(args.switch_score_margin)}.jsonl")
    if args.reverse:
        filename = filename.replace(".jsonl", "_reverse.jsonl")
    return filename


def groupc_generation_result(decoder, context):
    result = decoder.generate(context)
    return {
        "choices": [
            {
                "text": result["text"],
                "logprobs": {
                    "tokens": result["tokens"],
                    "token_logprobs": [],
                    "top_logprobs": [],
                },
                "groupc_metrics": result["metrics"],
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

    print(f"Loading big model on {args.big_device}: {args.big_model_name}")
    big_model, big_tokenizer = load_model_and_tokenizer(args.big_model_name, args.auth_token, args.big_device)
    print(f"Loading small model on {args.small_device}: {args.small_model_name}")
    small_model, small_tokenizer = load_model_and_tokenizer(args.small_model_name, args.auth_token, args.small_device)

    decoder_config = GroupADecodingConfig(
        max_new_tokens=args.max_tokens,
        entropy_threshold=args.entropy_threshold,
        draft_candidates=args.draft_candidates,
        max_draft_tokens=args.max_draft_tokens,
        max_fallback_tokens=args.max_fallback_tokens,
        big_temperature=args.big_temperature,
        big_top_p=args.big_top_p,
        small_temperature=args.small_temperature,
        small_top_p=args.small_top_p,
        path_length_weight_alpha=args.length_weight_alpha,
        path_length_weight_mode=args.length_weight_mode,
        switch_score_margin=args.switch_score_margin,
        use_prefix_cache_for_verify=not args.disable_prefix_cache_verify,
    )
    decoder = GroupAAsyncDecoder(big_model, big_tokenizer, small_model, small_tokenizer, decoder_config)

    correct, wrong = 0, 0
    try:
        for batch_idx in tqdm(range((len(inputs) + args.batch_size - 1) // args.batch_size)):
            batch = inputs[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            for exp in batch:
                full_prompt, _ = get_prompt_inputs(args.dt_name, args.prompts, exp, use_chatgpt=args.chatgpt)
                if args.verbal:
                    print("======================")
                    print(f'Index: {exp["index"]}\nQuestion: {exp.get("question", "")}')

                raw_results = groupc_generation_result(decoder, full_prompt)
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
                        "groupc_metrics": raw_results["choices"][0]["groupc_metrics"],
                    }
                )
                with jsonlines.open(filename, mode="a") as writer:
                    writer.write(exp)
                torch.cuda.empty_cache()
    finally:
        decoder.close()

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
