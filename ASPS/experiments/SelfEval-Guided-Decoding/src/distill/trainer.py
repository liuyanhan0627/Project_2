import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class DistillRecord:
    record_id: str
    question_id: Optional[int]
    dataset: str
    prefix_token_ids: List[int]
    big_topk_ids: List[int]
    big_topk_logits: List[float]
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LoRA distillation from ASPS trigger records.")
    parser.add_argument(
        "--records_path",
        required=True,
        nargs="+",
        help="trigger_records.jsonl file(s), or directories containing trigger_records.jsonl.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for adapter and metrics.")
    parser.add_argument("--small_model_name", default="meta-llama/Llama-3.2-1B-Instruct", type=str)
    parser.add_argument("--auth_token", default=os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN"), type=str)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--train_qid_max", default=69, type=int)
    parser.add_argument("--eval_qid_min", default=70, type=int)
    parser.add_argument("--max_prefix_tokens", default=4096, type=int)
    parser.add_argument("--max_train_records", default=0, type=int)
    parser.add_argument("--max_eval_records", default=0, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--lambda_anchor", default=0.1, type=float)
    parser.add_argument("--eval_teacher_mass_top_n", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        type=str,
        help="Comma-separated PEFT target modules.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dry_run", action="store_true", help="Validate records and write a plan without loading models.")
    return parser.parse_args()


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def discover_record_files(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("trigger_records.jsonl")))
        else:
            raise FileNotFoundError(path)

    deduped: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    if not deduped:
        raise FileNotFoundError(f"No trigger_records.jsonl files found in: {paths}")
    return deduped


def load_prefix_pools(record_files: Sequence[Path]) -> Dict[str, List[int]]:
    pools: Dict[str, List[int]] = {}
    for record_file in record_files:
        pool_path = record_file.parent / "prefix_pool.jsonl"
        if not pool_path.exists():
            continue
        with pool_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {pool_path}:{line_no}: {exc}") from exc
                prefix_hash = item.get("prefix_hash")
                prefix = item.get("prefix_token_ids")
                if isinstance(prefix_hash, str) and isinstance(prefix, list):
                    pools[prefix_hash] = [int(token_id) for token_id in prefix]
    return pools


def parse_question_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_prefix(record: Dict[str, Any], prefix_pools: Dict[str, List[int]]) -> List[int]:
    prefix = record.get("prefix_token_ids") or []
    if prefix:
        return [int(token_id) for token_id in prefix]
    prefix_hash = record.get("prefix_hash")
    if prefix_hash and prefix_hash in prefix_pools:
        return list(prefix_pools[prefix_hash])
    return []


def load_records(record_files: Sequence[Path], top_k: int, max_prefix_tokens: int) -> Tuple[List[DistillRecord], Dict[str, int]]:
    prefix_pools = load_prefix_pools(record_files)
    records: List[DistillRecord] = []
    skipped = {
        "empty_prefix": 0,
        "long_prefix": 0,
        "missing_teacher_topk": 0,
        "bad_json": 0,
    }
    for path in record_files:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    skipped["bad_json"] += 1
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc

                prefix = resolve_prefix(raw, prefix_pools)
                if not prefix:
                    skipped["empty_prefix"] += 1
                    continue
                if len(prefix) > max_prefix_tokens:
                    skipped["long_prefix"] += 1
                    continue

                ids = raw.get("big_topk_ids") or []
                logits = raw.get("big_topk_logits") or []
                if not ids or not logits:
                    skipped["missing_teacher_topk"] += 1
                    continue
                k = min(int(top_k), len(ids), len(logits))
                records.append(
                    DistillRecord(
                        record_id=str(raw.get("record_id") or f"{path}:{line_no}"),
                        question_id=parse_question_id(raw.get("question_id")),
                        dataset=str(raw.get("dataset") or ""),
                        prefix_token_ids=prefix,
                        big_topk_ids=[int(token_id) for token_id in ids[:k]],
                        big_topk_logits=[float(value) for value in logits[:k]],
                        source_file=str(path),
                    )
                )
    return records, skipped


def split_records(args: argparse.Namespace, records: Sequence[DistillRecord]) -> Tuple[List[DistillRecord], List[DistillRecord]]:
    train = [record for record in records if record.question_id is not None and record.question_id <= args.train_qid_max]
    eval_records = [record for record in records if record.question_id is not None and record.question_id >= args.eval_qid_min]

    if not train:
        train = list(records)
    if not eval_records:
        eval_records = []

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(eval_records)
    if args.max_train_records > 0:
        train = train[: args.max_train_records]
    if args.max_eval_records > 0:
        eval_records = eval_records[: args.max_eval_records]
    return train, eval_records


def summarize_records(records: Sequence[DistillRecord]) -> Dict[str, Any]:
    qids = [record.question_id for record in records if record.question_id is not None]
    prefix_lens = [len(record.prefix_token_ids) for record in records]
    topk_lens = [len(record.big_topk_ids) for record in records]
    return {
        "records": len(records),
        "unique_qids": len(set(qids)),
        "qid_min": min(qids) if qids else None,
        "qid_max": max(qids) if qids else None,
        "avg_prefix_tokens": mean(prefix_lens) if prefix_lens else None,
        "max_prefix_tokens": max(prefix_lens) if prefix_lens else None,
        "avg_top_k": mean(topk_lens) if topk_lens else None,
    }


def sanitized_args(args: argparse.Namespace) -> Dict[str, Any]:
    data = dict(vars(args))
    token = str(data.get("auth_token") or "")
    data["auth_token"] = "SET" if token and token != "YOUR_HF_TOKEN" else ""
    return data


def import_training_deps():
    try:
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.optim import AdamW
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "F1 offline distillation requires torch, transformers, and peft. "
            "Install PEFT with `pip install peft` if it is missing."
        ) from exc
    return torch, F, AdamW, AutoModelForCausalLM, AutoTokenizer, LoraConfig, TaskType, get_peft_model, set_seed


def dtype_from_name(torch, name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_lora_model(args: argparse.Namespace):
    (
        torch,
        _F,
        AdamW,
        AutoModelForCausalLM,
        AutoTokenizer,
        LoraConfig,
        TaskType,
        get_peft_model,
        set_seed,
    ) = import_training_deps()
    set_seed(args.seed)
    dtype = dtype_from_name(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.small_model_name,
        padding_side="left",
        trust_remote_code=True,
        use_auth_token=args.auth_token,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "use_auth_token": args.auth_token,
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if args.device == "cpu":
        load_kwargs["device_map"] = None
    else:
        load_kwargs["device_map"] = {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(args.small_model_name, **load_kwargs)
    if args.device == "cpu":
        model.to("cpu")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return torch, _F, AdamW, model, tokenizer


def iter_batches(records: Sequence[DistillRecord], batch_size: int, shuffle: bool, seed: int) -> Iterable[List[DistillRecord]]:
    indexes = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        yield [records[index] for index in indexes[start : start + batch_size]]


def collate_batch(torch, records: Sequence[DistillRecord], pad_token_id: int, device: str):
    max_len = max(len(record.prefix_token_ids) for record in records)
    input_rows = []
    mask_rows = []
    for record in records:
        prefix = record.prefix_token_ids
        pad_len = max_len - len(prefix)
        input_rows.append([pad_token_id] * pad_len + prefix)
        mask_rows.append([0] * pad_len + [1] * len(prefix))
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    topk_ids = torch.tensor([record.big_topk_ids for record in records], dtype=torch.long, device=device)
    topk_logits = torch.tensor([record.big_topk_logits for record in records], dtype=torch.float32, device=device)
    return input_ids, attention_mask, topk_ids, topk_logits


def next_token_logits(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return outputs.logits[:, -1, :].float()


def distill_loss(F, student_logits, teacher_ids, teacher_logits, tau: float):
    teacher_log_probs = F.log_softmax(teacher_logits / tau, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_on_teacher = student_logits.gather(-1, teacher_ids)
    student_log_probs = F.log_softmax(student_on_teacher / tau, dim=-1)
    cross_entropy = -(teacher_probs * student_log_probs).sum(dim=-1)
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return cross_entropy.mean(), kl.mean(), student_log_probs


def compute_anchor(F, model, input_ids, attention_mask, teacher_ids, student_log_probs, tau: float):
    with model.disable_adapter():
        base_logits = next_token_logits(model, input_ids, attention_mask)
    base_on_teacher = base_logits.gather(-1, teacher_ids)
    base_log_probs = F.log_softmax(base_on_teacher / tau, dim=-1)
    student_probs = student_log_probs.exp()
    return (student_probs * (student_log_probs - base_log_probs)).sum(dim=-1).mean()


def evaluate(torch, F, model, tokenizer, records: Sequence[DistillRecord], args: argparse.Namespace) -> Dict[str, Any]:
    if not records:
        return {"records": 0}
    model.eval()
    kls: List[float] = []
    ces: List[float] = []
    top1_matches = 0
    teacher_mass_hits: List[float] = []
    with torch.no_grad():
        for batch in iter_batches(records, args.batch_size, shuffle=False, seed=args.seed):
            input_ids, attention_mask, teacher_ids, teacher_logits = collate_batch(
                torch, batch, tokenizer.pad_token_id, args.device
            )
            logits = next_token_logits(model, input_ids, attention_mask)
            ce, kl, _student_log_probs = distill_loss(F, logits, teacher_ids, teacher_logits, args.tau)
            ces.append(float(ce.detach().cpu().item()))
            kls.append(float(kl.detach().cpu().item()))

            student_on_teacher = logits.gather(-1, teacher_ids)
            teacher_top1 = teacher_ids[:, 0]
            student_choice = student_on_teacher.argmax(dim=-1)
            student_top1 = teacher_ids.gather(1, student_choice.unsqueeze(1)).squeeze(1)
            top1_matches += int((student_top1 == teacher_top1).sum().detach().cpu().item())

            n = max(1, min(args.eval_teacher_mass_top_n, teacher_ids.shape[-1]))
            teacher_probs = F.softmax(teacher_logits / args.tau, dim=-1)
            student_rank = student_on_teacher.argsort(dim=-1, descending=True)[:, :n]
            batch_mass = teacher_probs.gather(1, student_rank).sum(dim=-1)
            teacher_mass_hits.extend(float(value) for value in batch_mass.detach().cpu().tolist())
    model.train()
    return {
        "records": len(records),
        "mean_forward_kl_topk": mean(kls) if kls else None,
        "mean_forward_ce_topk": mean(ces) if ces else None,
        "student_top1_matches_teacher_rate": top1_matches / len(records),
        f"teacher_mass_in_student_top{args.eval_teacher_mass_top_n}_within_teacher_topk": (
            mean(teacher_mass_hits) if teacher_mass_hits else None
        ),
    }


def train(torch, F, AdamW, model, tokenizer, train_records: Sequence[DistillRecord], args: argparse.Namespace) -> Dict[str, Any]:
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    step_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    global_step = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_losses: List[float] = []
        epoch_kls: List[float] = []
        batch_seed = args.seed + epoch + 1
        for batch in iter_batches(train_records, args.batch_size, shuffle=True, seed=batch_seed):
            input_ids, attention_mask, teacher_ids, teacher_logits = collate_batch(
                torch, batch, tokenizer.pad_token_id, args.device
            )
            logits = next_token_logits(model, input_ids, attention_mask)
            ce, kl, student_log_probs = distill_loss(F, logits, teacher_ids, teacher_logits, args.tau)
            if args.lambda_anchor > 0:
                anchor = compute_anchor(F, model, input_ids, attention_mask, teacher_ids, student_log_probs, args.tau)
                loss = ce + args.lambda_anchor * anchor
            else:
                anchor = torch.zeros((), device=teacher_logits.device)
                loss = ce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_value = float(loss.detach().cpu().item())
            ce_value = float(ce.detach().cpu().item())
            kl_value = float(kl.detach().cpu().item())
            anchor_value = float(anchor.detach().cpu().item())
            epoch_losses.append(loss_value)
            epoch_kls.append(kl_value)
            step_rows.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss_value,
                    "forward_ce_topk": ce_value,
                    "forward_kl_topk": kl_value,
                    "anchor_kl": anchor_value,
                }
            )
            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"step={global_step} epoch={epoch + 1} "
                    f"loss={loss_value:.6f} kl={kl_value:.6f} anchor={anchor_value:.6f}",
                    flush=True,
                )
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "steps": len(epoch_losses),
                "mean_loss": mean(epoch_losses) if epoch_losses else None,
                "mean_forward_kl_topk": mean(epoch_kls) if epoch_kls else None,
            }
        )
    return {
        "train_records": len(train_records),
        "epochs": args.epochs,
        "steps": global_step,
        "duration_sec": time.time() - start_time,
        "epochs_summary": epoch_rows,
        "steps_history": step_rows,
    }


def write_run_plan(args: argparse.Namespace, record_files: Sequence[Path], records: Sequence[DistillRecord], skipped: Dict[str, int]) -> None:
    output_dir = Path(args.output_dir)
    train_records, eval_records = split_records(args, records)
    plan = {
        "args": sanitized_args(args),
        "record_files": [str(path) for path in record_files],
        "skipped": skipped,
        "all": summarize_records(records),
        "train": summarize_records(train_records),
        "eval": summarize_records(eval_records),
        "adapter_dir": str(output_dir / "adapter"),
        "parameter_isolation": {
            "base_small_model_modified": False,
            "saved_artifact": "LoRA adapter only",
            "load_path_argument": "--small_lora_path",
        },
    }
    save_json(output_dir / "dry_run_summary.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_files = discover_record_files(args.records_path)
    records, skipped = load_records(record_files, args.top_k, args.max_prefix_tokens)
    if not records:
        raise ValueError("No usable distillation records found.")
    train_records, eval_records = split_records(args, records)
    if not train_records:
        raise ValueError("No train records after split.")

    metadata = {
        "args": sanitized_args(args),
        "record_files": [str(path) for path in record_files],
        "skipped": skipped,
        "all": summarize_records(records),
        "train": summarize_records(train_records),
        "eval": summarize_records(eval_records),
        "adapter_dir": str(output_dir / "adapter"),
        "parameter_isolation": {
            "base_small_model_modified": False,
            "saved_artifact": "LoRA adapter only",
            "load_path_argument": "--small_lora_path",
        },
    }
    save_json(output_dir / "trainer_metadata.json", metadata)

    if args.dry_run:
        write_run_plan(args, record_files, records, skipped)
        return 0

    torch, F, AdamW, model, tokenizer = load_lora_model(args)
    print("Trainable parameters:")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    eval_before = evaluate(torch, F, model, tokenizer, eval_records, args)
    save_json(output_dir / "eval_before.json", eval_before)
    print(f"eval_before={json.dumps(eval_before, sort_keys=True)}", flush=True)

    train_metrics = train(torch, F, AdamW, model, tokenizer, train_records, args)
    save_json(output_dir / "train_metrics.json", train_metrics)

    eval_after = evaluate(torch, F, model, tokenizer, eval_records, args)
    save_json(output_dir / "eval_after.json", eval_after)
    print(f"eval_after={json.dumps(eval_after, sort_keys=True)}", flush=True)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    save_json(
        output_dir / "done.json",
        {
            "adapter_dir": str(adapter_dir),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "eval_before": eval_before,
            "eval_after": eval_after,
        },
    )
    print(f"Saved Group F1 LoRA adapter to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
