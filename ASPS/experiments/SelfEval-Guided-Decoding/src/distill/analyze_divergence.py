import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ASPS trigger-local distillation records.")
    parser.add_argument("paths", nargs="+", help="trigger_records.jsonl files or directories containing them")
    parser.add_argument("--output-dir", default="", help="Analysis output directory. Defaults to <first input>/analysis")
    parser.add_argument("--bins", default=6, type=int, help="Number of bins for divergence-vs-rate plots")
    return parser.parse_args()


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
    deduped = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def load_records(files: Iterable[Path]) -> Iterable[Dict]:
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                record["_source_file"] = str(path)
                yield record


def log_probs_from_logits(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    max_logit = max(logits)
    denom = max_logit + math.log(sum(math.exp(value - max_logit) for value in logits))
    return [value - denom for value in logits]


def kl_from_log_probs(log_p: Sequence[float], log_q: Sequence[float]) -> float:
    total = 0.0
    for lp, lq in zip(log_p, log_q):
        p = math.exp(lp)
        total += p * (lp - lq)
    return total


def js_from_log_probs(log_p: Sequence[float], log_q: Sequence[float]) -> float:
    p = [math.exp(value) for value in log_p]
    q = [math.exp(value) for value in log_q]
    m = [0.5 * (left + right) for left, right in zip(p, q)]
    log_m = [math.log(max(value, 1e-45)) for value in m]
    return 0.5 * kl_from_log_probs(log_p, log_m) + 0.5 * kl_from_log_probs(log_q, log_m)


def map_logits(ids: Sequence[int], logits: Sequence[float]) -> Dict[int, float]:
    return {int(token_id): float(logit) for token_id, logit in zip(ids, logits)}


def union_logits(record: Dict) -> Tuple[List[int], List[float], List[float], bool]:
    big_topk_ids = [int(token_id) for token_id in record.get("big_topk_ids", [])]
    small_topk_ids = [int(token_id) for token_id in record.get("small_topk_ids", [])]
    big_map = map_logits(big_topk_ids, record.get("big_topk_logits", []))
    big_map.update(map_logits(small_topk_ids, record.get("big_logits_on_small_topk", [])))
    small_map = map_logits(small_topk_ids, record.get("small_topk_logits", []))
    small_map.update(map_logits(big_topk_ids, record.get("small_logits_on_big_topk", [])))

    union_ids = sorted(set(big_topk_ids) | set(small_topk_ids))
    exact = bool(union_ids) and all(token_id in big_map and token_id in small_map for token_id in union_ids)
    if not exact:
        union_ids = [token_id for token_id in union_ids if token_id in big_map and token_id in small_map]
    return union_ids, [big_map[token_id] for token_id in union_ids], [small_map[token_id] for token_id in union_ids], exact


def support_metrics(record: Dict) -> Dict[str, Optional[float]]:
    union_ids, big_logits, small_logits, exact_union = union_logits(record)
    out: Dict[str, Optional[float]] = {
        "kl_big_small_union": None,
        "kl_small_big_union": None,
        "js_union": None,
        "exact_union_logits": exact_union,
        "union_size": len(union_ids),
    }
    if union_ids:
        log_p_big = log_probs_from_logits(big_logits)
        log_p_small = log_probs_from_logits(small_logits)
        out["kl_big_small_union"] = kl_from_log_probs(log_p_big, log_p_small)
        out["kl_small_big_union"] = kl_from_log_probs(log_p_small, log_p_big)
        out["js_union"] = js_from_log_probs(log_p_big, log_p_small)

    big_ids = [int(token_id) for token_id in record.get("big_topk_ids", [])]
    small_on_big = record.get("small_logits_on_big_topk", [])
    if big_ids and len(small_on_big) == len(big_ids):
        log_p_big_topk = log_probs_from_logits(record.get("big_topk_logits", []))
        log_p_small_on_big = log_probs_from_logits(small_on_big)
        out["kl_big_small_big_topk"] = kl_from_log_probs(log_p_big_topk, log_p_small_on_big)
    else:
        out["kl_big_small_big_topk"] = None

    big_set = set(big_ids)
    small_set = {int(token_id) for token_id in record.get("small_topk_ids", [])}
    overlap_denom = len(big_set | small_set)
    out["topk_overlap_jaccard"] = len(big_set & small_set) / overlap_denom if overlap_denom else None
    out["entropy_gap"] = (
        float(record["small_entropy"]) - float(record["big_entropy"])
        if record.get("small_entropy") is not None and record.get("big_entropy") is not None
        else None
    )
    out["first_token_coverage"] = first_token_coverage(record)
    return out


def first_token_coverage(record: Dict) -> Optional[float]:
    big_ids = [int(token_id) for token_id in record.get("big_topk_ids", [])]
    big_logits = record.get("big_topk_logits", [])
    first_ids = {int(token_id) for token_id in record.get("draft_first_token_ids", [])}
    if not big_ids or not big_logits or not first_ids:
        return None
    log_probs = log_probs_from_logits(big_logits)
    return sum(math.exp(log_prob) for token_id, log_prob in zip(big_ids, log_probs) if token_id in first_ids)


def enrich_record(record: Dict) -> Dict:
    metrics = support_metrics(record)
    row = {
        "record_id": record.get("record_id"),
        "run_id": record.get("run_id"),
        "dataset": record.get("dataset"),
        "question_id": record.get("question_id"),
        "trigger_step": record.get("trigger_step"),
        "resolution": record.get("resolution"),
        "chosen_source": record.get("chosen_source"),
        "switch_happened": bool(record.get("switch_happened")),
        "final_correct": record.get("final_correct"),
        "verify_mode": record.get("verify_mode"),
        "big_entropy": record.get("big_entropy"),
        "small_entropy": record.get("small_entropy"),
        "draft_count": len(record.get("draft_path_token_ids", [])),
        "source_file": record.get("_source_file"),
    }
    row.update(metrics)
    return row


def mean_optional(rows: Sequence[Dict], key: str) -> Optional[float]:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def rate(rows: Sequence[Dict], key: str) -> Optional[float]:
    if not rows:
        return None
    return sum(1 for row in rows if row.get(key)) / len(rows)


def summarize(rows: Sequence[Dict]) -> Dict:
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset") or "")].append(row)

    def summarize_rows(items: Sequence[Dict]) -> Dict:
        return {
            "records": len(items),
            "verified_records": sum(1 for row in items if row.get("resolution") == "verified"),
            "exact_union_records": sum(1 for row in items if row.get("exact_union_logits")),
            "small_draft_win_rate": rate([row for row in items if row.get("resolution") == "verified"], "switch_happened"),
            "mean_kl_big_small_union": mean_optional(items, "kl_big_small_union"),
            "mean_kl_big_small_big_topk": mean_optional(items, "kl_big_small_big_topk"),
            "mean_js_union": mean_optional(items, "js_union"),
            "mean_topk_overlap_jaccard": mean_optional(items, "topk_overlap_jaccard"),
            "mean_first_token_coverage": mean_optional(items, "first_token_coverage"),
            "mean_entropy_gap": mean_optional(items, "entropy_gap"),
        }

    return {
        "overall": summarize_rows(rows),
        "by_dataset": {dataset: summarize_rows(items) for dataset, items in sorted(by_dataset.items())},
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    fieldnames = [
        "record_id",
        "run_id",
        "dataset",
        "question_id",
        "trigger_step",
        "resolution",
        "chosen_source",
        "switch_happened",
        "final_correct",
        "verify_mode",
        "big_entropy",
        "small_entropy",
        "entropy_gap",
        "kl_big_small_union",
        "kl_small_big_union",
        "kl_big_small_big_topk",
        "js_union",
        "topk_overlap_jaccard",
        "first_token_coverage",
        "union_size",
        "exact_union_logits",
        "draft_count",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def binned_rate(rows: Sequence[Dict], value_key: str, flag_key: str, bins: int) -> List[Dict]:
    usable = [row for row in rows if row.get(value_key) is not None]
    usable.sort(key=lambda row: row[value_key])
    if not usable:
        return []
    bins = max(1, min(bins, len(usable)))
    chunk_size = math.ceil(len(usable) / bins)
    out = []
    for start in range(0, len(usable), chunk_size):
        chunk = usable[start : start + chunk_size]
        values = [row[value_key] for row in chunk]
        out.append(
            {
                "x_mid": mean(values),
                "x_min": min(values),
                "x_max": max(values),
                "n": len(chunk),
                "rate": rate(chunk, flag_key),
            }
        )
    return out


def maybe_write_plots(output_dir: Path, rows: Sequence[Dict], bins: int) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written = []
    verified = [row for row in rows if row.get("resolution") == "verified"]
    for flag_key, name, ylabel in [
        ("switch_happened", "divergence_vs_small_win_rate.png", "Small draft win rate"),
        ("final_correct", "divergence_vs_correct_rate.png", "Final correct rate"),
    ]:
        plot_rows = verified
        if flag_key == "final_correct":
            plot_rows = [row for row in verified if row.get("final_correct") is not None]
        points = binned_rate(plot_rows, "kl_big_small_big_topk", flag_key, bins)
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([point["x_mid"] for point in points], [point["rate"] for point in points], width=0.02)
        ax.set_xlabel("KL(p_big || p_small) on big top-K")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by divergence bin")
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    by_dataset: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get("kl_big_small_big_topk")
        if value is not None:
            by_dataset[str(row.get("dataset") or "")].append(value)
    if by_dataset:
        fig, ax = plt.subplots(figsize=(7, 4))
        for dataset, values in sorted(by_dataset.items()):
            ax.hist(values, bins=20, alpha=0.45, label=dataset)
        ax.set_xlabel("KL(p_big || p_small) on big top-K")
        ax.set_ylabel("Trigger count")
        ax.set_title("Task-wise trigger divergence")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "taskwise_divergence_hist.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def default_output_dir(first_input: str) -> Path:
    path = Path(first_input)
    if path.is_file():
        return path.parent / "analysis"
    return path / "analysis"


def main() -> None:
    args = parse_args()
    files = discover_record_files(args.paths)
    if not files:
        raise SystemExit("No trigger_records.jsonl files found.")
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.paths[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [enrich_record(record) for record in load_records(files)]
    write_csv(output_dir / "trigger_metrics.csv", rows)
    summary = summarize(rows)
    summary["record_files"] = [str(path) for path in files]
    summary["plots"] = maybe_write_plots(output_dir, rows, args.bins)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
