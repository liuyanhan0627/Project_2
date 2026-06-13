import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


ACCURACY_PATTERN = re.compile(r"Accuracy:\s*([0-9.]+)\s*\((\d+)\s*/\s*(\d+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ASPS experiment outputs into a registry CSV.")
    parser.add_argument("--outputs", default="outputs", help="Root outputs directory to scan.")
    parser.add_argument("--registry", default="experiments/registry.csv", help="CSV file to write.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_accuracy(summary_path: Path) -> Dict[str, Any]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    match = ACCURACY_PATTERN.search(text)
    if not match:
        return {"accuracy": None, "correct": None, "total": None}
    return {
        "accuracy": float(match.group(1)),
        "correct": int(match.group(2)),
        "total": int(match.group(3)),
    }


def p90(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = int(0.9 * (len(ordered) - 1))
    return ordered[index]


def flatten_numbers(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            out.append(float(value))
        elif isinstance(value, list):
            out.extend(flatten_numbers(value))
    return out


def jsonl_records(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "index" in obj:
                    yield obj


def summarize_jsonl_metrics(result_dir: Path, group: str) -> Dict[str, Any]:
    jsonl_files = sorted(result_dir.glob("*.jsonl"))
    records = list(jsonl_records(jsonl_files))
    metric_key = {
        "group_a": "groupa_metrics",
        "group_c": "groupc_metrics",
        "group_b": "groupb_metrics",
        "group_d": "groupd_metrics",
        "group_e": "groupe_metrics",
        "group_f0": "groupa_metrics",
        "group_f1": "groupa_metrics",
    }.get(group)
    metrics = [record.get(metric_key, {}) for record in records] if metric_key else []
    metrics = [metric for metric in metrics if isinstance(metric, dict)]

    sample_wall_times = [float(metric["wall_time"]) for metric in metrics if isinstance(metric.get("wall_time"), (int, float))]
    accepted_tokens = flatten_numbers(metric.get("accepted_tokens_per_verify", []) for metric in metrics)
    avg_state_scores = [
        float(metric["avg_state_score"])
        for metric in metrics
        if isinstance(metric.get("avg_state_score"), (int, float))
    ]
    best_state_scores = [
        float(metric["best_state_score"])
        for metric in metrics
        if isinstance(metric.get("best_state_score"), (int, float))
    ]

    return {
        "jsonl_files": len(jsonl_files),
        "sample_count": len(records),
        "avg_sample_wall_time": statistics.mean(sample_wall_times) if sample_wall_times else None,
        "p90_sample_wall_time": p90(sample_wall_times),
        "entropy_triggers": sum(int(metric.get("entropy_triggers", 0)) for metric in metrics),
        "verify_calls": sum(int(metric.get("verify_calls", 0)) for metric in metrics),
        "switches": sum(int(metric.get("switches", 0)) for metric in metrics),
        "late_draft_drops": sum(int(metric.get("late_draft_drops", 0)) for metric in metrics),
        "small_draft_wins": sum(int(metric.get("small_draft_wins", 0)) for metric in metrics),
        "fallback_wins": sum(int(metric.get("fallback_wins", 0)) for metric in metrics),
        "length_weight_overrides": sum(int(metric.get("length_weight_overrides", 0)) for metric in metrics),
        "switch_margin_rejections": sum(int(metric.get("switch_margin_rejections", 0)) for metric in metrics),
        "avg_accepted_tokens_per_verify": statistics.mean(accepted_tokens) if accepted_tokens else None,
        "restart_count": sum(1 for metric in metrics if metric.get("should_restart") is True),
        "tot_generate_calls": sum(int(metric.get("generate_calls", 0)) for metric in metrics),
        "tot_evaluate_calls": sum(int(metric.get("evaluate_calls", 0)) for metric in metrics),
        "tot_final_calls": sum(int(metric.get("final_calls", 0)) for metric in metrics),
        "tot_generated_thoughts": sum(int(metric.get("generated_thoughts", 0)) for metric in metrics),
        "tot_evaluated_states": sum(int(metric.get("evaluated_states", 0)) for metric in metrics),
        "tot_avg_state_score": statistics.mean(avg_state_scores) if avg_state_scores else None,
        "tot_best_state_score": statistics.mean(best_state_scores) if best_state_scores else None,
        "big_input_tokens": sum(int(metric.get("big_input_tokens", 0)) for metric in metrics),
        "big_output_tokens": sum(int(metric.get("big_output_tokens", 0)) for metric in metrics),
    }


def job_lookup(run_dir: Path) -> Dict[tuple, Dict[str, Any]]:
    jobs = load_json(run_dir / "jobs.json").get("jobs", [])
    lookup = {}
    for job in jobs:
        if isinstance(job, dict):
            lookup[(job.get("dataset", ""), job.get("group", ""))] = job
    return lookup


def infer_dataset_group(run_dir: Path, summary_path: Path) -> Dict[str, str]:
    try:
        rel = summary_path.relative_to(run_dir)
    except ValueError:
        return {"dataset": "", "group": ""}
    parts = rel.parts
    if len(parts) >= 4 and parts[0] == "results":
        return {"dataset": parts[1], "group": parts[2]}
    return {"dataset": "", "group": ""}


def best_row_marker(best_metric: Dict[str, Any], summary_path: Path) -> bool:
    row = best_metric.get("row")
    if not isinstance(row, dict):
        return False
    return Path(str(row.get("summary_file", ""))).name == summary_path.name


def row_for_summary(run_dir: Path, summary_path: Path) -> Dict[str, Any]:
    metadata = load_json(run_dir / "metadata.json")
    best_metric = load_json(run_dir / "best_metric.json")
    config = load_yaml(run_dir / "config.yaml")
    parsed = parse_accuracy(summary_path)
    dataset_group = infer_dataset_group(run_dir, summary_path)
    jobs = job_lookup(run_dir)
    job = jobs.get((dataset_group["dataset"], dataset_group["group"]), {})
    metric_summary = summarize_jsonl_metrics(summary_path.parent, dataset_group["group"])
    experiment = config.get("experiment", {}) if isinstance(config, dict) else {}
    return {
        "run_id": run_dir.name,
        "experiment_name": experiment.get("name", ""),
        "status": metadata.get("status", ""),
        "started_at": metadata.get("started_at", ""),
        "finished_at": metadata.get("finished_at", ""),
        "run_dir": str(run_dir),
        "dataset": dataset_group["dataset"],
        "group": dataset_group["group"],
        "accuracy": parsed["accuracy"],
        "correct": parsed["correct"],
        "total": parsed["total"],
        "job_duration_sec": job.get("duration_sec"),
        "avg_sample_wall_time": metric_summary["avg_sample_wall_time"],
        "p90_sample_wall_time": metric_summary["p90_sample_wall_time"],
        "sample_count": metric_summary["sample_count"],
        "jsonl_files": metric_summary["jsonl_files"],
        "entropy_triggers": metric_summary["entropy_triggers"],
        "verify_calls": metric_summary["verify_calls"],
        "switches": metric_summary["switches"],
        "late_draft_drops": metric_summary["late_draft_drops"],
        "small_draft_wins": metric_summary["small_draft_wins"],
        "fallback_wins": metric_summary["fallback_wins"],
        "length_weight_overrides": metric_summary["length_weight_overrides"],
        "switch_margin_rejections": metric_summary["switch_margin_rejections"],
        "avg_accepted_tokens_per_verify": metric_summary["avg_accepted_tokens_per_verify"],
        "restart_count": metric_summary["restart_count"],
        "tot_generate_calls": metric_summary["tot_generate_calls"],
        "tot_evaluate_calls": metric_summary["tot_evaluate_calls"],
        "tot_final_calls": metric_summary["tot_final_calls"],
        "tot_generated_thoughts": metric_summary["tot_generated_thoughts"],
        "tot_evaluated_states": metric_summary["tot_evaluated_states"],
        "tot_avg_state_score": metric_summary["tot_avg_state_score"],
        "tot_best_state_score": metric_summary["tot_best_state_score"],
        "big_input_tokens": metric_summary["big_input_tokens"],
        "big_output_tokens": metric_summary["big_output_tokens"],
        "is_best": best_row_marker(best_metric, summary_path),
        "summary_file": str(summary_path),
        "config_path": metadata.get("config_path", ""),
    }


def row_for_run_without_summary(run_dir: Path) -> Dict[str, Any]:
    metadata = load_json(run_dir / "metadata.json")
    best_metric = load_json(run_dir / "best_metric.json")
    config = load_yaml(run_dir / "config.yaml")
    experiment = config.get("experiment", {}) if isinstance(config, dict) else {}
    return {
        "run_id": run_dir.name,
        "experiment_name": experiment.get("name", ""),
        "status": metadata.get("status", ""),
        "started_at": metadata.get("started_at", ""),
        "finished_at": metadata.get("finished_at", ""),
        "run_dir": str(run_dir),
        "dataset": "",
        "group": "",
        "accuracy": best_metric.get("value"),
        "correct": "",
        "total": "",
        "job_duration_sec": "",
        "avg_sample_wall_time": "",
        "p90_sample_wall_time": "",
        "sample_count": "",
        "jsonl_files": "",
        "entropy_triggers": "",
        "verify_calls": "",
        "switches": "",
        "late_draft_drops": "",
        "small_draft_wins": "",
        "fallback_wins": "",
        "length_weight_overrides": "",
        "switch_margin_rejections": "",
        "avg_accepted_tokens_per_verify": "",
        "restart_count": "",
        "tot_generate_calls": "",
        "tot_evaluate_calls": "",
        "tot_final_calls": "",
        "tot_generated_thoughts": "",
        "tot_evaluated_states": "",
        "tot_avg_state_score": "",
        "tot_best_state_score": "",
        "big_input_tokens": "",
        "big_output_tokens": "",
        "is_best": "",
        "summary_file": "",
        "config_path": metadata.get("config_path", ""),
    }


def collect_rows(outputs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_dirs = sorted(path.parent for path in outputs_root.glob("**/metadata.json"))
    for run_dir in run_dirs:
        summaries = sorted(run_dir.glob("results/**/*_summary.txt"))
        if summaries:
            rows.extend(row_for_summary(run_dir, summary) for summary in summaries)
        else:
            rows.append(row_for_run_without_summary(run_dir))
    return rows


def write_registry(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "experiment_name",
        "status",
        "started_at",
        "finished_at",
        "run_dir",
        "dataset",
        "group",
        "accuracy",
        "correct",
        "total",
        "job_duration_sec",
        "avg_sample_wall_time",
        "p90_sample_wall_time",
        "sample_count",
        "jsonl_files",
        "entropy_triggers",
        "verify_calls",
        "switches",
        "late_draft_drops",
        "small_draft_wins",
        "fallback_wins",
        "length_weight_overrides",
        "switch_margin_rejections",
        "avg_accepted_tokens_per_verify",
        "restart_count",
        "tot_generate_calls",
        "tot_evaluate_calls",
        "tot_final_calls",
        "tot_generated_thoughts",
        "tot_evaluated_states",
        "tot_avg_state_score",
        "tot_best_state_score",
        "big_input_tokens",
        "big_output_tokens",
        "is_best",
        "summary_file",
        "config_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = collect_rows(Path(args.outputs))
    write_registry(Path(args.registry), rows)
    print(f"Wrote {len(rows)} rows to {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
