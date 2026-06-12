import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import yaml


GROUP_SCRIPTS = {
    "baseline": "generate_code_baseline_llama3.1.py",
    "group_a": "generate_code_groupa_llama.py",
    "group_b": "generate_code_groupb_reflect_llama.py",
    "group_c": "generate_code_groupc_llama.py",
    "group_d": "generate_code_groupd_cautious_llama.py",
    "group_e": "generate_code_groupe_tot_llama.py",
}

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SRC_DIR = "ASPS/experiments/SelfEval-Guided-Decoding/src"
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
ACCURACY_PATTERN = re.compile(r"Accuracy:\s*([0-9.]+)\s*\((\d+)\s*/\s*(\d+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASPS experiments from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Create the run directory and print commands only.")
    return parser.parse_args()


def expand_env_string(value: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return ENV_PATTERN.sub(replace, value)


def expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_string(value)
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return expand_env_values(config)


def resolve_repo_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "experiment"


def create_run_dir(config: Dict[str, Any], config_path: Path) -> Path:
    experiment = config.get("experiment", {})
    output_root = resolve_repo_path(experiment.get("output_root", "outputs")).resolve()
    explicit_output_dir = experiment.get("output_dir")
    if explicit_output_dir:
        run_dir = resolve_repo_path(explicit_output_dir).resolve()
    else:
        name = slugify(str(experiment.get("name") or config_path.stem))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / f"{timestamp}_{name}"

    for child in ("logs", "results", "checkpoints"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    return run_dir


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def log_line(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def bool_arg_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def add_cli_arg(command: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def script_for_group(group_name: str, group_config: Dict[str, Any]) -> str:
    script = group_config.get("script")
    if script:
        return str(script)
    if group_name not in GROUP_SCRIPTS:
        raise ValueError(f"Unknown group '{group_name}'. Set groups.{group_name}.script explicitly.")
    return GROUP_SCRIPTS[group_name]


def base_args_for_group(
    group_name: str,
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    group_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    models = config.get("models", {})
    common_args = dict(config.get("common_args", {}))
    group_args = dict(group_config.get("args", {}))

    dataset_args = {
        "dt_name": dataset.get("dt_name", dataset.get("name")),
        "input_file": str(resolve_repo_path(dataset["input_file"]).resolve()) if dataset.get("input_file") else None,
        "start": dataset.get("start", common_args.pop("start", 0)),
        "end": dataset.get("end", common_args.pop("end", -1)),
        "output_dir": str(output_dir.resolve()),
    }
    for key, value in dataset.items():
        if key not in {"name", "dt_name", "input_file", "label"}:
            dataset_args[key] = value

    model_args: Dict[str, Any] = {}
    if group_name in {"group_a", "group_c"}:
        model_args.update(
            {
                "big_model_name": group_args.pop("big_model_name", models.get("big_model")),
                "small_model_name": group_args.pop("small_model_name", models.get("small_model")),
                "auth_token": group_args.pop("auth_token", models.get("auth_token")),
                "big_device": group_args.pop("big_device", models.get("big_device")),
                "small_device": group_args.pop("small_device", models.get("small_device")),
            }
        )
    else:
        model_args.update(
            {
                "model_name": group_args.pop("model_name", models.get("big_model")),
                "auth_token": group_args.pop("auth_token", models.get("auth_token")),
            }
        )
        if group_name in {"group_b", "group_e"}:
            model_args["device"] = group_args.pop("device", models.get("big_device"))

    merged: Dict[str, Any] = {}
    merged.update(common_args)
    merged.update(model_args)
    merged.update(dataset_args)
    merged.update(group_args)
    return {key: value for key, value in merged.items() if value is not None}


def build_jobs(config: Dict[str, Any], run_dir: Path) -> List[Dict[str, Any]]:
    experiment = config.get("experiment", {})
    runtime = config.get("runtime", {})
    src_dir = resolve_repo_path(experiment.get("src_dir", DEFAULT_SRC_DIR)).resolve()
    python_bin = runtime.get("python", sys.executable)

    datasets = config.get("datasets", [])
    groups = config.get("groups", {})
    if not datasets:
        raise ValueError("Config must include at least one dataset.")
    if not groups:
        raise ValueError("Config must include at least one group.")

    jobs: List[Dict[str, Any]] = []
    for dataset in datasets:
        dataset_name = str(dataset.get("name") or dataset.get("dt_name"))
        if not dataset_name or not dataset.get("input_file"):
            raise ValueError("Each dataset must include name/dt_name and input_file.")

        for group_name, raw_group_config in groups.items():
            group_config = raw_group_config or {}
            if not bool_arg_enabled(group_config.get("enabled", True)):
                continue

            result_dir = run_dir / "results" / dataset_name / group_name
            result_dir.mkdir(parents=True, exist_ok=False)
            script = script_for_group(group_name, group_config)
            script_path = Path(script)
            if not script_path.is_absolute() and not (src_dir / script_path).exists():
                raise FileNotFoundError(f"Script for group '{group_name}' not found: {src_dir / script_path}")
            if script_path.is_absolute() and not script_path.exists():
                raise FileNotFoundError(f"Script for group '{group_name}' not found: {script_path}")
            args = base_args_for_group(group_name, config, dataset, group_config, result_dir)

            command = [str(python_bin), str(script_path)]
            for key, value in args.items():
                add_cli_arg(command, key, value)

            job_env = dict(config.get("env", {}))
            job_env.update(group_config.get("env", {}) or {})
            jobs.append(
                {
                    "dataset": dataset_name,
                    "group": group_name,
                    "cwd": str(src_dir),
                    "command": command,
                    "env": {str(k): str(v) for k, v in job_env.items() if v is not None},
                    "result_dir": str(result_dir),
                }
            )
    return jobs


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_job(job: Dict[str, Any], log_path: Path) -> int:
    env = os.environ.copy()
    env.update(job.get("env", {}))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ cd {job['cwd']}\n")
        handle.write(f"$ {shell_join(job['command'])}\n\n")
        handle.flush()
        process = subprocess.Popen(
            job["command"],
            cwd=job["cwd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return process.wait()


def parse_summary(summary_path: Path, run_dir: Path) -> Dict[str, Any]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    accuracy = None
    correct = None
    total = None
    match = ACCURACY_PATTERN.search(text)
    if match:
        accuracy = float(match.group(1))
        correct = int(match.group(2))
        total = int(match.group(3))

    rel = summary_path.relative_to(run_dir)
    parts = rel.parts
    dataset = parts[1] if len(parts) > 2 and parts[0] == "results" else ""
    group = parts[2] if len(parts) > 3 and parts[0] == "results" else ""
    return {
        "summary_file": str(summary_path),
        "dataset": dataset,
        "group": group,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def collect_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    summaries = sorted(run_dir.glob("results/**/*_summary.txt"))
    return [parse_summary(path, run_dir) for path in summaries]


def select_best_metric(config: Dict[str, Any], metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_config = config.get("metric", {})
    metric_name = metric_config.get("name", "accuracy")
    mode = metric_config.get("mode", "max")
    valid = [row for row in metrics if row.get(metric_name) is not None]
    if not valid:
        return {"metric": metric_name, "mode": mode, "value": None, "row": None}
    reverse = mode != "min"
    best = sorted(valid, key=lambda row: row[metric_name], reverse=reverse)[0]
    return {"metric": metric_name, "mode": mode, "value": best[metric_name], "row": best}


def checkpoint_state(
    run_dir: Path,
    config_path: Path,
    jobs: List[Dict[str, Any]],
    job_results: List[Dict[str, Any]],
    best_metric: Dict[str, Any],
    status: str,
) -> Dict[str, Any]:
    return {
        "status": status,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "jobs": job_results,
        "job_count": len(jobs),
        "completed_jobs": sum(1 for row in job_results if row.get("returncode") == 0),
        "best_metric": best_metric,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.dry_run:
        config.setdefault("experiment", {})["dry_run"] = True

    run_dir = create_run_dir(config, config_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    train_log = run_dir / "logs" / "train.log"
    save_yaml(run_dir / "config.yaml", config)

    jobs = build_jobs(config, run_dir)
    metadata = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "started_at": started_at,
        "dry_run": bool(config.get("experiment", {}).get("dry_run", False)),
    }
    save_json(run_dir / "metadata.json", metadata)

    log_line(train_log, f"run_dir: {run_dir}")
    log_line(train_log, f"config: {config_path}")
    log_line(train_log, f"jobs: {len(jobs)}")

    dry_run = bool(config.get("experiment", {}).get("dry_run", False))
    stop_on_failure = bool_arg_enabled(config.get("experiment", {}).get("stop_on_failure", True))
    job_results: List[Dict[str, Any]] = []

    for index, job in enumerate(jobs, start=1):
        name = f"{index:03d}_{job['dataset']}_{job['group']}"
        job_log = run_dir / "logs" / f"{name}.log"
        command_text = shell_join(job["command"])
        log_line(train_log, f"[{name}] {command_text}")
        job_started_at = datetime.now().isoformat(timespec="seconds")
        job_started = time.time()
        if dry_run:
            log_line(job_log, f"$ cd {job['cwd']}")
            log_line(job_log, f"$ {command_text}")
            returncode = 0
        else:
            returncode = run_job(job, job_log)
        duration_sec = time.time() - job_started
        job_finished_at = datetime.now().isoformat(timespec="seconds")

        result = {
            "name": name,
            "dataset": job["dataset"],
            "group": job["group"],
            "returncode": returncode,
            "started_at": job_started_at,
            "finished_at": job_finished_at,
            "duration_sec": duration_sec,
            "log_file": str(job_log),
            "result_dir": job["result_dir"],
            "command": command_text,
        }
        job_results.append(result)
        save_json(run_dir / "jobs.json", {"jobs": job_results})
        save_json(run_dir / "checkpoints" / "experiment_state.json", {"jobs": job_results})
        if returncode != 0 and stop_on_failure:
            log_line(train_log, f"[{name}] failed with return code {returncode}; stopping.")
            break

    metrics = collect_metrics(run_dir)
    best_metric = select_best_metric(config, metrics)
    failed = [row for row in job_results if row["returncode"] != 0]
    status = "dry_run" if dry_run else ("failed" if failed else "completed")
    finished_at = datetime.now().isoformat(timespec="seconds")

    save_json(run_dir / "metrics.json", {"metrics": metrics})
    save_json(run_dir / "best_metric.json", best_metric)
    save_json(run_dir / "jobs.json", {"jobs": job_results})
    state = checkpoint_state(run_dir, config_path, jobs, job_results, best_metric, status)
    save_json(run_dir / "checkpoints" / "experiment_state.json", state)
    metadata.update({"finished_at": finished_at, "status": status})
    save_json(run_dir / "metadata.json", metadata)

    log_line(train_log, f"status: {status}")
    log_line(train_log, f"best_metric: {json.dumps(best_metric, ensure_ascii=False)}")
    print(f"Run directory: {run_dir}")
    print(f"Status: {status}")
    print(f"Best metric: {best_metric}")
    return 1 if failed and not dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
