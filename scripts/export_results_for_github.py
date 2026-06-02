import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


SKIP_DIRS = {"checkpoints", "__pycache__"}
SKIP_SUFFIXES = {
    ".bin",
    ".chk",
    ".ckpt",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy experiment results into a GitHub-friendly export directory.")
    parser.add_argument("--outputs", default="outputs", help="Root outputs directory to export from.")
    parser.add_argument("--registry", default="experiments/registry.csv", help="Registry CSV to include if it exists.")
    parser.add_argument("--export-root", default="experiments/result_exports", help="Directory for exported result bundles.")
    parser.add_argument("--name", default="", help="Export bundle name. Defaults to timestamped group_ac_results.")
    return parser.parse_args()


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return path.suffix in SKIP_SUFFIXES


def copy_tree(outputs_root: Path, export_dir: Path) -> Dict[str, int]:
    counts = {"files": 0, "bytes": 0}
    exported_outputs = export_dir / "outputs"
    for source in sorted(outputs_root.rglob("*")):
        if source.is_dir() or should_skip(source):
            continue
        rel = source.relative_to(outputs_root)
        target = exported_outputs / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        counts["files"] += 1
        counts["bytes"] += target.stat().st_size
    return counts


def write_manifest(export_dir: Path, args: argparse.Namespace, counts: Dict[str, int]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": args.outputs,
        "registry": args.registry,
        "export_dir": str(export_dir),
        "excluded_dirs": sorted(SKIP_DIRS),
        "excluded_suffixes": sorted(SKIP_SUFFIXES),
        "file_count": counts["files"],
        "byte_count": counts["bytes"],
    }
    with (export_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    outputs_root = Path(args.outputs)
    export_root = Path(args.export_root)
    name = args.name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_group_ac_results"
    export_dir = export_root / name
    export_dir.mkdir(parents=True, exist_ok=False)

    counts = copy_tree(outputs_root, export_dir)
    registry = Path(args.registry)
    if registry.exists():
        shutil.copy2(registry, export_dir / "registry.csv")
        counts["files"] += 1
        counts["bytes"] += (export_dir / "registry.csv").stat().st_size
    write_manifest(export_dir, args, counts)
    print(f"Exported {counts['files']} files to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
