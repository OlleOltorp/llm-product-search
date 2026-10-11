"""Run folders: ranked results, metadata and the committed results tables.

Each run lives in runs/<name>/ (gitignored):
    config.yaml                  Settings used.
    results.parquet              Top RESULT_DEPTH products per query.
    meta.json                    Git commit, machine, versions, cost.
    metrics_<split>.json         Mean and bootstrap interval per metric.
    metrics_per_query_<split>.parquet

results/<split>.csv (committed) holds one row of metric means per run.
"""

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from product_search.eval.cost import thread_settings
from product_search.eval.metrics import rank
from product_search.types import Results

REPO_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_DIR / "runs"
RESULTS_DIR = REPO_DIR / "results"
RESULT_DEPTH = 1000

PACKAGES = ("numpy", "pandas", "bm25s", "torch", "sentence-transformers", "transformers")


def create_run_dir(name: str, runs_dir: Path = RUNS_DIR, force: bool = False) -> Path:
    """Create runs/<name>/.

    Raises:
        FileExistsError: If the run exists and force is not set, so results
            are never overwritten by accident.
    """
    run_dir = runs_dir / name
    if run_dir.exists() and not force:
        raise FileExistsError(f"{run_dir} exists. Pick a new name or use force.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_results(results: Results, path: Path, depth: int = RESULT_DEPTH) -> None:
    """Save the top depth products per query as Parquet (query_id, product_id, rank, score)."""
    rows = [
        (query_id, product_id, position + 1, results[query_id][product_id])
        for query_id, ranking in rank(results, depth).items()
        for position, product_id in enumerate(ranking)
    ]
    table = pd.DataFrame(rows, columns=["query_id", "product_id", "rank", "score"])
    table.to_parquet(path, index=False)


def load_results(path: Path) -> Results:
    """Load results saved by save_results."""
    table = pd.read_parquet(path)
    results: Results = {}
    for query_id, product_id, score in zip(
        table["query_id"], table["product_id"], table["score"], strict=True
    ):
        results.setdefault(query_id, {})[product_id] = float(score)
    return results


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a dict as indented JSON."""
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def git_info() -> dict[str, Any]:
    """Current commit and whether there are uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def machine_info() -> dict[str, Any]:
    """Hardware, OS, power plan, threads and package versions, for meta.json."""
    info: dict[str, Any] = {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": sys.version.split()[0],
        "threads": thread_settings(),
        "gpu": _gpu_name(),
        "power_plan": _power_plan(),
        "packages": {},
    }
    for package in PACKAGES:
        try:
            info["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return info


def update_results_table(path: Path, run: str, summary: dict[str, dict[str, float]]) -> None:
    """Add or replace a run's row (metric means) in a results CSV."""
    row = {"run": run, **{metric: round(s["mean"], 4) for metric, s in summary.items()}}
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["run"])
    table = table[table["run"] != run]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _gpu_name() -> str:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "none"
    return output.stdout.strip()


def _power_plan() -> str:
    if sys.platform != "win32":
        return "n/a"
    try:
        output = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return output.stdout.strip()
