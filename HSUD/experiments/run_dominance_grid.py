#!/usr/bin/env python3
"""Run pre-registered dominance grid settings (A0/A1/A2) via run_benchmark.

After each setting finishes, re-analyze dominance with margin=0.1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_dominance import run_offline_analysis, _print_table
from utils.config import load_config, resolve_path
from utils.logger import get_logger

SETTING_CONFIGS = {
    "A0": "configs/dominance_grid/A0_baseline.yaml",
    "A1": "configs/dominance_grid/A1_reduce_reasoning.yaml",
    "A2": "configs/dominance_grid/A2_nli_cluster.yaml",
}


def run_one(setting: str, max_samples: int | None, datasets: list[str] | None) -> Path:
    logger = get_logger("dominance_grid")
    config_path = PROJECT_ROOT / SETTING_CONFIGS[setting]
    config = load_config(config_path)
    outputs_dir = Path(resolve_path(config["paths"]["outputs_dir"]))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_benchmark.py"),
        "--config",
        str(config_path),
        "--mode",
        "pipeline",
        "--source",
        "local",
    ]
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if datasets:
        cmd.append("--datasets")
        cmd.extend(datasets)
    else:
        cmd.extend(["--datasets", "truthfulqa", "ambigqa", "gsm8k"])

    logger.info("Running %s: %s", setting, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    full_json = outputs_dir / "benchmark_pipeline_full.json"
    if not full_json.exists():
        raise FileNotFoundError(full_json)
    analysis = run_offline_analysis(
        full_json,
        margins=[0.0, 0.1],
        min_term_value=0.1,
        output_dir=outputs_dir / "dominance_analysis",
    )
    logger.info("=== Dominance summary for %s ===", setting)
    _print_table(analysis)
    return full_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Dominance measurement grid A0/A1/A2.")
    parser.add_argument(
        "--setting",
        choices=["A0", "A1", "A2", "all"],
        default="A0",
        help="Which pre-registered setting to run.",
    )
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset override (default: truthfulqa ambigqa gsm8k).",
    )
    args = parser.parse_args()
    settings = ["A0", "A1", "A2"] if args.setting == "all" else [args.setting]
    for setting in settings:
        run_one(setting, max_samples=args.max_samples, datasets=args.datasets)


if __name__ == "__main__":
    main()
