#!/usr/bin/env python3
"""Launch cross-model small-budget suite (wrapper around run_benchmark).

This script does not invent API keys. It runs the listed configs sequentially
and then aggregates with summarize_cross_model.py.

Example (API required):
  export OPENAI_API_KEY=...
  PYTHONPATH=. python experiments/run_cross_model_small.py --execute

Dry-run (print commands only):
  PYTHONPATH=. python experiments/run_cross_model_small.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = (
    "configs/cross_model/small_gpt4o_mini.yaml",
    "configs/cross_model/small_gpt35_turbo.yaml",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="YAML configs to run",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually run benchmarks (default: print commands only)",
    )
    p.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Do not run summarize_cross_model.py after executes",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmds: list[list[str]] = []
    out_dirs: list[str] = []
    for cfg in args.configs:
        cmds.append(
            [
                sys.executable,
                "experiments/run_benchmark.py",
                "--config",
                cfg,
                "--mode",
                "pipeline",
                "--source",
                "local",
                "--fresh",
            ]
        )
        # Infer output dir from yaml without full parse dependency beyond Path name
        stem = Path(cfg).stem
        out_dirs.append(f"outputs/cross_model/{stem}")

    print("Cross-model small-budget plan:")
    for cmd in cmds:
        print(" ", " ".join(cmd))

    if not args.execute:
        print("\nDry-run only. Re-run with --execute after setting OPENAI_API_KEY.")
        return

    for cmd in cmds:
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    if not args.skip_aggregate:
        agg = [
            sys.executable,
            "experiments/summarize_cross_model.py",
            "--runs",
            *out_dirs,
            "--full-reference",
            "outputs/benchmark_g8_fixedR_nli",
            "--out-dir",
            "outputs/cross_model/summary",
        ]
        print("\n>>>", " ".join(agg), flush=True)
        subprocess.run(agg, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
