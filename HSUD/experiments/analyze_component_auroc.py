"""Compute per-component AUROC from benchmark full-score JSON (Table 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.evaluation.metrics import auroc
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


SCORE_FIELDS: tuple[tuple[str, str], ...] = (
    ("u_theta", "U_Theta"),
    ("u_z", "U_Z"),
    ("u_r", "U_R"),
    ("u_res", "U_res"),
    ("nested_total", "H(S|x)"),
    ("semantic_entropy", "Semantic Entropy"),
)


def load_full_results(path: Path) -> dict[str, Any]:
    """Load benchmark full JSON written by ``run_benchmark.py``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "scores" not in payload:
        raise ValueError(f"Missing 'scores' in {path}; re-run run_benchmark.py first.")
    return payload


def compute_component_auroc(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute AUROC for each uncertainty score field."""

    labels = np.array([int(item["label"]) for item in scores], dtype=int)
    overall: dict[str, float | None] = {}
    for field, _label in SCORE_FIELDS:
        values = np.array([float(item[field]) for item in scores], dtype=float)
        overall[field] = _safe_auroc(labels, values)

    per_dataset: dict[str, dict[str, float | None]] = {}
    for dataset in sorted({str(item["dataset"]) for item in scores}):
        subset = [item for item in scores if item["dataset"] == dataset]
        y = np.array([int(item["label"]) for item in subset], dtype=int)
        per_dataset[dataset] = {"count": len(subset)}
        for field, _label in SCORE_FIELDS:
            values = np.array([float(item[field]) for item in subset], dtype=float)
            per_dataset[dataset][field] = _safe_auroc(y, values)

    return {"overall": overall, "per_dataset": per_dataset}


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return AUROC or ``None`` when labels are single-class."""

    if len(np.unique(labels)) < 2:
        return None
    return float(auroc(labels, scores))


def default_full_path(config: dict[str, Any], mode: str) -> Path:
    """Resolve default ``benchmark_{mode}_full.json`` under config outputs dir."""

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    return outputs_dir / f"benchmark_{mode}_full.json"


def print_table(metrics: dict[str, Any]) -> None:
    """Print a compact AUROC table for paper copy-paste."""

    print("\n=== Component AUROC (Table 2) ===")
    header = f"{'Score':<20} {'Overall':>10}"
    datasets = sorted(metrics["per_dataset"])
    for dataset in datasets:
        header += f" {dataset:>12}"
    print(header)

    for field, label in SCORE_FIELDS:
        overall = metrics["overall"][field]
        row = f"{label:<20} {_format_auroc(overall):>10}"
        for dataset in datasets:
            value = metrics["per_dataset"][dataset][field]
            row += f" {_format_auroc(value):>12}"
        print(row)


def _format_auroc(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-component AUROC from benchmark full scores."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=("oracle", "pipeline"),
        default="pipeline",
        help="Benchmark mode used when generating full scores.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to benchmark_{mode}_full.json (default: from config paths).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)

    input_path = Path(args.input) if args.input else default_full_path(config, args.mode)
    if not input_path.is_file():
        raise SystemExit(
            f"Input not found: {input_path}\n"
            f"Run: python experiments/run_benchmark.py --config {args.config or 'configs/paper.yaml'} "
            f"--mode {args.mode}"
        )

    payload = load_full_results(input_path)
    metrics = compute_component_auroc(payload["scores"])
    print_table(metrics)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"component_auroc_{args.mode}.json"
    save_json(
        {
            "source": str(input_path.resolve()),
            "mode": payload.get("mode", args.mode),
            "budget": payload.get("budget", {}),
            "metrics": metrics,
        },
        output_path,
    )
    logger.info("Saved component AUROC to %s", output_path)


if __name__ == "__main__":
    main()
