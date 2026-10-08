#!/usr/bin/env python3
"""Offline AUROC bootstrap CIs from a saved benchmark_pipeline_full.json.

Does not call any LLM.

Example:
  PYTHONPATH=. python experiments/run_auroc_bootstrap.py \\
    --input outputs/benchmark_g8_fixedR_nli/benchmark_pipeline_full.json \\
    --out-dir outputs/auroc_bootstrap_g8_fixedR_nli
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_benchmark import BASELINE_FIELDS, _score_from_dict, compute_detection_metrics
from utils.logger import get_logger, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="benchmark_*_full.json path")
    p.add_argument("--out-dir", default="outputs/auroc_bootstrap")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep CI tables without risk-coverage curves."""

    return {
        "label_counts": metrics.get("label_counts"),
        "nested_auroc": metrics.get("nested_auroc"),
        "semantic_entropy_auroc": metrics.get("semantic_entropy_auroc"),
        "self_consistency_auroc": metrics.get("self_consistency_auroc"),
        "input_perturbation_auroc": metrics.get("input_perturbation_auroc"),
        "nested_beats_semantic_entropy_auroc": metrics.get(
            "nested_beats_semantic_entropy_auroc"
        ),
        "nested_beats_self_consistency_auroc": metrics.get(
            "nested_beats_self_consistency_auroc"
        ),
        "nested_beats_input_perturbation_auroc": metrics.get(
            "nested_beats_input_perturbation_auroc"
        ),
        "nested_beats_all_flat_baselines_auroc": metrics.get(
            "nested_beats_all_flat_baselines_auroc"
        ),
        "nested_beats_baseline_auroc": metrics.get("nested_beats_baseline_auroc"),
        "auroc_by_method": metrics.get("auroc_by_method"),
        "auroc_bootstrap": metrics.get("auroc_bootstrap"),
        "auroc_delta_bootstrap": metrics.get("auroc_delta_bootstrap"),
        "auroc_bootstrap_by_dataset": metrics.get("auroc_bootstrap_by_dataset"),
        "per_dataset": {
            ds: {k: v for k, v in vals.items() if k != "risk_coverage"}
            for ds, vals in (metrics.get("per_dataset") or {}).items()
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    scores = [_score_from_dict(row) for row in payload["scores"]]
    metrics = compute_detection_metrics(
        scores, n_boot=args.n_boot, bootstrap_seed=args.seed
    )
    summary = _compact(metrics)
    summary["source_input"] = str(Path(args.input).resolve())
    summary["bootstrap_settings"] = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "alpha": 0.05,
    }

    out_path = out_dir / "auroc_bootstrap_summary.json"
    save_json(summary, out_path)
    log.info("Wrote %s", out_path)

    # Human-readable markdown table
    lines = [
        "# AUROC bootstrap (95% percentile CI)",
        "",
        f"Source: `{summary['source_input']}`",
        "",
        "| Method | AUROC | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, blob in (summary.get("auroc_bootstrap") or {}).items():
        lines.append(
            f"| {name} | {blob['auroc']:.3f} | "
            f"[{blob['ci_low']:.3f}, {blob['ci_high']:.3f}] |"
        )
    lines.extend(["", "## Paired deltas (Nested − baseline)", ""])
    for key, blob in (summary.get("auroc_delta_bootstrap") or {}).items():
        lines.append(
            f"- **{key}**: Δ={blob['delta']:+.3f}, "
            f"95% CI [{blob['ci_low']:+.3f}, {blob['ci_high']:+.3f}], "
            f"P(Δ≤0)={blob['p_le_zero']:.3f}"
        )
    md_path = out_dir / "auroc_bootstrap_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
