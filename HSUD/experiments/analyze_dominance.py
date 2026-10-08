#!/usr/bin/env python3
"""Dominance diagnosis helpers: margin/uncertain + offline G8 reanalysis + grid launcher.

Offline (no API) — re-label G8 scores under margin rules:
  python experiments/analyze_dominance.py \\
    --input outputs/benchmark_g8_paper/benchmark_pipeline_full.json \\
    --margins 0.0 0.05 0.1 0.15

API grid (small, pre-registered A0/A1/A2):
  python experiments/run_dominance_grid.py --setting A0 --max-samples 10
  python experiments/run_dominance_grid.py --setting all --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.router import select_dominant_term
from utils.logger import get_logger, save_json

TERM_KEYS = ("U_Theta", "U_Z", "U_R", "U_res")
SETTING_CONFIGS = {
    "A0": "configs/dominance_grid/A0_baseline.yaml",
    "A1": "configs/dominance_grid/A1_reduce_reasoning.yaml",
    "A2": "configs/dominance_grid/A2_nli_cluster.yaml",
}


def _row_terms(row: dict[str, Any]) -> dict[str, float]:
    return {
        "U_Theta": float(row["u_theta"]),
        "U_Z": float(row["u_z"]),
        "U_R": float(row["u_r"]),
        "U_res": float(row["u_res"]),
    }


def analyze_scores(
    scores: list[dict[str, Any]],
    *,
    min_term_value: float = 0.1,
    min_dominance_margin: float = 0.0,
) -> dict[str, Any]:
    """Summarize dominant labels, margins, and component means."""

    per_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for row in scores:
        terms = _row_terms(row)
        dom, margin, uncertain = select_dominant_term(
            terms,
            min_term_value=min_term_value,
            min_dominance_margin=min_dominance_margin,
        )
        hs = float(row.get("nested_total") or sum(terms.values()))
        shares = {k: (terms[k] / hs if hs > 1e-12 else 0.0) for k in TERM_KEYS}
        rec = {
            "item_id": row.get("item_id"),
            "dataset": row.get("dataset"),
            "label": row.get("label"),
            "terms": terms,
            "shares": shares,
            "dominant": dom,
            "dominance_margin": margin,
            "is_uncertain": uncertain,
            "nested_total": hs,
        }
        records.append(rec)
        per_dataset[str(row.get("dataset"))].append(rec)

    def _summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
        dom_counts = Counter(r["dominant"] for r in recs)
        n = len(recs)
        margins = [r["dominance_margin"] for r in recs]
        means = {
            k: float(sum(r["terms"][k] for r in recs) / max(n, 1)) for k in TERM_KEYS
        }
        return {
            "n": n,
            "dominant_counts": dict(dom_counts),
            "dominant_rates": {k: v / max(n, 1) for k, v in dom_counts.items()},
            "uncertain_rate": float(sum(r["is_uncertain"] for r in recs) / max(n, 1)),
            "margin_mean": float(sum(margins) / max(n, 1)),
            "margin_median": float(sorted(margins)[n // 2]) if n else 0.0,
            "frac_margin_lt_0_05": float(sum(m < 0.05 for m in margins) / max(n, 1)),
            "frac_margin_lt_0_10": float(sum(m < 0.10 for m in margins) / max(n, 1)),
            "component_means": means,
            "u_r_dominant_rate_among_certain": (
                float(
                    sum(r["dominant"] == "U_R" for r in recs if not r["is_uncertain"])
                    / max(sum(not r["is_uncertain"] for r in recs), 1)
                )
            ),
        }

    by_ds = {ds: _summarize(recs) for ds, recs in sorted(per_dataset.items())}
    return {
        "min_term_value": min_term_value,
        "min_dominance_margin": min_dominance_margin,
        "overall": _summarize(records),
        "by_dataset": by_ds,
        "n_records": len(records),
    }


def run_offline_analysis(
    input_path: Path,
    margins: list[float],
    min_term_value: float,
    output_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    scores = payload.get("scores") or payload.get("pairs")
    if not scores:
        raise RuntimeError(f"No scores found in {input_path}")
    # intervention files use different schema — skip
    if "u_theta" not in scores[0]:
        raise RuntimeError("Input does not look like benchmark_pipeline_full.json scores")

    by_margin = {}
    for m in margins:
        by_margin[str(m)] = analyze_scores(
            scores,
            min_term_value=min_term_value,
            min_dominance_margin=float(m),
        )
    result = {
        "input": str(input_path),
        "n_scores": len(scores),
        "by_margin": by_margin,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, output_dir / "dominance_reanalysis_summary.json")
    # also write a compact markdown-friendly table to stdout via logger
    return result


def _print_table(result: dict[str, Any]) -> None:
    logger = get_logger("analyze_dominance")
    for margin, block in result["by_margin"].items():
        ov = block["overall"]
        logger.info(
            "margin=%s uncertain=%.1f%% U_R_dom=%.1f%% (among certain U_R=%.1f%%) "
            "margin_med=%.3f frac_m<0.1=%.1f%% means Θ=%.3f Z=%.3f R=%.3f res=%.3f",
            margin,
            100 * ov["uncertain_rate"],
            100 * ov["dominant_rates"].get("U_R", 0.0),
            100 * ov["u_r_dominant_rate_among_certain"],
            ov["margin_median"],
            100 * ov["frac_margin_lt_0_10"],
            ov["component_means"]["U_Theta"],
            ov["component_means"]["U_Z"],
            ov["component_means"]["U_R"],
            ov["component_means"]["U_res"],
        )
        for ds, sub in block["by_dataset"].items():
            logger.info(
                "  [%s] n=%d rates=%s uncertain=%.1f%%",
                ds,
                sub["n"],
                {k: f"{100*v:.0f}%" for k, v in sub["dominant_rates"].items()},
                100 * sub["uncertain_rate"],
            )


def main_analyze() -> None:
    parser = argparse.ArgumentParser(description="Offline dominance reanalysis.")
    parser.add_argument(
        "--input",
        default="outputs/benchmark_g8_paper/benchmark_pipeline_full.json",
    )
    parser.add_argument(
        "--margins",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15],
    )
    parser.add_argument("--min-term-value", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        default="outputs/dominance_reanalysis",
    )
    args = parser.parse_args()
    result = run_offline_analysis(
        Path(args.input),
        margins=list(args.margins),
        min_term_value=args.min_term_value,
        output_dir=Path(args.output_dir),
    )
    _print_table(result)
    get_logger("analyze_dominance").info(
        "Saved %s", Path(args.output_dir) / "dominance_reanalysis_summary.json"
    )


if __name__ == "__main__":
    main_analyze()
