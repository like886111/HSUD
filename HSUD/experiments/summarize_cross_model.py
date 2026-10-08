#!/usr/bin/env python3
"""Aggregate cross-model small-budget runs (+ optional full reference).

Reports detection AUROC (with bootstrap), default-chain mass-share ranks, and
optionally order/Shapley hard-match if decision_node summaries exist beside runs.

Example:
  PYTHONPATH=. python experiments/summarize_cross_model.py \\
    --runs outputs/cross_model/small_gpt4o_mini outputs/cross_model/small_gpt35_turbo \\
    --full-reference outputs/benchmark_g8_fixedR_nli \\
    --out-dir outputs/cross_model/summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_benchmark import _score_from_dict, compute_detection_metrics
from utils.logger import get_logger, save_json

COMPS = ("u_theta", "u_z", "u_r", "u_res")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="Run output dirs")
    p.add_argument("--full-reference", default=None, help="Optional full-budget dir")
    p.add_argument("--out-dir", default="outputs/cross_model/summary")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _find_full_json(run_dir: Path) -> Path | None:
    for name in (
        "benchmark_pipeline_full.json",
        "benchmark_oracle_full.json",
    ):
        path = run_dir / name
        if path.is_file():
            return path
    matches = sorted(run_dir.glob("benchmark_*_full.json"))
    return matches[0] if matches else None


def _share_profile(scores: list[Any]) -> dict[str, Any]:
    means = {
        c: float(np.mean([getattr(s, c) for s in scores]))
        for c in COMPS
    }
    h = float(np.mean([s.nested_total for s in scores]))
    shares = {
        c: (100.0 * means[c] / h if h > 0 else float("nan")) for c in COMPS
    }
    rank = sorted(COMPS, key=lambda c: (-shares[c], c))
    return {
        "H_mean": h,
        "component_means": means,
        "share_pct_ratio_of_means": shares,
        "share_rank": rank,
        "n": len(scores),
    }


def _analyze_run(run_dir: Path, n_boot: int, seed: int) -> dict[str, Any]:
    full = _find_full_json(run_dir)
    if full is None:
        return {"run_dir": str(run_dir), "status": "missing_full_json"}

    payload = json.loads(full.read_text(encoding="utf-8"))
    scores = [_score_from_dict(row) for row in payload["scores"]]
    metrics = compute_detection_metrics(
        scores, n_boot=n_boot, bootstrap_seed=seed
    )
    model_name = None
    # Best-effort: read sibling config note from summary if present
    summary_path = run_dir / "benchmark_pipeline_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = summary.get("model_name")

    decision = None
    for cand in (
        run_dir / "decision_node_v1_v2_summary.json",
        run_dir.parent / f"decision_{run_dir.name}" / "decision_node_v1_v2_summary.json",
    ):
        if cand.is_file():
            decision = json.loads(cand.read_text(encoding="utf-8"))
            break

    return {
        "run_dir": str(run_dir.resolve()),
        "full_json": str(full.resolve()),
        "status": "ok",
        "model_name": model_name,
        "budget": payload.get("budget"),
        "max_samples": payload.get("max_samples"),
        "n_items": len(scores),
        "share_profile": _share_profile(scores),
        "auroc_bootstrap": metrics.get("auroc_bootstrap"),
        "auroc_delta_bootstrap": metrics.get("auroc_delta_bootstrap"),
        "nested_beats_flags": {
            "vs_se": metrics.get("nested_beats_semantic_entropy_auroc"),
            "vs_sc": metrics.get("nested_beats_self_consistency_auroc"),
            "vs_ip": metrics.get("nested_beats_input_perturbation_auroc"),
            "vs_all_flat": metrics.get("nested_beats_all_flat_baselines_auroc"),
        },
        "decision_node": {
            "default_vs_shapley_hard_match_rate": (
                decision.get("v1_default_vs_shapley_hard_match_rate")
                if decision
                else None
            ),
            "present": decision is not None,
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger()

    runs = [Path(p) for p in args.runs]
    if args.full_reference:
        runs.append(Path(args.full_reference))

    analyzed = [_analyze_run(run, args.n_boot, args.seed) for run in runs]
    # Trend checks among successful small runs
    ok_runs = [r for r in analyzed if r.get("status") == "ok"]
    ranks = [tuple(r["share_profile"]["share_rank"]) for r in ok_runs]
    trend = {
        "n_ok_runs": len(ok_runs),
        "share_rank_identical_across_ok_runs": (
            len(set(ranks)) == 1 if ranks else False
        ),
        "share_ranks": {
            Path(r["run_dir"]).name: r["share_profile"]["share_rank"] for r in ok_runs
        },
        "success_criterion": (
            "Same share-rank order across models under the small-budget protocol "
            "supports 'not a one-model artifact' for multi-source mass; "
            "AUROC point estimates may still move."
        ),
    }

    payload = {"runs": analyzed, "trend": trend}
    save_json(payload, out_dir / "cross_model_summary.json")

    lines = [
        "# Cross-model summary",
        "",
        f"Share-rank identical across OK runs: **{trend['share_rank_identical_across_ok_runs']}**",
        "",
    ]
    for r in analyzed:
        name = Path(r["run_dir"]).name
        lines.append(f"## {name}")
        if r.get("status") != "ok":
            lines.append(f"- status: {r.get('status')}")
            lines.append("")
            continue
        shares = r["share_profile"]["share_pct_ratio_of_means"]
        lines.append(
            f"- n={r['n_items']}, budget={r.get('budget')}, rank={r['share_profile']['share_rank']}"
        )
        lines.append(
            "- shares(%): "
            + ", ".join(f"{c}={shares[c]:.1f}" for c in COMPS)
        )
        nested = r["auroc_bootstrap"]["Nested H(S|X)"]
        lines.append(
            f"- Nested AUROC {nested['auroc']:.3f} "
            f"[{nested['ci_low']:.3f}, {nested['ci_high']:.3f}]"
        )
        lines.append("")
    (out_dir / "cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_dir / "cross_model_summary.json")


if __name__ == "__main__":
    main()
