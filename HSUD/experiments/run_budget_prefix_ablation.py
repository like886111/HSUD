#!/usr/bin/env python3
"""Offline budget prefix ablation on persisted nested trees.

Recomputes ordered decompositions after restricting each item's leaves to
theta_0..M-1, z_id<K, r_id<J, l_id<L. Uses existing NLI cluster labels on the
kept leaves (no new LLM / NLI calls).

Example:
  PYTHONPATH=. python experiments/run_budget_prefix_ablation.py \\
    --paths-glob 'outputs/benchmark_g8_fixedR_nli/nested_paths/*.jsonl' \\
    --out-dir outputs/budget_prefix_ablation_g8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.estimators import decompose_uncertainty_ordered
from core.llm_backend import SampledPath
from utils.logger import get_logger, save_json

COMPS = ("u_theta", "u_z", "u_r", "u_res")
DEFAULT_BUDGETS = (
    (2, 2, 2, 2),
    (4, 2, 2, 2),
    (4, 4, 2, 2),
    (4, 4, 4, 2),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-glob", required=True)
    p.add_argument("--out-dir", default="outputs/budget_prefix_ablation")
    p.add_argument(
        "--budgets",
        nargs="+",
        default=["2,2,2,2", "4,2,2,2", "4,4,2,2", "4,4,4,2"],
        help="M,K,J,L tuples",
    )
    p.add_argument("--corrected", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _load_paths(path: Path) -> list[SampledPath]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        SampledPath(
            theta_id=str(r["theta_id"]),
            z_id=int(r["z_id"]),
            r_id=int(r["r_id"]),
            question=str(r.get("question", "")),
            z_text=str(r.get("z_text", "")),
            r_text=str(r.get("r_text", "")),
            answer_text=str(r.get("answer_text", "")),
            l_id=int(r.get("l_id", 0)),
            s_cluster_id=(
                int(r["s_cluster_id"]) if r.get("s_cluster_id") is not None else None
            ),
        )
        for r in rows
    ]


def _theta_rank(theta_id: str) -> int:
    # theta_0, theta_1, ...
    try:
        return int(str(theta_id).split("_")[-1])
    except ValueError:
        return 10**9


def _filter_budget(
    paths: list[SampledPath], M: int, K: int, J: int, L: int
) -> list[SampledPath]:
    kept = [
        p
        for p in paths
        if _theta_rank(p.theta_id) < M
        and p.z_id < K
        and p.r_id < J
        and p.l_id < L
    ]
    return kept


def _shares(decomp) -> dict[str, float]:
    vals = {c: float(getattr(decomp, c)) for c in COMPS}
    h = float(decomp.total_entropy)
    if h <= 0:
        return {c: float("nan") for c in COMPS}
    return {c: vals[c] / h for c in COMPS}


def _hard(decomp) -> str:
    terms = {c: float(getattr(decomp, c)) for c in COMPS}
    eligible = {k: v for k, v in terms.items() if v >= 0.1}
    if not eligible:
        return "none"
    mx = max(eligible.values())
    tops = [k for k, v in eligible.items() if abs(v - mx) < 1e-12]
    priority = {"u_z": 0, "u_theta": 1, "u_r": 2, "u_res": 3}
    return min(tops, key=lambda k: priority[k])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger()

    files = sorted(Path().glob(args.paths_glob))
    if not files:
        raise SystemExit(f"No files matched {args.paths_glob}")

    budgets = []
    for spec in args.budgets:
        parts = [int(x) for x in spec.split(",")]
        if len(parts) != 4:
            raise SystemExit(f"Bad budget {spec}")
        budgets.append(tuple(parts))

    # Full reference = densest budget present in list, else (4,4,4,2)
    ref = max(budgets, key=lambda b: b[0] * b[1] * b[2] * b[3])

    per_item: list[dict[str, Any]] = []
    for fp in files:
        paths = _load_paths(fp)
        item: dict[str, Any] = {"item_id": fp.stem, "budgets": {}}
        ref_paths = _filter_budget(paths, *ref)
        if len(ref_paths) < 2:
            continue
        ref_decomp = decompose_uncertainty_ordered(
            ref_paths, order=("Theta", "Z", "R"), corrected=args.corrected
        )
        ref_q = _shares(ref_decomp)
        item["ref_budget"] = list(ref)
        item["ref_shares"] = ref_q
        item["ref_hard"] = _hard(ref_decomp)
        item["ref_H"] = float(ref_decomp.total_entropy)

        for b in budgets:
            sub = _filter_budget(paths, *b)
            key = f"M{b[0]}_K{b[1]}_J{b[2]}_L{b[3]}"
            if len(sub) < 2:
                item["budgets"][key] = {"status": "too_few_leaves", "n_leaves": len(sub)}
                continue
            decomp = decompose_uncertainty_ordered(
                sub, order=("Theta", "Z", "R"), corrected=args.corrected
            )
            q = _shares(decomp)
            # Spearman vs ref on finite shares
            pairs = [
                (ref_q[c], q[c])
                for c in COMPS
                if np.isfinite(ref_q[c]) and np.isfinite(q[c])
            ]
            if len(pairs) == 4 and len({p[0] for p in pairs}) > 1:
                rho = float(
                    np.corrcoef(
                        [p[0] for p in pairs], [p[1] for p in pairs]
                    )[0, 1]
                )
            else:
                rho = float("nan")
            item["budgets"][key] = {
                "n_leaves": len(sub),
                "stage_gens": b[0] * b[1] + b[0] * b[1] * b[2] + b[0] * b[1] * b[2] * b[3],
                "H": float(decomp.total_entropy),
                "shares": q,
                "share_rank": sorted(COMPS, key=lambda c: (-q[c] if np.isfinite(q[c]) else 0, c)),
                "hard": _hard(decomp),
                "pearson_vs_ref_shares": rho,
                "l1_vs_ref_shares": float(
                    np.nansum([abs(ref_q[c] - q[c]) for c in COMPS])
                ),
                "top_share_match_ref": (
                    sorted(COMPS, key=lambda c: (-q[c] if np.isfinite(q[c]) else 0, c))[0]
                    == sorted(
                        COMPS, key=lambda c: (-ref_q[c] if np.isfinite(ref_q[c]) else 0, c)
                    )[0]
                ),
            }
        per_item.append(item)

    # Aggregate
    summary: dict[str, Any] = {"n_items": len(per_item), "budgets": {}}
    for b in budgets:
        key = f"M{b[0]}_K{b[1]}_J{b[2]}_L{b[3]}"
        rows = [it["budgets"][key] for it in per_item if key in it["budgets"] and "shares" in it["budgets"][key]]
        if not rows:
            continue
        mean_shares = {
            c: float(np.nanmean([r["shares"][c] for r in rows])) for c in COMPS
        }
        hard_counts = Counter(r["hard"] for r in rows)
        summary["budgets"][key] = {
            "M_K_J_L": list(b),
            "n": len(rows),
            "stage_gens": rows[0]["stage_gens"],
            "mean_H": float(np.nanmean([r["H"] for r in rows])),
            "mean_shares": mean_shares,
            "share_rank_of_means": sorted(
                COMPS, key=lambda c: (-mean_shares[c], c)
            ),
            "mean_l1_vs_ref": float(np.nanmean([r["l1_vs_ref_shares"] for r in rows])),
            "mean_pearson_vs_ref": float(
                np.nanmean([r["pearson_vs_ref_shares"] for r in rows])
            ),
            "top_share_match_rate": float(
                np.mean([1.0 if r["top_share_match_ref"] else 0.0 for r in rows])
            ),
            "hard_pct": {
                k: 100.0 * hard_counts.get(k, 0) / len(rows)
                for k in list(COMPS) + ["none"]
            },
        }

    save_json({"summary": summary, "items": per_item}, out_dir / "budget_prefix_full.json")
    save_json(summary, out_dir / "budget_prefix_summary.json")

    lines = [
        "# Budget prefix ablation (offline)",
        "",
        f"Items: {summary['n_items']}",
        "",
        "| Budget | gens | mean shares Θ/Z/R/res | rank | mean L1 vs full | top match | hard UR% |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for b in budgets:
        key = f"M{b[0]}_K{b[1]}_J{b[2]}_L{b[3]}"
        row = summary["budgets"].get(key)
        if not row:
            continue
        s = row["mean_shares"]
        lines.append(
            f"| {b} | {row['stage_gens']} | "
            f"{100*s['u_theta']:.1f}/{100*s['u_z']:.1f}/{100*s['u_r']:.1f}/{100*s['u_res']:.1f} | "
            f"{'≻'.join(row['share_rank_of_means'])} | "
            f"{row['mean_l1_vs_ref']:.3f} | {100*row['top_share_match_rate']:.1f}% | "
            f"{row['hard_pct'].get('u_r', 0):.1f} |"
        )
    (out_dir / "budget_prefix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", out_dir / "budget_prefix_summary.json")


if __name__ == "__main__":
    main()
