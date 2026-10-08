#!/usr/bin/env python3
"""Decision-node V1+V2 offline validation on persisted nested paths.

V1: all 6 (Theta,Z,R) orders + Shapley average
V2: mass shares q_i = U_i / H, Spearman on U and on q, and residualized
    (control-H) Spearman via linear residualization of each U against H

Uses existing cluster labels on paths when present (Primary NLI run).
Does NOT call the generation LLM or re-run DeBERTa.

Example:
  PYTHONPATH=. python experiments/run_decision_node_v1_v2.py \\
    --paths-glob 'outputs/benchmark_g8_fixedR_nli/nested_paths/*.jsonl' \\
    --labels-json outputs/benchmark_g8_fixedR_nli/benchmark_pipeline_full.json \\
    --out-dir outputs/decision_node_v1_v2
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Sequence

from core.estimators import (
    UncertaintyDecomposition,
    decompose_uncertainty_ordered,
    shapley_decomposition,
)
from core.llm_backend import SampledPath
from utils.logger import get_logger, save_json
from utils.storage import load_sample_paths

COMPS = ("u_theta", "u_z", "u_r", "u_res")
PRIORITY = {"u_z": 0, "u_theta": 1, "u_r": 2, "u_res": 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-glob", required=True)
    p.add_argument("--labels-json", default=None)
    p.add_argument("--out-dir", default="outputs/decision_node_v1_v2")
    p.add_argument("--limit", type=int, default=0, help="Max items (0=all).")
    p.add_argument(
        "--corrected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Miller-Madow (default true).",
    )
    return p.parse_args()


def _hard(decomp: UncertaintyDecomposition) -> str:
    terms = {
        "u_theta": decomp.u_theta,
        "u_z": decomp.u_z,
        "u_r": decomp.u_r,
        "u_res": decomp.u_res,
    }
    eligible = {k: v for k, v in terms.items() if v >= 0.1}
    if not eligible:
        return "none"
    mx = max(eligible.values())
    tops = [k for k, v in eligible.items() if abs(v - mx) < 1e-12]
    return min(tops, key=lambda k: PRIORITY[k])


def _soft(decomp: UncertaintyDecomposition, margin: float = 0.1) -> str:
    terms = {
        "u_theta": decomp.u_theta,
        "u_z": decomp.u_z,
        "u_r": decomp.u_r,
        "u_res": decomp.u_res,
    }
    ordered = sorted(terms, key=lambda k: (-terms[k], PRIORITY[k]))
    if terms[ordered[0]] - terms[ordered[1]] < margin:
        return "uncertain"
    return ordered[0]


def _row(decomp: UncertaintyDecomposition) -> dict[str, Any]:
    h = float(decomp.total_entropy)
    out = {
        "H": h,
        "u_theta": float(decomp.u_theta),
        "u_z": float(decomp.u_z),
        "u_r": float(decomp.u_r),
        "u_res": float(decomp.u_res),
        "hard": _hard(decomp),
        "soft_m0.1": _soft(decomp),
    }
    if h > 0:
        out["q_theta"] = out["u_theta"] / h
        out["q_z"] = out["u_z"] / h
        out["q_r"] = out["u_r"] / h
        out["q_res"] = out["u_res"] / h
    else:
        out["q_theta"] = out["q_z"] = out["q_r"] = out["q_res"] = 0.0
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(a: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def _residualize(y: Sequence[float], x: Sequence[float]) -> list[float]:
    """Residual of y after linear regression on x (control for H)."""

    n = len(y)
    mx, my = _mean(x), _mean(y)
    varx = sum((a - mx) ** 2 for a in x)
    if varx <= 0:
        return [float(v - my) for v in y]
    beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / varx
    alpha = my - beta * mx
    return [float(b - (alpha + beta * a)) for a, b in zip(x, y)]


def _agg_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    h = _mean([r["H"] for r in rows])
    means = {c: _mean([r[c] for r in rows]) for c in COMPS}
    return {
        "n": len(rows),
        "H_mean": h,
        "component_means": means,
        "mass_share_pct": {c: (100 * means[c] / h if h else 0.0) for c in COMPS},
        "hard_counts": dict(Counter(r["hard"] for r in rows)),
        "hard_pct": {
            k: 100 * v / len(rows) for k, v in Counter(r["hard"] for r in rows).items()
        },
        "soft_m0.1_counts": dict(Counter(r["soft_m0.1"] for r in rows)),
        "soft_m0.1_pct": {
            k: 100 * v / len(rows)
            for k, v in Counter(r["soft_m0.1"] for r in rows).items()
        },
    }


def _corr_block(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, float]:
    out = {}
    for a, b in combinations(fields, 2):
        out[f"{a}-{b}"] = _spearman([r[a] for r in rows], [r[b] for r in rows])
    return out


def main() -> None:
    args = parse_args()
    log = get_logger("decision_node_v1_v2")
    paths_files = sorted(Path().glob(args.paths_glob))
    if args.limit > 0:
        paths_files = paths_files[: args.limit]
    if not paths_files:
        raise SystemExit(f"No files matched: {args.paths_glob}")

    labels: dict[str, dict[str, Any]] = {}
    if args.labels_json:
        payload = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
        for row in payload.get("scores", []):
            labels[row["item_id"]] = row

    orders = list(permutations(("Theta", "Z", "R")))
    per_item: list[dict[str, Any]] = []
    by_order_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shapley_rows: list[dict[str, Any]] = []
    default_rows: list[dict[str, Any]] = []

    for i, path in enumerate(paths_files, 1):
        item_id = path.stem
        sample_paths = load_sample_paths(path)
        if not sample_paths:
            log.warning("Empty paths: %s", path)
            continue
        if any(p.s_cluster_id is None for p in sample_paths):
            log.warning("Missing s_cluster_id on %s; skip (need Primary NLI labels)", item_id)
            continue

        item: dict[str, Any] = {
            "item_id": item_id,
            "dataset": labels.get(item_id, {}).get("dataset"),
            "label": labels.get(item_id, {}).get("label"),
            "orders": {},
        }
        for order in orders:
            decomp = decompose_uncertainty_ordered(
                sample_paths, order=order, corrected=args.corrected
            )
            row = _row(decomp)
            key = "->".join(order)
            item["orders"][key] = row
            by_order_rows[key].append(row)

        default = item["orders"]["Theta->Z->R"]
        default_rows.append(default)
        shapley = shapley_decomposition(sample_paths, corrected=args.corrected)
        srow = _row(shapley)
        item["shapley"] = srow
        shapley_rows.append(srow)
        item["default_vs_shapley_hard_match"] = default["hard"] == srow["hard"]
        per_item.append(item)
        if i % 25 == 0 or i == len(paths_files):
            log.info("Processed %d/%d", i, len(paths_files))

    # V2 on default-order rows
    H = [r["H"] for r in default_rows]
    resid = {
        c: _residualize([r[c] for r in default_rows], H) for c in COMPS
    }
    resid_rows = [
        {c: resid[c][i] for c in COMPS} for i in range(len(default_rows))
    ]

    summary = {
        "n_items": len(per_item),
        "corrected": args.corrected,
        "v1_by_order": {k: _agg_rows(v) for k, v in by_order_rows.items()},
        "v1_shapley": _agg_rows(shapley_rows),
        "v1_default": _agg_rows(default_rows),
        "v1_default_vs_shapley_hard_match_rate": (
            sum(1 for it in per_item if it["default_vs_shapley_hard_match"]) / len(per_item)
            if per_item
            else float("nan")
        ),
        "v2_spearman_raw_U": _corr_block(default_rows, COMPS),
        "v2_spearman_shares_q": _corr_block(
            default_rows, ("q_theta", "q_z", "q_r", "q_res")
        ),
        "v2_spearman_U_residualized_on_H": _corr_block(resid_rows, COMPS),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json({"summary": summary, "items": per_item}, out / "decision_node_v1_v2_full.json")
    save_json(summary, out / "decision_node_v1_v2_summary.json")
    log.info("Wrote %s", out / "decision_node_v1_v2_summary.json")
    log.info(
        "V1 default UR hard%%=%.1f shapley UR hard%%=%.1f match_rate=%.3f",
        summary["v1_default"]["hard_pct"].get("u_r", 0.0),
        summary["v1_shapley"]["hard_pct"].get("u_r", 0.0),
        summary["v1_default_vs_shapley_hard_match_rate"],
    )
    log.info("V2 raw U Spearman sample: %s", summary["v2_spearman_raw_U"])
    log.info(
        "V2 residualized-on-H Spearman sample: %s",
        summary["v2_spearman_U_residualized_on_H"],
    )


if __name__ == "__main__":
    main()
