#!/usr/bin/env python3
"""Post-hoc reanalysis of persisted nested sample trees.

Supports:
  - string vs NLI reclustering
  - default-order vs Shapley decomposition
  - Plugin vs Miller--Madow
  - M/K/J/L prefix subsampling sensitivity
  - NLI entailment-threshold sweep

Does **not** call the generation LLM. Requires leaf ``answer_text`` paths
saved under ``outputs/*/nested_paths/*.jsonl``.

Existing G8 paper run (2026-09) did not persist leaves, so NLI/Shapley
reanalysis of that score table is blocked until a re-run with
``paths.persist_nested_paths: true``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from core.estimators import (
    UncertaintyDecomposition,
    decompose_uncertainty,
    shapley_decomposition,
)
from core.evaluation.metrics import auroc
from core.llm_backend import SampledPath
from core.semantic_nli import SemanticNLIClusterer
from utils.logger import get_logger, save_json
from utils.storage import load_sample_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-glob",
        required=True,
        help="Glob of nested path jsonl files, e.g. outputs/.../nested_paths/*.jsonl",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Optional benchmark_pipeline_full.json to attach error labels / AUROC.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/reanalysis_nested_paths",
        help="Directory for summary JSON.",
    )
    parser.add_argument("--nli", action="store_true", help="Enable NLI reclustering.")
    parser.add_argument(
        "--nli-model",
        default=(
            "/home/like/Desktop/A Hierarchical Bayesian Framework for "
            "Decomposing Uncertainty in Large Language Models/models/nli-deberta-v3-large"
        ),
    )
    parser.add_argument(
        "--nli-thresholds",
        default="0.65",
        help="Comma-separated NLI thresholds, e.g. 0.5,0.65,0.8",
    )
    parser.add_argument("--shapley", action="store_true", help="Also compute Shapley.")
    parser.add_argument(
        "--mm-ablation",
        action="store_true",
        help="Compare Miller-Madow vs plugin estimators.",
    )
    parser.add_argument(
        "--budget-subsample",
        action="store_true",
        help="Prefix-subsample M/K/J/L sensitivity grid.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device for NLI (e.g. cuda, cpu).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max items (0=all).")
    return parser.parse_args()


def _dominant(decomp: UncertaintyDecomposition) -> str:
    terms = {
        "U_Theta": decomp.u_theta,
        "U_Z": decomp.u_z,
        "U_R": decomp.u_r,
        "U_res": decomp.u_res,
    }
    eligible = {k: v for k, v in terms.items() if v >= 0.1}
    if not eligible:
        return "uncertain"
    # Tie-break matches G8 hard rule: U_Z > U_Theta > U_R > U_res
    priority = {"U_Z": 0, "U_Theta": 1, "U_R": 2, "U_res": 3}
    return max(eligible.items(), key=lambda kv: (kv[1], -priority[kv[0]]))[0]


def _item_id_from_path(path: Path) -> str:
    return path.stem


def _subsample_paths(
    paths: Sequence[SampledPath],
    M: int,
    K: int,
    J: int,
    L: int,
) -> list[SampledPath]:
    theta_keep = {f"theta_{m}" for m in range(M)}
    return [
        p
        for p in paths
        if p.theta_id in theta_keep and p.z_id < K and p.r_id < J and p.l_id < L
    ]


def _infer_budget(paths: Sequence[SampledPath]) -> dict[str, int]:
    thetas = sorted({p.theta_id for p in paths})
    return {
        "M": len(thetas),
        "K": max((p.z_id for p in paths), default=-1) + 1,
        "J": max((p.r_id for p in paths), default=-1) + 1,
        "L": max((p.l_id for p in paths), default=-1) + 1,
        "N": len(paths),
    }


def _recluster(
    paths: Sequence[SampledPath],
    clusterer: SemanticNLIClusterer,
) -> list[SampledPath]:
    labels = clusterer.cluster([p.answer_text for p in paths]).labels
    out: list[SampledPath] = []
    for path, label in zip(paths, labels):
        out.append(
            SampledPath(
                theta_id=path.theta_id,
                z_id=path.z_id,
                r_id=path.r_id,
                l_id=path.l_id,
                question=path.question,
                z_text=path.z_text,
                r_text=path.r_text,
                answer_text=path.answer_text,
                s_cluster_id=int(label),
            )
        )
    return out


def _decomp_row(
    decomp: UncertaintyDecomposition,
    *,
    mode: str,
    corrected: bool,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "miller_madow": corrected,
        "nested_total": decomp.total_entropy,
        "u_theta": decomp.u_theta,
        "u_z": decomp.u_z,
        "u_r": decomp.u_r,
        "u_res": decomp.u_res,
        "identity_residual": decomp.identity_residual,
        "dominant": _dominant(decomp),
    }


def _load_labels(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    scores = payload.get("scores", [])
    return {row["item_id"]: row for row in scores}


def _auroc_from_rows(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    labeled = [r for r in rows if r.get("label") is not None]
    if len(labeled) < 2:
        return None
    labels = [int(r["label"]) for r in labeled]
    if len(set(labels)) < 2:
        return None
    scores = [float(r[field]) for r in labeled]
    return float(auroc(labels, scores))


def analyze_item(
    item_id: str,
    paths: list[SampledPath],
    *,
    do_nli: bool,
    nli_thresholds: Sequence[float],
    nli_model: str,
    device: str | None,
    do_shapley: bool,
    do_mm: bool,
    do_budget: bool,
    label_row: dict[str, Any] | None,
) -> dict[str, Any]:
    budget = _infer_budget(paths)
    record: dict[str, Any] = {
        "item_id": item_id,
        "budget": budget,
        "label": None if label_row is None else label_row.get("label"),
        "dataset": None if label_row is None else label_row.get("dataset"),
        "analyses": [],
    }

    # Baseline: keep stored string-cluster labels if present; else recluster string.
    string_clusterer = SemanticNLIClusterer(use_model=False)
    string_paths = _recluster(paths, string_clusterer)

    corrected_flags = [True, False] if do_mm else [True]
    for corrected in corrected_flags:
        decomp = decompose_uncertainty(string_paths, corrected=corrected)
        row = _decomp_row(
            decomp,
            mode=f"string_default_order_{'mm' if corrected else 'plugin'}",
            corrected=corrected,
        )
        record["analyses"].append(row)
        if do_shapley:
            shapley = shapley_decomposition(string_paths, corrected=corrected)
            srow = _decomp_row(
                shapley,
                mode=f"string_shapley_{'mm' if corrected else 'plugin'}",
                corrected=corrected,
            )
            srow["dominant_match_default"] = srow["dominant"] == row["dominant"]
            record["analyses"].append(srow)

    if do_nli:
        for thr in nli_thresholds:
            clusterer = SemanticNLIClusterer(
                model_name=nli_model,
                entailment_threshold=float(thr),
                use_model=True,
                device=device,
            )
            nli_paths = _recluster(paths, clusterer)
            for corrected in corrected_flags:
                decomp = decompose_uncertainty(nli_paths, corrected=corrected)
                row = _decomp_row(
                    decomp,
                    mode=f"nli_default_order_t{thr}_{'mm' if corrected else 'plugin'}",
                    corrected=corrected,
                )
                row["nli_threshold"] = float(thr)
                record["analyses"].append(row)
                if do_shapley:
                    shapley = shapley_decomposition(nli_paths, corrected=corrected)
                    srow = _decomp_row(
                        shapley,
                        mode=f"nli_shapley_t{thr}_{'mm' if corrected else 'plugin'}",
                        corrected=corrected,
                    )
                    srow["nli_threshold"] = float(thr)
                    srow["dominant_match_default"] = srow["dominant"] == row["dominant"]
                    record["analyses"].append(srow)

    if do_budget:
        M, K, J, L = budget["M"], budget["K"], budget["J"], budget["L"]
        grid = []
        for m in sorted({1, 2, M}):
            for k in sorted({1, 2, K}):
                for j in sorted({1, 2, J}):
                    for ell in sorted({2, L}):
                        if m > M or k > K or j > J or ell > L:
                            continue
                        if m * k * j * ell < 4:
                            continue
                        grid.append((m, k, j, ell))
        # unique preserve order
        seen: set[tuple[int, int, int, int]] = set()
        for m, k, j, ell in grid:
            key = (m, k, j, ell)
            if key in seen:
                continue
            seen.add(key)
            sub = _subsample_paths(string_paths, m, k, j, ell)
            if len(sub) < 4:
                continue
            decomp = decompose_uncertainty(sub, corrected=True)
            row = _decomp_row(decomp, mode="string_budget_subsample", corrected=True)
            row["subsample_budget"] = {"M": m, "K": k, "J": j, "L": ell, "N": len(sub)}
            record["analyses"].append(row)

    # Fixed-R integrity check on stored leaves (same r_id => same r_text?)
    by_cell: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for p in paths:
        by_cell[(p.theta_id, p.z_id, p.r_id)].add(p.r_text)
    n_cells = len(by_cell)
    n_fixed = sum(1 for texts in by_cell.values() if len(texts) == 1)
    record["fixed_reasoning_integrity"] = {
        "n_cells": n_cells,
        "n_cells_single_r_text": n_fixed,
        "fraction_fixed": (n_fixed / n_cells) if n_cells else None,
        "interpretation": (
            "fixed_realized_reasoning_path"
            if n_cells and n_fixed == n_cells
            else "reasoning_branch_or_mixed_r_text"
        ),
    }
    return record


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        for row in rec["analyses"]:
            payload = dict(row)
            payload["item_id"] = rec["item_id"]
            payload["label"] = rec.get("label")
            payload["dataset"] = rec.get("dataset")
            by_mode[row["mode"]].append(payload)

    summary: dict[str, Any] = {"n_items": len(records), "by_mode": {}}
    for mode, rows in by_mode.items():
        dom = Counter(r["dominant"] for r in rows)
        means = {
            key: sum(float(r[key]) for r in rows) / len(rows)
            for key in ("nested_total", "u_theta", "u_z", "u_r", "u_res")
        }
        entry: dict[str, Any] = {
            "n": len(rows),
            "component_means": means,
            "dominant_counts": dict(dom),
            "dominant_rates": {k: v / len(rows) for k, v in dom.items()},
            "nested_auroc": _auroc_from_rows(rows, "nested_total"),
        }
        if any("dominant_match_default" in r for r in rows):
            matches = [bool(r.get("dominant_match_default")) for r in rows]
            entry["shapley_default_dominant_match_rate"] = sum(matches) / len(matches)
        summary["by_mode"][mode] = entry

    integrity = Counter(
        r["fixed_reasoning_integrity"]["interpretation"] for r in records
    )
    summary["fixed_reasoning_integrity_counts"] = dict(integrity)
    return summary


def main() -> None:
    args = parse_args()
    logger = get_logger("reanalyze_nested_paths")
    files = sorted(Path().glob(args.paths_glob))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(
            f"No path files matched {args.paths_glob!r}. "
            "Existing G8 paper outputs lack nested_paths/; re-run benchmark with "
            "paths.persist_nested_paths=true first."
        )

    labels = _load_labels(args.labels_json)
    thresholds = [float(x) for x in args.nli_thresholds.split(",") if x.strip()]
    records: list[dict[str, Any]] = []
    for path in files:
        item_id = _item_id_from_path(path)
        paths = load_sample_paths(path)
        if not paths:
            logger.warning("Empty paths file: %s", path)
            continue
        logger.info("Analyzing %s (%d leaves)", item_id, len(paths))
        records.append(
            analyze_item(
                item_id,
                paths,
                do_nli=args.nli,
                nli_thresholds=thresholds,
                nli_model=args.nli_model,
                device=args.device,
                do_shapley=args.shapley,
                do_mm=args.mm_ablation,
                do_budget=args.budget_subsample,
                label_row=labels.get(item_id),
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = {
        "n_files": len(files),
        "settings": {
            "nli": args.nli,
            "nli_thresholds": thresholds,
            "shapley": args.shapley,
            "mm_ablation": args.mm_ablation,
            "budget_subsample": args.budget_subsample,
        },
        "records": records,
        "aggregate": _aggregate(records),
    }
    save_json(full, out_dir / "reanalysis_full.json")
    save_json(full["aggregate"], out_dir / "reanalysis_summary.json")
    logger.info("Wrote %s", out_dir / "reanalysis_summary.json")
    print(json.dumps(full["aggregate"], indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
