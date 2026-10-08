#!/usr/bin/env python3
"""Offline NLI-threshold (tau) ablation on persisted nested leaves.

No LLM API calls. Computes bidirectional entailment scores once per item,
then applies multiple thresholds. Writes share ranks, hard rates, and
default-vs-Shapley hard agreement.

Example:
  python experiments/run_tau_ablation.py \\
    --paths-glob 'outputs/benchmark_g8_fixedR_nli/nested_paths/*.jsonl' \\
    --out-dir outputs/ablation_tau \\
    --device cpu --limit 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.estimators import (  # noqa: E402
    UncertaintyDecomposition,
    decompose_uncertainty,
    shapley_decomposition,
)
from core.llm_backend import SampledPath  # noqa: E402
from core.semantic_nli import (  # noqa: E402
    SemanticNLIClusterer,
    connected_components_union_find,
    softmax,
)
from utils.logger import get_logger, save_json  # noqa: E402
from utils.storage import load_sample_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--paths-glob",
        default="outputs/benchmark_g8_fixedR_nli/nested_paths/*.jsonl",
    )
    p.add_argument("--out-dir", default="outputs/ablation_tau")
    p.add_argument(
        "--nli-model",
        default=str(ROOT / "models/nli-deberta-v3-large"),
    )
    p.add_argument("--thresholds", default="0.50,0.65,0.80")
    p.add_argument("--device", default="cpu")
    p.add_argument("--shapley", action="store_true", default=True)
    p.add_argument("--no-shapley", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-item score caches under out-dir/score_cache/",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Write partial records JSON every N items (0=off).",
    )
    return p.parse_args()


def _dominant(decomp: UncertaintyDecomposition) -> str:
    terms = {
        "u_theta": decomp.u_theta,
        "u_z": decomp.u_z,
        "u_r": decomp.u_r,
        "u_res": decomp.u_res,
    }
    eligible = {k: v for k, v in terms.items() if v >= 0.1}
    if not eligible:
        return "none"
    priority = {"u_z": 0, "u_theta": 1, "u_r": 2, "u_res": 3}
    return max(eligible.items(), key=lambda kv: (kv[1], -priority[kv[0]]))[0]


def _shares(decomp: UncertaintyDecomposition) -> dict[str, float]:
    h = float(decomp.total_entropy)
    if h <= 1e-12:
        return {k: 0.0 for k in ("u_theta", "u_z", "u_r", "u_res")}
    return {
        "u_theta": 100.0 * decomp.u_theta / h,
        "u_z": 100.0 * decomp.u_z / h,
        "u_r": 100.0 * decomp.u_r / h,
        "u_res": 100.0 * decomp.u_res / h,
    }


def _dataset_of(item_id: str) -> str:
    if item_id.startswith("truthfulqa"):
        return "truthfulqa"
    if item_id.startswith("ambigqa"):
        return "ambigqa"
    if item_id.startswith("gsm8k"):
        return "gsm8k"
    return "unknown"


class BidirectionalScoreCache:
    """One CrossEncoder load; continuous min(forward, backward) score matrix."""

    def __init__(self, model_name: str, device: str, batch_size: int = 64) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        labels = getattr(self.model.model.config, "id2label", {})
        inverse = {str(v).lower(): int(k) for k, v in labels.items()}
        self.entailment_index = inverse.get("entailment")

    def _entail_probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=float)
        logits = np.asarray(self.model.predict(pairs, batch_size=self.batch_size))
        if logits.ndim == 1:
            return logits.astype(float)
        probs = np.vstack([softmax(row) for row in logits])
        idx = self.entailment_index
        if idx is None:
            idx = int(np.argmax(logits[0]))
        return probs[:, idx].astype(float)

    def score_matrix(self, answers: Sequence[str]) -> np.ndarray:
        """Bidirectional min-entailment scores; unique strings scored once then expanded."""
        n = len(answers)
        scores = np.eye(n, dtype=float)
        uniq: list[str] = []
        remap: list[int] = []
        index_of: dict[str, int] = {}
        for text in answers:
            key = text if text is not None else ""
            if key not in index_of:
                index_of[key] = len(uniq)
                uniq.append(key)
            remap.append(index_of[key])
        u = len(uniq)
        if u <= 1:
            return scores
        pairs_f: list[tuple[str, str]] = []
        pairs_b: list[tuple[str, str]] = []
        index: list[tuple[int, int]] = []
        for i in range(u):
            for j in range(i + 1, u):
                pairs_f.append((uniq[i], uniq[j]))
                pairs_b.append((uniq[j], uniq[i]))
                index.append((i, j))
        pf = self._entail_probs(pairs_f)
        pb = self._entail_probs(pairs_b)
        both = np.minimum(pf, pb)
        u_scores = np.eye(u, dtype=float)
        for k, (i, j) in enumerate(index):
            u_scores[i, j] = u_scores[j, i] = float(both[k])
        for i in range(n):
            for j in range(i + 1, n):
                val = float(u_scores[remap[i], remap[j]])
                scores[i, j] = scores[j, i] = val
        return scores


def labels_from_scores(scores: np.ndarray, tau: float) -> list[int]:
    adjacency = (scores >= float(tau)).astype(float)
    np.fill_diagonal(adjacency, 1.0)
    return connected_components_union_find(adjacency)


def apply_labels(paths: Sequence[SampledPath], labels: Sequence[int]) -> list[SampledPath]:
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


def analyze_item(
    item_id: str,
    paths: list[SampledPath],
    cache: BidirectionalScoreCache,
    thresholds: Sequence[float],
    do_shapley: bool,
    score_cache_dir: Path | None = None,
) -> dict[str, Any]:
    import time

    answers = [p.answer_text for p in paths]
    n_unique = len(set(answers))
    score_path = None
    scores = None
    if score_cache_dir is not None:
        score_cache_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_cache_dir / f"{item_id}.npy"
        if score_path.exists():
            scores = np.load(score_path)
            if scores.shape != (len(answers), len(answers)):
                scores = None
    t0 = time.perf_counter()
    if scores is None:
        scores = cache.score_matrix(answers)
        if score_path is not None:
            np.save(score_path, scores)
    score_s = time.perf_counter() - t0
    by_tau: dict[str, Any] = {}
    for tau in thresholds:
        labels = labels_from_scores(scores, tau)
        labeled = apply_labels(paths, labels)
        decomp = decompose_uncertainty(labeled, corrected=True)
        row: dict[str, Any] = {
            "n_clusters": len(set(labels)),
            "H": decomp.total_entropy,
            "u_theta": decomp.u_theta,
            "u_z": decomp.u_z,
            "u_r": decomp.u_r,
            "u_res": decomp.u_res,
            "identity_residual": decomp.identity_residual,
            "shares_pct": _shares(decomp),
            "hard": _dominant(decomp),
        }
        if do_shapley:
            shap = shapley_decomposition(labeled, corrected=True)
            row["shapley_hard"] = _dominant(shap)
            row["hard_match_shapley"] = row["hard"] == row["shapley_hard"]
            row["shapley_shares_pct"] = _shares(shap)
        by_tau[f"{tau:.2f}"] = row
    return {
        "item_id": item_id,
        "dataset": _dataset_of(item_id),
        "n_leaves": len(paths),
        "n_unique_answers": n_unique,
        "score_seconds": score_s,
        "by_tau": by_tau,
    }


def aggregate(records: list[dict[str, Any]], thresholds: Sequence[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_items": len(records), "by_tau": {}}
    for tau in thresholds:
        key = f"{tau:.2f}"
        rows = [r["by_tau"][key] for r in records]
        hard = Counter(r["hard"] for r in rows)
        share_keys = ("u_theta", "u_z", "u_r", "u_res")
        mean_u = {
            k: sum(float(r[k]) for r in rows) / len(rows) for k in share_keys + ("H",)
        }
        mean_q = {
            k: sum(float(r["shares_pct"][k]) for r in rows) / len(rows) for k in share_keys
        }
        rank = sorted(share_keys, key=lambda k: -mean_q[k])
        entry: dict[str, Any] = {
            "component_means": mean_u,
            "mass_share_pct_means": mean_q,
            "share_rank": rank,
            "hard_pct": {k: 100.0 * v / len(rows) for k, v in hard.items()},
            "hard_counts": dict(hard),
        }
        if rows and "hard_match_shapley" in rows[0]:
            entry["default_vs_shapley_hard_match_rate"] = sum(
                1 for r in rows if r.get("hard_match_shapley")
            ) / len(rows)
            shap_hard = Counter(r["shapley_hard"] for r in rows)
            entry["shapley_hard_pct"] = {
                k: 100.0 * v / len(rows) for k, v in shap_hard.items()
            }
        # by dataset
        by_ds: dict[str, Any] = {}
        for ds in sorted({r["dataset"] for r in records}):
            sub = [records[i]["by_tau"][key] for i, r in enumerate(records) if r["dataset"] == ds]
            if not sub:
                continue
            mq = {
                k: sum(float(r["shares_pct"][k]) for r in sub) / len(sub) for k in share_keys
            }
            by_ds[ds] = {
                "n": len(sub),
                "mass_share_pct_means": mq,
                "share_rank": sorted(share_keys, key=lambda k: -mq[k]),
                "hard_pct": {
                    k: 100.0 * v / len(sub)
                    for k, v in Counter(r["hard"] for r in sub).items()
                },
            }
        entry["by_dataset"] = by_ds
        out["by_tau"][key] = entry

    # Rank stability vs 0.65
    base = "0.65"
    if base in out["by_tau"]:
        base_rank = out["by_tau"][base]["share_rank"]
        out["rank_equals_065"] = {
            t: out["by_tau"][t]["share_rank"] == base_rank for t in out["by_tau"]
        }
    return out


def main() -> None:
    args = parse_args()
    logger = get_logger("tau_ablation")
    do_shapley = bool(args.shapley) and not bool(args.no_shapley)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    files = sorted(Path().glob(args.paths_glob))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No files matched {args.paths_glob!r}")

    logger.info(
        "tau ablation: %d items, thresholds=%s, device=%s, shapley=%s",
        len(files),
        thresholds,
        args.device,
        do_shapley,
    )
    cache = BidirectionalScoreCache(args.nli_model, args.device, args.batch_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_cache_dir = out_dir / "score_cache"
    records: list[dict[str, Any]] = []
    partial_path = out_dir / "tau_ablation_partial.json"
    done_ids: set[str] = set()
    if args.resume and partial_path.exists():
        prev = json.loads(partial_path.read_text(encoding="utf-8"))
        records = list(prev.get("records") or [])
        done_ids = {r["item_id"] for r in records}
        logger.info("Resume: loaded %d records", len(records))

    for i, path in enumerate(files, 1):
        if path.stem in done_ids:
            logger.info("[%d/%d] skip cached %s", i, len(files), path.stem)
            continue
        paths = load_sample_paths(path)
        logger.info("[%d/%d] %s (%d leaves)", i, len(files), path.stem, len(paths))
        records.append(
            analyze_item(
                path.stem,
                paths,
                cache,
                thresholds,
                do_shapley,
                score_cache_dir=score_cache_dir,
            )
        )
        if args.checkpoint_every > 0 and len(records) % args.checkpoint_every == 0:
            save_json({"records": records}, partial_path)
            logger.info("Checkpoint: %d records -> %s", len(records), partial_path)

    summary = aggregate(records, thresholds)
    save_json(
        {
            "settings": {
                "thresholds": thresholds,
                "device": args.device,
                "shapley": do_shapley,
                "n_items": len(records),
                "paths_glob": args.paths_glob,
            },
            "summary": summary,
            "records": records,
        },
        out_dir / "tau_ablation_full.json",
    )
    save_json(summary, out_dir / "tau_ablation_summary.json")
    save_json({"records": records}, partial_path)
    logger.info("Wrote %s", out_dir / "tau_ablation_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()
