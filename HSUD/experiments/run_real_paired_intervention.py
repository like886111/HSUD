#!/usr/bin/env python3
"""Minimal real-LLM paired intervention study (mechanism selectivity).

Pre-registered criteria live in configs/benchmark_g8_min_intervention.yaml and
docs/最小干预实验配置与判定标准.md. This script does not optimize for untreated
dominant-label aesthetics on benchmark tasks.
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

from core.benchmarks.data import BenchmarkItem, load_benchmark_split
from core.benchmarks.evaluator import evaluate_item_pipeline
from experiments.interventions import INTERVENTION_BY_NAME
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json

TERM_KEYS = ("U_Theta", "U_Z", "U_R", "U_res")
STRUCTURED = ("U_Theta", "U_Z", "U_R")


def _scores_to_terms(scores: Any) -> dict[str, float]:
    return {
        "U_Theta": float(scores.u_theta),
        "U_Z": float(scores.u_z),
        "U_R": float(scores.u_r),
        "U_res": float(scores.u_res),
        "H_S": float(scores.nested_total),
    }


def _deltas(baseline: dict[str, float], treated: dict[str, float]) -> dict[str, float]:
    return {k: treated[k] - baseline[k] for k in TERM_KEYS}


def _verify_pair(
    deltas: dict[str, float],
    target_term: str,
    min_target_delta: float,
    min_selectivity_ratio: float,
) -> dict[str, Any]:
    target_delta = float(deltas[target_term])
    others = {t: float(deltas[t]) for t in STRUCTURED if t != target_term}
    max_other = max(others.values()) if others else 0.0
    ratio = target_delta / max(max_other, 1e-12)
    checks = {
        "target_delta_sufficient": target_delta >= min_target_delta,
        "target_ge_max_other_structured": target_delta >= max_other * min_selectivity_ratio,
    }
    return {
        "target_term": target_term,
        "target_delta": target_delta,
        "other_structured_deltas": others,
        "max_other_structured": max_other,
        "selectivity_ratio": ratio,
        "checks": checks,
        "soft_pass": all(checks.values()),
    }


def _load_g8_correct_ids(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores", [])
    by_ds: dict[str, list[str]] = {}
    for row in scores:
        if int(row.get("label", 1)) != 0:
            continue
        by_ds.setdefault(str(row["dataset"]), []).append(str(row["item_id"]))
    return by_ds


def _select_base_items(config: dict[str, Any], limit_bases: int | None) -> list[BenchmarkItem]:
    cfg = config["real_paired_intervention"]
    datasets = list(cfg.get("datasets", ["gsm8k"]))
    bases_per_dataset = int(cfg.get("bases_per_dataset", 2))
    max_bases = int(cfg.get("max_bases", 6))
    if limit_bases is not None:
        max_bases = min(max_bases, int(limit_bases))

    prefer = bool(cfg.get("prefer_correct_from_g8", True))
    g8_ids = {}
    if prefer:
        g8_path = resolve_path(cfg.get("g8_full_json", "outputs/benchmark_g8_paper/benchmark_pipeline_full.json"))
        g8_ids = _load_g8_correct_ids(Path(g8_path))

    selected: list[BenchmarkItem] = []
    for dataset in datasets:
        if len(selected) >= max_bases:
            break
        need = min(bases_per_dataset, max_bases - len(selected))
        source = str(cfg.get("source", "local"))
        items = load_benchmark_split(dataset, source=source)  # type: ignore[arg-type]
        preferred = []
        if dataset in g8_ids:
            id_set = set(g8_ids[dataset])
            preferred = [it for it in items if it.id in id_set]
        pool = preferred if preferred else list(items)
        # Prefer G8 order when available; otherwise stable by id.
        if preferred:
            order = {iid: i for i, iid in enumerate(g8_ids[dataset])}
            pool = sorted(pool, key=lambda it: order.get(it.id, 10**9))
        else:
            pool = sorted(pool, key=lambda it: it.id)
        selected.extend(pool[:need])
    return selected[:max_bases]


def _aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, float]]] = {}
    for pair in pairs:
        by_name.setdefault(pair["intervention"], []).append(pair["deltas"])

    mean_by_intervention: dict[str, dict[str, float]] = {}
    target_is_argmax: dict[str, bool] = {}
    for name, delta_list in by_name.items():
        means = {
            term: float(sum(d[term] for d in delta_list) / len(delta_list))
            for term in TERM_KEYS
        }
        mean_by_intervention[name] = means
        target = next(p["target_term"] for p in pairs if p["intervention"] == name)
        structured_means = {t: means[t] for t in STRUCTURED}
        argmax = max(structured_means, key=structured_means.get)
        target_is_argmax[name] = argmax == target

    soft_passes = sum(1 for p in pairs if p["verification"]["soft_pass"])
    soft_rate = soft_passes / max(len(pairs), 1)
    return {
        "mean_deltas_by_intervention": mean_by_intervention,
        "aggregate_target_is_argmax": target_is_argmax,
        "soft_pair_pass_count": soft_passes,
        "soft_pair_pass_rate": soft_rate,
        "n_pairs": len(pairs),
    }


def _decide(aggregate: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    soft_rate = float(aggregate["soft_pair_pass_rate"])
    soft_thr = float(criteria.get("soft_pair_pass_rate", 0.5))
    all_argmax = all(aggregate["aggregate_target_is_argmax"].values())
    require_argmax = bool(criteria.get("aggregate_target_is_argmax", True))

    p1 = soft_rate >= soft_thr
    p1_partial = soft_rate >= 0.30
    p2 = all_argmax if require_argmax else True

    if p1 and p2:
        status = "pass"
    elif p2 and p1_partial:
        status = "partial"
    else:
        status = "fail"

    return {
        "status": status,
        "pass": status == "pass",
        "partial": status == "partial",
        "P1_soft_pair_rate_ok": p1,
        "P1_soft_pair_rate_partial_band": p1_partial and not p1,
        "P2_aggregate_target_argmax_ok": p2,
        "soft_pair_pass_rate": soft_rate,
        "soft_pair_threshold": soft_thr,
    }


def run_real_paired_intervention(
    config_path: str | Path,
    limit_bases: int | None = None,
    interventions: list[str] | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    logger = get_logger("real_paired_intervention")
    config = load_config(config_path)
    cfg = config["real_paired_intervention"]
    criteria = dict(cfg.get("criteria", {}))
    min_target = float(criteria.get("min_target_delta", 0.10))
    min_ratio = float(criteria.get("min_selectivity_ratio", 1.0))
    intervention_names = list(
        interventions
        if interventions is not None
        else cfg.get(
            "interventions",
            ["add_ambiguity", "replace_with_obscure_knowledge", "elongate_reasoning"],
        )
    )

    bases = _select_base_items(config, limit_bases=limit_bases)
    if not bases:
        raise RuntimeError("No baseline items selected. Check local splits / G8 JSON.")

    outputs_dir = Path(resolve_path(config["paths"]["outputs_dir"]))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outputs_dir / "real_paired_intervention_checkpoint.jsonl"
    done_keys: set[str] = set()
    prior_pairs: list[dict[str, Any]] = []
    if fresh and ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("Removed checkpoint for --fresh run: %s", ckpt_path)
    if bool(cfg.get("checkpoint", True)) and ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['base_item_id']}::{row['intervention']}"
            if row["intervention"] not in intervention_names:
                continue
            done_keys.add(key)
            prior_pairs.append(row)
        logger.info("Resumed %d checkpointed pairs from %s", len(prior_pairs), ckpt_path)

    pairs = list(prior_pairs)
    sampling = config["sampling"]
    budget = {
        "M": int(sampling["M"]),
        "K": int(sampling["K"]),
        "J": int(sampling["J"]),
        "L": int(sampling["L"]),
    }
    ambiguity_version = cfg.get("ambiguity_operator_version")
    if ambiguity_version is None:
        try:
            from experiments.interventions import AMBIGUITY_OPERATOR_VERSION

            ambiguity_version = AMBIGUITY_OPERATOR_VERSION
        except Exception:  # noqa: BLE001
            ambiguity_version = "unknown"


    for base in bases:
        for name in intervention_names:
            key = f"{base.id}::{name}"
            if key in done_keys:
                continue
            spec = INTERVENTION_BY_NAME[name]
            logger.info("Running %s on %s (%s)", name, base.id, base.dataset)

            baseline_item = BenchmarkItem(
                id=f"{base.id}__baseline",
                dataset=base.dataset,
                question=base.question,
                label=base.label,
                reference_answer=base.reference_answer,
                category=base.category,
                notes=base.notes,
                acceptable_answers=base.acceptable_answers,
                label_source=base.label_source,
                split=base.split,
            )
            treated_q = spec.perturb(base.question)
            treated_item = BenchmarkItem(
                id=f"{base.id}__{name}",
                dataset=base.dataset,
                question=treated_q,
                label=base.label,
                reference_answer=base.reference_answer,
                category=base.category,
                notes=f"intervention={name}",
                acceptable_answers=base.acceptable_answers,
                label_source=base.label_source,
                split=base.split,
            )

            baseline_scores = evaluate_item_pipeline(baseline_item, config)
            treated_scores = evaluate_item_pipeline(treated_item, config)
            baseline_terms = _scores_to_terms(baseline_scores)
            treated_terms = _scores_to_terms(treated_scores)
            deltas = _deltas(baseline_terms, treated_terms)
            verification = _verify_pair(
                deltas,
                target_term=spec.target_term,
                min_target_delta=min_target,
                min_selectivity_ratio=min_ratio,
            )
            row = {
                "base_item_id": base.id,
                "dataset": base.dataset,
                "base_question": base.question,
                "treated_question": treated_q,
                "intervention": name,
                "target_term": spec.target_term,
                "ambiguity_operator_version": ambiguity_version if name == "add_ambiguity" else None,
                "budget": budget,
                "baseline_metrics": baseline_terms,
                "treated_metrics": treated_terms,
                "deltas": deltas,
                "verification": verification,
            }
            pairs.append(row)
            if bool(cfg.get("checkpoint", True)):
                with ckpt_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(
                "  soft_pass=%s target_delta=%.3f ratio=%.3f",
                verification["soft_pass"],
                verification["target_delta"],
                verification["selectivity_ratio"],
            )

    aggregate = _aggregate(pairs)
    decision = _decide(aggregate, criteria)
    result = {
        "mode": "real_paired_intervention",
        "model": config.get("llm", {}).get("model_name"),
        "budget": budget,
        "criteria": criteria,
        "interventions": intervention_names,
        "ambiguity_operator_version": ambiguity_version,
        "base_item_ids": [b.id for b in bases],
        "n_bases": len(bases),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "aggregate": aggregate,
        "decision": decision,
        "pass": decision["pass"],
        "partial": decision["partial"],
        "status": decision["status"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal real paired intervention study.")
    parser.add_argument(
        "--config",
        default="configs/benchmark_g8_min_intervention.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--limit-bases",
        type=int,
        default=None,
        help="Optional cap on baseline questions (smoke).",
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=None,
        help="Optional subset of intervention names (e.g. add_ambiguity).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore/delete checkpoint and rerun selected pairs.",
    )
    args = parser.parse_args()
    logger = get_logger("real_paired_intervention")
    result = run_real_paired_intervention(
        args.config,
        limit_bases=args.limit_bases,
        interventions=args.interventions,
        fresh=args.fresh,
    )

    config = load_config(args.config)
    outputs_dir = Path(resolve_path(config["paths"]["outputs_dir"]))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    full_path = outputs_dir / "real_paired_intervention_full.json"
    summary = {
        k: result[k]
        for k in (
            "mode",
            "model",
            "budget",
            "criteria",
            "interventions",
            "ambiguity_operator_version",
            "base_item_ids",
            "n_bases",
            "n_pairs",
            "aggregate",
            "decision",
            "pass",
            "partial",
            "status",
        )
    }
    save_json(result, full_path)
    save_json(summary, outputs_dir / "real_paired_intervention_summary.json")
    logger.info("Status=%s soft_rate=%.2f", result["status"], result["aggregate"]["soft_pair_pass_rate"])
    logger.info("Saved %s", full_path)
    for name, flag in result["aggregate"]["aggregate_target_is_argmax"].items():
        means = result["aggregate"]["mean_deltas_by_intervention"][name]
        logger.info(
            "  %s argmax_ok=%s means Θ=%.3f Z=%.3f R=%.3f",
            name,
            flag,
            means["U_Theta"],
            means["U_Z"],
            means["U_R"],
        )


if __name__ == "__main__":
    main()
