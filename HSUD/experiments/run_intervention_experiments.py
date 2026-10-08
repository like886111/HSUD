"""Phase 4 paired intervention experiments for causal uncertainty attribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.controlled_oracle import (
    cluster_and_decompose,
    decomposition_to_dict,
    make_controlled_paths,
)
from experiments.interventions import (
    INTERVENTIONS,
    InterventionSpec,
    deterministic_rule,
)
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


TERM_KEYS = ("U_Theta", "U_Z", "U_R", "U_res")


def load_base_questions(path: Path | None = None, limit: int | None = None) -> list[str]:
    """Load clear baseline questions from category D in synthetic QA."""

    dataset_path = path or PROJECT_ROOT / "data" / "synthetic_qa.json"
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = [record["question"] for record in records if record["category"] == "D"]
    if limit is not None:
        questions = questions[:limit]
    return questions


def evaluate_pair(
    question: str,
    answer_rule,
    M: int,
    K: int,
    J: int,
    L: int,
) -> dict[str, float]:
    """Run controlled oracle decomposition for one (question, rule) pair."""

    paths = make_controlled_paths(question, M, K, J, L, answer_rule)
    decomposition = cluster_and_decompose(paths, corrected=False)
    return decomposition_to_dict(decomposition)


def compute_deltas(
    baseline_metrics: dict[str, float],
    treated_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute component-wise uncertainty changes."""

    return {
        term: treated_metrics[term] - baseline_metrics[term]
        for term in TERM_KEYS
    }


def run_intervention_pair(
    base_question: str,
    intervention: InterventionSpec,
    M: int,
    K: int,
    J: int,
    L: int,
) -> dict[str, Any]:
    """Run one baseline vs intervention paired experiment."""

    baseline_metrics = evaluate_pair(base_question, deterministic_rule, M, K, J, L)
    treated_question = intervention.perturb(base_question)
    treated_metrics = evaluate_pair(
        treated_question,
        intervention.answer_rule,
        M,
        K,
        J,
        L,
    )
    deltas = compute_deltas(baseline_metrics, treated_metrics)
    verification = verify_selective_response(
        deltas,
        target_term=intervention.target_term,
        min_target_delta=0.5,
        max_other_delta=0.05,
        min_selectivity_ratio=3.0,
    )
    return {
        "base_question": base_question,
        "treated_question": treated_question,
        "intervention": intervention.name,
        "target_term": intervention.target_term,
        "baseline_metrics": baseline_metrics,
        "treated_metrics": treated_metrics,
        "deltas": deltas,
        "verification": verification,
    }


def verify_selective_response(
    deltas: dict[str, float],
    target_term: str,
    min_target_delta: float,
    max_other_delta: float,
    min_selectivity_ratio: float,
) -> dict[str, Any]:
    """Verify that only the targeted uncertainty component increases."""

    target_delta = deltas[target_term]
    other_terms = [term for term in TERM_KEYS if term != target_term]
    other_deltas = {term: deltas[term] for term in other_terms}
    max_other = max(other_deltas.values())
    ratio = target_delta / max(max_other, 1e-12)

    checks = {
        "target_delta_sufficient": target_delta >= min_target_delta,
        "others_small": all(abs(value) <= max_other_delta for value in other_deltas.values()),
        "target_dominates": target_delta >= max_other * min_selectivity_ratio,
    }
    return {
        "target_term": target_term,
        "target_delta": target_delta,
        "other_deltas": other_deltas,
        "max_other_delta": max_other,
        "selectivity_ratio": ratio,
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_all_interventions(
    config_path: str | Path | None = None,
    question_limit: int | None = 10,
) -> dict[str, Any]:
    """Run all paired interventions over multiple baseline questions."""

    config = load_config(config_path)
    phase_cfg = config.get(
        "intervention_experiments",
        {
            "M": config["sanity_check"]["M"],
            "K": config["sanity_check"]["K"],
            "J": config["sanity_check"]["J"],
            "L": config["sanity_check"]["L"],
            "question_limit": 10,
            "min_target_delta": 0.5,
            "max_other_delta": 0.05,
            "min_selectivity_ratio": 3.0,
        },
    )
    M = int(phase_cfg.get("M", 4))
    K = int(phase_cfg.get("K", 4))
    J = int(phase_cfg.get("J", 4))
    L = int(phase_cfg.get("L", 4))
    limit = question_limit if question_limit is not None else phase_cfg.get("question_limit")
    limit = int(limit) if limit is not None else 10

    base_questions = load_base_questions(limit=limit)
    pairs: list[dict[str, Any]] = []
    for base_question in base_questions:
        for intervention in INTERVENTIONS:
            result = run_intervention_pair(base_question, intervention, M, K, J, L)
            pairs.append(result)

    aggregate = aggregate_intervention_deltas(pairs)
    all_pass = all(pair["verification"]["pass"] for pair in pairs)
    return {
        "budget": {"M": M, "K": K, "J": J, "L": L},
        "question_count": len(base_questions),
        "pairs": pairs,
        "aggregate_deltas": aggregate,
        "pass": all_pass,
    }


def aggregate_intervention_deltas(pairs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average deltas by intervention name."""

    buckets: dict[str, list[dict[str, float]]] = {}
    for pair in pairs:
        buckets.setdefault(pair["intervention"], []).append(pair["deltas"])

    aggregate: dict[str, dict[str, float]] = {}
    for intervention, delta_list in buckets.items():
        aggregate[intervention] = {
            term: float(sum(item[term] for item in delta_list) / len(delta_list))
            for term in TERM_KEYS
        }
    return aggregate


def plot_intervention_deltas(
    aggregate_deltas: dict[str, dict[str, float]],
    output_path: Path,
) -> Path:
    """Plot mean delta-U bars for each intervention."""

    rows = []
    label_map = {
        "add_ambiguity": "Add Ambiguity",
        "replace_with_obscure_knowledge": "Obscure Knowledge",
        "elongate_reasoning": "Elongate Reasoning",
    }
    for intervention, deltas in aggregate_deltas.items():
        for term in TERM_KEYS:
            rows.append(
                {
                    "intervention": label_map.get(intervention, intervention),
                    "term": term,
                    "delta": deltas[term],
                }
            )

    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=frame,
        x="intervention",
        y="delta",
        hue="term",
        hue_order=list(TERM_KEYS),
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Phase 4 Intervention Experiments: Mean ΔU by Perturbation")
    ax.set_xlabel("Perturbation Operator")
    ax.set_ylabel("Δ Uncertainty (nats)")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def print_summary(results: dict[str, Any]) -> None:
    """Print intervention summary to stdout."""

    print("\n=== Phase 4 Intervention Experiments ===")
    print("Budget:", results["budget"])
    print("Base questions:", results["question_count"])
    print("\nMean ΔU by intervention:")
    header = f"{'Intervention':<28} " + " ".join(f"{term:>8}" for term in TERM_KEYS)
    print(header)
    for intervention, deltas in results["aggregate_deltas"].items():
        row = f"{intervention:<28} " + " ".join(f"{deltas[t]:8.4f}" for t in TERM_KEYS)
        print(row)

    pass_count = sum(1 for pair in results["pairs"] if pair["verification"]["pass"])
    print(f"\nPairwise selective-response pass rate: {pass_count}/{len(results['pairs'])}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 4 intervention experiments.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)
    results = run_all_interventions(
        config_path=args.config,
        question_limit=args.question_limit,
    )
    print_summary(results)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "budget": results["budget"],
        "question_count": results["question_count"],
        "aggregate_deltas": results["aggregate_deltas"],
        "pass": results["pass"],
        "pair_pass_rate": sum(1 for pair in results["pairs"] if pair["verification"]["pass"])
        / len(results["pairs"]),
    }
    summary_path = outputs_dir / "intervention_experiments_summary.json"
    save_json(summary, summary_path)
    save_json(results, outputs_dir / "intervention_experiments_full.json")
    logger.info("Saved summary to %s", summary_path)

    if not args.no_plot:
        plot_path = reports_dir / "intervention_delta_u.png"
        plot_intervention_deltas(results["aggregate_deltas"], plot_path)
        logger.info("Saved plot to %s", plot_path)

    if not results["pass"]:
        raise SystemExit("Intervention experiments failed selective-response checks.")

    logger.info("Intervention experiments passed.")


if __name__ == "__main__":
    main()
