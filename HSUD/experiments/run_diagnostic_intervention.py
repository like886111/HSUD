"""Diagnostic intervention experiment with a 3x3 repair-gain heatmap.

Oracle mode simulates the paper's intended intervention logic:
- high U_Theta samples are repaired by RAG,
- high U_Z samples are repaired by clarification,
- high U_R samples are repaired by voting / self-consistency.

The same output schema can later be populated by real pipeline interventions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.controlled_oracle import (
    AnswerRule,
    ambiguous_rule,
    cluster_and_decompose,
    decomposition_to_dict,
    knowledge_blind_rule,
    make_controlled_paths,
    reasoning_rule,
)
from experiments.interventions import deterministic_rule
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


RISK_RULES: dict[str, AnswerRule] = {
    "Theta": knowledge_blind_rule,
    "Z": ambiguous_rule,
    "R": reasoning_rule,
}

TARGET_TERMS = {
    "Theta": "U_Theta",
    "Z": "U_Z",
    "R": "U_R",
}

MATCHING_STRATEGY = {
    "Theta": "rag",
    "Z": "clarification",
    "R": "voting",
}


def run_diagnostic_intervention(
    config_path: str | Path | None = "configs/experiments.yaml",
) -> dict[str, Any]:
    """Run the diagnostic intervention experiment."""

    config = load_config(config_path)
    cfg = config.get("diagnostic_intervention", {})
    M = int(cfg.get("M", 4))
    K = int(cfg.get("K", 4))
    J = int(cfg.get("J", 4))
    L = int(cfg.get("L", 4))
    items_per_type = int(cfg.get("items_per_risk_type", 12))
    strategies = list(cfg.get("strategies", ["rag", "clarification", "voting"]))

    rows: list[dict[str, Any]] = []
    for risk_type, rule in RISK_RULES.items():
        for item_index in range(items_per_type):
            question = f"diagnostic_{risk_type}_{item_index}"
            baseline = _evaluate(question, rule, M, K, J, L)
            for strategy in strategies:
                treated_rule = _repair_rule(risk_type, strategy, rule)
                treated = _evaluate(question, treated_rule, M, K, J, L)
                target = TARGET_TERMS[risk_type]
                rows.append(
                    {
                        "risk_type": risk_type,
                        "strategy": strategy,
                        "item_id": question,
                        "target_term": target,
                        "baseline": baseline,
                        "treated": treated,
                        "target_reduction": baseline[target] - treated[target],
                        "total_reduction": baseline["H(S|X)"] - treated["H(S|X)"],
                        "matched": MATCHING_STRATEGY[risk_type] == strategy,
                    }
                )

    heatmap = aggregate_heatmap(rows)
    return {
        "budget": {"M": M, "K": K, "J": J, "L": L},
        "items_per_risk_type": items_per_type,
        "rows": rows,
        "heatmap": heatmap,
        "interpretation": (
            "A strong diagonal means decomposed uncertainty identifies which "
            "repair strategy is most cost-effective."
        ),
    }


def aggregate_heatmap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate mean reductions by risk type and intervention strategy."""

    frame = pd.DataFrame(
        [
            {
                "risk_type": row["risk_type"],
                "strategy": row["strategy"],
                "target_reduction": row["target_reduction"],
                "total_reduction": row["total_reduction"],
                "matched": row["matched"],
            }
            for row in rows
        ]
    )
    grouped = (
        frame.groupby(["risk_type", "strategy", "matched"], as_index=False)
        .agg(
            mean_target_reduction=("target_reduction", "mean"),
            mean_total_reduction=("total_reduction", "mean"),
        )
        .sort_values(["risk_type", "strategy"])
    )
    return grouped.to_dict(orient="records")


def plot_heatmap(results: dict[str, Any], reports_dir: Path) -> Path:
    """Plot the 3x3 repair-gain heatmap."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results["heatmap"])
    pivot = frame.pivot(
        index="strategy",
        columns="risk_type",
        values="mean_target_reduction",
    ).reindex(index=["rag", "clarification", "voting"], columns=["Theta", "Z", "R"])

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Diagnostic Intervention: Target-Uncertainty Reduction")
    ax.set_xlabel("Dominant Risk Type")
    ax.set_ylabel("Repair Strategy")
    fig.tight_layout()
    path = reports_dir / "diagnostic_intervention_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path.resolve()


def _evaluate(
    question: str,
    rule: Callable[[int, int, int, int], str],
    M: int,
    K: int,
    J: int,
    L: int,
) -> dict[str, float]:
    paths = make_controlled_paths(question, M, K, J, L, rule)
    return decomposition_to_dict(cluster_and_decompose(paths, corrected=False))


def _repair_rule(risk_type: str, strategy: str, fallback: AnswerRule) -> AnswerRule:
    if MATCHING_STRATEGY[risk_type] == strategy:
        return deterministic_rule
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostic intervention experiment.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)
    results = run_diagnostic_intervention(args.config)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "diagnostic_intervention.json"
    save_json(results, output_path)
    logger.info("Saved diagnostic intervention results to %s", output_path)

    if not args.no_plot:
        plot_path = plot_heatmap(results, reports_dir)
        logger.info("Saved diagnostic intervention heatmap to %s", plot_path)


if __name__ == "__main__":
    main()
