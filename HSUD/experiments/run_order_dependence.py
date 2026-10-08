"""Order-dependence and Shapley symmetrization ablation."""

from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from core.estimators import decompose_uncertainty_ordered, shapley_decomposition
from experiments.controlled_oracle import (
    AnswerRule,
    ambiguous_rule,
    cluster_and_decompose,
    knowledge_blind_rule,
    make_controlled_paths,
    reasoning_rule,
)
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


RISK_RULES: dict[str, AnswerRule] = {
    "Theta": knowledge_blind_rule,
    "Z": ambiguous_rule,
    "R": reasoning_rule,
}
TERM_FIELDS = ("u_theta", "u_z", "u_r")


def run_order_dependence(
    config_path: str | Path | None = "configs/experiments.yaml",
) -> dict[str, Any]:
    """Compare default ordered decomposition with Shapley symmetrization."""

    config = load_config(config_path)
    cfg = config.get("order_dependence", {})
    M = int(cfg.get("M", 4))
    K = int(cfg.get("K", 4))
    J = int(cfg.get("J", 4))
    L = int(cfg.get("L", 4))
    items_per_type = int(cfg.get("items_per_type", 12))

    rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    for risk_type, rule in RISK_RULES.items():
        for item_index in range(items_per_type):
            question = f"order_{risk_type}_{item_index}"
            paths = make_controlled_paths(question, M, K, J, L, rule)
            cluster_and_decompose(paths, corrected=False)

            default = decompose_uncertainty_ordered(
                paths, order=("Theta", "Z", "R"), corrected=False
            )
            shapley = shapley_decomposition(paths, corrected=False)
            row = {
                "item_id": question,
                "risk_type": risk_type,
                "default_dominant": _dominant(default),
                "shapley_dominant": _dominant(shapley),
                "dominant_match": _dominant(default) == _dominant(shapley),
            }
            for field in TERM_FIELDS:
                row[f"default_{field}"] = float(getattr(default, field))
                row[f"shapley_{field}"] = float(getattr(shapley, field))
            rows.append(row)

            for order in permutations(("Theta", "Z", "R")):
                decomp = decompose_uncertainty_ordered(paths, order=order, corrected=False)
                order_rows.append(
                    {
                        "item_id": question,
                        "risk_type": risk_type,
                        "order": "->".join(order),
                        "dominant": _dominant(decomp),
                        "u_theta": decomp.u_theta,
                        "u_z": decomp.u_z,
                        "u_r": decomp.u_r,
                        "u_res": decomp.u_res,
                    }
                )

    summary = summarize_order_dependence(rows)
    return {
        "budget": {"M": M, "K": K, "J": J, "L": L},
        "items_per_type": items_per_type,
        "rows": rows,
        "order_rows": order_rows,
        "summary": summary,
    }


def summarize_order_dependence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute match rates and Spearman correlations."""

    frame = pd.DataFrame(rows)
    correlations: dict[str, float | None] = {}
    for field in TERM_FIELDS:
        corr, _p_value = spearmanr(frame[f"default_{field}"], frame[f"shapley_{field}"])
        correlations[field] = None if pd.isna(corr) else float(corr)

    per_type = (
        frame.groupby("risk_type")["dominant_match"]
        .mean()
        .reset_index()
        .rename(columns={"dominant_match": "dominant_match_rate"})
        .to_dict(orient="records")
    )
    return {
        "dominant_match_rate": float(frame["dominant_match"].mean()),
        "per_type": per_type,
        "spearman": correlations,
    }


def plot_results(results: dict[str, Any], reports_dir: Path) -> dict[str, str]:
    """Plot default-vs-Shapley scatter and dominant match rates."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths: dict[str, str] = {}

    frame = pd.DataFrame(results["rows"])
    long_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for field in TERM_FIELDS:
            long_rows.append(
                {
                    "risk_type": row["risk_type"],
                    "term": field,
                    "default": row[f"default_{field}"],
                    "shapley": row[f"shapley_{field}"],
                }
            )
    long_frame = pd.DataFrame(long_rows)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=long_frame,
        x="default",
        y="shapley",
        hue="term",
        style="risk_type",
        ax=ax,
    )
    ax.set_title("Default Order vs Shapley Contributions")
    fig.tight_layout()
    path = reports_dir / "order_default_vs_shapley.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["scatter"] = str(path.resolve())

    match_frame = pd.DataFrame(results["summary"]["per_type"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=match_frame, x="risk_type", y="dominant_match_rate", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Dominant Risk Agreement")
    fig.tight_layout()
    path = reports_dir / "order_dominant_match.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["dominant_match"] = str(path.resolve())

    return paths


def _dominant(decomposition: Any) -> str:
    terms = {
        "Theta": decomposition.u_theta,
        "Z": decomposition.u_z,
        "R": decomposition.u_r,
    }
    return max(terms, key=terms.get)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run order-dependence ablation.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)
    results = run_order_dependence(args.config)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "order_dependence.json"
    save_json(results, output_path)
    logger.info("Saved order-dependence results to %s", output_path)

    if not args.no_plot:
        paths = plot_results(results, reports_dir)
        logger.info("Saved order-dependence plots: %s", paths)


if __name__ == "__main__":
    main()
