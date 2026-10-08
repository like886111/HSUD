"""Risk prediction and orthogonal information-gain analysis.

The script consumes ``benchmark_{mode}_full.json`` produced by
``experiments/run_benchmark.py`` and builds paper-ready tables for:
- global AUROC comparison,
- per-dataset local weaknesses,
- logistic-regression feature importance for U_Theta, U_Z, U_R, U_res.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from core.evaluation.metrics import auroc
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


COMPONENT_FIELDS = ("u_theta", "u_z", "u_r", "u_res")
METHOD_FIELDS = {
    "Nested H(S|x)": "nested_total",
    "Semantic Entropy": "semantic_entropy",
    "Self-Consistency": "self_consistency",
    "Input Perturbation": "input_perturbation",
    "U_Theta": "u_theta",
    "U_Z": "u_z",
    "U_R": "u_r",
    "U_res": "u_res",
}


def run_risk_prediction_analysis(
    config_path: str | Path | None = "configs/experiments.yaml",
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze benchmark full scores for risk prediction."""

    config = load_config(config_path)
    full_path = _resolve_input_path(config, input_path)
    payload = json.loads(full_path.read_text(encoding="utf-8"))
    scores = _ensure_baseline_fields(payload["scores"])

    auroc_table = compute_auroc_table(scores)
    logistic = compute_logistic_importance(
        scores,
        folds=int(config.get("risk_prediction", {}).get("logistic_cv_folds", 5)),
    )
    correlations = compute_component_correlations(scores)
    return {
        "source": str(full_path.resolve()),
        "mode": payload.get("mode", "unknown"),
        "budget": payload.get("budget", {}),
        "datasets": sorted({item["dataset"] for item in scores}),
        "auroc_table": auroc_table,
        "logistic_importance": logistic,
        "component_correlations": correlations,
        "notes": {
            "self_consistency": "1 - majority vote share from flat CoT samples.",
            "input_perturbation": "Semantic entropy over answers from paraphrased inputs.",
            "token_entropy": "Not implemented; requires token logprob access from the generator.",
        },
    }


def compute_auroc_table(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute overall and per-dataset AUROC for every method field."""

    rows: list[dict[str, Any]] = []
    datasets = ["overall"] + sorted({str(item["dataset"]) for item in scores})
    for dataset in datasets:
        subset = scores if dataset == "overall" else [
            item for item in scores if item["dataset"] == dataset
        ]
        labels = np.array([int(item["label"]) for item in subset], dtype=int)
        for method, field in METHOD_FIELDS.items():
            values = np.array([float(item[field]) for item in subset], dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "field": field,
                    "auroc": _safe_auroc(labels, values),
                    "count": len(subset),
                }
            )
    return rows


def compute_logistic_importance(
    scores: list[dict[str, Any]],
    folds: int,
) -> dict[str, Any]:
    """Fit a simple component-feature risk classifier."""

    labels = np.array([int(item["label"]) for item in scores], dtype=int)
    features = np.array(
        [[float(item[field]) for field in COMPONENT_FIELDS] for item in scores],
        dtype=float,
    )
    if len(np.unique(labels)) < 2:
        return {"cv_auroc": None, "coefficients": {}}

    class_counts = np.bincount(labels)
    max_folds = int(min(folds, class_counts[class_counts > 0].min()))
    if max_folds < 2:
        cv_auroc = None
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        splitter = StratifiedKFold(n_splits=max_folds, shuffle=True, random_state=42)
        cv_auroc = float(
            np.mean(cross_val_score(model, features, labels, cv=splitter, scoring="roc_auc"))
        )

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    final_model.fit(features, labels)
    logistic = final_model.named_steps["logisticregression"]
    coefficients = {
        field: float(coef)
        for field, coef in zip(COMPONENT_FIELDS, logistic.coef_[0])
    }
    return {"cv_auroc": cv_auroc, "coefficients": coefficients}


def compute_component_correlations(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Return component correlation matrix for orthogonality diagnostics."""

    frame = pd.DataFrame(scores)
    corr = frame[list(COMPONENT_FIELDS)].corr(method="spearman").fillna(0.0)
    return {
        row: {col: float(corr.loc[row, col]) for col in COMPONENT_FIELDS}
        for row in COMPONENT_FIELDS
    }


def plot_results(results: dict[str, Any], reports_dir: Path) -> dict[str, str]:
    """Create AUROC and feature-importance plots."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths: dict[str, str] = {}

    auroc_frame = pd.DataFrame(results["auroc_table"])
    overall = auroc_frame[auroc_frame["dataset"] == "overall"].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=overall, x="method", y="auroc", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Global Risk Prediction AUROC")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = reports_dir / "risk_prediction_auroc.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["auroc"] = str(path.resolve())

    coefficients = results["logistic_importance"]["coefficients"]
    coef_frame = pd.DataFrame(
        [{"component": key, "coefficient": value} for key, value in coefficients.items()]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=coef_frame, x="component", y="coefficient", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Logistic Feature Importance")
    fig.tight_layout()
    path = reports_dir / "risk_prediction_feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["feature_importance"] = str(path.resolve())

    return paths


def _ensure_baseline_fields(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backfill legacy proxy fields when reading older benchmark JSON."""

    augmented: list[dict[str, Any]] = []
    for item in scores:
        record = dict(item)
        if "self_consistency" not in record:
            record["self_consistency"] = float(record.get("u_r", 0.0)) + float(record.get("u_res", 0.0))
        if "input_perturbation" not in record:
            record["input_perturbation"] = float(record.get("u_z", 0.0))
        augmented.append(record)
    return augmented


def _safe_auroc(labels: np.ndarray, values: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(auroc(labels, values))


def _resolve_input_path(config: dict[str, Any], input_path: str | Path | None) -> Path:
    if input_path is not None:
        return Path(input_path)
    risk_cfg = config.get("risk_prediction", {})
    candidate = PROJECT_ROOT / str(risk_cfg.get("input_full_json", ""))
    if candidate.is_file():
        return candidate
    fallback = PROJECT_ROOT / str(risk_cfg.get("fallback_full_json", "outputs/benchmark_oracle_full.json"))
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        "No benchmark full JSON found. Run run_benchmark.py first or pass --input."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run risk prediction analysis.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)
    results = run_risk_prediction_analysis(args.config, args.input)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "risk_prediction_analysis.json"
    save_json(results, output_path)
    logger.info("Saved risk prediction analysis to %s", output_path)

    if not args.no_plot:
        paths = plot_results(results, reports_dir)
        logger.info("Saved risk prediction plots: %s", paths)


if __name__ == "__main__":
    main()
