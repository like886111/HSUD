"""Statistical validation experiments for finite-sample guarantees.

This script covers:
1. Miller-Madow vs plug-in entropy bias at small sample sizes.
2. Bootstrap confidence interval convergence for U_Theta, U_Z, U_R, U_res.
3. Fano-style sensitivity of semantic entropy to injected clustering noise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.estimators import (
    UncertaintyDecomposition,
    decompose_uncertainty,
    entropy,
    miller_madow_entropy,
)
from core.llm_backend import SampledPath
from experiments.controlled_oracle import (
    cluster_and_decompose,
    knowledge_blind_rule,
    make_controlled_paths,
)
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


TERM_FIELDS = ("u_theta", "u_z", "u_r", "u_res")
TERM_LABELS = {
    "u_theta": "U_Theta",
    "u_z": "U_Z",
    "u_r": "U_R",
    "u_res": "U_res",
}


def run_statistical_validation(config_path: str | Path | None = None) -> dict[str, Any]:
    """Run all statistical validation sub-experiments."""

    config = load_config(config_path)
    cfg = config.get("statistical_validation", {})
    seed = int(cfg.get("seed", 20260617))
    rng = np.random.default_rng(seed)

    mm = run_miller_madow_bias(
        sample_sizes=cfg.get("mm_sample_sizes", [5, 10, 20, 50, 100, 200]),
        repeats=int(cfg.get("mm_repeats", 300)),
        rng=rng,
    )
    bootstrap = run_bootstrap_ci(
        sample_sizes=cfg.get("bootstrap_sample_sizes", [64, 128, 256, 512, 1024]),
        repeats=int(cfg.get("bootstrap_repeats", 200)),
        rng=rng,
    )
    fano = run_fano_sensitivity(
        noise_rates=cfg.get("fano_noise_rates", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]),
        repeats=int(cfg.get("fano_repeats", 200)),
        rng=rng,
    )
    return {"seed": seed, "miller_madow": mm, "bootstrap_ci": bootstrap, "fano": fano}


def run_miller_madow_bias(
    sample_sizes: list[int],
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compare small-sample entropy bias for plug-in vs Miller-Madow."""

    probabilities = np.array([0.55, 0.25, 0.15, 0.05], dtype=float)
    true_entropy = entropy(probabilities)
    rows: list[dict[str, float | int | str]] = []

    for n_samples in sample_sizes:
        for _ in range(repeats):
            labels = rng.choice(len(probabilities), size=int(n_samples), p=probabilities)
            counts = np.bincount(labels, minlength=len(probabilities))
            plugin = entropy(counts / counts.sum())
            corrected = miller_madow_entropy(labels.tolist())
            rows.append(
                {
                    "N": int(n_samples),
                    "estimator": "plugin",
                    "estimate": plugin,
                    "absolute_error": abs(plugin - true_entropy),
                }
            )
            rows.append(
                {
                    "N": int(n_samples),
                    "estimator": "miller_madow",
                    "estimate": corrected,
                    "absolute_error": abs(corrected - true_entropy),
                }
            )

    summary = _group_mean(rows, ["N", "estimator"], "absolute_error")
    return {
        "true_entropy": true_entropy,
        "sample_sizes": [int(value) for value in sample_sizes],
        "repeats": repeats,
        "rows": rows,
        "summary": summary,
    }


def run_bootstrap_ci(
    sample_sizes: list[int],
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Estimate non-parametric bootstrap CIs for decomposition terms."""

    base_paths = make_controlled_paths(
        "Synthetic knowledge-state bootstrap validation.",
        M=8,
        K=8,
        J=8,
        L=4,
        answer_rule=knowledge_blind_rule,
    )
    truth = cluster_and_decompose(base_paths, corrected=False)
    rows: list[dict[str, float | int | str]] = []

    for n_samples in sample_sizes:
        for _ in range(repeats):
            sampled = _resample_paths(base_paths, int(n_samples), rng)
            decomposition = decompose_uncertainty(
                sampled,
                corrected=True,
                clip_negative=False,
                verify_identity=True,
            )
            for field in TERM_FIELDS:
                rows.append(
                    {
                        "N": int(n_samples),
                        "term": TERM_LABELS[field],
                        "estimate": float(getattr(decomposition, field)),
                    }
                )

    summary: list[dict[str, float | int | str]] = []
    frame = pd.DataFrame(rows)
    for (n_samples, term), group in frame.groupby(["N", "term"]):
        values = group["estimate"].to_numpy(dtype=float)
        summary.append(
            {
                "N": int(n_samples),
                "term": str(term),
                "mean": float(np.mean(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
        )

    return {
        "truth": asdict(truth),
        "sample_sizes": [int(value) for value in sample_sizes],
        "repeats": repeats,
        "rows": rows,
        "summary": summary,
    }


def run_fano_sensitivity(
    noise_rates: list[float],
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Inject semantic-label noise and compare entropy drift to a Fano bound."""

    paths = make_controlled_paths(
        "Synthetic Fano sensitivity validation.",
        M=6,
        K=6,
        J=6,
        L=4,
        answer_rule=knowledge_blind_rule,
    )
    truth = cluster_and_decompose(paths, corrected=False)
    labels = np.array([path.s_cluster_id for path in paths], dtype=int)
    class_count = int(len(np.unique(labels)))
    rows: list[dict[str, float]] = []

    for eta in noise_rates:
        for _ in range(repeats):
            noisy_labels = _inject_label_noise(labels, float(eta), class_count, rng)
            noisy_paths = _paths_with_labels(paths, noisy_labels)
            noisy = decompose_uncertainty(
                noisy_paths,
                corrected=False,
                clip_negative=False,
                verify_identity=True,
            )
            drift = abs(noisy.total_entropy - truth.total_entropy)
            rows.append(
                {
                    "eta": float(eta),
                    "entropy_drift": float(drift),
                    "fano_bound": float(_fano_bound(float(eta), class_count)),
                }
            )

    summary = _group_mean(rows, ["eta"], "entropy_drift")
    for row in summary:
        row["fano_bound"] = _fano_bound(float(row["eta"]), class_count)
    return {
        "class_count": class_count,
        "truth": asdict(truth),
        "noise_rates": [float(value) for value in noise_rates],
        "repeats": repeats,
        "rows": rows,
        "summary": summary,
    }


def plot_results(results: dict[str, Any], reports_dir: Path) -> dict[str, str]:
    """Create the three statistical validation plots."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths: dict[str, str] = {}

    mm_frame = pd.DataFrame(results["miller_madow"]["summary"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=mm_frame, x="N", y="mean_absolute_error", hue="estimator", marker="o", ax=ax)
    ax.set_title("Miller-Madow Bias Correction")
    ax.set_ylabel("Mean absolute entropy error")
    fig.tight_layout()
    path = reports_dir / "stat_mm_bias.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["miller_madow"] = str(path.resolve())

    ci_frame = pd.DataFrame(results["bootstrap_ci"]["summary"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for term, group in ci_frame.groupby("term"):
        group = group.sort_values("N")
        x = group["N"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        low = group["ci_low"].to_numpy(dtype=float)
        high = group["ci_high"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", label=term)
        ax.fill_between(x, low, high, alpha=0.15)
    ax.set_title("Bootstrap CI Convergence")
    ax.set_xlabel("Leaf samples N")
    ax.set_ylabel("Uncertainty (nats)")
    ax.legend()
    fig.tight_layout()
    path = reports_dir / "stat_bootstrap_ci.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["bootstrap_ci"] = str(path.resolve())

    fano_frame = pd.DataFrame(results["fano"]["summary"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fano_frame["eta"], fano_frame["mean_entropy_drift"], marker="o", label="Observed drift")
    ax.plot(fano_frame["eta"], fano_frame["fano_bound"], linestyle="--", label="Fano bound")
    ax.set_title("Semantic Clustering Noise Sensitivity")
    ax.set_xlabel("Injected error rate eta")
    ax.set_ylabel("|H(noisy S) - H(S)|")
    ax.legend()
    fig.tight_layout()
    path = reports_dir / "stat_fano_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["fano"] = str(path.resolve())

    return paths


def _group_mean(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    value_key: str,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {key: value for key, value in zip(group_keys, keys)}
        row[f"mean_{value_key}"] = float(group[value_key].mean())
        out.append(row)
    return out


def _resample_paths(
    paths: list[SampledPath],
    n_samples: int,
    rng: np.random.Generator,
) -> list[SampledPath]:
    indices = rng.choice(len(paths), size=n_samples, replace=True)
    sampled: list[SampledPath] = []
    for new_index, source_index in enumerate(indices):
        source = paths[int(source_index)]
        sampled.append(
            SampledPath(
                theta_id=source.theta_id,
                z_id=source.z_id,
                r_id=source.r_id,
                l_id=new_index,
                question=source.question,
                z_text=source.z_text,
                r_text=source.r_text,
                answer_text=source.answer_text,
                s_cluster_id=source.s_cluster_id,
            )
        )
    return sampled


def _paths_with_labels(paths: list[SampledPath], labels: np.ndarray) -> list[SampledPath]:
    return [
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
        for path, label in zip(paths, labels)
    ]


def _inject_label_noise(
    labels: np.ndarray,
    eta: float,
    class_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy = labels.copy()
    flip_mask = rng.random(labels.shape[0]) < eta
    for index in np.where(flip_mask)[0]:
        choices = [label for label in range(class_count) if label != labels[index]]
        noisy[index] = int(rng.choice(choices))
    return noisy


def _fano_bound(eta: float, class_count: int) -> float:
    if eta <= 0.0:
        return 0.0
    if eta >= 1.0:
        eta = 1.0 - 1e-12
    binary_entropy = -eta * np.log(eta) - (1.0 - eta) * np.log(1.0 - eta)
    return float(binary_entropy + eta * np.log(max(class_count - 1, 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run statistical validation experiments.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)
    results = run_statistical_validation(args.config)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "statistical_validation.json"
    save_json(results, output_path)
    logger.info("Saved statistical validation results to %s", output_path)

    if not args.no_plot:
        paths = plot_results(results, reports_dir)
        logger.info("Saved statistical validation plots: %s", paths)


if __name__ == "__main__":
    main()
