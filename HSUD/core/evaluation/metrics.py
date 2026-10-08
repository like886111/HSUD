"""Evaluation metrics for uncertainty-based hallucination detection."""

from __future__ import annotations

from typing import Any

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC for binary labels and continuous uncertainty scores.

    Args:
        y_true: Binary labels where ``1`` indicates hallucination / error.
        y_score: Higher scores should indicate higher uncertainty / risk.

    Returns:
        Area under the ROC curve in ``[0, 1]``.
    """

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.size == 0:
        raise ValueError("Cannot compute AUROC on empty input.")
    if len(np.unique(labels)) < 2:
        # Smoke tests / tiny subsets may contain a single class; AUROC is undefined.
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    positives = labels == 1
    n_pos = int(np.sum(positives))
    n_neg = int(len(labels) - n_pos)
    rank_sum = float(np.sum(ranks[positives]))
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def risk_coverage_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage points by accepting lowest-uncertainty items first.

    Args:
        y_true: Binary error / hallucination labels.
        y_score: Uncertainty scores (lower = more confident).

    Returns:
        Tuple ``(coverages, risks)`` with values in ``[0, 1]``.
    """

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    order = np.argsort(scores)
    sorted_labels = labels[order]
    n = len(labels)
    coverages = np.arange(1, n + 1, dtype=float) / n
    risks = np.array(
        [float(np.mean(sorted_labels[: k + 1])) for k in range(n)],
        dtype=float,
    )
    return coverages, risks


def aurac(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the risk-coverage accuracy curve.

    Items are accepted from lowest uncertainty to highest uncertainty. Higher
    AURAC is better because it integrates ``1 - risk`` across coverage.
    """

    coverages, risks = risk_coverage_curve(y_true, y_score)
    accuracies = 1.0 - risks
    return float(np.trapezoid(accuracies, coverages))


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Non-parametric bootstrap CI for AUROC (percentile method).

    Rows are resampled with replacement. Bootstrap draws that contain a single
    class are skipped; if fewer than 50 valid draws remain, CI bounds are NaN.
    """

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    point = auroc(labels, scores)
    n = labels.size
    if n == 0 or not np.isfinite(point):
        return {
            "auroc": float(point) if np.isfinite(point) else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": float(n_boot),
            "n_valid": 0.0,
            "alpha": float(alpha),
            "seed": float(seed),
        }

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        values.append(auroc(boot_labels, scores[idx]))

    if len(values) < 50:
        return {
            "auroc": float(point),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": float(n_boot),
            "n_valid": float(len(values)),
            "alpha": float(alpha),
            "seed": float(seed),
        }

    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return {
        "auroc": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": float(n_boot),
        "n_valid": float(len(values)),
        "alpha": float(alpha),
        "seed": float(seed),
    }


def bootstrap_auroc_difference(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI for AUROC(A) - AUROC(B) on paired item resamples."""

    labels = np.asarray(y_true, dtype=int)
    score_a = np.asarray(y_score_a, dtype=float)
    score_b = np.asarray(y_score_b, dtype=float)
    point_a = auroc(labels, score_a)
    point_b = auroc(labels, score_b)
    point = float(point_a - point_b)
    n = labels.size
    if n == 0 or not (np.isfinite(point_a) and np.isfinite(point_b)):
        return {
            "delta": float("nan"),
            "auroc_a": float(point_a) if np.isfinite(point_a) else float("nan"),
            "auroc_b": float(point_b) if np.isfinite(point_b) else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_le_zero": float("nan"),
            "n_valid": 0,
            "n_boot": n_boot,
            "alpha": alpha,
            "seed": seed,
        }

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        deltas.append(auroc(boot_labels, score_a[idx]) - auroc(boot_labels, score_b[idx]))

    if len(deltas) < 50:
        return {
            "delta": point,
            "auroc_a": float(point_a),
            "auroc_b": float(point_b),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_le_zero": float("nan"),
            "n_valid": len(deltas),
            "n_boot": n_boot,
            "alpha": alpha,
            "seed": seed,
        }

    arr = np.asarray(deltas, dtype=float)
    # One-sided: fraction of bootstrap deltas <= 0 (A not better than B).
    p_le_zero = float(np.mean(arr <= 0.0))
    return {
        "delta": point,
        "auroc_a": float(point_a),
        "auroc_b": float(point_b),
        "ci_low": float(np.quantile(arr, alpha / 2.0)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
        "p_le_zero": p_le_zero,
        "n_valid": int(arr.size),
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
    }
