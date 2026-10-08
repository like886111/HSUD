"""Phase 3 synthetic sanity check with controlled oracle attribution and plots."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.controlled_oracle import (
    CATEGORY_RULES,
    cluster_and_decompose,
    decomposition_to_dict,
    dominant_term,
    make_controlled_paths,
)
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


TERM_KEYS = ("U_Theta", "U_Z", "U_R", "U_res")
CATEGORY_ORDER = ("A", "B", "C", "D")
CATEGORY_LABELS = {
    "A": "Ambiguous (U_Z)",
    "B": "Knowledge-blind (U_Theta)",
    "C": "Reasoning (U_R)",
    "D": "Clear / Residual (U_res)",
}


def load_synthetic_qa(path: Path | None = None) -> list[dict[str, str]]:
    """Load synthetic QA records from ``data/synthetic_qa.json``."""

    dataset_path = path or PROJECT_ROOT / "data" / "synthetic_qa.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def evaluate_record(
    record: dict[str, str],
    M: int,
    K: int,
    J: int,
    L: int,
) -> dict[str, Any]:
    """Run controlled oracle decomposition for one synthetic question."""

    category = record["category"]
    rule = CATEGORY_RULES[category]
    paths = make_controlled_paths(record["question"], M, K, J, L, rule)
    decomposition = cluster_and_decompose(paths, corrected=False)
    metrics = decomposition_to_dict(decomposition)
    predicted = dominant_term(decomposition)
    return {
        "id": record["id"],
        "category": category,
        "category_name": record["category_name"],
        "question": record["question"],
        "expected_high_term": record["expected_high_term"],
        "predicted_dominant": predicted,
        "orthogonality_pass": predicted == record["expected_high_term"],
        "metrics": metrics,
    }


def run_sanity_check(
    config_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate all synthetic records and aggregate category means."""

    config = load_config(config_path)
    sanity_cfg = config.get(
        "sanity_check",
        {
            "M": config["sampling"]["M"],
            "K": config["sampling"]["K"],
            "J": config["sampling"]["J"],
            "L": config["sampling"]["L"],
            "min_dominance_ratio": 1.5,
        },
    )
    M = int(sanity_cfg.get("M", config["sampling"]["M"]))
    K = int(sanity_cfg.get("K", config["sampling"]["K"]))
    J = int(sanity_cfg.get("J", config["sampling"]["J"]))
    L = int(sanity_cfg.get("L", config["sampling"]["L"]))

    records = load_synthetic_qa(Path(dataset_path) if dataset_path else None)
    per_item = [evaluate_record(record, M, K, J, L) for record in records]

    category_means: dict[str, dict[str, float]] = {}
    category_hits: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)

    for item in per_item:
        category = item["category"]
        category_counts[category] += 1
        category_hits[category] += int(item["orthogonality_pass"])

    for category in CATEGORY_ORDER:
        items = [item for item in per_item if item["category"] == category]
        category_means[category] = {
            term: float(
                sum(item["metrics"][term] for item in items) / max(len(items), 1)
            )
            for term in TERM_KEYS
        }

    orthogonality = verify_category_orthogonality(
        category_means,
        min_ratio=float(sanity_cfg.get("min_dominance_ratio", 1.5)),
    )

    return {
        "budget": {"M": M, "K": K, "J": J, "L": L},
        "per_item": per_item,
        "category_means": category_means,
        "category_accuracy": {
            category: category_hits[category] / category_counts[category]
            for category in CATEGORY_ORDER
        },
        "orthogonality": orthogonality,
    }


def verify_category_orthogonality(
    category_means: dict[str, dict[str, float]],
    min_ratio: float = 1.5,
) -> dict[str, Any]:
    """Check that each category's expected term dominates on average."""

    expected_map = {
        "A": "U_Z",
        "B": "U_Theta",
        "C": "U_R",
        "D": "U_res",
    }
    checks: dict[str, dict[str, float | bool | str]] = {}
    all_pass = True

    for category, expected in expected_map.items():
        means = category_means[category]
        expected_value = means[expected]
        other_values = [means[key] for key in TERM_KEYS if key != expected]
        ratio = expected_value / max(max(other_values), 1e-12)
        passed = expected_value >= max(other_values) and ratio >= min_ratio
        checks[category] = {
            "expected": expected,
            "expected_value": expected_value,
            "max_other": max(other_values),
            "ratio": ratio,
            "pass": passed,
        }
        all_pass = all_pass and passed

    return {"checks": checks, "pass": all_pass}


def plot_category_means(
    category_means: dict[str, dict[str, float]],
    output_path: Path,
) -> Path:
    """Plot mean uncertainty components by synthetic category."""

    rows = []
    for category in CATEGORY_ORDER:
        for term in TERM_KEYS:
            rows.append(
                {
                    "category": CATEGORY_LABELS[category],
                    "term": term,
                    "value": category_means[category][term],
                }
            )

    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=frame,
        x="category",
        y="value",
        hue="term",
        order=list(CATEGORY_LABELS.values()),
        hue_order=list(TERM_KEYS),
        ax=ax,
    )
    ax.set_title("Synthetic Sanity Check: Mean Uncertainty by Category")
    ax.set_xlabel("Question Type")
    ax.set_ylabel("Uncertainty (nats)")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def print_summary(results: dict[str, Any]) -> None:
    """Print human-readable attribution summary."""

    print("\n=== Phase 3 Synthetic Sanity Check ===")
    print("Budget:", results["budget"])
    print("\nCategory accuracy (dominant-term match):")
    for category in CATEGORY_ORDER:
        acc = results["category_accuracy"][category]
        print(f"  {category}: {acc:.1%}")

    print("\nMean uncertainty by category:")
    header = f"{'Cat':<5} " + " ".join(f"{term:>8}" for term in TERM_KEYS)
    print(header)
    for category in CATEGORY_ORDER:
        means = results["category_means"][category]
        row = f"{category:<5} " + " ".join(f"{means[t]:8.4f}" for t in TERM_KEYS)
        print(row)

    print("\nOrthogonality checks:")
    for category, check in results["orthogonality"]["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        print(
            f"  {category} expected={check['expected']} "
            f"ratio={check['ratio']:.2f} -> {status}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 3 synthetic sanity check.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to synthetic QA JSON (default: data/synthetic_qa.json).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)
    results = run_sanity_check(config_path=args.config, dataset_path=args.dataset)
    print_summary(results)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = outputs_dir / "sanity_check_summary.json"
    slim_results = {
        "budget": results["budget"],
        "category_means": results["category_means"],
        "category_accuracy": results["category_accuracy"],
        "orthogonality": results["orthogonality"],
        "per_item_count": len(results["per_item"]),
    }
    save_json(slim_results, summary_path)
    logger.info("Saved summary to %s", summary_path)

    if not args.no_plot:
        plot_path = reports_dir / "sanity_check_orthogonality.png"
        plot_category_means(results["category_means"], plot_path)
        logger.info("Saved plot to %s", plot_path)

    if not results["orthogonality"]["pass"]:
        raise SystemExit("Synthetic sanity check failed orthogonality checks.")

    logger.info("Synthetic sanity check passed.")


if __name__ == "__main__":
    main()
