"""Phase 6 adaptive router demo on canonical uncertainty regimes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.pipeline import evaluate_and_route
from core.router import route, RouterConfig
from experiments.controlled_oracle import CATEGORY_RULES, cluster_and_decompose, make_controlled_paths
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


def load_demo_cases(path: Path | None = None) -> list[dict[str, str]]:
    """Load router demo cases."""

    demo_path = path or PROJECT_ROOT / "data" / "router_demo_cases.json"
    return json.loads(demo_path.read_text(encoding="utf-8"))


def evaluate_case_oracle(
    case: dict[str, str],
    M: int,
    K: int,
    J: int,
    L: int,
    router_config: RouterConfig,
) -> dict[str, Any]:
    """Evaluate one demo case with controlled oracle decomposition."""

    rule = CATEGORY_RULES[case["category"]]
    paths = make_controlled_paths(case["question"], M, K, J, L, rule)
    decomposition = cluster_and_decompose(paths, corrected=False)
    routing = route(decomposition, router_config=router_config)
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_action": case["expected_action"],
        "expected_dominant": case["expected_dominant"],
        "predicted_action": routing.action,
        "predicted_dominant": routing.dominant_term,
        "pass": routing.action == case["expected_action"],
        "decomposition": asdict(decomposition),
        "routing": asdict(routing),
    }


def run_router_demo_oracle(config_path: str | Path | None = None) -> dict[str, Any]:
    """Run router demo using controlled oracle (recommended for CI)."""

    config = load_config(config_path)
    router_cfg = RouterConfig.from_config(config)
    sanity = config["sanity_check"]
    M = int(sanity["M"])
    K = int(sanity["K"])
    J = int(sanity["J"])
    L = int(sanity["L"])

    results = [
        evaluate_case_oracle(case, M, K, J, L, router_cfg)
        for case in load_demo_cases()
    ]
    return {
        "mode": "oracle",
        "results": results,
        "pass": all(item["pass"] for item in results),
    }


def run_router_demo_pipeline(
    config_path: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run router demo through the full pipeline."""

    results: list[dict[str, Any]] = []
    for case in load_demo_cases():
        routed = evaluate_and_route(
            question=case["question"],
            config_path=config_path,
            dry_run=dry_run,
            question_id=case["id"],
        )
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_action": case["expected_action"],
                "predicted_action": routed.routing.action,
                "predicted_dominant": routed.routing.dominant_term,
                "pass": routed.routing.action == case["expected_action"],
                "routing": asdict(routed.routing),
                "report_path": str(routed.report_path),
            }
        )
    return {
        "mode": "pipeline",
        "dry_run": dry_run,
        "results": results,
        "pass": all(item["pass"] for item in results),
    }


def plot_router_actions(results: dict[str, Any], output_path: Path) -> Path:
    """Plot predicted routing actions for demo cases."""

    rows = []
    for item in results["results"]:
        rows.append(
            {
                "case": item["id"],
                "expected": item["expected_action"],
                "predicted": item["predicted_action"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["match"] = frame["expected"] == frame["predicted"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=frame,
        x="case",
        y="predicted",
        hue="match",
        palette={True: "green", False: "red"},
        s=120,
        ax=ax,
    )
    for idx, row in frame.iterrows():
        ax.text(idx, row["expected"], "expected", fontsize=8, ha="center", va="bottom")
    ax.set_title("Phase 6 Router Demo: Predicted vs Expected Actions")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def print_summary(results: dict[str, Any]) -> None:
    """Print router demo summary."""

    print(f"\n=== Phase 6 Router Demo ({results['mode']}) ===")
    for item in results["results"]:
        status = "PASS" if item["pass"] else "FAIL"
        print(
            f"  {item['id']}: expected={item['expected_action']} "
            f"predicted={item['predicted_action']} dominant={item['predicted_dominant']} -> {status}"
        )
        if "routing" in item and "user_message" in item["routing"]:
            print(f"    message: {item['routing']['user_message']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 6 adaptive router demo.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=("oracle", "pipeline"),
        default="oracle",
        help="oracle=controlled decomposition; pipeline=evaluate_and_route",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if pipeline mode routing does not match expected actions.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)

    if args.mode == "oracle":
        results = run_router_demo_oracle(config_path=args.config)
    else:
        results = run_router_demo_pipeline(
            config_path=args.config,
            dry_run=bool(config["llm"].get("dry_run", True)),
        )

    print_summary(results)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = outputs_dir / f"router_demo_{results['mode']}_summary.json"
    save_json(results, summary_path)
    logger.info("Saved summary to %s", summary_path)

    if not args.no_plot:
        plot_path = reports_dir / f"router_demo_{results['mode']}.png"
        plot_router_actions(results, plot_path)
        logger.info("Saved plot to %s", plot_path)

    if not results["pass"]:
        if results["mode"] == "oracle" or args.strict:
            raise SystemExit("Router demo failed one or more expected action checks.")
        logger.warning(
            "Pipeline router demo did not match all expected actions "
            "(common with dummy LLM; use a real model or --mode oracle)."
        )
        return

    logger.info("Router demo passed.")


if __name__ == "__main__":
    main()
