"""Run the paper experiment suite in the order described in the report.

Default behavior is offline-friendly:
1. Run oracle benchmark on the configured dataset matrix.
2. Run statistical validation.
3. Run risk prediction analysis on the oracle full JSON.
4. Run diagnostic intervention.
5. Run order-dependence ablation.

For real LLM experiments, first run:
```
python experiments/run_benchmark.py --config configs/paper.yaml --mode pipeline
```
then pass ``--risk-input outputs/paper/benchmark_pipeline_full.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_benchmark import run_benchmark
from experiments.run_diagnostic_intervention import run_diagnostic_intervention
from experiments.run_order_dependence import run_order_dependence
from experiments.run_risk_prediction_analysis import run_risk_prediction_analysis
from experiments.run_statistical_validation import run_statistical_validation
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json


def run_suite(
    config_path: str | Path = "configs/experiments.yaml",
    skip_benchmark: bool = False,
    risk_input: str | Path | None = None,
) -> dict[str, Any]:
    """Run the configured experiment suite."""

    config = load_config(config_path)
    datasets = tuple(config.get("risk_prediction", {}).get("datasets", ("truthfulqa", "gsm8k")))
    results: dict[str, Any] = {"config": str(config_path), "datasets": list(datasets)}

    if not skip_benchmark:
        benchmark = run_benchmark(
            config_path=None,
            mode="oracle",
            datasets=datasets,
        )
        benchmark_path = PROJECT_ROOT / "outputs" / "benchmark_oracle_full.json"
        save_json(benchmark, benchmark_path)
        risk_input = risk_input or benchmark_path
        results["benchmark_oracle"] = benchmark

    results["statistical_validation"] = run_statistical_validation(config_path)
    results["risk_prediction"] = run_risk_prediction_analysis(config_path, risk_input)
    results["diagnostic_intervention"] = run_diagnostic_intervention(config_path)
    results["order_dependence"] = run_order_dependence(config_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full paper experiment suite.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip oracle benchmark generation and reuse an existing full JSON.",
    )
    parser.add_argument(
        "--risk-input",
        type=str,
        default=None,
        help="Full benchmark JSON for risk analysis, e.g. outputs/paper/benchmark_pipeline_full.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger()
    config = load_config(args.config)
    results = run_suite(args.config, args.skip_benchmark, args.risk_input)

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "paper_experiment_suite_summary.json"
    save_json(results, output_path)
    logger.info("Saved paper experiment suite summary to %s", output_path)


if __name__ == "__main__":
    main()
