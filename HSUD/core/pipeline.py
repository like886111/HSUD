"""End-to-end evaluation pipeline for hierarchical uncertainty decomposition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from core.estimators import UncertaintyDecomposition
from core.llm_backend import LLMGenerator
from core.monte_carlo import NestedMonteCarloEngine, NestedSamplingResult
from core.router import RoutingDecision, RouterConfig, route
from core.semantic_nli import SemanticNLIClusterer
from utils.config import load_config, resolve_path
from utils.logger import get_logger, save_json
from utils.storage import SampleFormat


@dataclass(slots=True)
class UncertaintyReport:
    """Final report returned by :func:`evaluate_question`."""

    question: str
    decomposition: UncertaintyDecomposition
    sample_tree_path: Path
    report_path: Path
    sample_tree: dict[str, Any]
    dry_run: bool


@dataclass(slots=True)
class RoutedReport:
    """Uncertainty report plus adaptive routing decision."""

    question: str
    decomposition: UncertaintyDecomposition
    routing: RoutingDecision
    sample_tree_path: Path
    report_path: Path
    sample_tree: dict[str, Any]
    dry_run: bool


def build_engine(config: dict[str, Any]) -> NestedMonteCarloEngine:
    """Construct a nested Monte Carlo engine from config."""

    project_seed = int(config.get("project", {}).get("seed", 42))
    return NestedMonteCarloEngine(
        generator=LLMGenerator.from_config(config),
        clusterer=SemanticNLIClusterer.from_config(config),
        base_seed=project_seed + 1009,
    )


def evaluate_question(
    question: str,
    config_path: str | Path | None = None,
    dry_run: Optional[bool] = None,
    output_dir: str | Path | None = None,
    question_id: str = "question",
) -> UncertaintyReport:
    """Run the full nested sampling + clustering + decomposition pipeline.

    Args:
        question: Observed input :math:`X`.
        config_path: Optional YAML config path. Defaults to ``configs/default.yaml``.
        dry_run: Override ``llm.dry_run`` from config when provided.
        output_dir: Directory for persisted sample trees and reports.
        question_id: Stable slug used in output filenames.

    Returns:
        ``UncertaintyReport`` with decomposition and artifact paths.
    """

    config = load_config(config_path)
    if dry_run is not None:
        config = _copy_config(config)
        config["llm"]["dry_run"] = dry_run

    sampling = config["sampling"]
    estimation = config["estimation"]
    paths_cfg = config["paths"]
    outputs_dir = Path(output_dir) if output_dir is not None else resolve_path(
        config, "paths", "outputs_dir"
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sample_format: SampleFormat = paths_cfg.get("samples_format", "jsonl")
    sample_tree_path = outputs_dir / f"{question_id}_samples.{sample_format}"

    engine = build_engine(config)
    logger = get_logger()
    logger.info(
        "Evaluating question with M=%s K=%s J=%s L=%s dry_run=%s",
        sampling["M"],
        sampling["K"],
        sampling["J"],
        sampling["L"],
        config["llm"]["dry_run"],
    )

    result = engine.run_nested_sampling(
        question=question,
        M=int(sampling["M"]),
        K=int(sampling["K"]),
        J=int(sampling["J"]),
        L=int(sampling["L"]),
        intent_temperature=float(sampling.get("intent_temperature", 0.8)),
        reasoning_temperature=float(sampling.get("reasoning_temperature", 1.0)),
        max_intent_tokens=int(sampling.get("max_intent_tokens", 96)),
        max_answer_tokens=int(sampling.get("max_answer_tokens", 192)),
        persist_path=sample_tree_path,
        persist_format=sample_format,
        miller_madow=bool(estimation.get("miller_madow", True)),
        clip_negative=bool(estimation.get("clip_negative", False)),
        identity_tolerance=float(estimation.get("identity_tolerance", 1e-10)),
        show_progress=True,
    )

    if result.decomposition is None:
        raise RuntimeError("Pipeline finished without a decomposition result.")

    report_path = outputs_dir / f"{question_id}_report.json"
    report_payload = {
        "question": question,
        "dry_run": bool(config["llm"]["dry_run"]),
        "decomposition": asdict(result.decomposition),
        "sample_tree_path": str(result.sample_tree_path or sample_tree_path),
        "budget": result.sample_tree["budget"],
    }
    save_json(report_payload, report_path)

    return UncertaintyReport(
        question=question,
        decomposition=result.decomposition,
        sample_tree_path=Path(result.sample_tree_path or sample_tree_path),
        report_path=report_path.resolve(),
        sample_tree=result.sample_tree,
        dry_run=bool(config["llm"]["dry_run"]),
    )


def evaluate_and_route(
    question: str,
    config_path: str | Path | None = None,
    dry_run: Optional[bool] = None,
    output_dir: str | Path | None = None,
    question_id: str = "question",
) -> RoutedReport:
    """Run uncertainty decomposition and choose a downstream routing action.

    Args:
        question: Observed input :math:`X`.
        config_path: Optional YAML config path.
        dry_run: Override ``llm.dry_run`` when provided.
        output_dir: Directory for persisted artifacts.
        question_id: Stable slug used in output filenames.

    Returns:
        ``RoutedReport`` containing decomposition and routing decision.
    """

    config = load_config(config_path)
    if dry_run is not None:
        config = _copy_config(config)
        config["llm"]["dry_run"] = dry_run

    report = evaluate_question(
        question=question,
        config_path=config_path,
        dry_run=dry_run,
        output_dir=output_dir,
        question_id=question_id,
    )
    router_config = RouterConfig.from_config(config)
    routing = route(report.decomposition, router_config=router_config)

    routed_path = report.report_path.with_name(f"{question_id}_routed_report.json")
    save_json(
        {
            "question": report.question,
            "dry_run": report.dry_run,
            "decomposition": asdict(report.decomposition),
            "routing": asdict(routing),
            "sample_tree_path": str(report.sample_tree_path),
        },
        routed_path,
    )

    return RoutedReport(
        question=report.question,
        decomposition=report.decomposition,
        routing=routing,
        sample_tree_path=report.sample_tree_path,
        report_path=routed_path.resolve(),
        sample_tree=report.sample_tree,
        dry_run=report.dry_run,
    )


def _copy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a config mapping for safe local overrides."""

    return json.loads(json.dumps(config))
