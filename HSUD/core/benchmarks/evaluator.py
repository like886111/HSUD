"""Oracle and pipeline evaluators for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.baselines.input_perturbation import input_perturbation_baseline
from core.baselines.self_consistency import self_consistency_baseline
from core.baselines.semantic_entropy import semantic_entropy_baseline
from core.benchmarks.comparability import ComparabilitySettings, build_fair_comparison_plan
from core.benchmarks.data import BenchmarkItem
from core.benchmarks.grading import is_correct_answer, majority_answer
from core.estimators import UncertaintyDecomposition
from core.llm_backend import LLMGenerator, SampledPath
from core.monte_carlo import NestedMonteCarloEngine
from core.semantic_nli import SemanticNLIClusterer
from experiments.controlled_oracle import (
    ambiguous_rule,
    cluster_and_decompose,
    knowledge_blind_rule,
    make_controlled_paths,
    reasoning_rule,
)
from experiments.interventions import deterministic_rule
from utils.config import resolve_path
from utils.prompts import PromptContext


@dataclass(slots=True)
class BenchmarkScores:
    """Uncertainty scores and baselines for one benchmark item."""

    item_id: str
    dataset: str
    label: int
    predicted_correct: bool
    nested_total: float
    semantic_entropy: float
    self_consistency: float
    input_perturbation: float
    u_theta: float
    u_z: float
    u_r: float
    u_res: float


def oracle_answer_rule(item: BenchmarkItem):
    """Choose a controlled answer rule correlated with benchmark label."""

    if item.label == 0:
        return deterministic_rule

    if item.dataset in {"truthfulqa", "triviaqa"}:
        return knowledge_blind_rule
    if item.dataset in {"gsm8k", "math"}:
        return reasoning_rule
    if item.dataset == "ambigqa":
        return ambiguous_rule
    return ambiguous_rule


def oracle_flat_answer_rule(item: BenchmarkItem):
    """Flat baseline rule: vary samples through observation index ``l``."""

    if item.label == 0:
        return deterministic_rule

    if item.dataset in {"truthfulqa", "triviaqa"}:

        def rule(theta: int, z: int, r: int, l: int) -> str:
            del theta, z, r
            return ["unknown", "possibly alpha", "possibly beta", "uncertain"][l % 4]

        return rule

    if item.dataset in {"gsm8k", "math"}:

        def rule(theta: int, z: int, r: int, l: int) -> str:
            del theta, z, r
            return ["42", "41", "43", "44"][l % 4]

        return rule

    def rule(theta: int, z: int, r: int, l: int) -> str:
        del theta, z, r
        return ["financial institution", "river edge", "data bank", "storage vault"][l % 4]

    return rule


def evaluate_item_oracle(
    item: BenchmarkItem,
    M: int,
    K: int,
    J: int,
    L: int,
) -> BenchmarkScores:
    """Evaluate one item with controlled oracle paths."""

    rule = oracle_answer_rule(item)
    nested_paths = make_controlled_paths(item.question, M, K, J, L, rule)
    nested = cluster_and_decompose(nested_paths, corrected=False)

    total_samples = M * K * J * L
    flat_rule = oracle_flat_answer_rule(item)
    flat_paths = make_controlled_paths(item.question, 1, 1, 1, total_samples, flat_rule)
    flat = cluster_and_decompose(flat_paths, corrected=False)

    sc_proxy = _self_consistency_from_paths(flat_paths)
    input_proxy = flat.u_z

    return _scores_from_decompositions(
        item,
        nested,
        flat.total_entropy,
        self_consistency=sc_proxy,
        input_perturbation=input_proxy,
        predicted_correct=item.label == 0,
    )


def evaluate_item_pipeline(
    item: BenchmarkItem,
    config: dict[str, Any],
) -> BenchmarkScores:
    """Evaluate one item through nested pipeline and flat baselines."""

    comparability = ComparabilitySettings.from_config(config)
    sampling = comparability.resolve_sampling(config)
    estimation = config["estimation"]
    M = int(sampling["M"])
    K = int(sampling["K"])
    J = int(sampling["J"])
    L = int(sampling["L"])
    fair = build_fair_comparison_plan(sampling, config, comparability)
    prompt_context = _prompt_context_for_item(item, comparability)
    baseline_kwargs = comparability.flat_baseline_kwargs(config) if comparability.enabled else {
        "max_tokens": int(sampling.get("max_answer_tokens", 192)),
        "base_seed": int(config.get("project", {}).get("seed", 42)),
    }

    generator = LLMGenerator.from_config(config)
    clusterer = SemanticNLIClusterer.from_config(config)
    engine = NestedMonteCarloEngine(generator=generator, clusterer=clusterer)

    persist_paths = bool(config.get("paths", {}).get("persist_nested_paths", True))
    persist_path = None
    if persist_paths:
        outputs_dir = resolve_path(config, "paths", "outputs_dir")
        nested_dir = outputs_dir / "nested_paths"
        nested_dir.mkdir(parents=True, exist_ok=True)
        persist_path = nested_dir / f"{item.id}.jsonl"

    fixed_reasoning_path = bool(sampling.get("fixed_reasoning_path", True))

    nested = engine.run_nested_sampling(
        question=item.question,
        M=M,
        K=K,
        J=J,
        L=L,
        persist_path=persist_path,
        persist_format="jsonl",
        show_progress=False,
        intent_temperature=fair.intent_temperature,
        reasoning_temperature=fair.answer_temperature,
        max_intent_tokens=int(sampling.get("max_intent_tokens", 96)),
        max_answer_tokens=int(sampling.get("max_answer_tokens", 192)),
        miller_madow=bool(estimation.get("miller_madow", True)),
        clip_negative=bool(estimation.get("clip_negative", False)),
        identity_tolerance=float(estimation.get("identity_tolerance", 1e-10)),
        prompt_context=prompt_context if comparability.enabled else None,
        skip_intent_rewrite=comparability.skip_intent_rewrite,
        fixed_reasoning_path=fixed_reasoning_path,
    )
    if nested.decomposition is None:
        raise RuntimeError(f"Missing decomposition for {item.id}")

    shared_kwargs = {
        **baseline_kwargs,
        "temperature": fair.answer_temperature,
        "prompt_context": prompt_context if comparability.enabled else None,
        "intent_temperature": fair.intent_temperature,
        "max_intent_tokens": int(sampling.get("max_intent_tokens", 96)),
    }
    baseline = semantic_entropy_baseline(
        question=item.question,
        num_samples=fair.se_sc_samples,
        generator=generator,
        clusterer=clusterer,
        corrected=bool(estimation.get("miller_madow", True)),
        **shared_kwargs,
    )
    sc = self_consistency_baseline(
        question=item.question,
        num_samples=fair.se_sc_samples,
        generator=generator,
        clusterer=clusterer,
        **shared_kwargs,
    )
    input_base = input_perturbation_baseline(
        question=item.question,
        num_samples=fair.ip_samples,
        generator=generator,
        clusterer=clusterer,
        corrected=bool(estimation.get("miller_madow", True)),
        **shared_kwargs,
    )

    predicted = majority_answer([path.answer_text for path in nested.paths])
    label, predicted_correct = resolve_error_label(item, predicted)

    scored_item = BenchmarkItem(
        id=item.id,
        dataset=item.dataset,
        question=item.question,
        label=label,
        reference_answer=item.reference_answer,
        category=item.category,
        notes=item.notes,
        acceptable_answers=item.acceptable_answers,
        label_source=item.label_source,
        split=item.split,
    )
    return _scores_from_decompositions(
        scored_item,
        nested.decomposition,
        baseline.semantic_entropy,
        self_consistency=sc.disagreement,
        input_perturbation=input_base.input_entropy,
        predicted_correct=predicted_correct,
    )


def resolve_error_label(item: BenchmarkItem, predicted_answer: str) -> tuple[int, bool]:
    """Return ``(error_label, predicted_correct)`` for AUROC evaluation."""

    if item.label_source == "correctness" or item.label < 0:
        acceptable = item.acceptable_answers if item.acceptable_answers else None
        correct = is_correct_answer(
            predicted_answer,
            item.reference_answer,
            item.dataset,
            acceptable,
        )
        return int(not correct), correct
    return item.label, item.label == 0


def _prompt_context_for_item(
    item: BenchmarkItem,
    comparability: ComparabilitySettings,
) -> PromptContext:
    """Build prompt context for comparable benchmark runs."""

    dataset = comparability.few_shot_dataset or item.dataset
    return PromptContext(
        dataset=dataset,
        few_shot_count=comparability.few_shot_count,
        use_fixed_cot=comparability.enabled,
        input_perturbation_mode=comparability.input_perturbation_mode,
    )


def _self_consistency_from_paths(paths: list[SampledPath]) -> float:
    answers = [path.answer_text for path in paths]
    if not answers:
        return 0.0
    from core.estimators import empirical_distribution

    distribution = empirical_distribution(answers)
    majority_share = max(distribution.values())
    return float(1.0 - majority_share)


def _scores_from_decompositions(
    item: BenchmarkItem,
    nested: UncertaintyDecomposition,
    semantic_entropy_value: float,
    self_consistency: float,
    input_perturbation: float,
    predicted_correct: bool,
) -> BenchmarkScores:
    """Pack nested decomposition and baselines into a score record."""

    return BenchmarkScores(
        item_id=item.id,
        dataset=item.dataset,
        label=item.label,
        predicted_correct=predicted_correct,
        nested_total=nested.total_entropy,
        semantic_entropy=semantic_entropy_value,
        self_consistency=self_consistency,
        input_perturbation=input_perturbation,
        u_theta=nested.u_theta,
        u_z=nested.u_z,
        u_r=nested.u_r,
        u_res=nested.u_res,
    )
