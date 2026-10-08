"""Self-consistency disagreement baseline."""

from __future__ import annotations

from dataclasses import dataclass

from core.baselines.semantic_entropy import flat_sample_paths, _extract_final_answer
from core.estimators import empirical_distribution
from core.llm_backend import LLMGenerator
from core.semantic_nli import SemanticNLIClusterer


@dataclass(slots=True)
class SelfConsistencyResult:
    """Disagreement score from multi-path majority voting."""

    question: str
    disagreement: float
    majority_answer: str
    num_samples: int
    majority_vote_share: float


def self_consistency_baseline(
    question: str,
    num_samples: int,
    generator: LLMGenerator,
    clusterer: SemanticNLIClusterer,
    **kwargs: object,
) -> SelfConsistencyResult:
    """Compute 1 - majority vote share over flat CoT samples.

    Higher disagreement indicates lower path consistency (common SC uncertainty proxy).
    """

    paths = flat_sample_paths(
        question=question,
        num_samples=num_samples,
        generator=generator,
        clusterer=clusterer,
        temperature=float(kwargs.get("temperature", 1.0)),
        max_tokens=int(kwargs.get("max_tokens", 192)),
        base_seed=int(kwargs.get("base_seed", 3031)),
        prompt_context=kwargs.get("prompt_context"),
    )
    answers = [_extract_final_answer(path.answer_text) for path in paths]
    distribution = empirical_distribution(answers)
    majority_answer = max(distribution, key=distribution.get)
    majority_share = float(distribution[majority_answer])
    disagreement = float(1.0 - majority_share)
    return SelfConsistencyResult(
        question=question,
        disagreement=disagreement,
        majority_answer=majority_answer,
        num_samples=num_samples,
        majority_vote_share=majority_share,
    )
