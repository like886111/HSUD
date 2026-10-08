"""Semantic Entropy baseline (Kuhn et al., 2023).

The baseline draws ``N`` i.i.d. answers, clusters them into semantic classes,
and returns :math:`H(S|X)` without nested :math:`\\Theta/Z/R` structure.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.estimators import semantic_entropy
from core.llm_backend import LLMGenerator, SampledPath
from core.semantic_nli import SemanticNLIClusterer
from utils.prompts import PromptContext, build_cot_prompt, build_flat_benchmark_cot_prompt


@dataclass(slots=True)
class SemanticEntropyResult:
    """Output of the flat semantic-entropy baseline."""

    question: str
    semantic_entropy: float
    num_samples: int
    num_clusters: int


def flat_sample_paths(
    question: str,
    num_samples: int,
    generator: LLMGenerator,
    clusterer: SemanticNLIClusterer,
    temperature: float = 1.0,
    max_tokens: int = 192,
    base_seed: int = 2025,
    prompt_context: PromptContext | None = None,
) -> list[SampledPath]:
    """Draw ``N`` flat answer samples without nested latent structure."""

    paths: list[SampledPath] = []
    for idx in range(num_samples):
        if prompt_context is not None and prompt_context.use_fixed_cot:
            prompt = build_flat_benchmark_cot_prompt(question, idx, prompt_context)
        else:
            prompt = build_cot_prompt(question, theta_index=0, r_id=idx, prompt_context=prompt_context)
        text = generator.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=base_seed + idx,
        )
        answer = _extract_final_answer(text)
        paths.append(
            SampledPath(
                theta_id="flat",
                z_id=0,
                r_id=idx,
                l_id=0,
                question=question,
                z_text=question,
                r_text=text,
                answer_text=answer,
            )
        )

    cluster_result = clusterer.cluster([path.answer_text for path in paths])
    for path, label in zip(paths, cluster_result.labels):
        path.s_cluster_id = label
    return paths


def semantic_entropy_baseline(
    question: str,
    num_samples: int,
    generator: LLMGenerator,
    clusterer: SemanticNLIClusterer,
    corrected: bool = True,
    **kwargs: object,
) -> SemanticEntropyResult:
    """Compute flat semantic entropy :math:`H(S|X)` for one question."""

    paths = flat_sample_paths(
        question=question,
        num_samples=num_samples,
        generator=generator,
        clusterer=clusterer,
        temperature=float(kwargs.get("temperature", 1.0)),
        max_tokens=int(kwargs.get("max_tokens", 192)),
        base_seed=int(kwargs.get("base_seed", 2025)),
        prompt_context=kwargs.get("prompt_context"),
    )
    entropy_value = semantic_entropy(paths, corrected=corrected)
    clusters = {path.s_cluster_id for path in paths}
    return SemanticEntropyResult(
        question=question,
        semantic_entropy=float(entropy_value),
        num_samples=num_samples,
        num_clusters=len(clusters),
    )


def _extract_final_answer(text: str) -> str:
    """Extract final answer line from a CoT generation."""

    import re

    match = re.search(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()
