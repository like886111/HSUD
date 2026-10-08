"""Nested Monte Carlo engine for latent uncertainty decomposition."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from core.estimators import UncertaintyDecomposition, decompose_uncertainty
from core.llm_backend import GenerationRequest, LLMGenerator, SampledPath
from core.semantic_nli import ClusterResult, SemanticNLIClusterer
from utils.prompts import (
    PromptContext,
    build_answer_given_reasoning_prompt,
    build_cot_prompt,
    build_intent_prompt,
    build_reasoning_only_prompt,
)
from utils.storage import SampleFormat, save_sample_paths


@dataclass(slots=True)
class NestedSamplingResult:
    """Result bundle returned by nested Monte Carlo sampling."""

    question: str
    paths: list[SampledPath]
    cluster_result: ClusterResult
    sample_tree: dict[str, Any]
    decomposition: Optional[UncertaintyDecomposition] = None
    sample_tree_path: Optional[Path] = None


class NestedMonteCarloEngine:
    """Controller for nested samples over :math:`\\Theta`, :math:`Z`, :math:`R`, and :math:`L`."""

    def __init__(
        self,
        generator: Optional[LLMGenerator] = None,
        clusterer: Optional[SemanticNLIClusterer] = None,
        base_seed: int = 1009,
    ) -> None:
        self.generator = generator or LLMGenerator(dry_run=True)
        self.clusterer = clusterer or SemanticNLIClusterer(use_model=False)
        self.base_seed = base_seed

    def run_nested_sampling(
        self,
        question: str,
        M: int,
        K: int,
        J: int,
        L: int = 1,
        intent_temperature: float = 0.8,
        reasoning_temperature: float = 1.0,
        max_intent_tokens: int = 96,
        max_answer_tokens: int = 192,
        estimate_decomposition: bool = True,
        show_progress: bool = True,
        persist_path: str | Path | None = None,
        persist_format: SampleFormat = "jsonl",
        miller_madow: bool = True,
        clip_negative: bool = False,
        identity_tolerance: float = 1e-10,
        prompt_context: PromptContext | None = None,
        skip_intent_rewrite: bool = False,
        fixed_reasoning_path: bool = True,
    ) -> NestedSamplingResult:
        """Run nested Monte Carlo sampling for one question.

        Sampling order:
            1. Draw ``M`` persona / parameter configurations :math:`\\Theta`.
            2. For each ``\\Theta``, draw ``K`` intent rewrites :math:`Z`.
            3. For each ``(\\Theta, Z)``, draw ``J`` reasoning paths :math:`R`.
            4. For each ``(\\Theta, Z, R)``, draw ``L`` answers :math:`Y`.
            5. Cluster all answers globally into semantic states :math:`S``.

        When ``fixed_reasoning_path`` is True (default), each ``r_id`` first
        realizes one CoT text, then ``L`` answer replicates are drawn
        *conditioned on that fixed text*. When False (legacy G8 mode), each
        of the ``L`` draws jointly regenerates CoT+answer under a shared
        reasoning-branch prompt index; ``U_R`` then measures
        reasoning-branch / prompt-condition uncertainty rather than
        variability across fixed realized paths.
        """

        _validate_positive(M=M, K=K, J=J, L=L)
        if L < 2:
            raise ValueError(
                "Observation layer L must be >= 2 so that U_res = H(S|Theta,Z,R,C) "
                f"does not degenerate to zero; got L={L}."
            )

        paths: list[SampledPath] = []
        iterator = range(M)
        if show_progress:
            iterator = tqdm(iterator, desc="Nested MC theta samples")

        for theta_index in iterator:
            theta_id = f"theta_{theta_index}"
            for z_id in range(K):
                if skip_intent_rewrite and K == 1:
                    z_text = question
                else:
                    intent_prompt = build_intent_prompt(question, theta_index, z_id)
                    z_text = self.generator.generate(
                        intent_prompt,
                        temperature=intent_temperature,
                        max_tokens=max_intent_tokens,
                        seed=self._seed(theta_index, z_id, -1, 0),
                    )
                for r_id in range(J):
                    if fixed_reasoning_path and not (
                        prompt_context is not None and prompt_context.use_fixed_cot
                    ):
                        reasoning_prompt = build_reasoning_only_prompt(
                            z_text, theta_index, r_id
                        )
                        r_text = self.generator.generate(
                            reasoning_prompt,
                            temperature=reasoning_temperature,
                            max_tokens=max_answer_tokens,
                            seed=self._seed(theta_index, z_id, r_id, -1),
                        )
                        requests = [
                            GenerationRequest(
                                prompt=build_answer_given_reasoning_prompt(
                                    z_text, theta_index, r_text
                                ),
                                temperature=reasoning_temperature,
                                max_tokens=max_answer_tokens,
                                seed=self._seed(theta_index, z_id, r_id, l_id),
                            )
                            for l_id in range(L)
                        ]
                        generations = self.generator.generate_batch(requests)
                        for l_id, raw_generation in enumerate(generations):
                            answer_text = extract_final_answer(raw_generation)
                            paths.append(
                                SampledPath(
                                    theta_id=theta_id,
                                    z_id=z_id,
                                    r_id=r_id,
                                    l_id=l_id,
                                    question=question,
                                    z_text=z_text,
                                    r_text=r_text,
                                    answer_text=answer_text,
                                )
                            )
                    else:
                        # Legacy / flat-CoT mode: L joint CoT+answer draws share
                        # only a prompt-level reasoning-branch index.
                        cot_prompt = build_cot_prompt(
                            z_text,
                            theta_index,
                            r_id,
                            prompt_context=prompt_context,
                        )
                        requests = [
                            GenerationRequest(
                                prompt=cot_prompt,
                                temperature=reasoning_temperature,
                                max_tokens=max_answer_tokens,
                                seed=self._seed(theta_index, z_id, r_id, l_id),
                            )
                            for l_id in range(L)
                        ]
                        generations = self.generator.generate_batch(requests)
                        for l_id, raw_generation in enumerate(generations):
                            answer_text = extract_final_answer(raw_generation)
                            paths.append(
                                SampledPath(
                                    theta_id=theta_id,
                                    z_id=z_id,
                                    r_id=r_id,
                                    l_id=l_id,
                                    question=question,
                                    z_text=z_text,
                                    r_text=raw_generation,
                                    answer_text=answer_text,
                                )
                            )

        cluster_result = self.clusterer.cluster([path.answer_text for path in paths])
        for path, cluster_id in zip(paths, cluster_result.labels):
            path.s_cluster_id = cluster_id

        decomposition = None
        if estimate_decomposition:
            decomposition = decompose_uncertainty(
                paths,
                corrected=miller_madow,
                clip_negative=clip_negative,
                verify_identity=True,
                identity_tolerance=identity_tolerance,
            )

        sample_tree_path: Optional[Path] = None
        if persist_path is not None:
            sample_tree_path = save_sample_paths(
                paths,
                persist_path,
                fmt=persist_format,
            )

        sample_tree = build_sample_tree_dict(
            question=question,
            M=M,
            K=K,
            J=J,
            L=L,
            paths=paths,
            cluster_result=cluster_result,
            decomposition=decomposition,
            sample_tree_path=sample_tree_path,
            fixed_reasoning_path=fixed_reasoning_path,
        )

        return NestedSamplingResult(
            question=question,
            paths=paths,
            cluster_result=cluster_result,
            sample_tree=sample_tree,
            decomposition=decomposition,
            sample_tree_path=sample_tree_path,
        )

    def _seed(self, theta_index: int, z_id: int, r_id: int, l_id: int) -> int:
        """Derive a stable node-level random seed."""

        return (
            self.base_seed
            + theta_index * 1_000_000
            + z_id * 10_000
            + (r_id + 1) * 100
            + l_id
        )


def run_nested_sampling(
    question: str,
    M: int,
    K: int,
    J: int,
    L: int = 1,
    generator: Optional[LLMGenerator] = None,
    clusterer: Optional[SemanticNLIClusterer] = None,
    base_seed: int = 1009,
    **kwargs: Any,
) -> NestedSamplingResult:
    """Convenience wrapper for :meth:`NestedMonteCarloEngine.run_nested_sampling`."""

    engine = NestedMonteCarloEngine(
        generator=generator,
        clusterer=clusterer,
        base_seed=base_seed,
    )
    return engine.run_nested_sampling(
        question=question,
        M=M,
        K=K,
        J=J,
        L=L,
        **kwargs,
    )


def build_sample_tree_dict(
    question: str,
    M: int,
    K: int,
    J: int,
    L: int,
    paths: list[SampledPath],
    cluster_result: ClusterResult,
    decomposition: Optional[UncertaintyDecomposition],
    sample_tree_path: Optional[Path],
    fixed_reasoning_path: bool = True,
) -> dict[str, Any]:
    """Build a JSON-serializable nested sample tree dictionary."""

    return {
        "question": question,
        "budget": {"M": M, "K": K, "J": J, "L": L, "N_total": M * K * J * L},
        "fixed_reasoning_path": fixed_reasoning_path,
        "r_semantics": (
            "fixed_realized_reasoning_path"
            if fixed_reasoning_path
            else "reasoning_branch_prompt_condition"
        ),
        "paths": [asdict(path) for path in paths],
        "cluster_result": {
            "labels": cluster_result.labels,
            "probabilities": cluster_result.probabilities,
            "clusters": cluster_result.clusters,
        },
        "decomposition": asdict(decomposition) if decomposition is not None else None,
        "sample_tree_path": str(sample_tree_path) if sample_tree_path else None,
    }


def extract_final_answer(text: str) -> str:
    """Extract the final answer :math:`Y` from a model generation."""

    match = re.search(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _validate_positive(**values: int) -> None:
    """Validate positive integer Monte Carlo sample counts."""

    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
