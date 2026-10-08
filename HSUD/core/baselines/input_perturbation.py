"""Input-perturbation uncertainty baseline (Hou et al.-style)."""

from __future__ import annotations

from dataclasses import dataclass

from core.baselines.semantic_entropy import _extract_final_answer
from core.estimators import semantic_entropy
from core.llm_backend import LLMGenerator, SampledPath
from core.semantic_nli import SemanticNLIClusterer
from utils.prompts import PromptContext, build_cot_prompt, build_input_perturbation_prompt, build_flat_benchmark_cot_prompt


@dataclass(slots=True)
class InputPerturbationResult:
    """Semantic entropy over answers from paraphrased / rewritten inputs."""

    question: str
    input_entropy: float
    num_samples: int
    num_clusters: int


def input_perturbation_baseline(
    question: str,
    num_samples: int,
    generator: LLMGenerator,
    clusterer: SemanticNLIClusterer,
    corrected: bool = True,
    **kwargs: object,
) -> InputPerturbationResult:
    """Perturb the input with intent rewrites, then measure answer entropy.

  This approximates input-sensitivity baselines that probe ambiguity via
  paraphrases / clarifications rather than nested parameter draws.
    """

    paths: list[SampledPath] = []
    base_seed = int(kwargs.get("base_seed", 4042))
    prompt_context = kwargs.get("prompt_context") or PromptContext()
    for z_id in range(num_samples):
        perturb_prompt = build_input_perturbation_prompt(question, z_id, prompt_context)
        z_text = generator.generate(
            perturb_prompt,
            temperature=float(kwargs.get("intent_temperature", 0.8)),
            max_tokens=int(kwargs.get("max_intent_tokens", 96)),
            seed=base_seed + z_id,
        )
        if prompt_context.use_fixed_cot:
            cot_prompt = build_flat_benchmark_cot_prompt(z_text, z_id, prompt_context)
        else:
            cot_prompt = build_cot_prompt(z_text, theta_index=0, r_id=0)
        answer_text = generator.generate(
            cot_prompt,
            temperature=float(kwargs.get("temperature", 1.0)),
            max_tokens=int(kwargs.get("max_tokens", 192)),
            seed=base_seed + 10_000 + z_id,
        )
        paths.append(
            SampledPath(
                theta_id="input_perturb",
                z_id=z_id,
                r_id=0,
                l_id=0,
                question=question,
                z_text=z_text,
                r_text=answer_text,
                answer_text=_extract_final_answer(answer_text),
            )
        )

    cluster_result = clusterer.cluster([path.answer_text for path in paths])
    for path, label in zip(paths, cluster_result.labels):
        path.s_cluster_id = label

    entropy_value = semantic_entropy(paths, corrected=corrected)
    clusters = {path.s_cluster_id for path in paths}
    return InputPerturbationResult(
        question=question,
        input_entropy=float(entropy_value),
        num_samples=num_samples,
        num_clusters=len(clusters),
    )
