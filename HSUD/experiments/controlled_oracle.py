"""Controlled oracle sampling for synthetic sanity-check experiments."""

from __future__ import annotations

from typing import Callable

from core.estimators import UncertaintyDecomposition, decompose_uncertainty
from core.llm_backend import SampledPath
from core.semantic_nli import SemanticNLIClusterer

AnswerRule = Callable[[int, int, int, int], str]


def make_controlled_paths(
    question: str,
    M: int,
    K: int,
    J: int,
    L: int,
    answer_rule: AnswerRule,
) -> list[SampledPath]:
    """Construct controlled ``Theta -> Z -> R -> L -> S`` samples.

    Args:
        question: Observed input :math:`X`.
        M: Number of :math:`\\Theta` draws.
        K: Number of :math:`Z` draws per :math:`\\Theta`.
        J: Number of :math:`R` draws per ``(Theta, Z)``.
        L: Number of observation replicates per ``(Theta, Z, R)``.
        answer_rule: Callable ``(theta, z, r, l) -> answer_text``.

    Returns:
        Leaf paths without semantic cluster IDs assigned.
    """

    if L < 2:
        raise ValueError("Controlled oracle requires L >= 2 for non-degenerate U_res.")

    paths: list[SampledPath] = []
    for theta in range(M):
        for z in range(K):
            for r in range(J):
                for l in range(L):
                    answer = answer_rule(theta, z, r, l)
                    paths.append(
                        SampledPath(
                            theta_id=f"theta_{theta}",
                            z_id=z,
                            r_id=r,
                            l_id=l,
                            question=question,
                            z_text=f"controlled intent {z}",
                            r_text=f"controlled rationale {r}\nFinal Answer: {answer}",
                            answer_text=answer,
                        )
                    )
    return paths


def ambiguous_rule(theta: int, z: int, r: int, l: int) -> str:
    """Category A: :math:`Z` controls semantic state."""

    del theta, r, l
    return ["financial institution", "river edge", "data bank", "storage vault"][z % 4]


def knowledge_blind_rule(theta: int, z: int, r: int, l: int) -> str:
    """Category B: :math:`\\Theta` controls semantic state."""

    del z, r, l
    return ["unknown", "possibly alpha", "possibly beta", "uncertain"][theta % 4]


def reasoning_rule(theta: int, z: int, r: int, l: int) -> str:
    """Category C: :math:`R` controls semantic state."""

    del theta, z, l
    return ["42", "41", "43", "44"][r % 4]


def residual_rule(theta: int, z: int, r: int, l: int) -> str:
    """Category D: observation layer :math:`L` controls semantic paraphrases."""

    del theta, z, r
    return ["indeed", "correct", "affirmative", "that is right"][l % 4]


CATEGORY_RULES: dict[str, AnswerRule] = {
    "A": ambiguous_rule,
    "B": knowledge_blind_rule,
    "C": reasoning_rule,
    "D": residual_rule,
}


def cluster_and_decompose(
    paths: list[SampledPath],
    corrected: bool = False,
) -> UncertaintyDecomposition:
    """Assign global semantic clusters and compute decomposition."""

    clusterer = SemanticNLIClusterer(use_model=False)
    clusters = clusterer.cluster([path.answer_text for path in paths])
    for path, label in zip(paths, clusters.labels):
        path.s_cluster_id = label
    return decompose_uncertainty(
        paths,
        corrected=corrected,
        clip_negative=False,
        verify_identity=True,
    )


def dominant_term(decomposition: UncertaintyDecomposition) -> str:
    """Return the largest uncertainty component key."""

    terms = {
        "U_Theta": decomposition.u_theta,
        "U_Z": decomposition.u_z,
        "U_R": decomposition.u_r,
        "U_res": decomposition.u_res,
    }
    return max(terms, key=terms.get)


def decomposition_to_dict(decomposition: UncertaintyDecomposition) -> dict[str, float]:
    """Convert decomposition dataclass to a flat metrics dictionary."""

    return {
        "H(S|X)": decomposition.total_entropy,
        "U_Theta": decomposition.u_theta,
        "U_Z": decomposition.u_z,
        "U_R": decomposition.u_r,
        "U_res": decomposition.u_res,
    }
