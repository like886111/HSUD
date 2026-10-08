"""Information-theoretic estimators for semantic uncertainty decomposition.

This module is pure math: it must not import or call any LLM / API code.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np
from scipy.special import xlogy

from core.llm_backend import SampledPath


LATENT_TO_FIELD = {
    "Theta": "U_Theta",
    "Z": "U_Z",
    "R": "U_R",
}


@dataclass(slots=True)
class UncertaintyDecomposition:
    """Four-term decomposition of total semantic entropy.

    Attributes:
        total_entropy: Estimated :math:`H(S | X)`.
        u_theta: :math:`I(S; \\Theta | X)`.
        u_z: :math:`I(S; Z | \\Theta, X)`.
        u_r: :math:`I(S; R | \\Theta, Z, X)`.
        u_res: :math:`H(S | \\Theta, Z, R, X)`.
        entropy_by_level: Intermediate conditional entropies from the chain rule.
        identity_residual: ``|(U_\\Theta + U_Z + U_R + U_res) - H(S|X)|``.
    """

    total_entropy: float
    u_theta: float
    u_z: float
    u_r: float
    u_res: float
    entropy_by_level: dict[str, float]
    identity_residual: float = 0.0


def entropy(probabilities: Sequence[float]) -> float:
    """Compute Shannon entropy :math:`H(P) = -\\sum_i p_i \\log p_i` in nats.

    Args:
        probabilities: Probability vector. Zero entries contribute ``0 log 0``.

    Returns:
        Entropy in nats.
    """

    p = np.asarray(probabilities, dtype=float).ravel()
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum()
    if p.size == 0:
        return 0.0
    return float(-np.sum(xlogy(p, p)))


def empirical_distribution(labels: Sequence[Hashable]) -> dict[Hashable, float]:
    """Estimate an empirical categorical distribution."""

    n_samples = len(labels)
    if n_samples == 0:
        return {}
    counts = Counter(labels)
    return {label: count / n_samples for label, count in counts.items()}


def miller_madow_entropy(labels: Sequence[Hashable]) -> float:
    """Compute Miller-Madow corrected empirical entropy.

    :math:`H_{corrected} = H_{empirical} + (K_{observed} - 1) / (2N)`.
    """

    n_samples = len(labels)
    if n_samples == 0:
        return 0.0
    distribution = empirical_distribution(labels)
    empirical = entropy(list(distribution.values()))
    k_observed = len(distribution)
    return float(empirical + (k_observed - 1) / (2 * n_samples))


def conditional_entropy(
    paths: Sequence[SampledPath],
    condition_key: Callable[[SampledPath], Hashable],
    corrected: bool = True,
) -> float:
    """Estimate :math:`H(S | C, X)` via grouped plug-in (+ optional Miller-Madow)."""

    if not paths:
        return 0.0

    groups: dict[Hashable, list[int]] = defaultdict(list)
    for path in paths:
        if path.s_cluster_id is None:
            raise ValueError("All paths must have s_cluster_id before estimation.")
        groups[condition_key(path)].append(path.s_cluster_id)

    n_total = len(paths)
    total = 0.0
    for labels in groups.values():
        weight = len(labels) / n_total
        group_entropy = miller_madow_entropy(labels) if corrected else entropy(
            list(empirical_distribution(labels).values())
        )
        total += weight * group_entropy
    return float(total)


def semantic_entropy(paths: Sequence[SampledPath], corrected: bool = True) -> float:
    """Estimate total semantic entropy :math:`H(S | X)`."""

    labels = _extract_cluster_labels(paths)
    if corrected:
        return miller_madow_entropy(labels)
    return entropy(list(empirical_distribution(labels).values()))


def decompose_uncertainty(
    paths: Sequence[SampledPath],
    corrected: bool = True,
    clip_negative: bool = False,
    verify_identity: bool = True,
    identity_tolerance: float = 1e-10,
) -> UncertaintyDecomposition:
    """Compute the four-term uncertainty decomposition.

    Chain rule:
    :math:`H(S|X) = I(S;\\Theta|X) + I(S;Z|\\Theta,X)
    + I(S;R|\\Theta,Z,X) + H(S|\\Theta,Z,R,X)`.
    """

    h_s = semantic_entropy(paths, corrected=corrected)
    h_s_given_theta = conditional_entropy(
        paths, lambda path: path.theta_id, corrected=corrected
    )
    h_s_given_theta_z = conditional_entropy(
        paths, lambda path: (path.theta_id, path.z_id), corrected=corrected
    )
    h_s_given_theta_z_r = conditional_entropy(
        paths, lambda path: (path.theta_id, path.z_id, path.r_id), corrected=corrected
    )

    u_theta = h_s - h_s_given_theta
    u_z = h_s_given_theta - h_s_given_theta_z
    u_r = h_s_given_theta_z - h_s_given_theta_z_r
    u_res = h_s_given_theta_z_r

    raw_sum = u_theta + u_z + u_r + u_res
    identity_residual = abs(raw_sum - h_s)

    if clip_negative:
        u_theta = max(0.0, u_theta)
        u_z = max(0.0, u_z)
        u_r = max(0.0, u_r)
        u_res = max(0.0, u_res)

    if verify_identity and not clip_negative:
        assert identity_residual < identity_tolerance, (
            f"Additive identity violated: residual={identity_residual}, "
            f"H(S|X)={h_s}, sum={raw_sum}"
        )

    return UncertaintyDecomposition(
        total_entropy=float(h_s),
        u_theta=float(u_theta),
        u_z=float(u_z),
        u_r=float(u_r),
        u_res=float(u_res),
        entropy_by_level={
            "H(S|X)": float(h_s),
            "H(S|Theta,X)": float(h_s_given_theta),
            "H(S|Theta,Z,X)": float(h_s_given_theta_z),
            "H(S|Theta,Z,R,X)": float(h_s_given_theta_z_r),
        },
        identity_residual=float(identity_residual),
    )


def decompose_uncertainty_ordered(
    paths: Sequence[SampledPath],
    order: Sequence[str] = ("Theta", "Z", "R"),
    corrected: bool = True,
    clip_negative: bool = False,
    verify_identity: bool = True,
    identity_tolerance: float = 1e-10,
) -> UncertaintyDecomposition:
    """Compute a chain-rule decomposition under an arbitrary latent order.

    The default order ``("Theta", "Z", "R")`` matches :func:`decompose_uncertainty`.
    This helper is used for order-dependence ablations and Shapley-style
    symmetrization.
    """

    normalized_order = tuple(order)
    if sorted(normalized_order) != ["R", "Theta", "Z"]:
        raise ValueError(
            "order must contain each latent exactly once: Theta, Z, R; "
            f"got {normalized_order}"
        )

    h_s = semantic_entropy(paths, corrected=corrected)
    previous_entropy = h_s
    contributions = {"U_Theta": 0.0, "U_Z": 0.0, "U_R": 0.0}
    entropy_by_level: dict[str, float] = {"H(S|X)": float(h_s)}

    prefix: list[str] = []
    for latent in normalized_order:
        prefix.append(latent)
        current_entropy = conditional_entropy(
            paths,
            lambda path, active=tuple(prefix): tuple(
                _latent_value(path, item) for item in active
            ),
            corrected=corrected,
        )
        contribution = previous_entropy - current_entropy
        contributions[LATENT_TO_FIELD[latent]] = float(contribution)
        entropy_by_level[f"H(S|{','.join(prefix)},X)"] = float(current_entropy)
        previous_entropy = current_entropy

    u_theta = contributions["U_Theta"]
    u_z = contributions["U_Z"]
    u_r = contributions["U_R"]
    u_res = previous_entropy
    raw_sum = u_theta + u_z + u_r + u_res
    identity_residual = abs(raw_sum - h_s)

    if clip_negative:
        u_theta = max(0.0, u_theta)
        u_z = max(0.0, u_z)
        u_r = max(0.0, u_r)
        u_res = max(0.0, u_res)

    if verify_identity and not clip_negative:
        assert identity_residual < identity_tolerance, (
            f"Ordered additive identity violated: residual={identity_residual}, "
            f"order={normalized_order}, H(S|X)={h_s}, sum={raw_sum}"
        )

    return UncertaintyDecomposition(
        total_entropy=float(h_s),
        u_theta=float(u_theta),
        u_z=float(u_z),
        u_r=float(u_r),
        u_res=float(u_res),
        entropy_by_level=entropy_by_level,
        identity_residual=float(identity_residual),
    )


def shapley_decomposition(
    paths: Sequence[SampledPath],
    corrected: bool = True,
    verify_identity: bool = True,
    identity_tolerance: float = 1e-10,
) -> UncertaintyDecomposition:
    """Average latent contributions across all six orders.

    This provides a computational reference for checking whether the default
    sequence ``Theta -> Z -> R`` changes dominant-risk decisions.
    """

    totals = {"U_Theta": 0.0, "U_Z": 0.0, "U_R": 0.0}
    order_count = 0
    residual_value = 0.0
    total_entropy = 0.0

    for order in permutations(("Theta", "Z", "R")):
        decomposition = decompose_uncertainty_ordered(
            paths,
            order=order,
            corrected=corrected,
            clip_negative=False,
            verify_identity=verify_identity,
            identity_tolerance=identity_tolerance,
        )
        totals["U_Theta"] += decomposition.u_theta
        totals["U_Z"] += decomposition.u_z
        totals["U_R"] += decomposition.u_r
        residual_value += decomposition.u_res
        total_entropy += decomposition.total_entropy
        order_count += 1

    u_theta = totals["U_Theta"] / order_count
    u_z = totals["U_Z"] / order_count
    u_r = totals["U_R"] / order_count
    u_res = residual_value / order_count
    h_s = total_entropy / order_count
    raw_sum = u_theta + u_z + u_r + u_res
    identity_residual = abs(raw_sum - h_s)

    if verify_identity:
        assert identity_residual < identity_tolerance, (
            f"Shapley additive identity violated: residual={identity_residual}, "
            f"H(S|X)={h_s}, sum={raw_sum}"
        )

    return UncertaintyDecomposition(
        total_entropy=float(h_s),
        u_theta=float(u_theta),
        u_z=float(u_z),
        u_r=float(u_r),
        u_res=float(u_res),
        entropy_by_level={
            "H(S|X)": float(h_s),
            "shapley_orders": float(order_count),
            "H(S|Theta,Z,R,X)": float(u_res),
        },
        identity_residual=float(identity_residual),
    )


def cluster_tensor_to_paths(
    cluster_ids: np.ndarray,
    question: str = "",
) -> list[SampledPath]:
    """Convert a ``[M, K, J, L]`` cluster tensor into ``SampledPath`` records.

    Args:
        cluster_ids: Integer semantic cluster IDs with shape ``(M, K, J, L)``.
        question: Observed question :math:`X` attached to every leaf.

    Returns:
        Flat list of leaf paths preserving nested indices.
    """

    if cluster_ids.ndim != 4:
        raise ValueError(
            f"cluster_ids must have shape [M, K, J, L], got {cluster_ids.shape}"
        )

    paths: list[SampledPath] = []
    m_size, k_size, j_size, l_size = cluster_ids.shape
    for m in range(m_size):
        for k in range(k_size):
            for j in range(j_size):
                for l in range(l_size):
                    cluster = int(cluster_ids[m, k, j, l])
                    paths.append(
                        SampledPath(
                            theta_id=f"theta_{m}",
                            z_id=k,
                            r_id=j,
                            l_id=l,
                            question=question,
                            z_text="",
                            r_text="",
                            answer_text=f"s_{cluster}",
                            s_cluster_id=cluster,
                        )
                    )
    return paths


def decompose_from_tensor(
    cluster_ids: np.ndarray,
    question: str = "",
    corrected: bool = True,
    clip_negative: bool = False,
    verify_identity: bool = True,
    identity_tolerance: float = 1e-10,
) -> UncertaintyDecomposition:
    """Decompose uncertainty directly from a ``[M, K, J, L]`` cluster tensor.

    Args:
        cluster_ids: Semantic cluster IDs for each nested leaf observation.
        question: Optional question string stored on generated paths.
        corrected: Whether to apply Miller-Madow correction.
        clip_negative: Whether to clip negative information terms to zero.
        verify_identity: Whether to assert the additive identity at runtime.
        identity_tolerance: Maximum allowed identity residual when verifying.

    Returns:
        Four-term decomposition in nats.
    """

    paths = cluster_tensor_to_paths(cluster_ids, question=question)
    return decompose_uncertainty(
        paths,
        corrected=corrected,
        clip_negative=clip_negative,
        verify_identity=verify_identity,
        identity_tolerance=identity_tolerance,
    )


def decompose_from_records(
    records: Iterable[dict[str, object]],
    corrected: bool = True,
    clip_negative: bool = False,
    verify_identity: bool = True,
    identity_tolerance: float = 1e-10,
) -> UncertaintyDecomposition:
    """Compute decomposition from dictionary-like records."""

    paths = [
        SampledPath(
            theta_id=str(record["theta_id"]),
            z_id=int(record["z_id"]),
            r_id=int(record["r_id"]),
            l_id=int(record.get("l_id", 0)),
            question=str(record.get("question", "")),
            z_text=str(record.get("z_text", "")),
            r_text=str(record.get("r_text", "")),
            answer_text=str(record.get("answer_text", "")),
            s_cluster_id=int(record["s_cluster_id"]),
        )
        for record in records
    ]
    return decompose_uncertainty(
        paths,
        corrected=corrected,
        clip_negative=clip_negative,
        verify_identity=verify_identity,
        identity_tolerance=identity_tolerance,
    )


def _extract_cluster_labels(paths: Sequence[SampledPath]) -> list[int]:
    """Extract semantic labels while validating cluster assignment."""

    labels: list[int] = []
    for path in paths:
        if path.s_cluster_id is None:
            raise ValueError("All paths must have s_cluster_id before estimation.")
        labels.append(path.s_cluster_id)
    return labels


def _latent_value(path: SampledPath, latent: str) -> Hashable:
    if latent == "Theta":
        return path.theta_id
    if latent == "Z":
        return path.z_id
    if latent == "R":
        return path.r_id
    raise ValueError(f"Unsupported latent: {latent}")
