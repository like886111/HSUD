"""Semantic equivalence clustering with bidirectional NLI entailment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-large"


@dataclass(slots=True)
class ClusterResult:
    """Output of global semantic clustering."""

    labels: list[int]
    probabilities: dict[int, float]
    clusters: dict[int, list[int]]
    similarity_matrix: np.ndarray


class UnionFind:
    """Disjoint-set union for global connected-component clustering."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node: int) -> int:
        """Find set representative with path compression."""

        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: int, right: int) -> None:
        """Merge two sets by rank."""

        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

    def connected_components(self) -> list[int]:
        """Return compact component labels for every node."""

        roots = [self.find(node) for node in range(len(self.parent))]
        remap: dict[int, int] = {}
        labels: list[int] = []
        next_label = 0
        for root in roots:
            if root not in remap:
                remap[root] = next_label
                next_label += 1
            labels.append(remap[root])
        return labels


class SemanticNLIClusterer:
    """Cluster answers using bidirectional entailment and global union-find."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        entailment_threshold: float = 0.65,
        use_model: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.use_model = use_model
        self.device = device
        self._model = None
        self._entailment_index: Optional[int] = None

        if use_model:
            self._load_model()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SemanticNLIClusterer:
        """Build a clusterer from the project YAML config."""

        semantic_cfg = config["semantic"]
        return cls(
            model_name=semantic_cfg.get("model_name", DEFAULT_NLI_MODEL),
            entailment_threshold=float(semantic_cfg.get("entailment_threshold", 0.65)),
            use_model=bool(semantic_cfg.get("use_model", False)),
            device=semantic_cfg.get("device"),
        )

    def _load_model(self) -> None:
        """Load the configured cross-encoder NLI model."""

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Semantic NLI clustering requires sentence-transformers."
            ) from exc

        if self.device is not None and str(self.device).startswith("cuda"):
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "semantic.device=cuda requires torch; install a CUDA build of PyTorch."
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "semantic.device is set to "
                    f"{self.device!r} but torch.cuda.is_available() is False. "
                    "Fix the NVIDIA driver / CUDA torch install, or set "
                    "semantic.device: cpu for a CPU NLI run."
                )

        kwargs = {"device": self.device} if self.device is not None else {}
        self._model = CrossEncoder(self.model_name, **kwargs)
        labels = getattr(self._model.model.config, "id2label", {})
        inverse = {str(value).lower(): int(key) for key, value in labels.items()}
        self._entailment_index = inverse.get("entailment")

    def cluster(self, answers: Sequence[str]) -> ClusterResult:
        """Cluster all answers in one global semantic space."""

        n_answers = len(answers)
        if n_answers == 0:
            return ClusterResult([], {}, {}, np.zeros((0, 0), dtype=float))

        similarity = self.build_similarity_matrix(answers)
        labels = connected_components_union_find(similarity)
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)
        probabilities = {
            cluster_id: len(indices) / n_answers
            for cluster_id, indices in clusters.items()
        }
        return ClusterResult(labels, probabilities, clusters, similarity)

    def build_similarity_matrix(self, answers: Sequence[str]) -> np.ndarray:
        """Build bidirectional semantic equivalence matrix."""

        n_answers = len(answers)
        matrix = np.eye(n_answers, dtype=float)
        for i in range(n_answers):
            for j in range(i + 1, n_answers):
                equivalent = self.are_equivalent(answers[i], answers[j])
                matrix[i, j] = matrix[j, i] = float(equivalent)
        return matrix

    def are_equivalent(self, left: str, right: str) -> bool:
        """Return whether two answers are bidirectionally entailing."""

        if not self.use_model:
            return normalize_answer(left) == normalize_answer(right)

        forward = self.entailment_probability(left, right)
        backward = self.entailment_probability(right, left)
        return (
            forward >= self.entailment_threshold
            and backward >= self.entailment_threshold
        )

    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        """Estimate ``P(entailment | premise, hypothesis)``."""

        if self._model is None:
            raise RuntimeError("NLI model is not loaded. Set use_model=True.")

        logits = np.asarray(self._model.predict([(premise, hypothesis)]))[0]
        if logits.ndim == 0:
            return float(logits)
        probs = softmax(logits)
        entailment_idx = self._entailment_index
        if entailment_idx is None:
            entailment_idx = int(np.argmax(logits))
        return float(probs[entailment_idx])


def normalize_answer(text: str) -> str:
    """Normalize answer strings for deterministic fallback clustering."""

    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9.]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def softmax(values: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax vector."""

    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def connected_components_union_find(adjacency: np.ndarray) -> list[int]:
    """Find connected components via union-find on an equivalence graph."""

    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return []

    dsu = UnionFind(n_nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > 0:
                dsu.union(i, j)
    return dsu.connected_components()


def connected_components(adjacency: np.ndarray) -> list[int]:
    """Backward-compatible alias for union-find connected components."""

    return connected_components_union_find(adjacency)
