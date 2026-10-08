"""Adaptive downstream router driven by decomposed uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.estimators import UncertaintyDecomposition


ACTION_MESSAGES: dict[str, str] = {
    "clarify": (
        "The question appears ambiguous. Please clarify the intended meaning "
        "before I commit to a final answer."
    ),
    "retrieve_or_refuse": (
        "My parameter/knowledge uncertainty is high. I should retrieve external "
        "evidence or abstain rather than guessing."
    ),
    "verify": (
        "My reasoning-path uncertainty is high. I will run additional "
        "verification or self-consistency checks before answering."
    ),
    "monitor_or_review": (
        "Uncertainty is dominated by residual semantic variation with no "
        "structured matched lever. Prefer monitoring, abstention, or human "
        "review rather than an automated generative fix."
    ),
    "escalate_or_abstain": (
        "No uncertainty source is clearly dominant. Prefer abstention, "
        "human review, or a total-risk threshold rather than a source-specific fix."
    ),
}

TERM_TO_ACTION: dict[str, str] = {
    "U_Z": "clarify",
    "U_Theta": "retrieve_or_refuse",
    "U_R": "verify",
    "U_res": "monitor_or_review",
    "uncertain": "escalate_or_abstain",
}

STRUCTURED_OR_ALL = ("U_Z", "U_Theta", "U_R", "U_res")


@dataclass(slots=True)
class RoutingDecision:
    """Routing action selected from uncertainty decomposition."""

    action: str
    dominant_term: str
    rationale: str
    user_message: str
    term_values: dict[str, float]
    dominance_margin: float = 0.0
    is_uncertain: bool = False


@dataclass(slots=True)
class RouterConfig:
    """Thresholds and priority for routing decisions."""

    min_term_value: float = 0.1
    min_dominance_margin: float = 0.0
    priority: tuple[str, ...] = ("U_Z", "U_Theta", "U_R", "U_res")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RouterConfig:
        """Build router config from YAML."""

        router_cfg = config.get("router", {})
        priority = tuple(
            router_cfg.get("priority", ["U_Z", "U_Theta", "U_R", "U_res"])
        )
        return cls(
            min_term_value=float(router_cfg.get("min_term_value", 0.1)),
            min_dominance_margin=float(router_cfg.get("min_dominance_margin", 0.0)),
            priority=priority,
        )


def select_dominant_term(
    terms: dict[str, float],
    *,
    min_term_value: float = 0.1,
    min_dominance_margin: float = 0.0,
    priority: tuple[str, ...] = ("U_Z", "U_Theta", "U_R", "U_res"),
) -> tuple[str, float, bool]:
    """Return ``(dominant_or_uncertain, margin, is_uncertain)``.

    ``margin`` is ``max - second_max`` over all four terms (0 if fewer than two).
    When the best eligible term does not beat the second-best by
    ``min_dominance_margin``, the label is ``uncertain``.
    """

    ordered_vals = sorted(terms.values(), reverse=True)
    margin = float(ordered_vals[0] - ordered_vals[1]) if len(ordered_vals) >= 2 else float(ordered_vals[0])

    eligible = {k: v for k, v in terms.items() if v >= min_term_value}
    if not eligible:
        # Legacy hard-argmax when no margin gate is requested.
        if min_dominance_margin <= 0.0:
            best_value = max(terms.values())
            selected = next(
                term
                for term in priority
                if abs(terms[term] - best_value) <= 1e-12
            )
            others = [v for k, v in terms.items() if k != selected]
            selected_margin = float(terms[selected] - max(others)) if others else float(terms[selected])
            return selected, selected_margin, False
        return "uncertain", margin, True

    best_value = max(eligible.values())
    # Priority breaks ties at the maximum.
    selected = next(
        term
        for term in priority
        if term in eligible and abs(eligible[term] - best_value) <= 1e-12
    )

    # Second-best among all terms except the selected one.
    others = [v for k, v in terms.items() if k != selected]
    second = max(others) if others else 0.0
    selected_margin = float(terms[selected] - second)
    if selected_margin < min_dominance_margin:
        return "uncertain", selected_margin, True
    return selected, selected_margin, False


def route(
    decomposition: UncertaintyDecomposition,
    router_config: RouterConfig | None = None,
) -> RoutingDecision:
    """Choose a downstream action from uncertainty components.

    Policy:
        - High ``U_Z`` -> ask clarifying question
        - High ``U_Theta`` -> refuse or retrieve external knowledge
        - High ``U_R`` -> add verification / self-consistency checks
        - High ``U_res`` -> monitor, abstain, or human review
        - Uncertain (small margin / all below floor) -> escalate or abstain
    """

    cfg = router_config or RouterConfig()
    terms = {
        "U_Theta": decomposition.u_theta,
        "U_Z": decomposition.u_z,
        "U_R": decomposition.u_r,
        "U_res": decomposition.u_res,
    }
    selected_term, margin, is_uncertain = select_dominant_term(
        terms,
        min_term_value=cfg.min_term_value,
        min_dominance_margin=cfg.min_dominance_margin,
        priority=cfg.priority,
    )
    action = TERM_TO_ACTION[selected_term]
    return RoutingDecision(
        action=action,
        dominant_term=selected_term,
        rationale=_rationale_for_term(selected_term),
        user_message=ACTION_MESSAGES[action],
        term_values=terms,
        dominance_margin=margin,
        is_uncertain=is_uncertain,
    )


def _rationale_for_term(term: str) -> str:
    """Return an internal rationale string for a dominant term."""

    mapping = {
        "U_Z": "Input ambiguity dominates; request intent clarification.",
        "U_Theta": "Parameter uncertainty dominates; retrieve evidence or abstain.",
        "U_R": "Reasoning instability dominates; run additional verification paths.",
        "U_res": "Residual semantic variation dominates; monitor, abstain, or review.",
        "uncertain": "No clear dominant source; escalate, abstain, or use total-risk only.",
    }
    return mapping[term]
