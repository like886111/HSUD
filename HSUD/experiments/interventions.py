"""Text perturbation operators for Phase 4 intervention experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from experiments.controlled_oracle import (
    AnswerRule,
    ambiguous_rule,
    knowledge_blind_rule,
    reasoning_rule,
)


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """Specification for one paired intervention experiment."""

    name: str
    target_term: str
    perturb: Callable[[str], str]
    answer_rule: AnswerRule
    description: str


# Pre-registered real-LLM ambiguity operator.
# underspecify_v2 (2026-09-04): drop clear ask; forbid listing alternatives.
# underspecify_v3 (2026-09-04): keep underspecification, but do NOT force a
# single short answer / forbid explanation — that collapsed nested entropy to 0
# on gsm8k_test_00588 in the v2 smoke contrast.
AMBIGUITY_OPERATOR_VERSION = "underspecify_v3_2026-09-04"


def add_ambiguity(question: str) -> str:
    """Inject genuine input underspecification into a previously clear question.

    Pre-registered operator ``underspecify_v3_2026-09-04``:
    1. Drop or blur a clear trailing ask when present.
    2. Replace it with a request that admits multiple valid referents/quantities.
    3. Avoid forcing a single short answer or forbidding explanation (v2 over-
       constrained generations and zeroed all uncertainty on some items).
    """

    stripped = question.strip().rstrip("?.!").strip()
    body = _drop_clear_ask(stripped)
    return (
        f"{body} "
        "What should be reported as the answer? "
        "The request is underspecified: several quantities or referents appear, "
        "and which one is required is not uniquely identified, so more than one "
        "reading is valid. "
        "Choose one valid reading and answer it; you may briefly show work, "
        "but do not enumerate every alternative reading."
    )


def _drop_clear_ask(text: str) -> str:
    """Remove a trailing clear interrogative/instruction clause when possible."""

    pattern = re.compile(
        r"(?is)(?<=[.!?\s])("
        r"(?:how many|how much|what is|what are|what was|what were|what will|what would|"
        r"who|which|when|where|find|calculate|compute)\b.+)$"
    )
    match = pattern.search(text)
    if match and match.start(1) >= 24:
        body = text[: match.start(1)].strip().rstrip(".,;")
        return f"{body}."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) >= 2 and sum(len(s) for s in sentences[:-1]) >= 24:
        return " ".join(sentences[:-1]).strip()
    return text


def replace_with_obscure_knowledge(question: str) -> str:
    """Replace a clear factual query with an obscure fictional-knowledge query."""

    stripped = question.strip().rstrip(".")
    token = _extract_anchor_phrase(stripped).replace(" ", "-").lower()
    entity = f"XQ-{abs(hash(token)) % 900 + 100} artifact"
    return (
        f"What is the verified property of the fictional {entity}, "
        f"instead of answering the original clear question: {stripped}?"
    )


def elongate_reasoning(question: str) -> str:
    """Append multi-step reasoning instructions to a clear question."""

    stripped = question.strip().rstrip(".")
    return (
        f"{stripped}. Before giving the final answer, perform at least five explicit "
        "intermediate calculations or logical steps and consider alternate solution paths."
    )


def _extract_anchor_phrase(question: str) -> str:
    """Extract a short anchor phrase from a clear question."""

    match = re.search(r"state\s+(.+?)\s+clearly", question, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    words = question.split()
    if len(words) >= 4:
        return " ".join(words[:4])
    return question


def deterministic_rule(theta: int, z: int, r: int, l: int) -> str:
    """Baseline rule with zero uncertainty across all latent layers."""

    del theta, z, r, l
    return "canonical answer"


INTERVENTIONS: tuple[InterventionSpec, ...] = (
    InterventionSpec(
        name="add_ambiguity",
        target_term="U_Z",
        perturb=add_ambiguity,
        answer_rule=ambiguous_rule,
        description=(
            "Input underspecification (underspecify_v3) should increase U_Z "
            "by admitting multiple valid readings of the ask."
        ),
    ),
    InterventionSpec(
        name="replace_with_obscure_knowledge",
        target_term="U_Theta",
        perturb=replace_with_obscure_knowledge,
        answer_rule=knowledge_blind_rule,
        description="Obscure-knowledge replacement should increase U_Theta.",
    ),
    InterventionSpec(
        name="elongate_reasoning",
        target_term="U_R",
        perturb=elongate_reasoning,
        answer_rule=reasoning_rule,
        description="Reasoning elongation should increase U_R.",
    ),
)

INTERVENTION_BY_NAME = {spec.name: spec for spec in INTERVENTIONS}
