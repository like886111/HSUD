"""Answer extraction and correctness grading for benchmark items."""

from __future__ import annotations

import re
from typing import Iterable


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_final_answer(text: str) -> str:
    """Extract a short final answer string from model output."""

    boxed = _BOXED_RE.search(text)
    if boxed:
        return boxed.group(1).strip()

    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip().splitlines()[0].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def normalize_text(value: str) -> str:
    """Normalize free-form answers for fuzzy matching."""

    text = value.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_numeric(value: str) -> str | None:
    """Extract the last numeric token from an answer string."""

    matches = _NUMERIC_RE.findall(value.replace(",", ""))
    if not matches:
        return None
    return matches[-1].lstrip("+")


def gsm8k_gold_answer(answer_field: str) -> str:
    """Parse GSM8K gold answer after the #### marker."""

    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()


def is_correct_answer(
    prediction: str,
    reference_answer: str,
    dataset: str,
    acceptable_answers: Iterable[str] | None = None,
) -> bool:
    """Check whether ``prediction`` matches dataset-specific gold answer(s)."""

    acceptable = [reference_answer]
    if acceptable_answers is not None:
        acceptable.extend(acceptable_answers)

    if dataset in {"gsm8k", "math"}:
        pred_num = extract_numeric(prediction)
        gold_nums = [extract_numeric(item) for item in acceptable]
        gold_nums = [item for item in gold_nums if item is not None]
        if pred_num is not None and gold_nums:
            try:
                pred_value = float(pred_num)
                return any(abs(pred_value - float(gold)) < 1e-6 for gold in gold_nums)
            except ValueError:
                return pred_num in gold_nums
        return normalize_text(prediction) in {normalize_text(item) for item in acceptable}

    normalized_prediction = normalize_text(prediction)
    normalized_gold = {normalize_text(item) for item in acceptable if item}
    if normalized_prediction in normalized_gold:
        return True

    # Substring match helps short factual answers.
    return any(
        normalized_prediction in gold or gold in normalized_prediction
        for gold in normalized_gold
        if gold
    )


def majority_answer(answer_texts: list[str]) -> str:
    """Return the modal answer string (ties broken lexicographically)."""

    if not answer_texts:
        return ""
    counts: dict[str, int] = {}
    for text in answer_texts:
        answer = extract_final_answer(text)
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts, key=lambda key: (counts[key], key))
