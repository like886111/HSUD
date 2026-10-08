"""HuggingFace loaders for full benchmark splits."""

from __future__ import annotations

from typing import Any, Iterator

from core.benchmarks.data import BenchmarkItem
from core.benchmarks.grading import gsm8k_gold_answer, normalize_text


SUPPORTED_DATASETS = ("truthfulqa", "triviaqa", "ambigqa", "gsm8k", "math")


def iter_hf_benchmark_items(
    dataset: str,
    split: str | None = None,
    max_samples: int | None = None,
) -> Iterator[BenchmarkItem]:
    """Yield benchmark items from a HuggingFace dataset."""

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    loader = _LOADER_REGISTRY[dataset]
    count = 0
    for record in loader(split=split):
        yield record
        count += 1
        if max_samples is not None and count >= max_samples:
            break


def load_hf_benchmark_split(
    dataset: str,
    split: str | None = None,
    max_samples: int | None = None,
) -> list[BenchmarkItem]:
    """Materialize a HuggingFace benchmark split."""

    return list(iter_hf_benchmark_items(dataset, split=split, max_samples=max_samples))


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace dataset loading requires `pip install datasets`."
        ) from exc
    return load_dataset


def _load_truthfulqa(split: str | None = None) -> Iterator[BenchmarkItem]:
    load_dataset = _require_datasets()
    chosen_split = split or "validation"
    dataset = load_dataset("truthful_qa", "generation", split=chosen_split)
    for index, row in enumerate(dataset):
        correct_answers = row.get("correct_answers") or []
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]
        reference = str(row.get("best_answer") or (correct_answers[0] if correct_answers else ""))
        yield BenchmarkItem(
            id=f"truthfulqa_{chosen_split}_{index:05d}",
            dataset="truthfulqa",
            question=str(row["question"]),
            label=-1,
            reference_answer=reference,
            category=str(row.get("category", "")),
            notes="label_from_correctness",
            acceptable_answers=tuple(str(item) for item in correct_answers),
            label_source="correctness",
            split=chosen_split,
        )


def _load_triviaqa(split: str | None = None) -> Iterator[BenchmarkItem]:
    load_dataset = _require_datasets()
    chosen_split = split or "validation"
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=chosen_split)
    for index, row in enumerate(dataset):
        answer = row.get("answer") or {}
        aliases = answer.get("aliases") or answer.get("normalized_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        value = str(answer.get("value", ""))
        acceptable = [value, *aliases]
        yield BenchmarkItem(
            id=f"triviaqa_{chosen_split}_{index:05d}",
            dataset="triviaqa",
            question=str(row["question"]),
            label=-1,
            reference_answer=value,
            category="trivia",
            notes="label_from_correctness",
            acceptable_answers=tuple(str(item) for item in acceptable if item),
            label_source="correctness",
            split=chosen_split,
        )


def _load_ambigqa(split: str | None = None) -> Iterator[BenchmarkItem]:
    load_dataset = _require_datasets()
    chosen_split = split or "validation"
    dataset = load_dataset("sewon/ambig_qa", "light", split=chosen_split)
    for index, row in enumerate(dataset):
        question = str(row.get("question", ""))
        annotations = row.get("annotations") or {}
        answers = annotations.get("answer") or []
        if isinstance(answers, dict):
            answers = list(answers.values())
        flat_answers: list[str] = []
        for entry in answers:
            if isinstance(entry, list):
                flat_answers.extend(str(item) for item in entry)
            else:
                flat_answers.append(str(entry))
        flat_answers = [item for item in flat_answers if item]
        reference = flat_answers[0] if flat_answers else ""
        ambiguous = len(flat_answers) > 1
        yield BenchmarkItem(
            id=f"ambigqa_{chosen_split}_{index:05d}",
            dataset="ambigqa",
            question=question,
            label=-1,
            reference_answer=reference,
            category="ambiguous" if ambiguous else "disambiguated",
            notes="label_from_correctness",
            acceptable_answers=tuple(flat_answers),
            label_source="correctness",
            split=chosen_split,
        )


def _load_gsm8k(split: str | None = None) -> Iterator[BenchmarkItem]:
    load_dataset = _require_datasets()
    chosen_split = split or "test"
    dataset = load_dataset("openai/gsm8k", "main", split=chosen_split)
    for index, row in enumerate(dataset):
        gold = gsm8k_gold_answer(str(row["answer"]))
        yield BenchmarkItem(
            id=f"gsm8k_{chosen_split}_{index:05d}",
            dataset="gsm8k",
            question=str(row["question"]),
            label=-1,
            reference_answer=gold,
            category="math_word_problem",
            notes="label_from_correctness",
            acceptable_answers=(gold,),
            label_source="correctness",
            split=chosen_split,
        )


MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _load_math(split: str | None = None) -> Iterator[BenchmarkItem]:
    load_dataset = _require_datasets()
    chosen_split = split or "test"
    index = 0

    try:
        datasets = [load_dataset("EleutherAI/hendrycks_math", "all", split=chosen_split)]
    except ValueError:
        datasets = [
            load_dataset("EleutherAI/hendrycks_math", subject, split=chosen_split)
            for subject in MATH_SUBJECTS
        ]

    for dataset in datasets:
        for row in dataset:
            problem = str(row.get("problem", ""))
            solution = str(row.get("solution", ""))
            gold = _extract_math_gold(solution)
            yield BenchmarkItem(
                id=f"math_{chosen_split}_{index:05d}",
                dataset="math",
                question=problem,
                label=-1,
                reference_answer=gold,
                category=str(row.get("type", row.get("subject", "math"))),
                notes="label_from_correctness",
                acceptable_answers=(gold, solution),
                label_source="correctness",
                split=chosen_split,
            )
            index += 1


def _extract_math_gold(solution: str) -> str:
    import re

    boxed = re.search(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed.group(1).strip()
    lines = [line.strip() for line in solution.splitlines() if line.strip()]
    return lines[-1] if lines else normalize_text(solution)


_LOADER_REGISTRY = {
    "truthfulqa": _load_truthfulqa,
    "triviaqa": _load_triviaqa,
    "ambigqa": _load_ambigqa,
    "gsm8k": _load_gsm8k,
    "math": _load_math,
}
