"""Benchmark sample schemas and loaders."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LabelSource = Literal["static", "correctness"]

SUPPORTED_BENCHMARK_DATASETS = (
    "truthfulqa",
    "triviaqa",
    "ambigqa",
    "gsm8k",
    "math",
)


@dataclass(slots=True)
class BenchmarkItem:
    """One benchmark example with hallucination / error label."""

    id: str
    dataset: str
    question: str
    label: int
    reference_answer: str = ""
    category: str = ""
    notes: str = ""
    acceptable_answers: tuple[str, ...] = ()
    label_source: LabelSource = "static"
    split: str = ""


def load_benchmark_split(
    dataset: str,
    split_path: Path | None = None,
    source: Literal["auto", "local", "hf", "pilot"] = "auto",
    hf_split: str | None = None,
    max_samples: int | None = None,
) -> list[BenchmarkItem]:
    """Load a benchmark split from local JSON or HuggingFace.

    Resolution order for ``source='auto'``:
        1. ``data/benchmark_splits/{dataset}.json`` (full export)
        2. ``data/benchmark_samples/{dataset}_sample.json`` (pilot)
        3. HuggingFace streaming load
    """

    if source == "hf":
        from core.benchmarks.hf_loaders import load_hf_benchmark_split

        return load_hf_benchmark_split(dataset, split=hf_split, max_samples=max_samples)

    if split_path is not None:
        return _load_json_split(dataset, split_path)

    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / "data" / "benchmark_splits" / f"{dataset}.json"
    pilot_path = project_root / "data" / "benchmark_samples" / f"{dataset}_sample.json"

    if source == "pilot":
        if pilot_path.is_file():
            items = _load_json_split(dataset, pilot_path)
            if max_samples is not None:
                return items[:max_samples]
            return items
        raise FileNotFoundError(f"No pilot sample found for {dataset}: {pilot_path}")

    if source == "local" or source == "auto":
        if full_path.is_file():
            items = _load_json_split(dataset, full_path)
            if max_samples is not None:
                return items[:max_samples]
            return items
        if pilot_path.is_file():
            items = _load_json_split(dataset, pilot_path)
            if max_samples is not None:
                return items[:max_samples]
            return items

    if source == "auto":
        from core.benchmarks.hf_loaders import load_hf_benchmark_split

        return load_hf_benchmark_split(dataset, split=hf_split, max_samples=max_samples)

    raise FileNotFoundError(
        f"No benchmark split found for {dataset}. "
        f"Run: python data/prepare_full_benchmarks.py --datasets {dataset}"
    )


def subsample_benchmark_items(
    items: list[BenchmarkItem],
    max_samples: int | None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Return a deterministic random subset for comparable benchmark runs."""

    if max_samples is None or max_samples >= len(items):
        return items
    indices = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return [items[indices[idx]] for idx in range(max_samples)]


def _load_json_split(dataset: str, split_path: Path) -> list[BenchmarkItem]:
    records = json.loads(split_path.read_text(encoding="utf-8"))
    return [_record_to_item(dataset, record) for record in records]


def _record_to_item(dataset: str, record: dict) -> BenchmarkItem:
    acceptable = record.get("acceptable_answers") or []
    if isinstance(acceptable, list):
        acceptable_tuple = tuple(str(item) for item in acceptable)
    else:
        acceptable_tuple = ()

    return BenchmarkItem(
        id=str(record["id"]),
        dataset=str(record.get("dataset", dataset)),
        question=str(record["question"]),
        label=int(record.get("label", -1)),
        reference_answer=str(record.get("reference_answer", "")),
        category=str(record.get("category", "")),
        notes=str(record.get("notes", "")),
        acceptable_answers=acceptable_tuple,
        label_source=str(record.get("label_source", "static")),
        split=str(record.get("split", "")),
    )
