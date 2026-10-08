"""Phase 5 real-benchmark evaluation against uncertainty baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.benchmarks.comparability import ComparabilitySettings, build_fair_comparison_plan
from core.benchmarks.data import BenchmarkItem, load_benchmark_split, subsample_benchmark_items
from core.benchmarks.evaluator import BenchmarkScores, evaluate_item_oracle, evaluate_item_pipeline
from core.evaluation.metrics import (
    aurac,
    auroc,
    bootstrap_auroc_ci,
    bootstrap_auroc_difference,
    risk_coverage_curve,
)
from utils.config import load_config, resolve_path
from utils.logger import get_logger, make_run_log_path, save_json


BASELINE_FIELDS: dict[str, str] = {
    "nested_total": "Nested H(S|X)",
    "semantic_entropy": "Semantic Entropy",
    "self_consistency": "Self-Consistency",
    "input_perturbation": "Input Perturbation",
}

COMPONENT_FIELDS: dict[str, str] = {
    "u_theta": "U_Theta",
    "u_z": "U_Z",
    "u_r": "U_R",
    "u_res": "U_res",
}

_SCORE_FIELD_NAMES = tuple(field.name for field in fields(BenchmarkScores))


def _score_from_dict(payload: dict[str, Any]) -> BenchmarkScores:
    """Rebuild ``BenchmarkScores`` from a checkpoint / JSON record."""

    return BenchmarkScores(**{name: payload[name] for name in _SCORE_FIELD_NAMES})


def _checkpoint_meta(
    mode: str,
    source: str,
    max_samples: int | None,
    datasets: tuple[str, ...],
    budget: dict[str, int],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "source": source,
        "max_samples": max_samples,
        "datasets": list(datasets),
        "budget": budget,
    }


def _meta_compatible(stored: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    """Allow exact match, or safe expansions of max_samples / datasets."""

    if stored.get("mode") != expected.get("mode"):
        return False, "mode differs"
    if stored.get("source") != expected.get("source"):
        return False, "source differs"
    if stored.get("budget") != expected.get("budget"):
        return False, "budget differs"

    stored_datasets = list(stored.get("datasets") or [])
    expected_datasets = list(expected.get("datasets") or [])
    if not set(stored_datasets).issubset(set(expected_datasets)):
        return False, (
            f"checkpoint datasets {stored_datasets} are not a subset of "
            f"requested {expected_datasets}"
        )

    stored_n = stored.get("max_samples")
    expected_n = expected.get("max_samples")
    if stored_n is None and expected_n is not None:
        return False, "checkpoint has unlimited max_samples; refuse to shrink scope"
    if stored_n is not None and expected_n is not None and int(expected_n) < int(stored_n):
        return False, f"refusing to shrink max_samples from {stored_n} to {expected_n}"

    return True, "ok"


def load_checkpoint_scores(
    checkpoint_path: Path,
    expected_meta: dict[str, Any],
) -> tuple[dict[str, BenchmarkScores], bool]:
    """Load scores from JSONL checkpoint.

    Returns:
        completed: item_id -> scores
        meta_upgraded: True if header should be rewritten to ``expected_meta``
            (e.g. max_samples 20→50 or datasets expanded).
    """

    if not checkpoint_path.is_file():
        return {}, False

    completed: dict[str, BenchmarkScores] = {}
    stored_meta: dict[str, Any] | None = None
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if line_no == 1 and record.get("_type") == "meta":
                stored_meta = {key: record.get(key) for key in expected_meta}
                ok, reason = _meta_compatible(stored_meta, expected_meta)
                if not ok:
                    raise ValueError(
                        "Checkpoint metadata incompatible; refuse to resume. "
                        f"reason={reason}; stored={stored_meta}, expected={expected_meta}. "
                        "Pass --fresh to start a new run."
                    )
                continue
            if record.get("_type") == "meta":
                continue
            score = _score_from_dict(record)
            completed[score.item_id] = score

    meta_upgraded = stored_meta is not None and stored_meta != expected_meta
    return completed, meta_upgraded


def init_checkpoint(checkpoint_path: Path, meta: dict[str, Any]) -> None:
    """Create / overwrite a checkpoint file with a metadata header."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    header = {"_type": "meta", **meta}
    checkpoint_path.write_text(
        json.dumps(header, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def rewrite_checkpoint(
    checkpoint_path: Path,
    meta: dict[str, Any],
    scores: dict[str, BenchmarkScores],
) -> None:
    """Rewrite checkpoint with upgraded meta while keeping existing scores."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_type": "meta", **meta}, ensure_ascii=False)]
    for score in scores.values():
        lines.append(json.dumps(asdict(score), ensure_ascii=False))
    checkpoint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_checkpoint(checkpoint_path: Path, score: BenchmarkScores) -> None:
    """Append one completed item to the JSONL checkpoint."""

    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(score), ensure_ascii=False) + "\n")


def append_item_ledger(
    ledger_path: Path,
    *,
    score: BenchmarkScores,
    elapsed_sec: float,
    newly_run: int,
    skipped: int,
    planned_so_far: int,
) -> None:
    """Append a human-auditable per-item completion record."""

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "newly_run": int(newly_run),
        "skipped": int(skipped),
        "planned_so_far": int(planned_so_far),
        **asdict(score),
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_benchmark(
    config_path: str | Path | None = None,
    mode: str = "oracle",
    datasets: tuple[str, ...] = ("truthfulqa", "gsm8k"),
    source: Literal["auto", "local", "hf", "pilot"] = "auto",
    max_samples: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Evaluate benchmark splits and compare uncertainty scores.

    When ``checkpoint_path`` is set, each finished item is appended to a JSONL
    file so interrupted API runs can resume without re-billing completed items.
    Expanding ``max_samples`` (e.g. 20→50) or adding datasets reuses overlapping
    ``item_id``s and only evaluates the new remainder.
    """

    log = logger or get_logger()
    config = load_config(config_path)
    bench_cfg = config.get("benchmark", {})
    comparability = ComparabilitySettings.from_config(config)
    effective_sampling = comparability.resolve_sampling(config)
    M = int(effective_sampling["M"])
    K = int(effective_sampling["K"])
    J = int(effective_sampling["J"])
    L = int(effective_sampling["L"])
    fair_plan = build_fair_comparison_plan(effective_sampling, config, comparability)
    source = str(bench_cfg.get("source", source))
    if max_samples is None and bench_cfg.get("max_samples") is not None:
        max_samples = int(bench_cfg["max_samples"])

    budget = {"M": M, "K": K, "J": J, "L": L}
    meta = _checkpoint_meta(mode, source, max_samples, datasets, budget)
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None

    completed: dict[str, BenchmarkScores] = {}
    if ckpt_path is not None:
        if resume and ckpt_path.is_file():
            completed, meta_upgraded = load_checkpoint_scores(ckpt_path, meta)
            if meta_upgraded:
                rewrite_checkpoint(ckpt_path, meta, completed)
                log.info(
                    "Upgraded checkpoint meta to %s (kept %d scored items)",
                    meta,
                    len(completed),
                )
            log.info(
                "Resuming from checkpoint %s (%d items already done)",
                ckpt_path,
                len(completed),
            )
        else:
            init_checkpoint(ckpt_path, meta)
            log.info("Initialized checkpoint at %s", ckpt_path)

    evaluator: Callable[[BenchmarkItem], BenchmarkScores]
    if mode == "oracle":
        evaluator = lambda item: evaluate_item_oracle(item, M, K, J, L)
    elif mode == "pipeline":
        evaluator = lambda item: evaluate_item_pipeline(item, config)
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    all_scores: list[BenchmarkScores] = []
    planned = 0
    skipped = 0
    newly_run = 0
    ledger_path = (
        ckpt_path.parent / "item_ledger.jsonl" if ckpt_path is not None else None
    )
    for dataset in datasets:
        items = load_benchmark_split(dataset, source=source, max_samples=None)
        items = subsample_benchmark_items(
            items,
            max_samples=max_samples,
            seed=comparability.subsample_seed if comparability.enabled else int(
                config.get("project", {}).get("seed", 42)
            ),
        )
        for item in items:
            planned += 1
            if item.id in completed:
                all_scores.append(completed[item.id])
                skipped += 1
                log.info("Skip (checkpoint hit): %s", item.id)
                continue
            log.info(
                "Evaluating %s (%s) [done=%d skipped=%d]",
                item.id,
                dataset,
                newly_run,
                skipped,
            )
            t0 = time.perf_counter()
            try:
                score = evaluator(item)
            except Exception:
                log.exception("Failed on item %s; progress kept in checkpoint", item.id)
                raise
            elapsed = time.perf_counter() - t0
            all_scores.append(score)
            newly_run += 1
            if ckpt_path is not None:
                append_checkpoint(ckpt_path, score)
                if ledger_path is not None:
                    append_item_ledger(
                        ledger_path,
                        score=score,
                        elapsed_sec=elapsed,
                        newly_run=newly_run,
                        skipped=skipped,
                        planned_so_far=planned,
                    )
                log.info(
                    "Recorded %s | elapsed=%.1fs | label=%d | "
                    "H=%.4f UΘ=%.4f UZ=%.4f UR=%.4f Ures=%.4f | "
                    "SE=%.4f SC=%.4f IP=%.4f | new=%d",
                    score.item_id,
                    elapsed,
                    score.label,
                    score.nested_total,
                    score.u_theta,
                    score.u_z,
                    score.u_r,
                    score.u_res,
                    score.semantic_entropy,
                    score.self_consistency,
                    score.input_perturbation,
                    newly_run,
                )

    log.info(
        "Benchmark loop finished: planned=%d resumed=%d newly_run=%d",
        planned,
        skipped,
        newly_run,
    )
    metrics = compute_detection_metrics(all_scores)
    return {
        "mode": mode,
        "source": source,
        "max_samples": max_samples,
        "budget": budget,
        "flat_baseline_samples": fair_plan.se_sc_samples,
        "fair_comparison": fair_plan.to_metadata(),
        "comparability": comparability.to_metadata(),
        "datasets": list(datasets),
        "scores": [asdict(score) for score in all_scores],
        "metrics": metrics,
        "pass": metrics["nested_beats_semantic_entropy_auroc"],
        "checkpoint": {
            "path": str(ckpt_path) if ckpt_path is not None else None,
            "resumed_items": skipped,
            "newly_run_items": newly_run,
            "planned_items": planned,
        },
    }


def compute_detection_metrics(
    scores: list[BenchmarkScores],
    *,
    n_boot: int = 2000,
    bootstrap_seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute AUROC and risk-coverage summaries for all methods."""

    labels = np.array([score.label for score in scores], dtype=int)
    method_arrays = {
        field: np.array([getattr(score, field) for score in scores], dtype=float)
        for field in BASELINE_FIELDS
    }

    aurocs = {field: auroc(labels, values) for field, values in method_arrays.items()}
    auracs = {field: aurac(labels, values) for field, values in method_arrays.items()}

    nested_cov, nested_risk = risk_coverage_curve(labels, method_arrays["nested_total"])
    risk_coverage = {
        field: {
            "coverage": cov.tolist(),
            "risk": risk.tolist(),
        }
        for field, (cov, risk) in {
            field: risk_coverage_curve(labels, values)
            for field, values in method_arrays.items()
        }.items()
    }

    per_dataset: dict[str, Any] = {}
    for dataset in sorted({score.dataset for score in scores}):
        subset = [score for score in scores if score.dataset == dataset]
        y = np.array([score.label for score in subset], dtype=int)
        per_dataset[dataset] = {
            "count": len(subset),
            **{
                f"{field}_auroc": auroc(y, np.array([getattr(score, field) for score in subset], dtype=float))
                for field in BASELINE_FIELDS
            },
        }

    nested_auroc = aurocs["nested_total"]
    semantic_auroc = aurocs["semantic_entropy"]
    sc_auroc = aurocs["self_consistency"]
    ip_auroc = aurocs["input_perturbation"]

    def _beats(candidate: float, reference: float) -> bool:
        return bool(
            np.isfinite(candidate) and np.isfinite(reference) and candidate >= reference
        )

    beats_semantic = _beats(nested_auroc, semantic_auroc)
    beats_self_consistency = _beats(nested_auroc, sc_auroc)
    beats_input_perturbation = _beats(nested_auroc, ip_auroc)
    # Beats *all* flat baselines (SE, SC, IP). Do not alias to SE-only.
    beats_all_flat = bool(
        beats_semantic and beats_self_consistency and beats_input_perturbation
    )

    boot_kw = {"n_boot": n_boot, "alpha": alpha, "seed": bootstrap_seed}
    auroc_bootstrap = {
        field: bootstrap_auroc_ci(labels, method_arrays[field], **boot_kw)
        for field in BASELINE_FIELDS
    }
    auroc_delta_bootstrap = {
        "nested_minus_semantic_entropy": bootstrap_auroc_difference(
            labels,
            method_arrays["nested_total"],
            method_arrays["semantic_entropy"],
            **boot_kw,
        ),
        "nested_minus_self_consistency": bootstrap_auroc_difference(
            labels,
            method_arrays["nested_total"],
            method_arrays["self_consistency"],
            **boot_kw,
        ),
        "nested_minus_input_perturbation": bootstrap_auroc_difference(
            labels,
            method_arrays["nested_total"],
            method_arrays["input_perturbation"],
            **boot_kw,
        ),
    }
    per_dataset_bootstrap: dict[str, Any] = {}
    for dataset in per_dataset:
        subset = [score for score in scores if score.dataset == dataset]
        y = np.array([score.label for score in subset], dtype=int)
        per_dataset_bootstrap[dataset] = {
            field: bootstrap_auroc_ci(
                y,
                np.array([getattr(score, field) for score in subset], dtype=float),
                **boot_kw,
            )
            for field in BASELINE_FIELDS
        }

    return {
        "nested_auroc": nested_auroc,
        "semantic_entropy_auroc": semantic_auroc,
        "self_consistency_auroc": sc_auroc,
        "input_perturbation_auroc": ip_auroc,
        "nested_aurac": auracs["nested_total"],
        "semantic_entropy_aurac": auracs["semantic_entropy"],
        "self_consistency_aurac": auracs["self_consistency"],
        "input_perturbation_aurac": auracs["input_perturbation"],
        "nested_beats_semantic_entropy_auroc": beats_semantic,
        "nested_beats_self_consistency_auroc": beats_self_consistency,
        "nested_beats_input_perturbation_auroc": beats_input_perturbation,
        "nested_beats_all_flat_baselines_auroc": beats_all_flat,
        # Backward-compatible alias: historically mis-copied SE-only flag.
        "nested_beats_baseline_auroc": beats_all_flat,
        "auroc_by_method": {BASELINE_FIELDS[field]: value for field, value in aurocs.items()},
        "aurac_by_method": {BASELINE_FIELDS[field]: value for field, value in auracs.items()},
        "auroc_bootstrap": {
            BASELINE_FIELDS[field]: payload for field, payload in auroc_bootstrap.items()
        },
        "auroc_delta_bootstrap": auroc_delta_bootstrap,
        "auroc_bootstrap_by_dataset": per_dataset_bootstrap,
        "risk_coverage": risk_coverage,
        "per_dataset": per_dataset,
        "label_counts": {
            "n": int(labels.size),
            "n_positive": int(np.sum(labels == 1)),
            "n_negative": int(np.sum(labels == 0)),
        },
    }


def plot_risk_coverage(metrics: dict[str, Any], output_path: Path) -> Path:
    """Plot risk-coverage curves for nested vs baseline methods."""

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"nested_total": "o", "semantic_entropy": "s", "self_consistency": "^", "input_perturbation": "D"}
    for field, label in BASELINE_FIELDS.items():
        curve = metrics["risk_coverage"][field]
        auroc_value = metrics["auroc_by_method"][label]
        label_text = (
            f"{label} (AUROC={auroc_value:.3f})"
            if np.isfinite(auroc_value)
            else f"{label} (AUROC=nan)"
        )
        ax.plot(
            curve["coverage"],
            curve["risk"],
            marker=markers[field],
            label=label_text,
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    ax.set_title("Risk-Coverage Curves: Hallucination / Error Detection")
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path.resolve()


def plot_auroc_summary(metrics: dict[str, Any], output_path: Path) -> Path:
    """Plot AUROC comparison bar chart."""

    rows = [
        {"method": label, "auroc": metrics["auroc_by_method"][label]}
        for label in BASELINE_FIELDS.values()
    ]
    for dataset, values in metrics["per_dataset"].items():
        for field, label in BASELINE_FIELDS.items():
            rows.append(
                {
                    "method": f"{label} ({dataset})",
                    "auroc": values[f"{field}_auroc"],
                }
            )

    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=frame, x="method", y="auroc", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("AUROC for Hallucination / Error Detection")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def print_summary(results: dict[str, Any]) -> None:
    """Print benchmark summary."""

    metrics = results["metrics"]
    print("\n=== Phase 5 Benchmark Evaluation ===")
    print("Mode:", results["mode"])
    print("Source:", results.get("source", "auto"))
    print("Budget:", results["budget"])
    print(f"Items evaluated: {len(results['scores'])}")
    if "label_counts" in metrics:
        counts = metrics["label_counts"]
        print(
            f"Labels: n={counts['n']}, "
            f"error/pos={counts['n_positive']}, "
            f"correct/neg={counts['n_negative']}"
        )
    for label, value in metrics["auroc_by_method"].items():
        value_str = f"{value:.4f}" if np.isfinite(value) else "nan (need both classes)"
        print(f"{label:24s} AUROC: {value_str}")
    print("Per-dataset nested AUROC:")
    for dataset, values in metrics["per_dataset"].items():
        def _fmt(key: str) -> str:
            v = values[key]
            return f"{v:.4f}" if np.isfinite(v) else "nan"

        print(
            f"  {dataset}: nested={_fmt('nested_total_auroc')}, "
            f"semantic={_fmt('semantic_entropy_auroc')}, "
            f"sc={_fmt('self_consistency_auroc')}, "
            f"input={_fmt('input_perturbation_auroc')}, n={values['count']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 5 benchmark evaluation.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names, e.g. truthfulqa gsm8k ambigqa.",
    )
    parser.add_argument(
        "--mode",
        choices=("oracle", "pipeline"),
        default="oracle",
        help="oracle=controlled benchmark; pipeline=dry-run/full LLM pipeline",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "local", "hf", "pilot"),
        default=None,
        help="Data source: auto (splits->pilot->HF), local JSON only, or HF streaming.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per dataset (useful for smoke tests).",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and start a new run (still writes a new checkpoint).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable per-item checkpointing (not recommended for API runs).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional explicit log path; default is logs/benchmark_<mode>_<timestamp>.log",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = load_config(args.config)
    bench_cfg = config.get("benchmark", {})
    datasets = tuple(args.datasets) if args.datasets else tuple(
        bench_cfg.get("datasets", config.get("risk_prediction", {}).get("datasets", ("truthfulqa", "gsm8k")))
    )
    source = args.source or str(bench_cfg.get("source", "auto"))
    mode = args.mode

    outputs_dir = resolve_path(config, "paths", "outputs_dir")
    reports_dir = resolve_path(config, "paths", "reports_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else make_run_log_path(f"benchmark_{mode}")
    logger = get_logger(log_file=log_path)
    latest_log = PROJECT_ROOT / "logs" / f"benchmark_{mode}_latest.log"
    try:
        if latest_log.exists() or latest_log.is_symlink():
            latest_log.unlink()
        latest_log.symlink_to(log_path.resolve())
    except OSError:
        latest_log.write_text(f"See: {log_path.resolve()}\n", encoding="utf-8")

    checkpoint_path = None if args.no_checkpoint else outputs_dir / f"benchmark_{mode}_checkpoint.jsonl"
    logger.info(
        "Starting benchmark mode=%s source=%s datasets=%s max_samples=%s checkpoint=%s fresh=%s",
        mode,
        source,
        list(datasets),
        args.max_samples,
        checkpoint_path,
        args.fresh,
    )

    results = run_benchmark(
        config_path=args.config,
        mode=mode,
        datasets=datasets,
        source=source,
        max_samples=args.max_samples,
        checkpoint_path=checkpoint_path,
        resume=not args.fresh,
        logger=logger,
    )
    print_summary(results)

    summary_path = outputs_dir / f"benchmark_{mode}_summary.json"
    full_path = outputs_dir / f"benchmark_{mode}_full.json"
    save_json(
        {
            "mode": results["mode"],
            "source": results["source"],
            "max_samples": results["max_samples"],
            "budget": results["budget"],
            "metrics": results["metrics"],
            "pass": results["pass"],
            "checkpoint": results.get("checkpoint"),
            "log_file": str(log_path.resolve()),
        },
        summary_path,
    )
    save_json(results, full_path)
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved full scores to %s", full_path)

    if not args.no_plot:
        rc_path = reports_dir / f"benchmark_{mode}_risk_coverage.png"
        auroc_path = reports_dir / f"benchmark_{mode}_auroc.png"
        plot_risk_coverage(results["metrics"], rc_path)
        plot_auroc_summary(results["metrics"], auroc_path)
        logger.info("Saved plots to %s and %s", rc_path, auroc_path)

    logger.info("Benchmark evaluation complete. Log: %s", log_path)


if __name__ == "__main__":
    main()
