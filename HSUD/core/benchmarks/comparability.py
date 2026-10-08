"""Benchmark comparability settings aligned with published baseline papers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class FairComparisonPlan:
    """Matched-budget plan for nested vs flat baseline comparisons.

    Absolute fairness is impossible once methods differ structurally; this plan
    makes the matching axis explicit and removes silent compute asymmetries.
    """

    mode: str
    M: int
    K: int
    J: int
    L: int
    nested_intent_calls: int
    nested_answer_calls: int
    nested_total_calls: int
    se_sc_samples: int
    ip_samples: int
    answer_temperature: float
    intent_temperature: float
    notes: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "M": self.M,
            "K": self.K,
            "J": self.J,
            "L": self.L,
            "nested_intent_calls": self.nested_intent_calls,
            "nested_answer_calls": self.nested_answer_calls,
            "nested_total_calls": self.nested_total_calls,
            "se_sc_samples": self.se_sc_samples,
            "ip_samples": self.ip_samples,
            "ip_total_calls": 2 * self.ip_samples,
            "answer_temperature": self.answer_temperature,
            "intent_temperature": self.intent_temperature,
            "notes": list(self.notes),
        }


def build_fair_comparison_plan(
    sampling: Mapping[str, Any],
    config: Mapping[str, Any],
    comparability: "ComparabilitySettings",
) -> FairComparisonPlan:
    """Build an explicit nested/flat matching plan from config."""

    M = int(sampling["M"])
    K = int(sampling["K"])
    J = int(sampling["J"])
    L = int(sampling["L"])
    nested_intent = M * K
    nested_answer = M * K * J * L
    nested_total = nested_intent + nested_answer

    bench_cfg = config.get("benchmark", {}) or {}
    fair_cfg = bench_cfg.get("fair_comparison") or {}
    # Default: answer-count match for SE/SC; API-count match for IP (rewrite+answer).
    mode = str(fair_cfg.get("mode", "hybrid")).lower()
    if comparability.enabled:
        # Published-setting alignment overrides hybrid defaults.
        mode = "comparability"
        se_sc = int(comparability.flat_baseline_samples)
        ip = se_sc
        answer_T = float(comparability.flat_baseline_temperature)
        intent_T = float(sampling.get("intent_temperature", 0.8))
        notes = (
            "comparability.enabled: SE/SC/IP use flat_baseline_samples and "
            "flat_baseline_temperature from the published-alignment block.",
        )
    elif mode == "api_matched":
        se_sc = nested_total
        ip = max(1, nested_total // 2)
        answer_T = float(sampling.get("reasoning_temperature", 1.0))
        intent_T = float(sampling.get("intent_temperature", 0.8))
        notes = (
            "SE/SC sample count equals nested total API calls "
            f"(intents+answers={nested_total}).",
            f"IP uses {ip} rewrite+answer pairs (~{2 * ip} calls) to match nested API.",
        )
    elif mode == "answer_matched":
        se_sc = nested_answer
        ip = nested_answer  # legacy; may give IP 2x API calls — disclosed in notes
        answer_T = float(sampling.get("reasoning_temperature", 1.0))
        intent_T = float(sampling.get("intent_temperature", 0.8))
        notes = (
            f"SE/SC/IP sample count equals nested answer leaves ({nested_answer}).",
            "WARNING: IP then uses ~2x API calls vs nested answers; prefer hybrid.",
        )
    else:  # hybrid (recommended)
        mode = "hybrid"
        se_sc = nested_answer
        ip = max(1, nested_total // 2)
        answer_T = float(sampling.get("reasoning_temperature", 1.0))
        intent_T = float(sampling.get("intent_temperature", 0.8))
        notes = (
            f"SE/SC matched on answer leaves N={se_sc}.",
            f"IP matched on total API calls: {ip} pairs ≈ {2 * ip} vs nested {nested_total}.",
            "Structural gap remains: nested varies persona/intent by design; "
            "flat SE/SC use fixed persona_0 on the original question.",
        )

    # Optional explicit overrides
    if "se_sc_samples" in fair_cfg:
        se_sc = int(fair_cfg["se_sc_samples"])
    if "ip_samples" in fair_cfg:
        ip = int(fair_cfg["ip_samples"])

    return FairComparisonPlan(
        mode=mode,
        M=M,
        K=K,
        J=J,
        L=L,
        nested_intent_calls=nested_intent,
        nested_answer_calls=nested_answer,
        nested_total_calls=nested_total,
        se_sc_samples=se_sc,
        ip_samples=ip,
        answer_temperature=answer_T,
        intent_temperature=intent_T,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class ComparabilitySettings:
    """Settings for fair comparison with Hou et al. (2024) GSM8K Table 1."""

    enabled: bool = False
    reference: str = ""
    flat_baseline_samples: int = 10
    flat_baseline_temperature: float = 0.5
    few_shot_dataset: str = ""
    few_shot_count: int = 0
    input_perturbation_mode: str = "intent"
    subsample_seed: int = 42
    skip_intent_rewrite: bool = False
    reference_baselines: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ComparabilitySettings:
        """Parse ``benchmark.comparability`` from a run config."""

        bench_cfg = config.get("benchmark", {})
        raw = bench_cfg.get("comparability") or {}
        if not raw:
            return cls(enabled=False)

        ref_baselines = {
            str(key): float(value)
            for key, value in (raw.get("reference_baselines") or {}).items()
        }
        return cls(
            enabled=bool(raw.get("enabled", False)),
            reference=str(raw.get("reference", "")),
            flat_baseline_samples=int(raw.get("flat_baseline_samples", 10)),
            flat_baseline_temperature=float(raw.get("flat_baseline_temperature", 0.5)),
            few_shot_dataset=str(raw.get("few_shot_dataset", "")),
            few_shot_count=int(raw.get("few_shot_count", 0)),
            input_perturbation_mode=str(raw.get("input_perturbation_mode", "intent")),
            subsample_seed=int(raw.get("subsample_seed", config.get("project", {}).get("seed", 42))),
            skip_intent_rewrite=bool(raw.get("skip_intent_rewrite", False)),
            reference_baselines=ref_baselines,
        )

    def resolve_sampling(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Return effective sampling block (nested budget + temperatures).

        Prefers ``sampling.*``, then fills missing ``M/K/J/L`` from
        ``benchmark.*`` so experiment YAMLs without a full sampling block
        still run.
        """

        sampling = dict(config.get("sampling", {}) or {})
        bench_cfg = config.get("benchmark", {}) or {}
        for key in ("M", "K", "J", "L"):
            if key not in sampling and key in bench_cfg:
                sampling[key] = bench_cfg[key]
        defaults = {"M": 4, "K": 4, "J": 4, "L": 4}
        for key, value in defaults.items():
            sampling.setdefault(key, value)

        if not self.enabled:
            return sampling

        override = (bench_cfg.get("comparability") or {}).get("nested") or {}
        for key in ("M", "K", "J", "L", "reasoning_temperature", "intent_temperature"):
            if key in override:
                sampling[key] = override[key]
        return sampling

    def flat_baseline_kwargs(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Keyword arguments shared by flat baseline samplers."""

        sampling = self.resolve_sampling(config)
        return {
            "temperature": self.flat_baseline_temperature,
            "max_tokens": int(sampling.get("max_answer_tokens", 192)),
            "base_seed": int(config.get("project", {}).get("seed", 42)),
        }

    def to_metadata(self) -> dict[str, Any]:
        """Serialize settings for benchmark JSON output."""

        return {
            "enabled": self.enabled,
            "reference": self.reference,
            "flat_baseline_samples": self.flat_baseline_samples,
            "flat_baseline_temperature": self.flat_baseline_temperature,
            "few_shot_dataset": self.few_shot_dataset,
            "few_shot_count": self.few_shot_count,
            "input_perturbation_mode": self.input_perturbation_mode,
            "subsample_seed": self.subsample_seed,
            "skip_intent_rewrite": self.skip_intent_rewrite,
            "reference_baselines": dict(self.reference_baselines),
        }
