"""LLM backend abstractions and base sampling data types.

Model serving is isolated from information-theoretic estimation logic.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Optional

BackendName = Literal["dummy", "transformers", "vllm", "openai"]


@dataclass(slots=True)
class GenerationRequest:
    """One generation request for batched / concurrent inference."""

    prompt: str
    temperature: float = 0.7
    max_tokens: int = 128
    seed: Optional[int] = None


@dataclass(slots=True)
class GenerationResult:
    """Container for a raw LLM generation."""

    text: str
    prompt: str
    temperature: float
    max_tokens: int
    backend: str


@dataclass(slots=True)
class SampledPath:
    """SampledPath leaf in ``Theta -> Z -> R -> S -> Y``.

    Attributes:
        theta_id: Simulated parameter draw :math:`\\Theta`.
        z_id: Sampled intent index :math:`Z`.
        r_id: Sampled reasoning path index :math:`R`.
        l_id: Observation-layer replicate index :math:`L`.
        question: Original observed input :math:`X`.
        z_text: Intent-conditioned question representing :math:`Z`.
        r_text: Realized reasoning text for :math:`R` (shared across ``l_id``
            when ``fixed_reasoning_path=True``; otherwise may differ per leaf
            under legacy reasoning-branch sampling).
        answer_text: Final answer :math:`Y` before semantic clustering.
        s_cluster_id: Semantic cluster ID for :math:`S` (assigned after clustering).
    """

    theta_id: str
    z_id: int
    r_id: int
    question: str
    z_text: str
    r_text: str
    answer_text: str
    l_id: int = 0
    s_cluster_id: Optional[int] = None


class LLMGenerator:
    """Unified text generation interface for LLM backends."""

    def __init__(
        self,
        backend: BackendName = "dummy",
        model_name: str = "distilgpt2",
        seed: int = 13,
        device: Optional[str] = None,
        dry_run: bool = True,
        max_concurrency: int = 4,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        api_type: Literal["chat", "completion"] = "chat",
        system_prompt: str = "You are a helpful assistant.",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        trust_env: bool = True,
    ) -> None:
        self.backend: BackendName = "dummy" if dry_run else backend
        self.model_name = model_name
        self.seed = seed
        self.device = device
        self.dry_run = dry_run
        self.max_concurrency = max(1, max_concurrency)
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.api_type = api_type
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.trust_env = trust_env
        self._tokenizer = None
        self._model = None
        self._vllm = None
        self._openai_client = None

        if self.backend == "transformers":
            self._init_transformers_backend()
        elif self.backend == "vllm":
            self._init_vllm_backend()
        elif self.backend == "openai":
            self._init_openai_backend()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LLMGenerator:
        """Build a generator from the project YAML config."""

        llm_cfg = config["llm"]
        project_seed = int(config.get("project", {}).get("seed", 13))
        api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_key = llm_cfg.get("api_key")
        if api_key is None:
            api_key = os.environ.get(api_key_env)
        base_url = llm_cfg.get("base_url")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        return cls(
            backend=llm_cfg.get("backend", "dummy"),
            model_name=llm_cfg.get("model_name", "distilgpt2"),
            seed=project_seed,
            device=llm_cfg.get("device"),
            dry_run=bool(llm_cfg.get("dry_run", True)),
            max_concurrency=int(llm_cfg.get("max_concurrency", 4)),
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            api_type=llm_cfg.get("api_type", "chat"),
            system_prompt=str(llm_cfg.get("system_prompt", "You are a helpful assistant.")),
            timeout=float(llm_cfg.get("timeout", 120.0)),
            max_retries=int(llm_cfg.get("max_retries", 3)),
            retry_delay=float(llm_cfg.get("retry_delay", 2.0)),
            trust_env=bool(llm_cfg.get("trust_env", True)),
        )

    def _init_transformers_backend(self) -> None:
        """Load a HuggingFace causal LM."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The transformers backend requires torch and transformers."
            ) from exc

        torch.manual_seed(self.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.device is not None:
            self._model = self._model.to(self.device)
        self._model.eval()

    def _init_vllm_backend(self) -> None:
        """Load a vLLM engine."""

        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "The vllm backend requires `pip install -r requirements-vllm.txt`."
            ) from exc

        kwargs = {"model": self.model_name}
        if self.device is not None:
            kwargs["device"] = self.device
        self._vllm = LLM(**kwargs)

    def _init_openai_backend(self) -> None:
        """Initialize an OpenAI-compatible HTTP API client."""

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai backend requires `pip install openai`."
            ) from exc

        if not self.api_key:
            raise ValueError(
                f"Missing API key for backend=openai. Set llm.api_key in config "
                f"or export {self.api_key_env}."
            )

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if not self.trust_env:
            import httpx

            client_kwargs["http_client"] = httpx.Client(
                trust_env=False,
                timeout=self.timeout,
            )
        self._openai_client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 128,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt."""

        if self.backend == "dummy":
            return self._dummy_generate(prompt, temperature, max_tokens, seed)
        if self.backend == "transformers":
            return self._transformers_generate(prompt, temperature, max_tokens, seed)
        if self.backend == "vllm":
            return self._vllm_generate(prompt, temperature, max_tokens, seed)
        if self.backend == "openai":
            return self._openai_generate(prompt, temperature, max_tokens, seed)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def generate_batch(self, requests: list[GenerationRequest]) -> list[str]:
        """Generate multiple prompts concurrently."""

        if not requests:
            return []

        if len(requests) == 1 or self.max_concurrency == 1:
            return [
                self.generate(
                    request.prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    seed=request.seed,
                )
                for request in requests
            ]

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [
                executor.submit(
                    self.generate,
                    request.prompt,
                    request.temperature,
                    request.max_tokens,
                    request.seed,
                )
                for request in requests
            ]
        return [future.result() for future in futures]

    def _dummy_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        """Generate a lightweight deterministic mock response."""

        digest = hashlib.sha256(
            f"{self.seed}:{seed}:{temperature:.3f}:{prompt}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        lower_prompt = prompt.lower()

        if "rewrite" in lower_prompt or "intent" in lower_prompt:
            variants = [
                "Clarify the most common interpretation",
                "Ask for the literal interpretation",
                "Resolve the entity reference explicitly",
                "Focus on the mathematical quantities",
            ]
            return f"{rng.choice(variants)}: {self._shorten(prompt, max_tokens)}"

        if "final answer" in lower_prompt or "answer" in lower_prompt:
            answer_bank = self._select_dummy_answers(lower_prompt)
            rationale = rng.choice(
                [
                    "I compare the plausible interpretations.",
                    "I compute the result step by step.",
                    "I rely on uncertain background knowledge.",
                    "I sample one consistent reasoning path.",
                ]
            )
            return f"Reasoning: {rationale}\nFinal Answer: {rng.choice(answer_bank)}"

        return f"Dummy generation: {self._shorten(prompt, max_tokens)}"

    @staticmethod
    def _select_dummy_answers(lower_prompt: str) -> list[str]:
        """Choose answer variants for synthetic sanity-check prompts."""

        if "bank" in lower_prompt:
            return ["financial institution", "river edge", "data bank"]
        if "xq-17" in lower_prompt or "unknown" in lower_prompt:
            return ["unknown", "possibly alpha", "possibly beta"]
        if "math" in lower_prompt or "calculate" in lower_prompt:
            return ["42", "41", "43"]
        return ["yes", "no", "uncertain"]

    @staticmethod
    def _shorten(text: str, max_tokens: int) -> str:
        """Return a word-limited rendering of ``text`` for dummy responses."""

        words = text.strip().split()
        return " ".join(words[: max(1, min(max_tokens, len(words)))])

    def _transformers_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        """Generate text with a local HuggingFace causal language model."""

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Transformers backend has not been initialized.")

        import torch

        if seed is not None:
            torch.manual_seed(seed)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device is not None:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated[len(prompt) :].strip() or generated.strip()

    def _vllm_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        """Generate text with vLLM."""

        if self._vllm is None:
            raise RuntimeError("vLLM backend has not been initialized.")

        from vllm import SamplingParams

        params = SamplingParams(
            temperature=max(temperature, 1e-5),
            max_tokens=max_tokens,
            seed=seed,
        )
        outputs = self._vllm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def _openai_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        """Generate text through an OpenAI-compatible chat/completions API."""

        if self._openai_client is None:
            raise RuntimeError("OpenAI backend has not been initialized.")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.api_type == "chat":
                    request_kwargs: dict[str, Any] = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": max(temperature, 0.0),
                        "max_tokens": max_tokens,
                    }
                    if seed is not None:
                        request_kwargs["seed"] = seed
                    response = self._openai_client.chat.completions.create(**request_kwargs)
                    content = response.choices[0].message.content
                    return (content or "").strip()

                response = self._openai_client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=max(temperature, 0.0),
                    max_tokens=max_tokens,
                )
                return (response.choices[0].text or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"OpenAI API generation failed after {self.max_retries} attempts: {last_error}. "
            "If you see SSL/proxy errors, unset HTTP(S)_PROXY or set llm.trust_env: false in config."
        ) from last_error
