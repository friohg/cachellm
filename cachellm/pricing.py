"""Token counting and cost calculation.

Real provider usage numbers are always preferred.  When a provider omits usage
(common for streamed responses from OpenAI-compatible servers), tokens are
estimated - either with ``tiktoken`` if installed, or with a configurable
characters-per-token heuristic.

Pricing is per model and fully user-configured; nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .config import ModelPricing, PricingConfig
from .extract import message_text
from .logging_utils import get_logger

log = get_logger("cachellm.pricing")


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
        }


@dataclass(slots=True)
class CostBreakdown:
    model: str
    currency: str
    actual_cost: float
    cost_without_cache: float

    @property
    def savings(self) -> float:
        return max(0.0, self.cost_without_cache - self.actual_cost)

    @property
    def savings_pct(self) -> float:
        if self.cost_without_cache <= 0:
            return 0.0
        return 100.0 * self.savings / self.cost_without_cache

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "currency": self.currency,
            "actual_cost": round(self.actual_cost, 8),
            "cost_without_cache": round(self.cost_without_cache, 8),
            "savings": round(self.savings, 8),
            "savings_pct": round(self.savings_pct, 4),
        }


class TokenCounter:
    """Counts tokens, preferring provider-reported usage."""

    def __init__(self, cfg: PricingConfig) -> None:
        self.cfg = cfg
        self._encoders: dict[str, Any] = {}

    # -- provider usage ---------------------------------------------------
    @staticmethod
    def usage_from_response(payload: Mapping[str, Any] | None) -> TokenUsage | None:
        """Extract usage from a chat-completions or responses payload."""
        if not isinstance(payload, Mapping):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            return None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if prompt is None and completion is None:
            # /v1/responses naming
            prompt = usage.get("input_tokens")
            completion = usage.get("output_tokens")
        if prompt is None and completion is None:
            return None
        return TokenUsage(int(prompt or 0), int(completion or 0), estimated=False)

    # -- estimation -------------------------------------------------------
    def _tiktoken_encoder(self, model: str) -> Any | None:
        if self.cfg.token_estimator != "tiktoken":
            return None
        if model in self._encoders:
            return self._encoders[model]
        try:
            import tiktoken  # type: ignore
        except ImportError:
            log.debug("tiktoken not installed; using heuristic estimator")
            self._encoders[model] = None
            return None
        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover
                encoder = None
        self._encoders[model] = encoder
        return encoder

    def count_text(self, text: str, model: str = "") -> int:
        if not text:
            return 0
        encoder = self._tiktoken_encoder(model)
        if encoder is not None:
            try:
                return len(encoder.encode(text))
            except Exception:  # pragma: no cover
                pass
        chars_per_token = self.cfg.chars_per_token or 4.0
        return max(1, int(len(text) / chars_per_token + 0.5))

    def estimate_request_tokens(self, payload: Mapping[str, Any], model: str = "") -> int:
        parts: list[str] = []
        for field_name in ("instructions", "system", "prompt"):
            value = payload.get(field_name)
            if isinstance(value, str):
                parts.append(value)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raw_input = payload.get("input")
            messages = raw_input if isinstance(raw_input, list) else []
            if isinstance(raw_input, str):
                parts.append(raw_input)
        for message in messages or []:
            parts.append(message_text(message))
            if isinstance(message, Mapping) and message.get("tool_calls"):
                parts.append(str(message.get("tool_calls")))
        for tool in payload.get("tools") or []:
            parts.append(str(tool))
        overhead = 4 * len(messages or [])  # per-message framing
        return self.count_text("\n".join(p for p in parts if p), model) + overhead

    def estimate_response_tokens(self, payload: Mapping[str, Any], model: str = "") -> int:
        texts: list[str] = []
        for choice in payload.get("choices") or []:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message") or choice.get("delta") or {}
            texts.append(message_text(message))
            if isinstance(message, Mapping) and message.get("tool_calls"):
                texts.append(str(message["tool_calls"]))
        for item in payload.get("output") or []:  # /v1/responses
            texts.append(str(item))
        if isinstance(payload.get("output_text"), str):
            texts.append(payload["output_text"])
        return self.count_text("\n".join(t for t in texts if t), model)

    def usage_for(
        self,
        *,
        request: Mapping[str, Any],
        response: Mapping[str, Any] | None,
        model: str,
    ) -> TokenUsage:
        """Real usage when available, else an estimate."""
        real = self.usage_from_response(response)
        if real is not None and real.total_tokens > 0:
            return real
        return TokenUsage(
            prompt_tokens=self.estimate_request_tokens(request, model),
            completion_tokens=(
                self.estimate_response_tokens(response, model) if response else 0
            ),
            estimated=True,
        )


class CostCalculator:
    """Turns token counts into money using user-supplied per-model prices."""

    def __init__(self, cfg: PricingConfig) -> None:
        self.cfg = cfg

    @property
    def currency(self) -> str:
        return self.cfg.currency

    def pricing_for(self, model: str) -> ModelPricing:
        pricing = self.cfg.models.get(model)
        if pricing is not None:
            return pricing
        # Allow prefix matches so 'gpt-4o-mini-2024-07-18' inherits 'gpt-4o-mini'.
        best: ModelPricing | None = None
        best_len = -1
        for name, candidate in self.cfg.models.items():
            if model.startswith(name) and len(name) > best_len:
                best, best_len = candidate, len(name)
        return best or self.cfg.default

    def upsert(self, model: str, pricing: ModelPricing) -> None:
        self.cfg.models[model] = pricing

    def remove(self, model: str) -> bool:
        return self.cfg.models.pop(model, None) is not None

    def cost(self, model: str, usage: TokenUsage, *, cached: bool = False) -> float:
        pricing = self.pricing_for(model)
        if cached:
            return 0.0  # served locally: no upstream spend
        input_rate = pricing.input_per_1m / 1_000_000.0
        output_rate = pricing.output_per_1m / 1_000_000.0
        return usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate

    def breakdown(
        self, model: str, usage: TokenUsage, *, cached: bool
    ) -> CostBreakdown:
        without_cache = self.cost(model, usage, cached=False)
        actual = 0.0 if cached else without_cache
        return CostBreakdown(
            model=model,
            currency=self.currency,
            actual_cost=actual,
            cost_without_cache=without_cache,
        )

    def totals(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate a set of usage rows into a savings summary."""
        cost = 0.0
        without = 0.0
        for row in rows:
            cost += float(row.get("cost") or 0.0)
            without += float(row.get("cost_without_cache") or 0.0)
        savings = max(0.0, without - cost)
        return {
            "currency": self.currency,
            "actual_cost": round(cost, 6),
            "cost_without_cache": round(without, 6),
            "savings": round(savings, 6),
            "savings_pct": round(100.0 * savings / without, 3) if without > 0 else 0.0,
        }

    def table(self) -> list[dict[str, Any]]:
        return [
            {
                "model": model,
                "input_per_1m": p.input_per_1m,
                "output_per_1m": p.output_per_1m,
                "cached_input_per_1m": p.cached_input_per_1m,
            }
            for model, p in sorted(self.cfg.models.items())
        ]
