import math
import os
from dataclasses import dataclass
from time import monotonic

from .config import AIConfigurationError


# https://developers.openai.com/api/docs/models/gpt-6.2-astra
# Published short-context USD rates. Hosted search input is billed at model rates.
_SEARCH_USD = 0.01
# Reserve an estimated allowance for hosted input, which has no exact request-side cap.
HOSTED_INPUT_RESERVE = 50_000


def estimate_cost_usd(
    model: str, tier: str, input_tokens: int, output_tokens: int,
    cached_tokens: int | None, search_calls: int,
) -> float | None:
    if model != "gpt-6.2-astra":
        return None
    multipliers = {"default": 1, "fast": 2, "priority": 2}
    multiplier = multipliers.get(tier)
    if multiplier is None:
        return None
    cached = cached_tokens or 0
    input_multiplier = 2 if input_tokens > 272_000 else 1
    output_multiplier = 1.5 if input_tokens > 272_000 else 1
    return multiplier * (
        (8 * (input_tokens - cached) + 0.4 * cached) * input_multiplier
        + 40 * output_tokens * output_multiplier
    ) / 1_000_000 + _SEARCH_USD * search_calls


@dataclass
class EvaluationBudget:
    deadline: float
    max_estimated_usd: float
    accounted_usd: float = 0.0

    @classmethod
    def from_env(cls, start: float) -> "EvaluationBudget":
        try:
            timeout = float(os.getenv("AI_EVALUATION_TIMEOUT_SECONDS", "240"))
            max_cost = float(os.getenv("AI_EVALUATION_MAX_ESTIMATED_USD", "4"))
        except ValueError:
            raise AIConfigurationError("Invalid evaluation time or cost limit.") from None
        if any(not math.isfinite(value) or value <= 0 for value in (timeout, max_cost)):
            raise AIConfigurationError("Invalid evaluation time or cost limit.")
        return cls(start + timeout, max_cost)

    def remaining_seconds(self) -> float:
        return self.deadline - monotonic()

    def permits(self, reservation_usd: float | None) -> bool:
        return reservation_usd is not None and self.accounted_usd + reservation_usd <= self.max_estimated_usd

    def record(self, reservation_usd: float | None, observed_usd: float | None) -> None:
        assert reservation_usd is not None
        self.accounted_usd += observed_usd if observed_usd is not None else reservation_usd
