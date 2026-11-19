import math
import os
from dataclasses import dataclass
from enum import StrEnum


class Component(StrEnum):
    RESEARCH = "research"
    PREDICTION = "prediction"
    DECISION = "decision"
    CHAT = "chat"


class AIConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class AISettings:
    model: str
    reasoning_effort: str
    service_tier: str
    max_output_tokens: int
    timeout: float
    max_tool_calls: int | None
    search_context_size: str
    chat_tool_budget: int = 6
    chat_max_rounds: int = 8

    @classmethod
    def from_env(cls, component: Component) -> "AISettings":
        prefix = f"AI_{component.value.upper()}"
        research = component == Component.RESEARCH
        model = os.getenv(f"{prefix}_MODEL", "gpt-6.2-astra").strip()
        effort = os.getenv(f"{prefix}_REASONING_EFFORT", "medium" if research else "low")
        tier = os.getenv(f"{prefix}_SERVICE_TIER", os.getenv("AI_SERVICE_TIER", "fast"))
        context = os.getenv("AI_RESEARCH_SEARCH_CONTEXT_SIZE", "low") if research else "medium"
        if not model or effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise AIConfigurationError("Invalid AI model or reasoning effort.")
        if model == "gpt-6.2-astra" and effort in {"none", "minimal"}:
            raise AIConfigurationError("GPT-6.2 Astra requires reasoning effort of low or higher.")
        if tier not in {"auto", "default", "fast", "priority", "flex", "scale"}:
            raise AIConfigurationError("Invalid AI service tier.")
        if context not in {"low", "medium", "high"}:
            raise AIConfigurationError("Invalid research search context size.")
        try:
            default_output = {Component.RESEARCH: 8000, Component.PREDICTION: 2000, Component.DECISION: 2000}
            output = int(os.getenv(f"{prefix}_MAX_OUTPUT_TOKENS", str(default_output.get(component, 25000))))
            timeout = float(os.getenv("AI_TIMEOUT_SECONDS", "120"))
            budget = int(os.getenv("AI_RESEARCH_MAX_TOOL_CALLS", "5")) if research else None
            chat_budget = int(os.getenv("AI_CHAT_TOOL_BUDGET", "6")) if component == Component.CHAT else 6
            chat_rounds = int(os.getenv("AI_CHAT_MAX_ROUNDS", "8")) if component == Component.CHAT else 8
        except ValueError:
            raise AIConfigurationError("Invalid AI output, timeout, or search budget.") from None
        if output < 16 or not math.isfinite(timeout) or timeout <= 0 or (budget is not None and not 1 <= budget <= 5):
            raise AIConfigurationError("Invalid AI output, timeout, or search budget.")
        if not 1 <= chat_budget <= 30 or not 2 <= chat_rounds <= 32:
            raise AIConfigurationError("Invalid chat tool budget or round limit.")
        return cls(model, effort, tier, output, timeout, budget, context, chat_budget, chat_rounds)
