from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from time import monotonic
from uuid import UUID

from .config import Component


class ProgressKind(StrEnum):
    COMPANY_REGISTERED = "company_registered"
    INITIAL_TIER_ASSIGNED = "initial_tier_assigned"
    OUTPUT_CORRECTION_STARTED = "output_correction_started"
    CHAT_TOOL_STARTED = "chat_tool_started"
    CHAT_TOOL_COMPLETED = "chat_tool_completed"
    CHAT_RESPONSE_STREAMING = "chat_response_streaming"
    CUSTOMER_REGISTERED = "customer_registered"
    CUSTOMER_UPDATED = "customer_updated"
    CUSTOMER_ACTIVITY_PERSISTED = "customer_activity_persisted"
    REEVALUATION_GATE_ASSESSED = "reevaluation_gate_assessed"
    EVALUATION_COMPLETED = "evaluation_completed"
    RESEARCH_STARTED = "research_started"
    WEB_SEARCH_STARTED = "web_search_started"
    WEB_SEARCH_COMPLETED = "web_search_completed"
    RESEARCH_COMPLETED = "research_completed"
    PREDICTION_STARTED = "prediction_started"
    PREDICTION_COMPLETED = "prediction_completed"
    RECOMMENDATION_STARTED = "recommendation_started"
    RECOMMENDATION_COMPLETED = "recommendation_completed"
    EVALUATION_FAILED = "evaluation_failed"


@dataclass(frozen=True)
class ProgressEvent:
    kind: ProgressKind
    component: Component | None
    timestamp: datetime
    tool_call_id: str | None = None
    completed_tool_calls: int = 0
    workflow_stage: str | None = None
    initial_tier: int | None = None
    gate_status: str | None = None
    conversation_id: UUID | None = None
    tool_name: str | None = None
    tool_status: str | None = None
    response_delta: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]


class SearchProgress:
    def __init__(self, callback: ProgressCallback | None):
        self.callback = callback
        self.started: set[str] = set()
        self.completed: set[str] = set()
        self.started_at: dict[str, float] = {}
        self.completed_at: dict[str, float] = {}

    def observe(self, kind: ProgressKind, item_id: str, *, streamed: bool = True) -> None:
        seen = self.started if kind == ProgressKind.WEB_SEARCH_STARTED else self.completed
        if item_id in seen:
            return
        seen.add(item_id)
        if streamed:
            timestamps = self.started_at if seen is self.started else self.completed_at
            timestamps[item_id] = monotonic()
        if self.callback is not None:
            self.callback(ProgressEvent(kind, Component.RESEARCH, datetime.now(UTC), item_id, len(self.completed)))
