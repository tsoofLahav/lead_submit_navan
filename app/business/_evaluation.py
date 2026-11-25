from datetime import UTC, datetime
from types import MappingProxyType

from sqlalchemy.orm import Session, sessionmaker

from ..ai.client import AIClient
from ..ai.config import Component
from ..ai.progress import ProgressCallback, ProgressEvent, ProgressKind
from ..ai.prompts import PromptRenderer, Task
from ..data.reference_data import get_business_policy, get_prompt


class EvaluationWorkflow:
    def __init__(
        self, sessions: sessionmaker[Session], *, ai_client: AIClient | None = None,
        reference_sessions: sessionmaker[Session] | None = None,
    ):
        self.sessions = sessions
        self.reference_sessions = reference_sessions if reference_sessions is not None else sessions
        self.ai_client = ai_client

    @staticmethod
    def _emit(
        callback: ProgressCallback | None, kind: ProgressKind, component: Component | None = None,
        *, stage: str | None = None, tier: int | None = None, gate_status: str | None = None,
    ) -> None:
        if callback is not None:
            callback(ProgressEvent(
                kind, component, datetime.now(UTC), workflow_stage=stage, initial_tier=tier, gate_status=gate_status,
            ))

    def _ai_context(self, client: AIClient, system_key: str, *tasks: Task, policy: str | None = None) -> tuple[str, PromptRenderer]:
        with self.reference_sessions() as session:
            if policy is None:
                policy = get_business_policy(session)
            templates = MappingProxyType({key: get_prompt(session, key) for key in (system_key, *tasks)})
        return policy, PromptRenderer(None, client.tools, templates=templates)
