import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, field_validator
from pydantic_core import to_jsonable_python
from sqlalchemy.exc import SQLAlchemyError

from .ai.client import AIClient, AIExecutionError, ExecutionMetadata
from .ai.config import AIConfigurationError, AISettings, Component
from .ai.progress import ProgressCallback
from .ai.prompts import PromptRenderError, PromptRenderer, Task
from .ai.schemas import StructuredOutput
from .ai.tools import ToolError
from .chat_tools import build_chat_tools
from .data_access import WorkspaceData
from .human_approval import HumanApproval, RecommendationNotFound
from .models import Conversation, Evaluation, Message, ProductActivity
from .orchestrator import (
    CustomerWorkflowFailure, LeadEvaluationResult, LeadResult, LeadWorkflowFailure, Orchestrator,
)
from .reference_data import PromptNotFound
from .validation import CustomerActivityInput, LeadInput

DEFAULT_TITLE = "New conversation"
MAX_MESSAGE_LENGTH = 8000


class MessageKind(StrEnum):
    CHAT = "chat"
    WORKFLOW_REQUEST = "workflow_request"
    EVALUATION = "evaluation_result"
    ACTIVITY = "activity_update"
    WORKFLOW_FAILURE = "workflow_failure"
    DECISION = "decision_confirmation"


@dataclass(frozen=True)
class MessageMetadata:
    kind: MessageKind


@dataclass(frozen=True)
class ChatReplyMetadata(MessageMetadata):
    reply_to: int


@dataclass(frozen=True)
class WorkflowRequestMetadata(MessageMetadata):
    operation: str


@dataclass(frozen=True)
class WorkflowMetadata(ChatReplyMetadata):
    company_id: UUID | None
    evaluation_id: UUID | None
    activity_id: UUID | None
    result: dict
    retried_failure_id: int | None = None


@dataclass(frozen=True)
class DecisionMetadata(MessageMetadata):
    decided_evaluation_id: UUID
    confirmation_key: str
    status: str


@dataclass(frozen=True)
class ConversationView:
    id: UUID
    title: str
    created_at: datetime


@dataclass(frozen=True)
class MessageView:
    id: int
    conversation_id: UUID
    role: str
    content: str
    metadata: dict
    created_at: datetime


@dataclass(frozen=True)
class MessageDraft:
    conversation_id: UUID
    content: str
    metadata: MessageMetadata


@dataclass(frozen=True)
class ChatTurn:
    user_message: MessageView
    assistant_message: MessageView
    execution: ExecutionMetadata | None


class ConversationNotFound(LookupError):
    pass


class ChatFailure(RuntimeError):
    def __init__(self, user_message: MessageView, cause: Exception):
        self.user_message = user_message
        self.reason = cause.kind.value if isinstance(cause, AIExecutionError) else type(cause).__name__
        self.execution = cause.metadata if isinstance(cause, AIExecutionError) else None
        super().__init__(f"Chat failed ({self.reason}).")


class ConversationPersistenceError(RuntimeError):
    def __init__(self, draft: MessageDraft, business_result=None):
        self.draft = draft
        self.business_result = business_result
        super().__init__("The prepared conversation message could not be saved.")


class ChatAnswer(StructuredOutput):
    answer: str = Field(min_length=1, max_length=8000)

    @field_validator("answer")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("An answer must contain text.")
        return value.strip()


def _view(message: Message, evaluation: Evaluation | None = None) -> MessageView:
    metadata = dict(message.message_metadata or {})
    if metadata.get("kind") == MessageKind.EVALUATION:
        metadata["decision_status"] = evaluation.decision_status if evaluation is not None else None
        metadata["approval_available"] = (
            evaluation is not None and evaluation.decision_status == "pending"
            and HumanApproval.is_actionable_tier_change(evaluation.current_tier, evaluation.recommended_tier)
        )
    return MessageView(message.id, message.conversation_id, message.role, message.content, metadata, message.created_at)


def _evaluation_id(message: Message) -> UUID | None:
    value = (message.message_metadata or {}).get("evaluation_id")
    try:
        return UUID(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _views(data: WorkspaceData, messages: list[Message]) -> list[MessageView]:
    evaluations = data.evaluations_by_id({key for row in messages if (key := _evaluation_id(row)) is not None})
    return [_view(row, evaluations.get(_evaluation_id(row))) for row in messages]


def _chat_context(row: MessageView) -> dict:
    metadata = row.metadata
    result = metadata.get("result", {})
    selected = {}
    if metadata.get("kind") in {MessageKind.EVALUATION, MessageKind.ACTIVITY}:
        selected = {
            key: metadata.get(key) for key in ("kind", "company_id", "evaluation_id", "activity_id",
                                              "decision_status", "approval_available")
        }
        selected.update({
            key: result[key] for key in ("initial_tier", "current_tier", "gate", "recommendation")
            if key in result
        })
    return {"role": row.role, "content": row.content, "metadata": selected}


def _routed(callback: ProgressCallback | None, conversation_id: UUID) -> ProgressCallback | None:
    if callback is None:
        return None

    def forward(event):
        callback(replace(event, conversation_id=conversation_id))

    return forward


class ChatAssistant:
    def __init__(self, sessions, *, reference_sessions=None, orchestrator=None, sdk=None, history_limit=None):
        self.sessions = sessions
        self.reference_sessions = sessions if reference_sessions is None else reference_sessions
        self.orchestrator = orchestrator if orchestrator is not None else Orchestrator(
            sessions, reference_sessions=self.reference_sessions,
        )
        self.tools = build_chat_tools(sessions, self.reference_sessions)
        self.client = AIClient(self.tools, sdk=sdk)
        self.history_limit = int(os.getenv("AI_CHAT_HISTORY_LIMIT", "20")) if history_limit is None else history_limit
        if type(self.history_limit) is not int or not 1 <= self.history_limit <= 100:
            raise ValueError("Chat history limit must be between 1 and 100.")

    @staticmethod
    def _conversation(data: WorkspaceData, conversation_id: UUID) -> Conversation:
        conversation = data.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFound("Conversation not found in this workspace.")
        return conversation

    def create_conversation(self, workspace_id: UUID) -> ConversationView:
        with self.sessions.begin() as session:
            conversation = Conversation(workspace_id=workspace_id)
            session.add(conversation)
            session.flush()
            return ConversationView(conversation.id, conversation.title, conversation.created_at)

    def get_conversation(self, workspace_id: UUID, conversation_id: UUID) -> ConversationView:
        with self.sessions() as session:
            row = self._conversation(WorkspaceData(session, workspace_id), conversation_id)
            return ConversationView(row.id, row.title, row.created_at)

    def rename_conversation(self, workspace_id: UUID, conversation_id: UUID, title: str) -> ConversationView:
        title = " ".join(title.split())
        if not title or len(title) > 80:
            raise ValueError("Conversation title must contain 1 to 80 characters.")
        with self.sessions.begin() as session:
            row = self._conversation(WorkspaceData(session, workspace_id), conversation_id)
            row.title = title
            return ConversationView(row.id, row.title, row.created_at)

    def list_conversations(self, workspace_id: UUID, limit: int = 50) -> list[ConversationView]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Conversation limit must be between 1 and 100.")
        with self.sessions() as session:
            return [ConversationView(row.id, row.title, row.created_at)
                    for row in WorkspaceData(session, workspace_id).list_conversations(limit)]

    def list_messages(self, workspace_id: UUID, conversation_id: UUID, limit: int = 50,
                      before_id: int | None = None) -> list[MessageView]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Message limit must be between 1 and 100.")
        with self.sessions() as session:
            data = WorkspaceData(session, workspace_id)
            self._conversation(data, conversation_id)
            return _views(data, data.conversation_messages(conversation_id, limit, before_id))

    def _save_user(self, workspace_id: UUID, conversation_id: UUID | None, content: str,
                   metadata: MessageMetadata) -> MessageView:
        if not isinstance(content, str) or not content.strip() or len(content) > MAX_MESSAGE_LENGTH or "\x00" in content:
            raise ValueError("Invalid user message.")
        with self.sessions.begin() as session:
            data = WorkspaceData(session, workspace_id)
            if conversation_id is None:
                recent = data.list_conversations(1) if isinstance(metadata, WorkflowRequestMetadata) else []
                if recent:
                    conversation = recent[0]
                else:
                    conversation = Conversation(workspace_id=workspace_id, title=DEFAULT_TITLE)
                    session.add(conversation)
                    session.flush()
            else:
                conversation = self._conversation(data, conversation_id)
            if conversation.title == DEFAULT_TITLE and not data.conversation_messages(conversation.id, 1):
                conversation.title = " ".join(content.split())[:80].rstrip()
            row = Message(conversation_id=conversation.id, role="user", content=content.strip(),
                          message_metadata=to_jsonable_python(asdict(metadata)))
            session.add(row)
            session.flush()
            return _view(row)

    def save_prepared_message(self, workspace_id: UUID, draft: MessageDraft) -> MessageView:
        """Retry a message write without rerunning AI, approval, or a business workflow."""
        with self.sessions.begin() as session:
            data = WorkspaceData(session, workspace_id)
            self._conversation(data, draft.conversation_id)
            metadata = draft.metadata
            if isinstance(metadata, WorkflowMetadata):
                if metadata.company_id is not None and data.get_company(metadata.company_id) is None:
                    raise ConversationNotFound("Workflow company not found in this workspace.")
                for model, record_id in ((Evaluation, metadata.evaluation_id), (ProductActivity, metadata.activity_id)):
                    if record_id is not None:
                        record = data.get_company_record(model, record_id)
                        if record is None or record.company_id != metadata.company_id:
                            raise ConversationNotFound("Workflow result not found for this company.")
            if isinstance(metadata, ChatReplyMetadata):
                request = data.get_message(metadata.reply_to)
                if request is None or request.role != "user" or request.conversation_id != draft.conversation_id:
                    raise ConversationNotFound("Original user message not found in this conversation.")
                key, value = "reply_to", str(metadata.reply_to)
            elif isinstance(metadata, DecisionMetadata):
                if not data.conversation_message_by_metadata(
                    draft.conversation_id, "evaluation_id", str(metadata.decided_evaluation_id),
                ):
                    raise RecommendationNotFound("Evaluation not found in this conversation.")
                key, value = "confirmation_key", metadata.confirmation_key
            else:
                raise ValueError("Invalid assistant message metadata.")
            existing = data.conversation_message_by_metadata(draft.conversation_id, key, value)
            if existing is not None:
                return _views(data, [existing])[0]
            row = Message(conversation_id=draft.conversation_id, role="assistant", content=draft.content,
                          message_metadata=to_jsonable_python(asdict(metadata)))
            session.add(row)
            session.flush()
            return _views(data, [row])[0]

    def _persist(self, workspace_id: UUID, draft: MessageDraft, business_result=None) -> MessageView:
        try:
            return self.save_prepared_message(workspace_id, draft)
        except (SQLAlchemyError, ConversationNotFound):
            raise ConversationPersistenceError(draft, business_result) from None

    def send_message(self, workspace_id: UUID, conversation_id: UUID | None, content: str,
                     *, on_progress: ProgressCallback | None = None) -> ChatTurn:
        user = self._save_user(workspace_id, conversation_id, content, MessageMetadata(MessageKind.CHAT))
        return self.retry_message(workspace_id, user.conversation_id, user.id, on_progress=on_progress)

    def retry_message(self, workspace_id: UUID, conversation_id: UUID, user_message_id: int,
                      *, on_progress: ProgressCallback | None = None) -> ChatTurn:
        with self.sessions() as session:
            data = WorkspaceData(session, workspace_id)
            row = data.get_message(user_message_id)
            kind = (row.message_metadata or {}).get("kind", MessageKind.CHAT) if row is not None else None
            if row is None or row.conversation_id != conversation_id or row.role != "user" or kind != MessageKind.CHAT:
                raise ConversationNotFound("User chat message not found in this conversation.")
            user = _view(row)
            existing = data.conversation_message_by_metadata(conversation_id, "reply_to", str(user_message_id))
            if existing is not None:
                return ChatTurn(user, _view(existing), None)
            history = (
                [_chat_context(item) for item in _views(
                    data, data.conversation_messages(conversation_id, self.history_limit - 1, before_id=user_message_id),
                )] if self.history_limit > 1 else []
            )
        try:
            with self.reference_sessions() as session:
                prompt = PromptRenderer(session, self.tools).render(
                    Task.CHAT, policy="", inputs={"message": user.content, "history": history},
                    settings=AISettings.from_env(Component.CHAT),
                )
            result = self.client.execute(
                prompt, chat_schema=ChatAnswer, workspace_id=workspace_id,
                on_progress=_routed(on_progress, conversation_id),
            )
        except (AIExecutionError, AIConfigurationError, PromptNotFound, PromptRenderError, SQLAlchemyError, ToolError) as error:
            raise ChatFailure(user, error) from None
        draft = MessageDraft(conversation_id, result.output.answer, ChatReplyMetadata(MessageKind.CHAT, user.id))
        return ChatTurn(user, self._persist(workspace_id, draft), result.metadata)

    def submit_lead(self, workspace_id: UUID, conversation_id: UUID | None, lead: LeadInput,
                    *, on_progress: ProgressCallback | None = None) -> MessageView:
        return self._workflow(workspace_id, conversation_id, f"Submit lead for {lead.company_name}",
                              "lead_submit", self.orchestrator.submit_lead, lead, on_progress)

    def submit_customer_reevaluation(self, workspace_id: UUID, conversation_id: UUID | None,
                                     activity: CustomerActivityInput,
                                     *, on_progress: ProgressCallback | None = None) -> MessageView:
        return self._workflow(workspace_id, conversation_id, f"Update customer activity for {activity.company_name}",
                              "re_evaluation", self.orchestrator.submit_customer_reevaluation, activity, on_progress)

    def retry_workflow(self, workspace_id: UUID, conversation_id: UUID, failure_message_id: int,
                       *, on_progress: ProgressCallback | None = None) -> MessageView:
        with self.sessions() as session:
            data = WorkspaceData(session, workspace_id)
            message = data.get_message(failure_message_id)
            metadata = (message.message_metadata or {}) if message is not None else {}
            if message is None or message.conversation_id != conversation_id or metadata.get("kind") != MessageKind.WORKFLOW_FAILURE:
                raise ConversationNotFound("Workflow failure not found in this conversation.")
            detail = metadata.get("result", {})
            if detail.get("stage") == "evaluation_persistence" or str(detail.get("reason", "")).startswith("stale_"):
                raise ValueError("This workflow failure cannot be safely retried.")
            request = data.get_message(metadata["reply_to"])
            if request is None or request.conversation_id != conversation_id:
                raise ConversationNotFound("Original workflow request not found.")
            if data.conversation_message_by_metadata(conversation_id, "retried_failure_id", str(failure_message_id)):
                raise ValueError("This failure already has a saved retry result.")
            operation = (request.message_metadata or {}).get("operation")
            if operation == "lead_submit" and metadata.get("company_id"):
                run, value = self.orchestrator.retry_lead_evaluation, UUID(metadata["company_id"])
            elif operation == "re_evaluation" and metadata.get("activity_id"):
                run, value = self.orchestrator.retry_customer_reevaluation, UUID(metadata["activity_id"])
            else:
                raise ValueError("This workflow failure cannot be retried.")

        def mark_retry(draft: MessageDraft) -> MessageDraft:
            if not isinstance(draft.metadata, WorkflowMetadata):
                return draft
            return replace(draft, metadata=replace(draft.metadata, retried_failure_id=failure_message_id))

        return self._workflow(workspace_id, conversation_id, "Retry evaluation", operation, run, value, on_progress, mark_retry,
                              correlation_id=detail.get("evaluation_correlation_id"))

    def _workflow(self, workspace_id, conversation_id, content, operation, run, value, on_progress, prepare=None, *,
                  correlation_id=None) -> MessageView:
        user = self._save_user(workspace_id, conversation_id, content,
                               WorkflowRequestMetadata(MessageKind.WORKFLOW_REQUEST, operation))
        try:
            retry_context = {"correlation_id": correlation_id} if correlation_id else {}
            result = run(workspace_id, value, on_progress=_routed(on_progress, user.conversation_id), **retry_context)
        except (LeadWorkflowFailure, CustomerWorkflowFailure) as failure:
            diagnostic = failure.executions[-1].validation_diagnostic if failure.executions else None
            metadata = WorkflowMetadata(
                MessageKind.WORKFLOW_FAILURE, user.id, failure.company_id, None, getattr(failure, "activity_id", None),
                {"stage": failure.stage, "reason": failure.reason, "diagnostic": asdict(diagnostic) if diagnostic else None,
                 "research_id": getattr(failure, "research_id", None),
                 "evaluation_correlation_id": getattr(failure, "evaluation_correlation_id", None)},
            )
            draft = MessageDraft(user.conversation_id, str(failure), metadata)
            failure.conversation_message = self._persist(workspace_id, prepare(draft) if prepare else draft, failure)
            raise
        draft = self._workflow_draft(user, result)
        return self._persist(workspace_id, prepare(draft) if prepare else draft, result)

    @staticmethod
    def _workflow_draft(user: MessageView, result) -> MessageDraft:
        if isinstance(result, LeadResult):
            text = f"The company is already registered in Tier {result.current_tier}."
            metadata = WorkflowMetadata(MessageKind.ACTIVITY, user.id, result.company_id, None, None,
                                        {"duplicate_lead": True, "current_tier": result.current_tier})
            return MessageDraft(user.conversation_id, text, metadata)
        lead = isinstance(result, LeadEvaluationResult)
        recommendation = result.recommendation
        name = result.company.company_name if lead else result.customer.company_name
        tier = result.current_tier if lead else result.customer.current_tier
        if recommendation is None:
            text = f"{name}: Activity is unchanged. No new evaluation was needed."
        else:
            outcome = (f"{recommendation.outcome.capitalize()} Tier {recommendation.recommended_tier}"
                       if recommendation.recommended_tier is not None else "Recommendation undetermined")
            text = f"{name} is in Tier {tier}. {outcome} ({recommendation.confidence} confidence). {recommendation.rationale}"
        body = {
            "company_name": name, "current_tier": tier, "predictions": result.predictions,
            "recommendation": recommendation, "evaluation_type": "lead_submit" if lead else "re_evaluation",
        }
        if lead:
            body.update(initial_tier=result.initial_tier, research_id=result.research_id, research=result.research)
        else:
            body["gate"] = result.gate
            body["unchanged"] = result.unchanged
            body["submitted_activity"] = result.customer
        metadata = WorkflowMetadata(
            MessageKind.EVALUATION if result.evaluation_id is not None else MessageKind.ACTIVITY,
            user.id, result.company_id, result.evaluation_id, None if lead else result.activity_id,
            to_jsonable_python(body),
        )
        return MessageDraft(user.conversation_id, text, metadata)

    def decide_recommendation(self, workspace_id: UUID, conversation_id: UUID, evaluation_id: UUID,
                              *, accept: bool) -> MessageView:
        with self.sessions() as session:
            self._conversation_evaluation(WorkspaceData(session, workspace_id), conversation_id, evaluation_id)
        approval = HumanApproval(self.sessions)
        (approval.accept if accept else approval.reject)(workspace_id, evaluation_id)
        return self.record_decision_confirmation(workspace_id, conversation_id, evaluation_id)

    @staticmethod
    def _conversation_evaluation(data: WorkspaceData, conversation_id: UUID, evaluation_id: UUID) -> Evaluation:
        message = data.conversation_message_by_metadata(conversation_id, "evaluation_id", str(evaluation_id))
        evaluation = data.get_company_record(Evaluation, evaluation_id)
        if message is None or evaluation is None:
            raise RecommendationNotFound("Evaluation not found in this conversation.")
        return evaluation

    def record_decision_confirmation(self, workspace_id: UUID, conversation_id: UUID, evaluation_id: UUID) -> MessageView:
        with self.sessions() as session:
            evaluation = self._conversation_evaluation(WorkspaceData(session, workspace_id), conversation_id, evaluation_id)
            status = evaluation.decision_status
            if status not in {"accepted", "rejected"}:
                raise ValueError("Only a recorded acceptance or rejection can be confirmed.")
            text = (f"Recommendation accepted. Tier changed from {evaluation.current_tier} to {evaluation.recommended_tier}."
                    if status == "accepted" else "Recommendation rejected. This decision did not change the company tier.")
        draft = MessageDraft(conversation_id, text, DecisionMetadata(
            MessageKind.DECISION, evaluation_id, f"{evaluation_id}:{status}", status,
        ))
        return self._persist(workspace_id, draft, {"evaluation_id": evaluation_id, "status": status})
