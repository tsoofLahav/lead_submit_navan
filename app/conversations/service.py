"""Workspace-scoped conversation and message persistence."""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic_core import to_jsonable_python

from ..data.access import WorkspaceData
from ..business.human_approval.service import RecommendationNotFound
from ..data.models import Conversation, Evaluation, Message, ProductActivity
from ..business.rules.tier_eligibility import is_actionable_tier_change

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


class ConversationNotFound(LookupError):
    pass


class ConversationPersistenceError(RuntimeError):
    def __init__(self, draft: MessageDraft, business_result=None):
        self.draft = draft
        self.business_result = business_result
        super().__init__("The prepared conversation message could not be saved.")


def _view(message: Message, evaluation: Evaluation | None = None) -> MessageView:
    metadata = dict(message.message_metadata or {})
    if metadata.get("kind") == MessageKind.EVALUATION:
        metadata["decision_status"] = evaluation.decision_status if evaluation is not None else None
        metadata["approval_available"] = (
            evaluation is not None and evaluation.decision_status == "pending"
            and is_actionable_tier_change(evaluation.current_tier, evaluation.recommended_tier)
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


class ConversationService:
    def __init__(self, sessions):
        self.sessions = sessions

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

    def save_user(self, workspace_id: UUID, conversation_id: UUID | None, content: str,
                  metadata: MessageMetadata, *, reuse_latest: bool = False) -> MessageView:
        if not isinstance(content, str) or not content.strip() or len(content) > MAX_MESSAGE_LENGTH or "\x00" in content:
            raise ValueError("Invalid user message.")
        with self.sessions.begin() as session:
            data = WorkspaceData(session, workspace_id)
            if conversation_id is None:
                recent = data.list_conversations(1) if reuse_latest else []
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

    def reply_context(self, workspace_id: UUID, conversation_id: UUID, user_message_id: int,
                      history_limit: int) -> tuple[MessageView, MessageView | None, list[MessageView]]:
        with self.sessions() as session:
            data = WorkspaceData(session, workspace_id)
            row = data.get_message(user_message_id)
            kind = (row.message_metadata or {}).get("kind", MessageKind.CHAT) if row is not None else None
            if row is None or row.conversation_id != conversation_id or row.role != "user" or kind != MessageKind.CHAT:
                raise ConversationNotFound("User chat message not found in this conversation.")
            user = _view(row)
            existing = data.conversation_message_by_metadata(conversation_id, "reply_to", str(user_message_id))
            if existing is not None:
                return user, _view(existing), []
            history = _views(
                data, data.conversation_messages(conversation_id, history_limit, before_id=user_message_id),
            ) if history_limit else []
            return user, None, history

    def workflow_retry_context(self, workspace_id: UUID, conversation_id: UUID,
                               failure_message_id: int) -> tuple[str | None, dict]:
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
            return operation, metadata

    def get_linked_evaluation(self, workspace_id: UUID, conversation_id: UUID, evaluation_id: UUID) -> Evaluation:
        with self.sessions() as session:
            return self._conversation_evaluation(WorkspaceData(session, workspace_id), conversation_id, evaluation_id)

    @staticmethod
    def _conversation_evaluation(data: WorkspaceData, conversation_id: UUID, evaluation_id: UUID) -> Evaluation:
        message = data.conversation_message_by_metadata(conversation_id, "evaluation_id", str(evaluation_id))
        evaluation = data.get_company_record(Evaluation, evaluation_id)
        if message is None or evaluation is None:
            raise RecommendationNotFound("Evaluation not found in this conversation.")
        return evaluation
