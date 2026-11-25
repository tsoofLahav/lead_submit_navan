"""Coordinate request operations and attach their results to conversations."""

import os
from dataclasses import asdict, dataclass, replace
from uuid import UUID

from pydantic import Field, field_validator
from pydantic_core import to_jsonable_python
from sqlalchemy.exc import SQLAlchemyError

from ..ai.client import AIClient, AIExecutionError, ExecutionMetadata
from ..ai.config import AIConfigurationError
from ..ai.progress import ProgressCallback
from ..ai.prompts import PromptRenderError, PromptRenderer, Task
from ..ai.schemas import StructuredOutput
from ..ai.tools import ToolError
from ..conversations.tools import build_chat_tools
from ..conversations.service import (
    ChatReplyMetadata, ConversationNotFound, ConversationPersistenceError, ConversationService,
    DecisionMetadata, MessageDraft, MessageKind, MessageMetadata, MessageView, WorkflowMetadata,
    WorkflowRequestMetadata,
)
from ..business.customer_reevaluation.workflow import CustomerReevaluation, CustomerWorkflowFailure
from ..business.lead_submission.workflow import (
    LeadEvaluationResult, LeadResult, LeadSubmission, LeadWorkflowFailure,
)
from ..business.human_approval.service import HumanApproval
from ..data.reference_data import PromptNotFound
from ..business.rules.validation import CustomerActivityInput, LeadInput


@dataclass(frozen=True)
class ChatTurn:
    user_message: MessageView
    assistant_message: MessageView
    execution: ExecutionMetadata | None


class ChatFailure(RuntimeError):
    def __init__(self, user_message: MessageView, cause: Exception):
        self.user_message = user_message
        self.reason = cause.kind.value if isinstance(cause, AIExecutionError) else type(cause).__name__
        self.execution = cause.metadata if isinstance(cause, AIExecutionError) else None
        super().__init__(f"Chat failed ({self.reason}).")


class ChatAnswer(StructuredOutput):
    answer: str = Field(min_length=1, max_length=8000)

    @field_validator("answer")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("An answer must contain text.")
        return value.strip()


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


class Orchestrator:
    def __init__(self, sessions, *, reference_sessions=None, evaluation_client=None, sdk=None, history_limit=None):
        self.reference_sessions = sessions if reference_sessions is None else reference_sessions
        self.conversations = ConversationService(sessions)
        self.leads = LeadSubmission(sessions, ai_client=evaluation_client, reference_sessions=self.reference_sessions)
        self.reevaluation = CustomerReevaluation(
            sessions, ai_client=evaluation_client, reference_sessions=self.reference_sessions,
        )
        self.approval = HumanApproval(sessions)
        self.tools = build_chat_tools(sessions, self.reference_sessions)
        self.client = AIClient(self.tools, sdk=sdk)
        self.history_limit = int(os.getenv("AI_CHAT_HISTORY_LIMIT", "20")) if history_limit is None else history_limit
        if type(self.history_limit) is not int or not 1 <= self.history_limit <= 100:
            raise ValueError("Chat history limit must be between 1 and 100.")

    def _persist(self, workspace_id: UUID, draft: MessageDraft, business_result=None) -> MessageView:
        try:
            return self.conversations.save_prepared_message(workspace_id, draft)
        except (SQLAlchemyError, ConversationNotFound):
            raise ConversationPersistenceError(draft, business_result) from None

    def reply_to_chat(self, workspace_id: UUID, conversation_id: UUID | None, content: str,
                     *, on_progress: ProgressCallback | None = None) -> ChatTurn:
        user = self.conversations.save_user(workspace_id, conversation_id, content, MessageMetadata(MessageKind.CHAT))
        return self.retry_message(workspace_id, user.conversation_id, user.id, on_progress=on_progress)

    def retry_message(self, workspace_id: UUID, conversation_id: UUID, user_message_id: int,
                      *, on_progress: ProgressCallback | None = None) -> ChatTurn:
        user, existing, history = self.conversations.reply_context(
            workspace_id, conversation_id, user_message_id, self.history_limit - 1,
        )
        if existing is not None:
            return ChatTurn(user, existing, None)
        try:
            with self.reference_sessions() as session:
                prompt = PromptRenderer(session, self.tools).render(
                    Task.CHAT, policy="", inputs={"message": user.content, "history": [_chat_context(item) for item in history]},
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
                              "lead_submit", self.leads.submit_lead, lead, on_progress)

    def submit_customer_reevaluation(self, workspace_id: UUID, conversation_id: UUID | None,
                                     activity: CustomerActivityInput,
                                     *, on_progress: ProgressCallback | None = None) -> MessageView:
        return self._workflow(workspace_id, conversation_id, f"Update customer activity for {activity.company_name}",
                              "re_evaluation", self.reevaluation.submit_customer_reevaluation, activity, on_progress)

    def retry_workflow(self, workspace_id: UUID, conversation_id: UUID, failure_message_id: int,
                       *, on_progress: ProgressCallback | None = None) -> MessageView:
        operation, metadata = self.conversations.workflow_retry_context(
            workspace_id, conversation_id, failure_message_id,
        )
        if operation == "lead_submit" and metadata.get("company_id"):
            run, value = self.leads.retry_lead_evaluation, UUID(metadata["company_id"])
        elif operation == "re_evaluation" and metadata.get("activity_id"):
            run, value = self.reevaluation.retry_customer_reevaluation, UUID(metadata["activity_id"])
        else:
            raise ValueError("This workflow failure cannot be retried.")

        def mark_retry(draft: MessageDraft) -> MessageDraft:
            if not isinstance(draft.metadata, WorkflowMetadata):
                return draft
            return replace(draft, metadata=replace(draft.metadata, retried_failure_id=failure_message_id))

        return self._workflow(workspace_id, conversation_id, "Retry evaluation", operation, run, value, on_progress, mark_retry,
                              correlation_id=metadata.get("result", {}).get("evaluation_correlation_id"))

    def _workflow(self, workspace_id, conversation_id, content, operation, run, value, on_progress, prepare=None, *,
                  correlation_id=None) -> MessageView:
        user = self.conversations.save_user(
            workspace_id, conversation_id, content,
            WorkflowRequestMetadata(MessageKind.WORKFLOW_REQUEST, operation), reuse_latest=True,
        )
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
        self.conversations.get_linked_evaluation(workspace_id, conversation_id, evaluation_id)
        (self.approval.accept if accept else self.approval.reject)(workspace_id, evaluation_id)
        return self.record_decision_confirmation(workspace_id, conversation_id, evaluation_id)

    def record_decision_confirmation(self, workspace_id: UUID, conversation_id: UUID, evaluation_id: UUID) -> MessageView:
        evaluation = self.conversations.get_linked_evaluation(workspace_id, conversation_id, evaluation_id)
        status = evaluation.decision_status
        if status not in {"accepted", "rejected"}:
            raise ValueError("Only a recorded acceptance or rejection can be confirmed.")
        text = (f"Recommendation accepted. Tier changed from {evaluation.current_tier} to {evaluation.recommended_tier}."
                if status == "accepted" else "Recommendation rejected. This decision did not change the company tier.")
        draft = MessageDraft(conversation_id, text, DecisionMetadata(
            MessageKind.DECISION, evaluation_id, f"{evaluation_id}:{status}", status,
        ))
        return self._persist(workspace_id, draft, {"evaluation_id": evaluation_id, "status": status})
