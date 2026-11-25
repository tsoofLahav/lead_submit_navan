"""Serialize public API data and map application errors to HTTP responses."""

from dataclasses import asdict

from pydantic_core import to_jsonable_python
from werkzeug.exceptions import HTTPException

from ..conversations.service import ConversationNotFound
from ..business.customer_reevaluation.workflow import CustomerWorkflowFailure
from ..business.lead_submission.workflow import LeadWorkflowFailure
from ..business.human_approval.service import RecommendationConflict, RecommendationNotFound
from ..business.rules.validation import ValidationError
from ..data.company_lock import CompanyBusy
from .actions import ChatFailure


class RequestError(ValueError):
    pass


def json_value(value):
    return to_jsonable_python(asdict(value))


def public_message(message):
    payload = json_value(message)
    payload["id"] = str(message.id)
    metadata = payload["metadata"]
    if metadata.get("kind") == "workflow_failure":
        detail = metadata.get("result", {})
        metadata["result"] = {
            "stage": detail.get("stage"),
            "can_retry": _workflow_retryable(detail.get("stage"), detail.get("reason"))
            and bool(metadata.get("company_id") or metadata.get("activity_id")),
        }
        payload["content"] = "The evaluation stopped. Any saved company or customer activity was preserved."
    for field in ("reply_to", "retried_failure_id"):
        if metadata.get(field) is not None:
            metadata[field] = str(metadata[field])
    return payload


def _workflow_retryable(stage, reason):
    return (
        stage in {"context", "research", "research_persistence", "prediction", "recommendation"}
        and not str(reason).startswith("stale_")
        and reason not in {"evaluation_already_completed", "activity_evaluation_in_progress", "lead_not_found", "activity_not_found"}
    )


def error_payload(error):
    if isinstance(error, (RequestError, ValidationError)):
        return {"code": "invalid_input", "message": str(error)}, 400
    if isinstance(error, (ConversationNotFound, RecommendationNotFound)):
        return {"code": "not_found", "message": "This record is unavailable in the current workspace."}, 404
    if isinstance(error, CompanyBusy):
        return {"code": "busy", "message": str(error)}, 409
    if isinstance(error, RecommendationConflict):
        return {"code": "stale_recommendation", "message": "This recommendation is no longer actionable. Refresh to see its current status."}, 409
    if isinstance(error, ChatFailure):
        return {"code": "chat_failed", "message": "The assistant could not answer. Your message was saved.",
                "retry": {"kind": "chat", "message_id": str(error.user_message.id)}}, 502
    if isinstance(error, (LeadWorkflowFailure, CustomerWorkflowFailure)):
        if error.reason == "CompanyConflict":
            return {"code": "company_conflict", "message": "The company name and domain conflict with an existing company."}, 409
        stale = error.reason.startswith("stale_") or error.reason in {"evaluation_already_completed"}
        info = {"code": "stale_workflow" if stale else "workflow_failed",
                "message": "The saved evaluation context is no longer current." if stale else
                "The evaluation stopped. Any saved company or customer activity was preserved."}
        message = getattr(error, "conversation_message", None)
        if _workflow_retryable(error.stage, error.reason) and message is not None and (
            isinstance(error, LeadWorkflowFailure) and error.company_id is not None
            or isinstance(error, CustomerWorkflowFailure) and error.activity_id is not None
        ):
            info["retry"] = {"kind": "workflow", "message_id": str(message.id)}
        return info, 409 if stale else 502
    if isinstance(error, HTTPException):
        return {"code": "request_error", "message": error.description}, error.code
    return {"code": "unavailable", "message": "The operation could not be completed. Refresh the conversation before retrying."}, 503
