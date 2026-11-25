"""Run operations in a worker and stream progress and terminal events."""

import json
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from secrets import token_urlsafe
from threading import Lock, Thread
from time import monotonic
from uuid import UUID

from flask import Response, current_app

from ..conversations.service import ConversationPersistenceError
from ..business.customer_reevaluation.workflow import CustomerWorkflowFailure
from ..business.lead_submission.workflow import LeadWorkflowFailure
from .actions import ChatFailure
from .responses import error_payload, json_value, public_message


@dataclass(frozen=True)
class RequestContext:
    workspace_id: UUID
    conversation_id: UUID


class OperationState:
    def __init__(self):
        self.lock = Lock()
        self.active = set()
        self.drafts = {}

    def start(self, key):
        with self.lock:
            if key in self.active or len(self.active) >= 4 or sum(item[0] == key[0] for item in self.active) >= 2:
                return False
            self.active.add(key)
            return True

    def release(self, key):
        with self.lock:
            self.active.discard(key)

    def is_active(self, key):
        with self.lock:
            return key in self.active

    def keep_draft(self, key, draft):
        with self.lock:
            now = monotonic()
            self.drafts = {token: row for token, row in self.drafts.items() if row[0] > now}
            if len(self.drafts) >= 128:
                self.drafts.pop(next(iter(self.drafts)))
            token = token_urlsafe(24)
            self.drafts[token] = now + 900, key, draft
            return token

    def draft(self, key, token):
        with self.lock:
            row = self.drafts.get(token)
            if row and row[1] == key and row[0] > monotonic():
                return row[2]
            return None

    def forget(self, token):
        with self.lock:
            self.drafts.pop(token, None)


def stream_operation(conversations, context: RequestContext, operation):
    workspace_id, conversation_id = context.workspace_id, context.conversation_id
    conversations.get_conversation(workspace_id, conversation_id)
    key = workspace_id, conversation_id
    app = current_app._get_current_object()
    state = app.extensions["http_operations"]
    if not state.start(key):
        return {"error": {"code": "busy", "message": "An operation is already running or the demo is busy."}}, 409
    events = Queue()

    def progress(item):
        events.put(("progress", json_value(item)))

    def work():
        try:
            message = operation(progress)
            if message.metadata.get("result", {}).get("duplicate_lead"):
                terminal = "error", {"code": "duplicate_company", "message": "This company is already registered."}
            else:
                terminal = "result", {"message": public_message(message)}
        except ConversationPersistenceError as error:
            _log_stream_failure(app.logger, conversation_id, error)
            terminal = "error", {
                "code": "message_persistence_failed", "message": "The operation finished, but its conversation message could not be saved.",
                "retry": {"kind": "message", "token": state.keep_draft(key, error.draft)},
            }
        except Exception as error:
            details, status = error_payload(error)
            if status >= 500:
                _log_stream_failure(app.logger, conversation_id, error)
            terminal = "error", details
        finally:
            state.release(key)
        events.put(terminal)

    try:
        Thread(target=work, daemon=True).start()
    except RuntimeError:
        state.release(key)
        raise

    def generate():
        yield ": connected\n\n"
        while True:
            try:
                event, payload = events.get(timeout=10)
            except Empty:
                yield ": heartbeat\n\n"
                continue
            yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if event != "progress":
                return

    return Response(generate(), mimetype="text/event-stream", headers={"X-Accel-Buffering": "no"})


def _log_stream_failure(logger, conversation_id, error):
    """Log only explicitly safe diagnostics. SDK errors can contain request data."""
    context = {"conversation_id": str(conversation_id), "exception": type(error).__name__}
    failure = error.business_result if isinstance(error, ConversationPersistenceError) else error
    execution = None
    if isinstance(failure, (LeadWorkflowFailure, CustomerWorkflowFailure)):
        context.update(
            stage=failure.stage, reason=failure.reason, cause=failure.exception_type,
            evaluation_correlation_id=getattr(failure, "evaluation_correlation_id", None),
            company_id=failure.company_id, research_id=getattr(failure, "research_id", None),
            activity_id=getattr(failure, "activity_id", None),
            duration_seconds=round(failure.duration_seconds, 3),
        )
        execution = next((item for item in reversed(failure.executions) if item.failure is not None), None)
    elif isinstance(failure, ChatFailure):
        context.update(stage="chat", reason=failure.reason, user_message_id=str(failure.user_message.id))
        execution = failure.execution
    if execution is not None:
        context.update(
            underlying_exception=execution.exception_type, component=execution.component, model=execution.model,
            service_tier=execution.actual_service_tier, ai_duration_seconds=round(execution.duration_seconds, 3),
            completed_hosted_tool_calls=execution.completed_hosted_tool_calls,
        )
        if execution.api_error is not None:
            context["api_error"] = execution.api_error
        if execution.validation_diagnostic is not None:
            context["validation"] = asdict(execution.validation_diagnostic)
        if execution.correction is not None:
            context["correction"] = asdict(execution.correction)
        if execution.usage is not None:
            context["usage"] = asdict(execution.usage)
    logger.warning("Stream operation failed: %s", json.dumps(context, default=str))
