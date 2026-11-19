"""Same-origin HTTP and operation streams for the Flask demo."""

import json
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from queue import Empty, Queue
from secrets import token_urlsafe
from threading import Lock, Thread
from time import monotonic

from flask import Blueprint, Response, current_app, g, make_response, render_template, request
from pydantic_core import to_jsonable_python
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException

from .chat import ChatAssistant, ChatFailure, ConversationNotFound, ConversationPersistenceError
from .data_access import WorkspaceData
from .human_approval import RecommendationConflict, RecommendationNotFound
from .orchestrator import CustomerWorkflowFailure, LeadWorkflowFailure
from .validation import CustomerActivityInput, LeadInput, ValidationError, validate_activity, validate_lead
from .workspaces import resolve_workspace, seed_demo_companies

api = Blueprint("api", __name__, url_prefix="/api")


class RequestError(ValueError):
    pass


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


def _json(value):
    return to_jsonable_python(asdict(value))


def _public_message(message):
    payload = _json(message)
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
        and reason not in {"evaluation_already_completed", "activity_not_triggered", "lead_not_found", "activity_not_found"}
    )


def _error(error):
    if isinstance(error, (RequestError, ValidationError)):
        return {"code": "invalid_input", "message": str(error)}, 400
    if isinstance(error, (ConversationNotFound, RecommendationNotFound)):
        return {"code": "not_found", "message": "This record is unavailable in the current workspace."}, 404
    if isinstance(error, RecommendationConflict):
        return {"code": "stale_recommendation", "message": "This recommendation is no longer actionable. Refresh to see its current status."}, 409
    if isinstance(error, ChatFailure):
        return {"code": "chat_failed", "message": "The assistant could not answer. Your message was saved.",
                "retry": {"kind": "chat", "message_id": str(error.user_message.id)}}, 502
    if isinstance(error, (LeadWorkflowFailure, CustomerWorkflowFailure)):
        if error.reason == "CompanyConflict":
            return {"code": "company_conflict", "message": "The company name and domain conflict with an existing company."}, 409
        stale = error.reason.startswith("stale_") or error.reason in {"evaluation_already_completed", "activity_not_triggered"}
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


def _assistant():
    factory = current_app.extensions.get("http_chat_factory", ChatAssistant)
    return factory(current_app.extensions["db_sessions"])


def _body(allowed):
    if not request.is_json:
        raise RequestError("Send a JSON request.")
    data = request.get_json()
    if not isinstance(data, dict) or set(data) - set(allowed):
        raise RequestError("The request contains unexpected fields.")
    return data


def _integer(value, name, optional=False):
    if value is None and optional:
        return None
    if type(value) is not int:
        raise RequestError(f"{name} must be a whole number.")
    return value


def _amount(value, name, optional=False):
    if value is None and optional:
        return None
    if type(value) not in (str, int, float) or len(str(value)) > 64:
        raise RequestError(f"{name} must be a decimal number.")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        raise RequestError(f"{name} must be a decimal number.") from None


def _company(data):
    name, domain = data.get("company_name"), data.get("company_domain")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
        raise RequestError("Enter a company name of up to 200 characters.")
    return name, domain


@api.before_request
def workspace():
    if request.method == "POST" and (
        request.headers.get("X-Navan-Request") != "1"
        or request.headers.get("Sec-Fetch-Site") == "cross-site"
        or (request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/"))
    ):
        return {"error": {"code": "forbidden", "message": "This request must come from the application."}}, 403
    g.workspace_id = resolve_workspace()


@api.after_request
def private_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@api.errorhandler(Exception)
def api_error(error):
    data, status = _error(error)
    if status >= 500:
        current_app.logger.warning("HTTP operation failed (%s).", type(error).__name__)
    return {"error": data}, status


@api.get("/conversations")
def conversations():
    return {"conversations": [_json(row) for row in _assistant().list_conversations(g.workspace_id)]}


@api.post("/conversations")
def create_conversation():
    _body(())
    return {"conversation": _json(_assistant().create_conversation(g.workspace_id))}, 201


@api.get("/conversations/<uuid:conversation_id>")
def conversation(conversation_id):
    assistant = _assistant()
    workspace_id = g.workspace_id
    row = assistant.get_conversation(workspace_id, conversation_id)
    before_id = request.args.get("before_id")
    if before_id is not None:
        before_id = _message_id(before_id)
    return {
        **_json(row),
        "messages": [_public_message(item) for item in assistant.list_messages(workspace_id, conversation_id, 100, before_id)],
        "active": current_app.extensions["http_operations"].is_active((workspace_id, conversation_id)),
    }


@api.get("/companies")
def companies():
    with current_app.extensions["db_sessions"].begin() as session:
        seed_demo_companies(session, g.workspace_id)
        rows = WorkspaceData(session, g.workspace_id).company_list(101)
        return {"companies": [{"id": str(row.id), "name": row.name, "tier": row.current_tier} for row in rows[:100]],
                "more": len(rows) > 100}


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


def _stream(assistant, conversation_id, operation):
    workspace_id = g.workspace_id
    assistant.get_conversation(workspace_id, conversation_id)
    key = workspace_id, conversation_id
    app = current_app._get_current_object()
    state = app.extensions["http_operations"]
    if not state.start(key):
        return {"error": {"code": "busy", "message": "An operation is already running or the demo is busy."}}, 409
    events = Queue()

    def progress(item):
        events.put(("progress", _json(item)))

    def work():
        try:
            message = operation(progress)
            if message.metadata.get("result", {}).get("duplicate_lead"):
                terminal = "error", {"code": "duplicate_company", "message": "This company is already registered."}
            else:
                terminal = "result", {"message": _public_message(message)}
        except ConversationPersistenceError as error:
            _log_stream_failure(app.logger, conversation_id, error)
            terminal = "error", {
                "code": "message_persistence_failed", "message": "The operation finished, but its conversation message could not be saved.",
                "retry": {"kind": "message", "token": state.keep_draft(key, error.draft)},
            }
        except Exception as error:
            details, status = _error(error)
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


@api.post("/conversations/<uuid:conversation_id>/chat")
def chat(conversation_id):
    message = _body(("message",)).get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 8000 or "\x00" in message:
        raise RequestError("Enter a message of up to 8,000 characters.")
    assistant, workspace_id = _assistant(), g.workspace_id
    return _stream(assistant, conversation_id, lambda progress:
                   assistant.send_message(workspace_id, conversation_id, message, on_progress=progress).assistant_message)


@api.post("/conversations/<uuid:conversation_id>/leads")
def leads(conversation_id):
    data = _body(("company_name", "company_domain", "submitted_annual_gbv", "employee_count"))
    lead = validate_lead(LeadInput(
        *_company(data), _amount(data.get("submitted_annual_gbv"), "Annual GBV"),
        _integer(data.get("employee_count"), "Employee count", optional=True),
    ))
    assistant, workspace_id = _assistant(), g.workspace_id
    return _stream(assistant, conversation_id, lambda progress:
                   assistant.submit_lead(workspace_id, conversation_id, lead, on_progress=progress))


@api.post("/conversations/<uuid:conversation_id>/customer-activity")
def activity(conversation_id):
    required_amounts = {
        "total_gbv": "Total GBV since signup (USD)",
        "avg_monthly_bookings": "Average monthly bookings",
    }
    optional_amounts = ("avg_monthly_active_users", "avg_monthly_support_requests")
    trends = ("travel_spend_trend", "company_size_trend", "product_adoption_trend")
    data = _body(("company_name", "company_domain", "current_tier", "months_since_signup", "employee_count", *required_amounts, *optional_amounts, *trends))
    for key, label in required_amounts.items():
        if data.get(key) is None or data[key] == "":
            raise RequestError(f"{label} is required.")
    snapshot = validate_activity(CustomerActivityInput(
        *_company(data), _integer(data.get("current_tier"), "Current tier"),
        _integer(data.get("months_since_signup"), "Months since signup"),
        employee_count=_integer(data.get("employee_count"), "Employee count", optional=True),
        **{key: _amount(data[key], label) for key, label in required_amounts.items()},
        **{key: _amount(data.get(key), key.replace("_", " "), optional=True) for key in optional_amounts},
        **{key: data.get(key, "unknown") for key in trends},
    ))
    assistant, workspace_id = _assistant(), g.workspace_id
    return _stream(assistant, conversation_id, lambda progress:
                   assistant.submit_customer_reevaluation(workspace_id, conversation_id, snapshot, on_progress=progress))


@api.post("/conversations/<uuid:conversation_id>/evaluations/<uuid:evaluation_id>/decision")
def decide(conversation_id, evaluation_id):
    accept = _body(("accept",)).get("accept")
    if type(accept) is not bool:
        raise RequestError("Choose Accept or Reject.")
    assistant, workspace_id = _assistant(), g.workspace_id
    return _stream(assistant, conversation_id, lambda _:
                   assistant.decide_recommendation(workspace_id, conversation_id, evaluation_id, accept=accept))


def _message_id(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or len(value) > 19:
        raise RequestError("Invalid message identifier.")
    result = int(value)
    if not 0 < result <= 2**63 - 1:
        raise RequestError("Invalid message identifier.")
    return result


@api.post("/conversations/<uuid:conversation_id>/retry")
def retry(conversation_id):
    data = _body(("kind", "message_id", "token"))
    assistant, workspace_id = _assistant(), g.workspace_id
    kind = data.get("kind")
    if kind == "chat":
        message_id = _message_id(data.get("message_id"))
        return _stream(assistant, conversation_id, lambda progress:
                       assistant.retry_message(workspace_id, conversation_id, message_id, on_progress=progress).assistant_message)
    if kind == "workflow":
        message_id = _message_id(data.get("message_id"))

        def run(progress):
            try:
                return assistant.retry_workflow(workspace_id, conversation_id, message_id, on_progress=progress)
            except ValueError as error:
                # ChatAssistant only raises these fixed messages for non-retryable saved failures.
                raise RequestError(str(error)) from None

        return _stream(assistant, conversation_id, run)
    if kind == "message":
        token = data.get("token")
        if not isinstance(token, str) or len(token) > 64:
            raise RequestError("Invalid message retry token.")
        state = current_app.extensions["http_operations"]
        draft = state.draft((workspace_id, conversation_id), token)
        if draft is None:
            raise ConversationNotFound("Prepared message unavailable.")

        def save(_):
            try:
                result = assistant.save_prepared_message(workspace_id, draft)
            except SQLAlchemyError:
                raise ConversationPersistenceError(draft) from None
            state.forget(token)
            return result

        return _stream(assistant, conversation_id, save)
    raise RequestError("Choose a supported retry action.")


def configure_http(app):
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
    app.extensions["http_operations"] = OperationState()
    app.register_blueprint(api)

    @app.errorhandler(404)
    def missing_route(error):
        if request.path.startswith("/api/"):
            return {"error": {"code": "not_found", "message": "This API record or route is unavailable."}}, 404
        return error

    @app.get("/")
    def index():
        response = make_response(render_template("index.html"))
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
