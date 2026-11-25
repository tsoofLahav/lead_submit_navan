"""Register Flask routes and select the operation for each HTTP request."""

from decimal import Decimal, InvalidOperation

from flask import Blueprint, current_app, g, make_response, render_template, request
from sqlalchemy.exc import SQLAlchemyError

from ..conversations.service import ConversationNotFound, ConversationPersistenceError
from ..data.access import WorkspaceData
from ..business.rules.validation import CustomerActivityInput, LeadInput, validate_activity, validate_lead
from .workspaces import resolve_workspace, seed_demo_companies
from .actions import Orchestrator
from .responses import RequestError, error_payload, json_value, public_message
from .streaming import OperationState, RequestContext, stream_operation

api = Blueprint("api", __name__, url_prefix="/api")


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
    data, status = error_payload(error)
    if status >= 500:
        current_app.logger.warning("HTTP operation failed (%s).", type(error).__name__)
    return {"error": data}, status


@api.get("/conversations")
def conversations():
    return {"conversations": [json_value(row) for row in _orchestrator().conversations.list_conversations(g.workspace_id)]}


@api.post("/conversations")
def create_conversation():
    _body(())
    return {"conversation": json_value(_orchestrator().conversations.create_conversation(g.workspace_id))}, 201


@api.get("/conversations/<uuid:conversation_id>")
def conversation(conversation_id):
    orchestrator = _orchestrator()
    workspace_id = g.workspace_id
    row = orchestrator.conversations.get_conversation(workspace_id, conversation_id)
    before_id = request.args.get("before_id")
    if before_id is not None:
        before_id = _message_id(before_id)
    return {
        **json_value(row),
        "messages": [public_message(item) for item in orchestrator.conversations.list_messages(workspace_id, conversation_id, 100, before_id)],
        "active": current_app.extensions["http_operations"].is_active((workspace_id, conversation_id)),
    }


@api.get("/companies")
def companies():
    with current_app.extensions["db_sessions"].begin() as session:
        seed_demo_companies(session, g.workspace_id)
        rows = WorkspaceData(session, g.workspace_id).company_list(101)
        return {"companies": [{"id": str(row.id), "name": row.name, "tier": row.current_tier} for row in rows[:100]],
                "more": len(rows) > 100}


@api.post("/conversations/<uuid:conversation_id>/chat")
def chat(conversation_id):
    message = _body(("message",)).get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 8000 or "\x00" in message:
        raise RequestError("Enter a message of up to 8,000 characters.")
    orchestrator = _orchestrator()
    context = RequestContext(g.workspace_id, conversation_id)
    return stream_operation(orchestrator.conversations, context, lambda progress:
                   orchestrator.reply_to_chat(context.workspace_id, context.conversation_id, message, on_progress=progress).assistant_message)


@api.post("/conversations/<uuid:conversation_id>/leads")
def leads(conversation_id):
    data = _body(("company_name", "company_domain", "submitted_annual_gbv", "employee_count"))
    lead = validate_lead(LeadInput(
        *_company(data), _amount(data.get("submitted_annual_gbv"), "Annual GBV"),
        _integer(data.get("employee_count"), "Employee count", optional=True),
    ))
    orchestrator = _orchestrator()
    context = RequestContext(g.workspace_id, conversation_id)
    return stream_operation(orchestrator.conversations, context, lambda progress:
                   orchestrator.submit_lead(context.workspace_id, context.conversation_id, lead, on_progress=progress))


@api.post("/conversations/<uuid:conversation_id>/customer-activity")
def activity(conversation_id):
    amounts = ("total_gbv", "avg_monthly_bookings", "avg_monthly_active_users", "avg_monthly_support_requests")
    trends = ("travel_spend_trend", "company_size_trend", "product_adoption_trend")
    data = _body(("company_name", "company_domain", "current_tier", "months_since_signup", "employee_count", *amounts, *trends))
    snapshot = validate_activity(CustomerActivityInput(
        *_company(data), _integer(data.get("current_tier"), "Current tier"),
        _integer(data.get("months_since_signup"), "Months since signup"),
        employee_count=_integer(data.get("employee_count"), "Employee count", optional=True),
        **{key: _amount(data.get(key), key.replace("_", " "), optional=True) for key in amounts},
        **{key: data.get(key, "unknown") for key in trends},
    ))
    orchestrator = _orchestrator()
    context = RequestContext(g.workspace_id, conversation_id)
    return stream_operation(orchestrator.conversations, context, lambda progress:
                   orchestrator.submit_customer_reevaluation(context.workspace_id, context.conversation_id, snapshot, on_progress=progress))


@api.post("/conversations/<uuid:conversation_id>/evaluations/<uuid:evaluation_id>/decision")
def decide(conversation_id, evaluation_id):
    accept = _body(("accept",)).get("accept")
    if type(accept) is not bool:
        raise RequestError("Choose Accept or Reject.")
    orchestrator = _orchestrator()
    context = RequestContext(g.workspace_id, conversation_id)
    return stream_operation(orchestrator.conversations, context, lambda _:
                   orchestrator.decide_recommendation(context.workspace_id, context.conversation_id, evaluation_id, accept=accept))


@api.post("/conversations/<uuid:conversation_id>/retry")
def retry(conversation_id):
    data = _body(("kind", "message_id", "token"))
    orchestrator = _orchestrator()
    context = RequestContext(g.workspace_id, conversation_id)
    kind = data.get("kind")
    if kind == "chat":
        message_id = _message_id(data.get("message_id"))
        return stream_operation(orchestrator.conversations, context, lambda progress:
                       orchestrator.retry_message(context.workspace_id, context.conversation_id, message_id, on_progress=progress).assistant_message)
    if kind == "workflow":
        message_id = _message_id(data.get("message_id"))

        def run(progress):
            try:
                return orchestrator.retry_workflow(context.workspace_id, context.conversation_id, message_id, on_progress=progress)
            except ValueError as error:
                # The orchestrator raises these fixed messages for non-retryable saved failures.
                raise RequestError(str(error)) from None

        return stream_operation(orchestrator.conversations, context, run)
    if kind == "message":
        token = data.get("token")
        if not isinstance(token, str) or len(token) > 64:
            raise RequestError("Invalid message retry token.")
        state = current_app.extensions["http_operations"]
        draft = state.draft((context.workspace_id, context.conversation_id), token)
        if draft is None:
            raise ConversationNotFound("Prepared message unavailable.")

        def save(_):
            try:
                result = orchestrator.conversations.save_prepared_message(context.workspace_id, draft)
            except SQLAlchemyError:
                raise ConversationPersistenceError(draft) from None
            state.forget(token)
            return result

        return stream_operation(orchestrator.conversations, context, save)
    raise RequestError("Choose a supported retry action.")


def _orchestrator():
    factory = current_app.extensions.get("http_orchestrator_factory", Orchestrator)
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


def _message_id(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or len(value) > 19:
        raise RequestError("Invalid message identifier.")
    result = int(value)
    if not 0 < result <= 2**63 - 1:
        raise RequestError("Invalid message identifier.")
    return result
