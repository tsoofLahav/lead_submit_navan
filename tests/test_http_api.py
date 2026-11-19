import json
from threading import Event
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app import create_app
from app.chat import ChatAssistant
from app.chat_tools import build_chat_tools
from app.ai.config import Component
from app.models import Company, CompanyResearch, DemoCompany, Evaluation, ProductActivity, TierChange
from app.orchestrator import Orchestrator
from test_ai import message as response_message, mock_client, scoped_evidence, terminal
from test_chat import call, final, references
from test_customer_reevaluation import assessment, recommendation
from test_lead_submission import decision_data, prediction_data, research_data, response as lead_response

pytestmark = pytest.mark.write_integration
HEADERS = {"X-Navan-Request": "1"}


@pytest.fixture
def web(sessions, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("SECRET_KEY", "stage7-test-only")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")
    app = create_app()
    app.extensions["db_sessions"] = sessions
    yield app
    app.extensions["db_engine"].dispose()


def post(client, path, data, **kwargs):
    return client.post(path, json=data, headers=HEADERS, **kwargs)


def thread(client):
    response = post(client, "/api/conversations", {})
    assert response.status_code == 201
    return "/api/conversations/" + response.json["conversation"]["id"]


def frames(response):
    assert response.status_code == 200 and response.mimetype == "text/event-stream"
    parsed = []
    for frame in response.get_data(as_text=True).split("\n\n"):
        if frame.startswith("event: "):
            event, data = frame.splitlines()
            parsed.append((event[7:], json.loads(data[6:])))
    return parsed


def payload(name, **values):
    return {"company_name": name, "company_domain": f"{name.lower()}.test", "current_tier": 3,
            "months_since_signup": 6, **values}


def test_frontend_and_workspace_reload_security(web):
    client = web.test_client()
    root = client.get("/")
    assert root.status_code == 200
    assert b"About this demo" in root.data and b'id="history" hidden' in root.data
    assert b'src="/static/api.js"' in root.data and b'src="/static/app.js"' in root.data
    for file in ("api.js", "app.js"):
        assert client.get("/static/" + file).status_code == 200
    before = client.get("/api/conversations")
    assert before.json == {"conversations": []}
    assert "HttpOnly" in before.headers["Set-Cookie"]
    assert client.get("/api/companies").json == {"companies": [], "more": False}
    assert client.post("/api/conversations", json={}).status_code == 403
    assert client.post("/api/conversations", json={}, headers={**HEADERS, "Origin": "https://anywhere.test"}).status_code == 403
    path = thread(client)
    assert client.get("/").status_code == 200
    assert len(client.get("/api/conversations").json["conversations"]) == 1
    assert client.get(path).json["messages"] == []
    outsider = web.test_client()
    assert outsider.get(path).status_code == 404
    assert post(outsider, path + "/chat", {"message": "hello"}).status_code == 404
    assert post(client, "/api/conversations", {"workspace_id": "forged"}).status_code == 400
    assert post(client, path + "/chat", {"message": "x" * 34000}).status_code == 413


def test_lead_real_persistence_stream_and_duplicate(web, sessions, references):
    business, requests = mock_client(
        lead_response(research_data(), research=True), lead_response(prediction_data()), lead_response(decision_data()),
    )
    web.extensions["http_chat_factory"] = lambda _: ChatAssistant(
        sessions, orchestrator=Orchestrator(sessions, ai_client=business),
    )
    client = web.test_client()
    path = thread(client)
    data = {"company_name": "Acme", "company_domain": "acme.test", "submitted_annual_gbv": "100000", "employee_count": None}
    events = frames(post(client, path + "/leads", data))
    assert [item["kind"] for event, item in events if event == "progress"] == [
        "company_registered", "initial_tier_assigned", "research_started", "web_search_completed",
        "research_completed", "prediction_started", "prediction_completed", "recommendation_started",
        "recommendation_completed", "evaluation_completed",
    ]
    assert all(item["conversation_id"] == path.rsplit("/", 1)[1] for event, item in events if event == "progress")
    message = events[-1][1]["message"]
    assert events[-1][0] == "result" and isinstance(message["id"], str)
    assert message["metadata"]["result"]["company_name"] == "Acme"
    assert message["metadata"]["decision_status"] == "not_applicable"
    assert client.get(path).json["messages"][-1] == message
    assert client.get("/api/companies").json["companies"][0]["tier"] == 2
    assert frames(post(client, path + "/leads", data))[-1][1]["code"] == "duplicate_company"
    assert len(requests) == 3


def test_gate_only_and_decimal_null_validation(web, sessions):
    client = web.test_client()
    path = thread(client)
    assert post(client, path + "/customer-activity", payload("Bad", current_tier=None)).status_code == 400
    assert post(client, path + "/customer-activity", payload("Bad", avg_monthly_bookings="invalid")).status_code == 400
    for name, values, status in [
        ("Quiet", {"total_gbv": "0", "employee_count": 40, "avg_monthly_bookings": None}, "NO_TRIGGER"),
        ("Sparse", {}, "INSUFFICIENT_DATA"),
    ]:
        events = frames(post(client, path + "/customer-activity", payload(name, **values)))
        assert events[-1][0] == "result"
        metadata = events[-1][1]["message"]["metadata"]
        assert metadata["kind"] == "activity_update" and metadata["evaluation_id"] is None
        assert metadata["result"]["gate"]["status"] == status
        assert not any(event.get("component") for kind, event in events if kind == "progress")
    with sessions() as session:
        snapshots = {company.name: activity for company, activity in session.execute(
            select(Company, ProductActivity).join(ProductActivity, ProductActivity.company_id == Company.id),
        )}
        assert snapshots["Quiet"].total_gbv == 0 and snapshots["Quiet"].avg_monthly_bookings is None
        assert snapshots["Sparse"].total_gbv is None
        assert session.scalar(select(func.count(Evaluation.id))) == 0


@pytest.mark.parametrize("accept", [True, False])
def test_triggered_customer_and_approval_or_rejection(web, sessions, references, accept):
    business, _ = mock_client([terminal([response_message(json.dumps(assessment()))])],
                              [terminal([response_message(json.dumps(recommendation()))])])
    web.extensions["http_chat_factory"] = lambda _: ChatAssistant(sessions, orchestrator=Orchestrator(sessions, ai_client=business))
    client, outsider = web.test_client(), web.test_client()
    path = thread(client)
    events = frames(post(client, path + "/customer-activity", payload("Growth", total_gbv="360000", employee_count=220)))
    message = events[-1][1]["message"]
    assert message["metadata"]["decision_status"] == "pending"
    assert message["metadata"]["result"]["gate"]["status"] == "TRIGGERED"
    decision = f'{path}/evaluations/{message["metadata"]["evaluation_id"]}/decision'
    assert post(outsider, decision, {"accept": True}).status_code == 404
    confirmation = frames(post(client, decision, {"accept": accept}))[-1][1]["message"]
    assert confirmation["metadata"]["status"] == ("accepted" if accept else "rejected")
    assert client.get("/api/companies").json["companies"][0]["tier"] == (1 if accept else 3)
    assert client.get(path).json["messages"][1]["metadata"]["approval_available"] is False
    assert frames(post(client, decision, {"accept": accept}))[-1][1]["code"] == "stale_recommendation"


def test_chat_tool_and_explicit_retry(web, sessions, references):
    sdk, requests = mock_client(500, [terminal([call("get_business_policy", {})])], final("A valid policy answer."))
    web.extensions["http_chat_factory"] = lambda _: ChatAssistant(sessions, sdk=sdk._sdk)
    client = web.test_client()
    path = thread(client)
    failure = frames(post(client, path + "/chat", {"message": "What is the eligibility policy?"}))[-1]
    assert failure[1]["code"] == "chat_failed"
    result = frames(post(client, path + "/retry", failure[1]["retry"]))
    assert [item["tool_name"] for event, item in result if event == "progress"] == [
        "get_business_policy", "get_business_policy",
    ]
    assert result[-1][1]["message"]["content"] == "A valid policy answer."
    assert len(client.get(path).json["messages"]) == 2 and len(requests) == 3


def test_workflow_and_message_only_retry(web, sessions, references, monkeypatch):
    business, _ = mock_client(500, lead_response(research_data(), research=True),
                              lead_response(prediction_data()), lead_response(decision_data()))
    service = ChatAssistant(sessions, orchestrator=Orchestrator(sessions, ai_client=business))
    web.extensions["http_chat_factory"] = lambda _: service
    client = web.test_client()
    path = thread(client)
    failure = frames(post(client, path + "/leads", {
        "company_name": "Acme", "company_domain": "acme.test", "submitted_annual_gbv": "100000",
    }))[-1][1]
    assert failure["retry"]["kind"] == "workflow"
    assert client.get(path).json["messages"][-1]["metadata"]["result"] == {"stage": "research", "can_retry": True}
    assert frames(post(client, path + "/retry", failure["retry"]))[-1][0] == "result"

    with monkeypatch.context() as patch:
        def fail(*args):
            raise SQLAlchemyError("private details")
        patch.setattr(service, "save_prepared_message", fail)
        save_failure = frames(post(client, path + "/customer-activity", payload("Quiet")))[-1][1]
    assert save_failure["code"] == "message_persistence_failed"
    assert "private" not in json.dumps(save_failure)
    assert post(client, thread(client) + "/retry", save_failure["retry"]).status_code == 404
    assert frames(post(client, path + "/retry", save_failure["retry"]))[-1][0] == "result"
    with sessions() as session:
        assert session.scalar(select(func.count(ProductActivity.id))) == 1
        assert session.scalar(select(func.count(Evaluation.id))) == 1


def test_disconnect_does_not_cancel_and_other_thread_can_load(web, sessions, monkeypatch):
    service = ChatAssistant(sessions)
    web.extensions["http_chat_factory"] = lambda _: service
    original = service.submit_customer_reevaluation
    started, release, done = Event(), Event(), Event()

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            done.set()

    monkeypatch.setattr(service, "submit_customer_reevaluation", delayed)
    client = web.test_client()
    first, second = thread(client), thread(client)
    response = post(client, first + "/customer-activity", payload("Quiet"), buffered=False)
    assert started.wait(2)
    try:
        assert client.get(second).status_code == 200
        assert client.get(first).json["active"]
        assert post(client, first + "/customer-activity", payload("Quiet")).status_code == 409
        response.close()
    finally:
        release.set()
    assert done.wait(5)
    assert len(client.get(first).json["messages"]) == 2
    assert not client.get(second).json["messages"]


def test_research_failure_logs_safe_diagnostic_and_retry_creates_no_duplicate(
    web, sessions, references, caplog, monkeypatch, tmp_path,
):
    monkeypatch.setenv("AI_DIAGNOSTICS_REPORT_DIR", str(tmp_path))
    invalid = research_data()
    invalid["signals"][0].update(status="missing", evidence=[scoped_evidence(
        "private unsourced observation", context="private customer context",
    )])
    bad = lead_response(invalid, research=True)
    business, requests = mock_client(
        bad, lead_response(invalid), bad, lead_response(invalid), lead_response(research_data(), research=True),
        lead_response(prediction_data()), lead_response(decision_data()),
    )
    web.extensions["http_chat_factory"] = lambda _: ChatAssistant(
        sessions, orchestrator=Orchestrator(sessions, ai_client=business),
    )
    client = web.test_client()
    path = thread(client)
    updates = frames(post(client, path + "/leads", {
        "company_name": "Acme", "company_domain": "acme.test", "submitted_annual_gbv": "100000",
    }))
    failure = updates[-1][1]
    assert failure["code"] == "workflow_failed"
    assert frames(post(client, path + "/retry", failure["retry"]))[-1][0] == "error"
    rows = client.get(path).json["messages"]
    assert rows[-1]["metadata"]["retried_failure_id"] == rows[1]["id"]
    assert rows[-1]["metadata"]["company_id"] == rows[1]["metadata"]["company_id"]
    assert "research.missing_evidence" not in json.dumps({"sse": updates, "messages": rows})

    records = [json.loads(record.getMessage().split(": ", 1)[1]) for record in caplog.records
               if record.getMessage().startswith("Stream operation failed: ")]
    assert len(records) == 2
    for record in records:
        assert record["conversation_id"] == path.rsplit("/", 1)[1]
        assert record["stage"] == "research" and record["reason"] == "validation"
        assert record["cause"] == "AIExecutionError"
        assert record["underlying_exception"] == "AIOutputValidationError"
        assert record["validation"]["rule_id"] == "research.missing_evidence"
        assert record["validation"]["field_path"] == "signals[0].evidence"
        assert record["validation"]["observed"] == {"count": 1}
    assert "private unsourced observation" not in caplog.text and "private customer context" not in caplog.text
    evidence_schema = requests[0]["text"]["format"]["schema"]["$defs"]["ScopedSignalEvidence"]
    assert "at least one actual supporting source" in evidence_schema["properties"]["sources"]["description"]

    latest_failure = {"kind": "workflow", "message_id": rows[-1]["id"]}
    assert frames(post(client, path + "/retry", latest_failure))[-1][0] == "result"
    with sessions() as session:
        assert session.scalar(select(func.count(Company.id))) == 1
        assert session.scalar(select(func.count(Evaluation.id))) == 1

    report_files = list(tmp_path.glob("*.json"))
    reports = [json.loads(path.read_text()) for path in report_files]
    assert sorted(report["status"] for report in reports) == ["completed", "failed", "failed"]
    correlation, = {report["correlation_id"] for report in reports}
    assert len({report["attempt_id"] for report in reports}) == 3
    assert sum(report["retry"] for report in reports) == 2
    assert all(record["evaluation_correlation_id"] == correlation for record in records)
    assert correlation not in json.dumps(rows)
    assert "private unsourced observation" not in json.dumps(reports)
    completed = next(report for report in reports if report["status"] == "completed")
    assert [stage["stage"] for stage in completed["stages"]] == [
        "context", "research", "research_persistence", "prediction", "recommendation", "evaluation_persistence",
    ]
    assert all(stage["duration_seconds"] >= 0 for stage in completed["stages"])
    assert completed["totals"]["measured_tokens"]["input_tokens"] == 30
    assert completed["totals"]["estimated_cost_usd"] > 0
    for report in reports:
        for stage in report["executions"]:
            assert stage["evaluation_correlation_id"] == correlation
            assert stage["evaluation_attempt_id"] == report["attempt_id"]
    failed = next(report for report in reports if report["status"] == "failed")
    first, correction = failed["executions"][0]["requests"]
    assert not first["correction"] and first["hosted_tool_limit"] == 5
    assert correction["correction"] and correction["hosted_tool_limit"] == 0
    assert correction["correction_context_utf8_bytes"] > 0


@pytest.mark.parametrize("accept", [True, False])
def test_demo_company_copies_are_interactive_and_isolated(web, sessions, references, accept):
    examples = {
        "Gradient Labs": "gradient-labs.test", "FintechOS": "fintechos.test",
        "NextGen Cloud": "nextgen-cloud.test", "Valarian": "valarian.test",
    }
    with sessions.begin() as session:
        session.add_all(DemoCompany(
            name=name, domain=domain, submitted_annual_gbv=100_000, employee_count=100, current_tier=3,
        ) for name, domain in examples.items())
    business, requests = mock_client([terminal([response_message(json.dumps(assessment()))])],
                                     [terminal([response_message(json.dumps(recommendation()))])])
    web.extensions["http_chat_factory"] = lambda _: ChatAssistant(
        sessions, orchestrator=Orchestrator(sessions, ai_client=business),
    )
    visitor, outsider = web.test_client(), web.test_client()

    def companies(client):
        return {row["name"]: row for row in client.get("/api/companies").json["companies"]}

    first, second = companies(visitor), companies(outsider)
    assert first.keys() == second.keys() == examples.keys()
    assert {row["id"] for row in first.values()}.isdisjoint(row["id"] for row in second.values())
    assert companies(visitor) == first
    assert not requests
    with sessions() as session:
        template_ids = set(session.scalars(select(DemoCompany.id)))
        copies = list(session.scalars(select(Company)))
        assert len(copies) == 8
        assert all(
            row.id not in template_ids and row.is_customer and row.submitted_annual_gbv == 100_000
            and row.employee_count == 100 and row.domain == examples[row.name]
            for row in copies
        )
        for model in (CompanyResearch, ProductActivity, Evaluation, TierChange):
            assert session.scalar(select(func.count()).select_from(model)) == 0
        workspace_id = session.get(Company, UUID(first["Gradient Labs"]["id"])).workspace_id

    tools = build_chat_tools(sessions, sessions)
    selector = json.dumps({"company_id": first["Gradient Labs"]["id"], "company_name": None})
    details = json.loads(tools.dispatch(Component.CHAT, workspace_id, "get_company", selector))
    assert details["submitted_annual_gbv"] == "100000.00" and "sample data" in details["submitted_data_provenance"]

    path = thread(visitor)
    snapshot = payload("Gradient Labs", company_domain=examples["Gradient Labs"],
                       total_gbv="360000", employee_count=220)
    last = frames(post(visitor, path + "/customer-activity", snapshot))[-1]
    assert last[0] == "result"
    metadata = last[1]["message"]["metadata"]
    assert metadata["company_id"] == first["Gradient Labs"]["id"]
    assert metadata["result"]["gate"]["status"] == "TRIGGERED"
    decision = f'{path}/evaluations/{metadata["evaluation_id"]}/decision'
    assert post(outsider, decision, {"accept": True}).status_code == 404
    result = frames(post(visitor, decision, {"accept": accept}))[-1]
    assert result[1]["message"]["metadata"]["status"] == ("accepted" if accept else "rejected")
    after = companies(visitor)
    assert len(after) == 4 and after["Gradient Labs"]["tier"] == (1 if accept else 3)
    assert companies(outsider) == second
    assert len(requests) == 2
    with sessions() as session:
        copy = session.get(Company, UUID(first["Gradient Labs"]["id"]))
        template = session.scalar(select(DemoCompany).where(DemoCompany.name == "Gradient Labs"))
        other = session.get(Company, UUID(second["Gradient Labs"]["id"]))
        assert copy.employee_count == 220 and template.employee_count == other.employee_count == 100
        assert template.current_tier == other.current_tier == 3
        for model, expected in ((ProductActivity, 1), (Evaluation, 1), (TierChange, int(accept)), (CompanyResearch, 0)):
            assert session.scalar(select(func.count()).select_from(model).where(model.company_id == copy.id)) == expected
