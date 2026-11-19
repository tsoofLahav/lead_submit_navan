from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal as D
from uuid import UUID, uuid4

import pytest
from flask import request
from sqlalchemy import func, select

from app import create_app
from app.data_access import CompanyConflict, WorkspaceData
from app.human_approval import HumanApproval, RecommendationConflict, RecommendationNotFound
from app.models import (
    Company, CompanyResearch, Conversation, Evaluation, Message, ProductActivity, TierChange, Workspace,
)
from app.orchestrator import Orchestrator
from app.reevaluation_gate import GateStatus
from app.validation import CustomerActivityInput, LeadInput, ValidationError
from app.workspaces import COOKIE_WORKSPACE_KEY, resolve_workspace

pytestmark = pytest.mark.write_integration

LEAD = LeadInput("  Acme   Inc ", " EXAMPLE.COM. ", D("500000"), 160)
ACTIVITY = CustomerActivityInput("Acme Inc", "example.com", 3, 4)


@pytest.fixture
def browser_app(engine, sessions, monkeypatch):
    raw_url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", raw_url)
    monkeypatch.setenv("SECRET_KEY", "test-only-secret-key")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")
    application = create_app()
    application.extensions["db_sessions"] = sessions

    @application.post("/workspace-test")
    def workspace_test():
        # The fixture accepts a body just like a future endpoint would.
        request.get_json(silent=True)
        return {"id": str(resolve_workspace())}

    yield application
    application.extensions["db_engine"].dispose()


def submit_workspace(client, **kwargs):
    return client.post("/workspace-test", base_url="https://localhost", **kwargs)


def record_evaluation(sessions, company_id, current=3, recommended=2, **kwargs):
    with sessions.begin() as session:
        evaluation = Evaluation(
            company_id=company_id, evaluation_type="lead_submit", predictions={},
            current_tier=current, recommended_tier=recommended, confidence="high",
            rationale="Isolated test recommendation.", **kwargs,
        )
        session.add(evaluation)
        session.flush()
        return evaluation.id


def changes(session, company_id):
    return list(session.scalars(select(TierChange).where(TierChange.company_id == company_id)))


def test_signed_workspace_creation_reuse_and_body_isolation(browser_app, sessions):
    first = browser_app.test_client()
    response = submit_workspace(first)
    workspace_id = response.json["id"]
    cookie = response.headers["Set-Cookie"]
    assert all(attribute in cookie for attribute in ("navan_workspace=", "HttpOnly", "Secure", "SameSite=Lax"))
    assert submit_workspace(first).json["id"] == workspace_id
    second = browser_app.test_client()
    other_id = submit_workspace(second, json={"workspace_id": workspace_id}).json["id"]
    assert other_id != workspace_id
    with sessions() as session:
        assert session.get(Workspace, UUID(workspace_id)) is not None
        assert session.get(Workspace, UUID(other_id)) is not None


@pytest.mark.parametrize("kind", ["tampered", "malformed_id", "missing_row"])
def test_invalid_workspace_cookie_creates_workspace(browser_app, kind):
    client = browser_app.test_client()
    signer = browser_app.session_interface.get_signing_serializer(browser_app)
    raw_id = "bad uuid" if kind == "malformed_id" else str(uuid4())
    cookie = signer.dumps({COOKIE_WORKSPACE_KEY: raw_id})
    if kind == "tampered":
        payload, separator, signature = cookie.rpartition(".")
        first_character = "A" if signature[0] != "A" else "B"
        cookie = payload + separator + first_character + signature[1:]
    client.set_cookie("navan_workspace", cookie)
    response = submit_workspace(client)
    assert UUID(response.json["id"]) != raw_id
    assert submit_workspace(client).json == response.json


def test_workspace_scopes_every_related_lookup(sessions, workspace_ids):
    owner, outsider = workspace_ids
    company = Orchestrator(sessions).submit_customer_activity(owner, ACTIVITY)
    evaluation_id = record_evaluation(sessions, company.company_id)
    HumanApproval(sessions).accept(owner, evaluation_id)
    with sessions.begin() as session:
        research = CompanyResearch(company_id=company.company_id, signals={})
        conversation = Conversation(workspace_id=owner)
        session.add_all((research, conversation))
        session.flush()
        message = Message(conversation_id=conversation.id, role="user", content="isolated")
        session.add(message)
        session.flush()
        owned = WorkspaceData(session, owner)
        other = WorkspaceData(session, outsider)
        assert owned.get_company(company.company_id) is not None
        assert other.get_company(company.company_id) is None
        for model, record_id in (
            (CompanyResearch, research.id), (ProductActivity, company.activity_id),
            (Evaluation, evaluation_id), (TierChange, changes(session, company.company_id)[0].id),
        ):
            assert owned.get_company_record(model, record_id) is not None
            assert other.get_company_record(model, record_id) is None
        assert owned.get_conversation(conversation.id) is conversation
        assert owned.get_message(message.id) is message
        assert other.get_conversation(conversation.id) is None
        assert other.get_message(message.id) is None


def test_normalized_lead_matching_and_domain_conflicts(sessions, workspace_ids):
    owner, outsider = workspace_ids
    orchestrator = Orchestrator(sessions)
    first = orchestrator.register_lead(owner, LEAD)
    duplicate = orchestrator.register_lead(owner, replace(LEAD, company_name="acME\tinc", submitted_annual_gbv=D("0")))
    second = orchestrator.register_lead(outsider, LEAD)
    assert first.created and second.created and first.company_id != second.company_id
    assert not duplicate.created and duplicate.company_id == first.company_id and duplicate.current_tier == 1
    other_lead = replace(LEAD, company_name="Different Company", company_domain="different.com")
    orchestrator.register_lead(owner, other_lead)
    with pytest.raises(CompanyConflict):
        orchestrator.register_lead(owner, replace(other_lead, company_domain="example.com"))
    with pytest.raises(CompanyConflict):
        orchestrator.submit_customer_activity(owner, replace(ACTIVITY, company_domain="different.com"))
    with sessions() as session:
        company = session.get(Company, first.company_id)
        assert (company.name, company.domain, company.submitted_annual_gbv) == ("Acme Inc", "example.com", 500000)
        assert not company.is_customer


@pytest.mark.parametrize(
    ("trend", "gate_status"),
    [
        ("increasing", GateStatus.TRIGGERED),
        ("stable", GateStatus.NO_TRIGGER),
        ("unknown", GateStatus.INSUFFICIENT_DATA),
    ],
)
def test_independent_customer_registration_saves_every_gate_outcome(sessions, workspace_ids, trend, gate_status):
    owner, _ = workspace_ids
    result = Orchestrator(sessions).submit_customer_activity(owner, replace(ACTIVITY, travel_spend_trend=trend))
    assert result.created and not result.external_tier_changed
    assert result.gate.status is gate_status
    with sessions() as session:
        company = session.get(Company, result.company_id)
        assert company.is_customer and company.current_tier == 3
        assert company.submitted_annual_gbv is None and company.employee_count is None
        assert changes(session, company.id) == []
        assert session.get(ProductActivity, result.activity_id).travel_spend_trend == trend
        assert session.scalar(select(func.count(Evaluation.id))) == 0


def test_convert_lead_preserve_history_and_supersede(sessions, workspace_ids):
    owner, _ = workspace_ids
    orchestrator = Orchestrator(sessions)
    lead = orchestrator.register_lead(owner, LEAD)
    pending = record_evaluation(sessions, lead.company_id, current=1, recommended=2)
    with sessions.begin() as session:
        session.add(CompanyResearch(company_id=lead.company_id, signals={}))
    first = orchestrator.submit_customer_activity(owner, ACTIVITY)
    assert not first.created and first.external_tier_changed and first.superseded_count == 1
    assert first.gate.status is GateStatus.INSUFFICIENT_DATA
    assert first.gate.metrics.adoption_rate_percent is None
    duplicate_lead = orchestrator.register_lead(owner, LEAD)
    assert not duplicate_lead.created and duplicate_lead.current_tier == 3

    second = orchestrator.submit_customer_activity(
        owner, replace(ACTIVITY, employee_count=150, company_domain="new.example.com")
    )
    assert not second.external_tier_changed and second.activity_id != first.activity_id
    assert second.gate.status is GateStatus.TRIGGERED
    with sessions() as session:
        company = session.get(Company, lead.company_id)
        assert company.is_customer and company.current_tier == 3 and company.employee_count == 150
        assert company.domain == "new.example.com" and company.submitted_annual_gbv == 500000
        history, = changes(session, company.id)
        assert (history.previous_tier, history.new_tier, history.evaluation_id) == (1, 3, None)
        assert session.get(ProductActivity, first.activity_id).employee_count is None
        assert session.get(ProductActivity, second.activity_id).employee_count == 150
        assert session.scalar(select(func.count(CompanyResearch.id))) == 1
        old = session.get(Evaluation, pending)
        assert old.decision_status == "superseded" and old.decided_at is not None
        assert session.scalar(select(func.count(Evaluation.id))) == 1


def test_customer_update_is_atomic(sessions, workspace_ids, monkeypatch):
    owner, _ = workspace_ids
    orchestrator = Orchestrator(sessions)
    lead = orchestrator.register_lead(owner, LEAD)
    pending = record_evaluation(sessions, lead.company_id, current=1, recommended=2)
    original_supersede = HumanApproval.supersede_pending

    def fail_after_supersession(data, company_id):
        original_supersede(data, company_id)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(HumanApproval, "supersede_pending", staticmethod(fail_after_supersession))
    with pytest.raises(RuntimeError, match="injected failure"):
        orchestrator.submit_customer_activity(owner, ACTIVITY)
    with sessions() as session:
        company = session.get(Company, lead.company_id)
        assert not company.is_customer and company.current_tier == 1
        assert changes(session, lead.company_id) == []
        assert session.scalar(select(func.count(ProductActivity.id))) == 0
        assert session.get(Evaluation, pending).decision_status == "pending"


def test_invalid_activity_is_not_written(sessions, workspace_ids):
    owner, _ = workspace_ids
    with pytest.raises(ValidationError):
        Orchestrator(sessions).submit_customer_activity(owner, replace(ACTIVITY, months_since_signup=0))
    with sessions() as session:
        assert session.scalar(select(func.count(Company.id))) == 0


def test_accept_recommendation_once_and_supersede_others(sessions, workspace_ids):
    owner, _ = workspace_ids
    company = Orchestrator(sessions).submit_customer_activity(owner, ACTIVITY)
    evaluation_id = record_evaluation(sessions, company.company_id)
    obsolete_id = record_evaluation(sessions, company.company_id, recommended=1)
    approval = HumanApproval(sessions)
    result = approval.accept(owner, evaluation_id)
    assert result.status == "accepted" and result.current_tier == 2
    for stale_id in (evaluation_id, obsolete_id):
        with pytest.raises(RecommendationConflict):
            approval.accept(owner, stale_id)
    with sessions() as session:
        assert session.get(Company, company.company_id).current_tier == 2
        history, = changes(session, company.company_id)
        assert (history.evaluation_id, history.previous_tier, history.new_tier) == (evaluation_id, 3, 2)
        accepted = session.get(Evaluation, evaluation_id)
        assert accepted.decision_status == "accepted" and accepted.decided_at is not None
        assert session.get(Evaluation, obsolete_id).decision_status == "superseded"


def test_rejection_updates_only_evaluation(sessions, workspace_ids):
    owner, _ = workspace_ids
    company = Orchestrator(sessions).submit_customer_activity(owner, ACTIVITY)
    evaluation_id = record_evaluation(sessions, company.company_id)
    assert HumanApproval(sessions).reject(owner, evaluation_id).current_tier == 3
    with sessions() as session:
        assert session.get(Company, company.company_id).current_tier == 3
        rejected = session.get(Evaluation, evaluation_id)
        assert rejected.decision_status == "rejected" and rejected.decided_at is not None
        assert not changes(session, company.company_id)


@pytest.mark.parametrize("operation", ["accept", "reject"])
def test_recommendations_cannot_be_decided_from_another_workspace(sessions, workspace_ids, operation):
    owner, outsider = workspace_ids
    company = Orchestrator(sessions).submit_customer_activity(owner, ACTIVITY)
    evaluation_id = record_evaluation(sessions, company.company_id)
    approval = HumanApproval(sessions)
    with pytest.raises(RecommendationNotFound):
        getattr(approval, operation)(outsider, evaluation_id)
    with sessions.begin() as session:
        assert approval.supersede_pending(WorkspaceData(session, outsider), company.company_id) == 0
        assert session.get(Evaluation, evaluation_id).decision_status == "pending"
        assert not changes(session, company.company_id)


@pytest.mark.parametrize("recommended", [None, 3])
@pytest.mark.parametrize("operation", ["accept", "reject"])
def test_non_actionable_recommendations(sessions, workspace_ids, recommended, operation):
    owner, _ = workspace_ids
    company = Orchestrator(sessions).submit_customer_activity(owner, ACTIVITY)
    evaluation_id = record_evaluation(sessions, company.company_id, recommended=recommended)
    with pytest.raises(RecommendationConflict):
        getattr(HumanApproval(sessions), operation)(owner, evaluation_id)


@pytest.mark.parametrize("stale_reason", ["tier_changed", "newer_activity", "superseded"])
def test_stale_recommendations_cannot_be_accepted(sessions, workspace_ids, stale_reason):
    owner, _ = workspace_ids
    orchestrator = Orchestrator(sessions)
    company = orchestrator.submit_customer_activity(owner, ACTIVITY)
    approval = HumanApproval(sessions)
    details = {"current": 1} if stale_reason == "tier_changed" else {}
    if stale_reason == "newer_activity":
        details["created_at"] = datetime(2000, 1, 1, tzinfo=UTC)
    evaluation_id = record_evaluation(sessions, company.company_id, **details)
    if stale_reason == "superseded":
        orchestrator.submit_customer_activity(owner, ACTIVITY)
    with pytest.raises(RecommendationConflict):
        approval.accept(owner, evaluation_id)
    with sessions() as session:
        assert session.get(Company, company.company_id).current_tier == 3
        assert not changes(session, company.company_id)


def test_acceptance_rolls_back_all_changes(sessions, workspace_ids, monkeypatch):
    owner, _ = workspace_ids
    company = Orchestrator(sessions).submit_customer_activity(owner, ACTIVITY)
    evaluation_id = record_evaluation(sessions, company.company_id)
    approval = HumanApproval(sessions)
    original_supersede = approval.supersede_pending

    def fail_after_tier_update(data, company_id):
        original_supersede(data, company_id)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(approval, "supersede_pending", fail_after_tier_update)
    with pytest.raises(RuntimeError, match="injected failure"):
        approval.accept(owner, evaluation_id)
    with sessions() as session:
        assert session.get(Company, company.company_id).current_tier == 3
        evaluation = session.get(Evaluation, evaluation_id)
        assert evaluation.decision_status == "pending" and evaluation.decided_at is None
        assert not changes(session, company.company_id)
