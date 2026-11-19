import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app.ai.progress import ProgressKind
from app.human_approval import HumanApproval
from app.models import BusinessPolicy, Company, CompanyResearch, Evaluation, ProductActivity, Prompt, TierChange
from app.orchestrator import CustomerStage, CustomerWorkflowFailure, Orchestrator
from app.reevaluation_gate import GateStatus
from app.validation import CustomerActivityInput, LeadInput
from test_ai import POLICY, SYSTEM, message, mock_client, terminal

pytestmark = pytest.mark.write_integration
EMPTY = CustomerActivityInput("Synthetic Customer", "example.com", 3, 6)
UPGRADE = replace(
    EMPTY, total_gbv=Decimal("360000"), employee_count=220, avg_monthly_bookings=Decimal("180"),
    avg_monthly_active_users=Decimal("110"), avg_monthly_support_requests=Decimal("2"),
    travel_spend_trend="increasing", company_size_trend="stable", product_adoption_trend="increasing",
)
RETAIN = replace(UPGRADE, current_tier=2, total_gbv=Decimal("90000"), employee_count=100,
                 avg_monthly_active_users=Decimal("120"), avg_monthly_support_requests=Decimal("9"))
DOWNGRADE = replace(RETAIN, total_gbv=Decimal("10000"), employee_count=50, travel_spend_trend="decreasing")


def store_reference_data(sessions):
    with sessions.begin() as session:
        session.add(BusinessPolicy(id=1, content=POLICY))
        session.add_all(Prompt(prompt_key=key, content=value) for key, value in {
            "reevaluation_system": SYSTEM,
            "reevaluation_prediction": (
                "{{validated_product_data}} {{derived_metrics}} {{trigger_results}} {{product_assessment_schema}}"
            ),
            "reevaluation_recommendation": "{{current_tier}} {{validated_product_assessment}} {{recommendation_schema}}",
        }.items())


def response(data):
    return [terminal([message(json.dumps(data))])]


def assessment(confidence="medium"):
    return {
        "confidence": confidence, "findings": [{
            "evidence": ["annualized_gbv", "employee_count"], "interpretation": "Assess the submitted activity.",
        }],
        "rationale": "Based on the user-reported snapshot.", "information_gaps": ["Monthly stability is unknown."],
    }


def recommendation(outcome="upgrade", tier=1, confidence="medium"):
    return {
        "outcome": outcome, "recommended_tier": tier, "confidence": confidence,
        "key_findings": ["The annualized rate is indicative."], "supporting_evidence": ["annualized_gbv"],
        "rationale": "The indicative rate supports this tier assessment.",
        "follow_up_actions": ["Confirm observations after another period."],
        "information_gaps": ["Sustained activity is unknown."],
    }


def counts(sessions):
    with sessions() as session:
        return tuple(session.scalar(select(func.count()).select_from(model))
                     for model in (Company, ProductActivity, Evaluation, CompanyResearch))


@pytest.mark.parametrize(("activity", "status"), [
    (replace(EMPTY, travel_spend_trend="stable"), GateStatus.NO_TRIGGER),
    (EMPTY, GateStatus.INSUFFICIENT_DATA),
])
def test_gate_only_saves_activity_without_reading_policy_or_calling_ai(sessions, workspace_ids, activity, status):
    owner, _ = workspace_ids
    client, requests = mock_client()
    events = []
    result = Orchestrator(sessions, ai_client=client).submit_customer_reevaluation(owner, activity, on_progress=events.append)
    assert result.gate.status == status
    assert result.gate.explanation and result.gate.information_gaps
    assert not result.approval_available and result.predictions is None and result.evaluation_id is None
    assert counts(sessions) == (1, 1, 0, 0) and not requests
    assert [e.kind for e in events] == [
        ProgressKind.CUSTOMER_REGISTERED, ProgressKind.CUSTOMER_ACTIVITY_PERSISTED, ProgressKind.REEVALUATION_GATE_ASSESSED,
    ]
    assert events[-1].gate_status == status


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_upgrade_context_persistence_and_human_decision(sessions, workspace_ids, decision):
    store_reference_data(sessions)
    owner, _ = workspace_ids
    client, requests = mock_client(response(assessment()), response(recommendation()))
    orchestrator = Orchestrator(sessions, ai_client=client)
    existing = orchestrator.register_lead(owner, LeadInput("Synthetic Customer", "example.com", Decimal("999999")))
    with sessions.begin() as session:
        research = CompanyResearch(company_id=existing.company_id, signals={"history": "do not send past research"})
        session.add(research)
        session.flush()
        prior = Evaluation(
            company_id=existing.company_id, research_id=research.id, evaluation_type="lead_submit",
            current_tier=1, recommended_tier=2, predictions={"history": "do not send prior predictions"},
            confidence="medium", rationale="do not send prior recommendation",
        )
        session.add(prior)
        session.flush()
        prior_id = prior.id
    events = []

    def progress(event):
        events.append(event)
        if event.kind == ProgressKind.PREDICTION_STARTED:
            assert counts(sessions) == (1, 1, 1, 1)
            with sessions.begin() as session:
                session.get(BusinessPolicy, 1).content = "New policy after snapshot."
        if event.kind == ProgressKind.EVALUATION_COMPLETED:
            assert counts(sessions) == (1, 1, 2, 1)

    result = orchestrator.submit_customer_reevaluation(owner, UPGRADE, on_progress=progress)
    assert result.company_id == existing.company_id and result.customer.current_tier == 3
    assert result.approval_available and result.gate.metrics.annualized_gbv == 720000
    assert [e.kind for e in events] == [
        ProgressKind.CUSTOMER_UPDATED, ProgressKind.CUSTOMER_ACTIVITY_PERSISTED, ProgressKind.REEVALUATION_GATE_ASSESSED,
        ProgressKind.PREDICTION_STARTED, ProgressKind.PREDICTION_COMPLETED, ProgressKind.RECOMMENDATION_STARTED,
        ProgressKind.RECOMMENDATION_COMPLETED, ProgressKind.EVALUATION_COMPLETED,
    ]
    assert [r["text"]["format"]["name"] for r in requests] == ["ProductAssessmentOutput", "RecommendationOutput"]
    for request in requests:
        assert request["tools"] == [] and request["tool_choice"] == "none"
        assert request["input"][0]["content"].count(POLICY) == 1
        assert "Never attribute them" in request["input"][0]["content"]
        assert "previous_response_id" not in request
        content = request["input"][1]["content"]
        assert all(term not in content for term in ("999999", "do not send", "validated_research_findings"))
        data = json.loads(content)
        assert data["validated_product_data"]["company_name"] == UPGRADE.company_name
        assert Decimal(data["derived_metrics"]["annualized_gbv"]) == Decimal("720000")
        assert "potential_tier_mismatch" in data["trigger_results"]["triggered_conditions"]
    assert "Reserve the recommendation outcome" in requests[0]["input"][0]["content"]
    assert json.loads(requests[1]["input"][1]["content"])["allowed_supporting_evidence"] == ["annualized_gbv", "employee_count"]
    with sessions() as session:
        evaluation = session.get(Evaluation, result.evaluation_id)
        assert (evaluation.evaluation_type, evaluation.research_id, evaluation.decision_status) == ("re_evaluation", None, "pending")
        assert evaluation.predictions["activity_snapshot_id"] == str(result.activity_id)
        assert evaluation.predictions["findings"] == assessment()["findings"]
        assert session.get(Evaluation, prior_id).decision_status == "superseded"
        assert session.get(Company, result.company_id).current_tier == 3
        external = session.scalar(select(TierChange).where(TierChange.evaluation_id.is_(None)))
        assert (external.previous_tier, external.new_tier) == (1, 3)
    approved = getattr(HumanApproval(sessions), decision)(owner, result.evaluation_id)
    assert approved.current_tier == (1 if decision == "accept" else 3)
    with sessions() as session:
        history = session.scalar(select(TierChange).where(TierChange.evaluation_id == result.evaluation_id))
        assert (history is not None) == (decision == "accept")


def test_triggered_retain_keeps_adoption_caveat(sessions, workspace_ids):
    store_reference_data(sessions)
    owner, _ = workspace_ids
    client, requests = mock_client(response(assessment()), response(recommendation("retain", 2)))
    result = Orchestrator(sessions, ai_client=client).submit_customer_reevaluation(owner, RETAIN)
    assert result.gate.metrics.adoption_rate_percent == 120
    assert any("exceeds 100%" in gap for gap in result.gate.information_gaps)
    assert "exceeds 100%" in requests[0]["input"][1]["content"]
    assert result.customer.current_tier == 2 and not result.approval_available
    with sessions() as session:
        evaluation = session.get(Evaluation, result.evaluation_id)
        assert evaluation.decision_status == "not_applicable" and evaluation.research_id is None


@pytest.mark.parametrize(("activity", "batches", "stage", "rule"), [
    (UPGRADE, [500], CustomerStage.PREDICTION, None),
    (UPGRADE, [response(assessment("high"))], CustomerStage.PREDICTION, None),
    (UPGRADE, [response(assessment()), response(recommendation(confidence="high")), response(recommendation(confidence="high"))],
     CustomerStage.RECOMMENDATION, "recommendation.snapshot_high_confidence"),
    (DOWNGRADE, [response(assessment()), response(recommendation("downgrade", 3)), response(recommendation("downgrade", 3))],
     CustomerStage.RECOMMENDATION, "recommendation.adoption_rule_unset"),
])
def test_technical_or_invalid_output_keeps_activity(sessions, workspace_ids, activity, batches, stage, rule):
    store_reference_data(sessions)
    owner, _ = workspace_ids
    client, requests = mock_client(*batches)
    events = []
    with pytest.raises(CustomerWorkflowFailure) as caught:
        Orchestrator(sessions, ai_client=client).submit_customer_reevaluation(owner, activity, on_progress=events.append)
    failure = caught.value
    assert failure.stage == stage and failure.activity_id is not None and failure.gate.status == GateStatus.TRIGGERED
    assert events[-1].kind == ProgressKind.EVALUATION_FAILED and events[-1].workflow_stage == stage
    assert counts(sessions) == (1, 1, 0, 0) and len(requests) == len(batches)
    if rule:
        assert failure.executions[-1].validation_diagnostic.rule_id == rule


def test_retry_reuses_saved_activity_and_blocks_cross_workspace_or_completed_retry(sessions, workspace_ids):
    store_reference_data(sessions)
    owner, outsider = workspace_ids
    client, requests = mock_client(500, response(assessment()), response(recommendation()))
    orchestrator = Orchestrator(sessions, ai_client=client)
    with pytest.raises(CustomerWorkflowFailure) as failure:
        orchestrator.submit_customer_reevaluation(owner, UPGRADE)
    activity_id = failure.value.activity_id
    with pytest.raises(CustomerWorkflowFailure, match="activity_not_found") as denied:
        orchestrator.retry_customer_reevaluation(outsider, activity_id)
    assert denied.value.activity_id is None and len(requests) == 1
    result = orchestrator.retry_customer_reevaluation(owner, activity_id)
    assert result.activity_id == activity_id and counts(sessions) == (1, 1, 1, 0)
    with pytest.raises(CustomerWorkflowFailure, match="evaluation_already_completed"):
        orchestrator.retry_customer_reevaluation(owner, activity_id)
    assert len(requests) == 3


@pytest.mark.parametrize("mutate_during", [ProgressKind.REEVALUATION_GATE_ASSESSED, ProgressKind.RECOMMENDATION_COMPLETED])
@pytest.mark.parametrize("change", ["activity", "tier"])
def test_stale_activity_or_tier_is_rejected(sessions, workspace_ids, mutate_during, change):
    store_reference_data(sessions)
    owner, _ = workspace_ids
    client, requests = mock_client(response(assessment()), response(recommendation()))
    orchestrator = Orchestrator(sessions, ai_client=client)

    def progress(event):
        if event.kind != mutate_during:
            return
        if change == "activity":
            orchestrator.submit_customer_activity(owner, UPGRADE)
        else:
            with sessions.begin() as session:
                company = session.scalar(select(Company))
                snapshot = session.scalar(select(ProductActivity))
                company.current_tier = 2
                session.add(TierChange(
                    company_id=company.id, previous_tier=3, new_tier=2,
                    created_at=snapshot.created_at + timedelta(microseconds=1),
                ))

    with pytest.raises(CustomerWorkflowFailure, match=f"stale_{change}") as failure:
        orchestrator.submit_customer_reevaluation(owner, UPGRADE, on_progress=progress)
    assert failure.value.stage == (
        CustomerStage.CONTEXT if mutate_during == ProgressKind.REEVALUATION_GATE_ASSESSED
        else CustomerStage.EVALUATION_PERSISTENCE
    )
    assert len(requests) == (0 if mutate_during == ProgressKind.REEVALUATION_GATE_ASSESSED else 2)
    with pytest.raises(CustomerWorkflowFailure, match=f"stale_{change}"):
        orchestrator.retry_customer_reevaluation(owner, failure.value.activity_id)
    assert counts(sessions)[2] == 0
