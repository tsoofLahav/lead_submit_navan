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
        "confidence": confidence, "material_eligibility_uncertainty": False, "findings": [{
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
def test_first_valid_submission_evaluates_even_without_business_indicators(sessions, workspace_ids, activity, status):
    store_reference_data(sessions)
    owner, _ = workspace_ids
    client, requests = mock_client(*uncertain_responses())
    result = Orchestrator(sessions, ai_client=client).submit_customer_reevaluation(owner, activity)
    assert result.gate.status == status and not result.gate.triggered_conditions
    assert result.recommendation.outcome == "undetermined" and not result.approval_available
    assert counts(sessions) == (1, 1, 1, 0) and len(requests) == 2
    inputs = json.loads(requests[0]["input"][1]["content"])
    assert inputs["trigger_results"]["business_indicators"] == []
    assert "status" not in inputs["trigger_results"]


def uncertain_responses():
    observed = {**assessment("insufficient"), "findings": [], "information_gaps": ["Eligibility is unknown."]}
    decision = {**recommendation("undetermined", None, "insufficient"), "supporting_evidence": []}
    return response(observed), response(decision)


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
        assert "potential_tier_mismatch" in data["trigger_results"]["business_indicators"]
    assert "Reserve the recommendation outcome" in requests[0]["input"][0]["content"]
    citations = json.loads(requests[1]["input"][1]["content"])["allowed_supporting_evidence"]
    assert {"annualized_gbv", "employee_count", "avg_monthly_bookings"} <= set(citations)
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
    (UPGRADE, [response(assessment()), response(recommendation("downgrade", 1)), response(recommendation("downgrade", 1))],
     CustomerStage.RECOMMENDATION, "recommendation.direction"),
    (DOWNGRADE, [response(assessment()), response({**recommendation("downgrade", 3), "supporting_evidence": ["fake_metric"]}),
                 response({**recommendation("downgrade", 3), "supporting_evidence": ["fake_metric"]})],
     CustomerStage.RECOMMENDATION, "recommendation.evidence_references"),
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


@pytest.mark.parametrize(("current", "gbv", "headcount", "adoption", "target", "decision"), [
    (1, "100000", None, "unknown", 2, "accept"),
    (1, "99999", 150, "declining", 2, "reject"),
    (1, "99999", 149, "unknown", 3, "accept"),
    (2, "99999", 149, "declining", 3, "stale"),
])
def test_final_downgrade_workflow(sessions, workspace_ids, current, gbv, headcount, adoption, target, decision):
    from app.human_approval import RecommendationConflict
    from scripts.smoke_revised_policy import seed_revised_policy

    seed_revised_policy(sessions)
    owner, _ = workspace_ids
    value = replace(
        EMPTY, months_since_signup=12, current_tier=current, total_gbv=Decimal(gbv),
        employee_count=headcount, travel_spend_trend="decreasing",
        product_adoption_trend="decreasing" if adoption == "declining" else "unknown",
        avg_monthly_active_users=Decimal(1) if adoption == "declining" else None,
    )
    keys = ["annualized_gbv"] if Decimal(gbv) >= 100_000 else ["annualized_gbv", "employee_count"]
    prediction = {**assessment(), "findings": [{"evidence": keys, "interpretation": "Current eligibility evidence."}]}
    proposed = {**recommendation("downgrade", target), "supporting_evidence": keys}
    client, requests = mock_client(response(prediction), response(proposed))
    workflow = Orchestrator(sessions, ai_client=client)
    result = workflow.submit_customer_reevaluation(owner, value)
    assert result.approval_available and result.recommendation.recommended_tier == target
    guidance = json.loads(requests[1]["input"][1]["content"])["quantitative_tier_guidance"]
    assert guidance["indicative_run_rate_meets_tier_1_threshold"] is False
    assert guidance["headcount_meets_tier_2_threshold"] == (None if headcount is None else headcount >= 150)
    assert len(requests) == 2 and all(not request["tools"] for request in requests)
    approval = HumanApproval(sessions)
    if decision == "stale":
        workflow.submit_customer_activity(owner, value)
        with pytest.raises(RecommendationConflict):
            approval.accept(owner, result.evaluation_id)
    else:
        getattr(approval, decision)(owner, result.evaluation_id)
    with sessions() as session:
        row = session.get(Company, result.company_id)
        assert row.current_tier == (target if decision == "accept" else current)
        change = session.scalar(select(TierChange).where(TierChange.evaluation_id == result.evaluation_id))
        if decision == "accept":
            assert (change.previous_tier, change.new_tier) == (current, target)
        else:
            assert change is None


def test_undetermined_is_valid_with_no_business_evidence_but_retain_requires_support():
    from app.ai.schemas import ActivityEvidence, ProductAssessmentOutput, RecommendationOutput
    from app.reevaluation_gate import ReevaluationGate
    from app.validation import AIOutputValidationError, validate_recommendation

    activity = EMPTY
    gate = ReevaluationGate.assess(activity)
    observed = ProductAssessmentOutput(
        confidence="insufficient", material_eligibility_uncertainty=True,
        findings=[{"evidence": [ActivityEvidence.MONTHS], "interpretation": "Completed months."}],
        rationale="No informative activity measurements were supplied.", information_gaps=[],
    )
    context = dict(current_tier=3, assessment=observed, activity=activity, metrics=gate.metrics, triggers=gate)
    proposal = {**recommendation("retain", 3, "low"), "supporting_evidence": ["months_since_signup"]}
    with pytest.raises(AIOutputValidationError, match="tier preference requires"):
        validate_recommendation(RecommendationOutput(**proposal), **context)
    proposal.update(outcome="undetermined", recommended_tier=None, confidence="insufficient")
    validate_recommendation(RecommendationOutput(**proposal), **context)


def test_activity_changes_compare_latest_normalized_values_and_policy(sessions, workspace_ids):
    store_reference_data(sessions)
    owner, _ = workspace_ids
    client, requests = mock_client(*(uncertain_responses() * 5))
    workflow = Orchestrator(sessions, ai_client=client)
    first = workflow.submit_customer_reevaluation(owner, EMPTY)
    with_bookings = replace(EMPTY, avg_monthly_bookings=Decimal("1.00"))
    second = workflow.submit_customer_reevaluation(owner, with_bookings)
    equivalent = workflow.submit_customer_reevaluation(owner, replace(with_bookings, avg_monthly_bookings=Decimal(1)))
    assert equivalent.unchanged and equivalent.activity_id == second.activity_id
    assert not second.gate.triggered_conditions
    third = workflow.submit_customer_reevaluation(owner, EMPTY)
    changed_tier = replace(EMPTY, current_tier=2)
    fourth = workflow.submit_customer_reevaluation(owner, changed_tier)
    with sessions.begin() as session:
        session.get(BusinessPolicy, 1).content += "\nNew policy revision."
    fifth = workflow.submit_customer_reevaluation(owner, changed_tier)
    assert len({item.activity_id for item in (first, second, third, fourth, fifth)}) == 5
    assert len(requests) == 10
    assert "1.00" not in requests[4]["input"][1]["content"]
    assert counts(sessions) == (1, 5, 5, 0)


def test_failed_sparse_activity_can_retry(sessions, workspace_ids):
    store_reference_data(sessions)
    owner, _ = workspace_ids
    client, requests = mock_client(500, *uncertain_responses())
    workflow = Orchestrator(sessions, ai_client=client)
    with pytest.raises(CustomerWorkflowFailure) as failed:
        workflow.submit_customer_reevaluation(owner, EMPTY)
    retry = workflow.retry_customer_reevaluation(owner, failed.value.activity_id)
    assert retry.recommendation.outcome == "undetermined"
    assert counts(sessions) == (1, 1, 1, 0) and len(requests) == 3


def test_concurrent_activity_is_blocked_before_any_ai_request(engine):
    from uuid import uuid4
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(engine)
    workflow = Orchestrator(factory)
    workspace_id = uuid4()
    with factory.begin() as competing:
        workflow._lock_customer_evaluation(competing, workspace_id, EMPTY.company_name)
        client, requests = mock_client()
        with pytest.raises(CustomerWorkflowFailure, match="activity_evaluation_in_progress"):
            Orchestrator(factory, ai_client=client).submit_customer_reevaluation(workspace_id, EMPTY)
        assert not requests
