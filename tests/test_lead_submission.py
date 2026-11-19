import json
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.ai.progress import ProgressKind
from app.ai.schemas import Signal
from app.human_approval import HumanApproval
from app.models import BusinessPolicy, Company, CompanyResearch, Evaluation, Prompt, TierChange
from app.orchestrator import LeadResult, LeadStage, LeadWorkflowFailure, Orchestrator
from app.validation import CustomerActivityInput, LeadInput
from test_ai import POLICY, SYSTEM, message, missing_research, mock_client, scoped_evidence, search_item, terminal

pytestmark = pytest.mark.write_integration
LEAD = LeadInput("Acme", "example.com", Decimal("100000"))
UNKNOWN = {
    "level": "unknown", "confidence": "insufficient", "supporting_signals": [],
    "rationale": "Evidence unavailable.", "information_gaps": ["Evidence unavailable."],
}


@pytest.fixture(autouse=True)
def reference_data(sessions):
    with sessions.begin() as session:
        session.add(BusinessPolicy(id=1, content=POLICY))
        session.add_all(Prompt(prompt_key=key, content=value) for key, value in {
            "lead_submit_system": SYSTEM,
            "lead_submit_research": "{{company_profile}} {{research_schema}}",
            "lead_submit_prediction": "{{company_profile}} {{validated_research_findings}} {{prediction_schema}}",
            "lead_submit_recommendation": "{{company_profile}} {{validated_predictions}} {{recommendation_schema}}",
        }.items())


def response(data, *, research=False):
    output = [search_item()] if research else []
    return [terminal([*output, message(json.dumps(data))])]


def research_data(*, mixed=False):
    data = missing_research().model_dump(mode="json")
    if not mixed:
        return data
    for finding in data["signals"]:
        signal = finding["signal"]
        if signal in {Signal.COMPANY_SIZE, Signal.EXTERNAL_ACTIVITY, Signal.GEOGRAPHY, Signal.GROWTH}:
            bounds = {"lower": 510, "upper": 510, "scope": "evaluated_population", "basis": "employees",
                      "as_of": None, "submitted_count_resolution": None} if signal == Signal.COMPANY_SIZE else None
            finding.update(status="found", information_gap=None, evidence=[scoped_evidence(
                510 if bounds else "Documented activity", employee_bounds=bounds,
                growth_direction="increasing" if signal == Signal.GROWTH else None,
                sources=[{"url": "https://example.com/report", "title": None, "publication_date": None}],
            )])
        if signal == Signal.EVENTS:
            finding.update(status="conflicting", evidence=[
                scoped_evidence(value, reliability="medium", sources=[
                    {"url": f"https://example.com/events/{value}", "title": None, "publication_date": None},
                ]) for value in (10, 20)
            ])
    return data


def prediction_data(*, mixed=False):
    travel = {key: value for key, value in UNKNOWN.items() if key != "level"}
    travel["intensity_ratings"] = []
    growth = UNKNOWN
    if mixed:
        travel.update(confidence="high",
                      supporting_signals=[Signal.COMPANY_SIZE, Signal.EXTERNAL_ACTIVITY, Signal.GEOGRAPHY],
                      intensity_ratings=[{"signal": signal, "rating": 2, "evidence_indices": [0], "material_uncertainty": False} for signal in (Signal.EXTERNAL_ACTIVITY, Signal.GEOGRAPHY)])
        growth = {**UNKNOWN, "level": "medium", "confidence": "low", "supporting_signals": [Signal.GROWTH]}
    return {"travel_spend_potential": travel, "growth_potential": growth, "expected_support_needs": UNKNOWN}


def decision_data(outcome="undetermined", tier=None, confidence="insufficient"):
    return {
        "outcome": outcome, "recommended_tier": tier, "confidence": confidence, "material_suitability_uncertainty": False,
        "key_findings": ["Evidence considered."], "supporting_evidence": ["travel_spend_potential", "submitted_annual_gbv"] if outcome == "upgrade" else [],
        "rationale": "Qualitative research informs the assessment.",
        "follow_up_actions": ["Follow up on evidence gaps."], "information_gaps": ["More observation needed."],
    }


def counts(sessions):
    with sessions() as session:
        return tuple(session.scalar(select(func.count()).select_from(model)) for model in (Company, CompanyResearch, Evaluation))


def test_complete_lead_with_missing_research_and_progress(sessions, workspace_ids):
    owner, _ = workspace_ids
    search_started = {"type": "response.web_search_call.in_progress", "item_id": "ws_test",
                      "output_index": 0, "sequence_number": 1}
    client, requests = mock_client(
        [search_started, *response(research_data(), research=True)], response(prediction_data()), response(decision_data()),
    )
    orchestrator = Orchestrator(sessions, ai_client=client)
    events = []

    def on_progress(event):
        events.append(event.kind)
        if event.kind == ProgressKind.RESEARCH_STARTED:
            assert counts(sessions) == (1, 0, 0)
        elif event.kind == ProgressKind.PREDICTION_STARTED:
            assert counts(sessions) == (1, 1, 0)
            with sessions.begin() as session:
                session.get(BusinessPolicy, 1).content = "Changed after the evaluation took its snapshot."
                session.get(Prompt, "lead_submit_prediction").content = "Changed after snapshot."
        elif event.kind == ProgressKind.RECOMMENDATION_STARTED:
            assert counts(sessions) == (1, 1, 0)
        elif event.kind == ProgressKind.EVALUATION_COMPLETED:
            assert counts(sessions) == (1, 1, 1)

    result = orchestrator.submit_lead(owner, LEAD, on_progress=on_progress)
    assert result.initial_tier == result.current_tier == 2
    assert not result.approval_available and result.recommendation.recommended_tier is None
    assert len(result.executions) == 3 and result.duration_seconds >= 0
    assert events == [
        ProgressKind.COMPANY_REGISTERED, ProgressKind.INITIAL_TIER_ASSIGNED, ProgressKind.RESEARCH_STARTED,
        ProgressKind.WEB_SEARCH_STARTED, ProgressKind.WEB_SEARCH_COMPLETED, ProgressKind.RESEARCH_COMPLETED,
        ProgressKind.PREDICTION_STARTED, ProgressKind.PREDICTION_COMPLETED, ProgressKind.RECOMMENDATION_STARTED,
        ProgressKind.RECOMMENDATION_COMPLETED, ProgressKind.EVALUATION_COMPLETED,
    ]
    assert [request["text"]["format"]["name"] for request in requests] == [
        "ScopedResearchOutput", "LeadAssessmentOutput", "LeadRecommendationOutput",
    ]
    for request in requests:
        assert request["input"][0]["content"].count(POLICY) == 1
        assert "previous_response_id" not in request
    assert requests[0]["tools"][0]["type"] == "web_search"
    assert not requests[1]["tools"] and not requests[2]["tools"]
    assert len(json.loads(requests[1]["input"][1]["content"])["validated_research_findings"]["signals"]) == 6
    decision_context = json.loads(requests[2]["input"][1]["content"])
    assert decision_context["current_tier"] == 2 and decision_context["validated_predictions"] == result.predictions.model_dump(mode="json")
    assert "submitted_annual_gbv" in decision_context["allowed_supporting_evidence"]
    assert "copy only exact identifiers" in requests[2]["input"][0]["content"]

    with sessions() as session:
        stored = session.get(Evaluation, result.evaluation_id)
        assert stored.research_id == result.research_id and stored.company_id == result.company_id
        assert (stored.evaluation_type, stored.current_tier, stored.decision_status) == ("lead_submit", 2, "not_applicable")
        assert stored.predictions["recommendation_evidence"]["outcome"] == "undetermined"
        assert stored.follow_up_actions == result.recommendation.follow_up_actions
        assert stored.information_gaps == result.recommendation.information_gaps
        assert session.get(CompanyResearch, result.research_id).signals == research_data()

    duplicate = orchestrator.submit_lead(owner, LEAD)
    assert duplicate == LeadResult(result.company_id, 2, False)
    assert len(requests) == 3 and counts(sessions) == (1, 1, 1)


@pytest.mark.parametrize(("stage", "expected_research"), [
    (LeadStage.RESEARCH, 0), (LeadStage.PREDICTION, 1), (LeadStage.RECOMMENDATION, 1),
])
def test_invalid_stage_preserves_only_committed_records(sessions, workspace_ids, stage, expected_research):
    owner, _ = workspace_ids
    research = response(research_data(), research=True)
    prediction = response(prediction_data())
    batches = {
        LeadStage.RESEARCH: [response({"signals": []}, research=True), response({"signals": []})],
        LeadStage.PREDICTION: [research, response({})],
        LeadStage.RECOMMENDATION: [research, prediction, response(decision_data("retain", 3, "low")),
                                   response(decision_data("retain", 3, "low"))],
    }
    client, requests = mock_client(*batches[stage])
    events = []
    with pytest.raises(LeadWorkflowFailure) as error:
        Orchestrator(sessions, ai_client=client).submit_lead(owner, LEAD, on_progress=events.append)
    failure = error.value
    assert failure.stage == stage and failure.company_id is not None
    assert (failure.research_id is not None) == bool(expected_research)
    assert failure.executions[-1].validation_passed is False
    assert (failure.predictions is not None) == (stage == LeadStage.RECOMMENDATION)
    assert len(requests) == len(batches[stage])
    assert events[-1].kind == ProgressKind.EVALUATION_FAILED and events[-1].workflow_stage == stage
    assert counts(sessions) == (1, expected_research, 0)


def test_scope_retry_and_actionable_human_approval(sessions, workspace_ids):
    owner, outsider = workspace_ids
    client, requests = mock_client(
        500, response(research_data(mixed=True), research=True), response(prediction_data(mixed=True)),
        response(decision_data("upgrade", 1, "high")),
        response(research_data(), research=True), response(prediction_data()),
        response(decision_data()),
    )
    orchestrator = Orchestrator(sessions, ai_client=client)
    with pytest.raises(LeadWorkflowFailure) as initial:
        orchestrator.submit_lead(owner, LeadInput("Acme", "example.com", Decimal("700000")))
    company_id = initial.value.company_id
    with pytest.raises(LeadWorkflowFailure, match="lead_not_found") as denied:
        orchestrator.retry_lead_evaluation(outsider, company_id)
    assert denied.value.company_id is None
    with pytest.raises(LeadWorkflowFailure, match="lead_not_found"):
        orchestrator.retry_lead_evaluation(owner, uuid4())
    assert len(requests) == 1
    Orchestrator(sessions).submit_customer_activity(
        owner, CustomerActivityInput("Acme", "example.com", 2, 10),
    )

    result = orchestrator.retry_lead_evaluation(owner, company_id)
    assert (result.initial_tier, result.current_tier, result.is_customer) == (1, 2, True)
    assert result.approval_available and counts(sessions) == (1, 1, 1)
    replacement = orchestrator.retry_lead_evaluation(owner, company_id)
    assert not replacement.approval_available and replacement.research_id != result.research_id
    with sessions() as session:
        assert session.get(Company, company_id).current_tier == 2
        assert session.get(Evaluation, result.evaluation_id).decision_status == "pending"
    approved = HumanApproval(sessions).accept(owner, result.evaluation_id)
    assert approved.current_tier == 1
    with sessions() as session:
        assert session.scalar(select(TierChange).where(TierChange.evaluation_id == result.evaluation_id)) is not None


@pytest.mark.parametrize("changed_tier", [False, True])
def test_newer_activity_or_tier_prevents_actionable_persistence(sessions, workspace_ids, changed_tier):
    owner, _ = workspace_ids
    lead = LeadInput("Acme", "example.com", Decimal("700000"))
    company = Orchestrator(sessions).register_lead(owner, lead)
    Orchestrator(sessions).submit_customer_activity(owner, CustomerActivityInput("Acme", "example.com", 2, 8))
    client, _ = mock_client(
        response(research_data(mixed=True), research=True), response(prediction_data(mixed=True)),
        response(decision_data("upgrade", 1, "high")),
    )

    def on_progress(event):
        if event.kind == ProgressKind.RECOMMENDATION_COMPLETED:
            Orchestrator(sessions).submit_customer_activity(
                owner, CustomerActivityInput("Acme", "example.com", 1 if changed_tier else 2, 9),
            )

    with pytest.raises(LeadWorkflowFailure, match="stale_company_state") as failure:
        Orchestrator(sessions, ai_client=client).retry_lead_evaluation(owner, company.company_id, on_progress=on_progress)
    assert failure.value.stage == LeadStage.EVALUATION_PERSISTENCE
    assert counts(sessions) == (1, 1, 0)


@pytest.mark.parametrize(
    ("failed_component", "expected_rule"),
    [
        ("research", "research.missing_evidence"),
        ("prediction", "prediction.travel_rating_references"),
        ("decision", "recommendation.direction"),
    ],
)
def test_workflow_corrects_only_failed_stage_with_tools_disabled(sessions, workspace_ids, failed_component, expected_rule):
    owner, _ = workspace_ids
    research = research_data(mixed=True)
    prediction = prediction_data(mixed=True)
    valid_decision = decision_data()
    if failed_component == "research":
        invalid = research_data(mixed=True)
        size = next(row for row in invalid["signals"] if row["signal"] == Signal.COMPANY_SIZE)
        size.update(status="missing", information_gap="Scope needs verification.")
        batches = [response(invalid, research=True), response(research), response(prediction), response(valid_decision)]
    elif failed_component == "prediction":
        invalid = prediction_data(mixed=True)
        invalid["travel_spend_potential"]["intensity_ratings"][0]["evidence_indices"] = [99]
        batches = [response(research, research=True), response(invalid), response(prediction), response(valid_decision)]
    else:
        batches = [response(research, research=True), response(prediction), response(decision_data("retain", 3, "low")),
                   response(valid_decision)]

    client, requests = mock_client(*batches)
    events = []
    result = Orchestrator(sessions, ai_client=client).submit_lead(owner, LEAD, on_progress=events.append)
    assert counts(sessions) == (1, 1, 1)
    assert len(requests) == 4
    assert [request["text"]["format"]["name"] for request in requests].count("ScopedResearchOutput") == (
        2 if failed_component == "research" else 1
    )
    assert sum(request["tool_choice"] == "required" for request in requests) == 1
    correction, = [request for request in requests if len(request["input"]) == 4]
    assert correction["tools"] == [] and correction["tool_choice"] == "none" and "max_tool_calls" not in correction
    original = requests[requests.index(correction) - 1]
    assert correction["input"][:2] == original["input"]
    context = json.loads(correction["input"][-1]["content"])
    assert context["validation_error"]["rule_id"] == expected_rule
    assert json.loads(context["invalid_structured_output"])
    started, = [event for event in events if event.kind == ProgressKind.OUTPUT_CORRECTION_STARTED]
    assert started.component == failed_component
    execution, = [item for item in result.executions if item.correction is not None]
    assert execution.validation_passed and execution.correction.succeeded
    assert execution.correction.initial_diagnostic.rule_id == expected_rule
    assert execution.correction.duration_seconds >= 0
    assert execution.request_rounds == 2


def test_v3_medium_forecast_upgrade_and_approval_below_quantitative_threshold(sessions, workspace_ids):
    owner, _ = workspace_ids
    lead = LeadInput("Acme", "example.com", Decimal("50000"), 150)
    observed = research_data(mixed=True)
    complexity = next(row for row in observed["signals"] if row["signal"] == Signal.COMPLEXITY)
    complexity.update(status="found", information_gap=None, evidence=[scoped_evidence(
        "Multi-region rollout coordination",
        sources=[{"url": "https://example.com/rollout", "title": None, "publication_date": None}],
    )])
    predicted = prediction_data(mixed=True)
    predicted["expected_support_needs"] = {
        **UNKNOWN, "level": "high", "confidence": "medium", "supporting_signals": [Signal.COMPLEXITY],
    }
    proposed = decision_data("upgrade", 1, "medium")
    proposed["supporting_evidence"] = ["travel_spend_potential", "expected_support_needs"]
    client, _ = mock_client(response(observed, research=True), response(predicted), response(proposed))
    result = Orchestrator(sessions, ai_client=client).submit_lead(owner, lead)
    assert result.current_tier == 2 and result.recommendation.recommended_tier == 1
    assert result.recommendation.confidence == "medium" and result.approval_available
    assert HumanApproval(sessions).accept(owner, result.evaluation_id).current_tier == 1


def test_v3_actionable_lead_change_rejects_unsupported_forecast(sessions, workspace_ids):
    owner, _ = workspace_ids
    proposal = decision_data("upgrade", 1, "medium")
    proposal["supporting_evidence"] = ["submitted_annual_gbv"]
    client, requests = mock_client(
        response(research_data(), research=True), response(prediction_data()), response(proposal), response(proposal),
    )
    with pytest.raises(LeadWorkflowFailure) as failure:
        Orchestrator(sessions, ai_client=client).submit_lead(owner, LEAD)
    assert failure.value.executions[-1].validation_diagnostic.rule_id == "recommendation.eligibility"
    assert counts(sessions) == (1, 1, 0) and len(requests) == 4
