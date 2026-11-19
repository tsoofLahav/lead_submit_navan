import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app import revised_policy as policy, validation
from app.ai.config import AISettings, Component
from app.ai.prompts import PromptRenderer, Task
from app.ai.schemas import (
    ActivityEvidence, EmployeeBounds, EvidenceRating, LeadAssessmentOutput, Prediction, ProductAssessmentOutput, RecommendationOutput,
    ResearchSource, ScopedResearchOutput, ScopedResearchSignal, ScopedSignalEvidence, Signal, TravelAssessment,
)
from app.ai.tools import ToolRegistry
from app.human_approval import HumanApproval, RecommendationConflict
from app.models import Company, Evaluation
from app.orchestrator import Orchestrator
from app.reevaluation_gate import ReevaluationGate
from app.validation import CustomerActivityInput, LeadInput
from scripts.smoke_revised_policy import seed_revised_policy
from test_ai import message, mock_client, search_item, terminal

SEED = Path(__file__).resolve().parents[1] / "seeds/business_prediction_policy.txt"
LEAD = LeadInput("Demo", "demo.test", Decimal("60000"))


def evidence(value, *, lower=None, upper=None, scope="evaluated_population", treatment="usable",
             reliability="high", as_of="2026-01-01", approximation=False, sourced=True, direction=None,
             key=None, period=None, replacement=None):
    bounds = EmployeeBounds(lower=lower, upper=upper, scope=scope, basis="employees",
                            as_of=as_of, submitted_count_resolution=replacement) if lower is not None or upper is not None else None
    return ScopedSignalEvidence(
        value=value, estimated_range=None, reliability=reliability, evidence_type="direct", context="Source context.",
        sources=[ResearchSource(url="https://example.test/source", title=None, publication_date=None)] if sourced else [],
        applicable=True, approximation=approximation, treatment=treatment, employee_bounds=bounds,
        observation_key=key, observation_period=period, growth_direction=direction,
    )


def research(employees, **other):
    signals = []
    for signal in Signal:
        observations = employees if signal == Signal.COMPANY_SIZE else other.get(signal.value, [])
        signals.append(ScopedResearchSignal(
            signal=signal, evidence=observations, status="conflicting" if len(observations) > 1 else
            "found" if observations else "missing",
            information_gap="Unavailable or uncertain" if len(observations) != 1 else None,
        ))
    return ScopedResearchOutput(signals=signals)


def prediction(level="unknown", confidence="insufficient", refs=None):
    return Prediction(level=level, confidence=confidence, supporting_signals=refs or [],
                      rationale="Evidence interprets the business case.", information_gaps=[])


def draft(*, extra=False, growth=None, support=None):
    axes = [Signal.SALES, Signal.CUSTOMER_FACING]
    if extra:
        axes.append(Signal.INTERNAL_COLLABORATION)
    return LeadAssessmentOutput(
        travel_spend_potential=TravelAssessment(
            supporting_signals=[Signal.COMPANY_SIZE, *axes], rationale="Travel demand reflects the referenced ratings.",
            confidence="high", information_gaps=[],
            intensity_ratings=[
                EvidenceRating(signal=axis, rating=grade, evidence_indices=[0], material_uncertainty=False)
                for axis, grade in zip(axes, (0, 1, 2))
            ],
        ),
        growth_potential=growth or prediction(),
        expected_support_needs=support or prediction(),
    )


@pytest.mark.parametrize(("observations", "expected"), [
    ([evidence(510, lower=510, upper=510), evidence(810, lower=810, upper=810)], "very_large"),
    ([evidence(30, lower=30, upper=30, scope="other"), evidence(510, lower=510, upper=510)], "very_large"),
    ([evidence("More than 500 employees in a verified subset", lower=501, scope="verified_subset")], "very_large"),
    ([evidence("Subset employed 100–120 people", lower=100, upper=120, scope="verified_subset")], None),
    ([evidence(480, lower=480, upper=480), evidence(550, lower=550, upper=550)], None),
    ([evidence("149–151 employees", lower=149, upper=151)], None),
    ([evidence(30, lower=510, upper=510)], None),
    ([evidence(30, lower=30, upper=30, scope="unknown")], None),
    ([evidence(40, lower=40, upper=40, treatment="weaker_or_outdated", reliability="medium", as_of="2020-01-01"),
      evidence(600, lower=600, upper=600, treatment="preferred")], "very_large"),
    ([evidence(40, lower=40, upper=40, treatment="weaker_or_outdated", as_of="2020-01-01"),
      evidence(600, lower=600, upper=600, treatment="preferred", reliability="medium")], None),
])
def test_scope_bands_and_stronger_recent_evidence(observations, expected):
    assert policy.company_scale(research(observations)) == expected


def test_sourced_approximation_and_isolated_exclusion_preserve_other_findings():
    result = policy.normalize_research(research(
        [evidence(600, lower=600, upper=600, approximation=True), evidence(800, sourced=False)],
        sales_and_account_management=[evidence("Regional enterprise sales")],
        hiring_activity=[evidence("Unsourced job count", sourced=False)],
    ))
    rows = {item.signal: item for item in result.signals}
    assert rows[Signal.COMPANY_SIZE].status == "estimated"
    assert len(rows[Signal.COMPANY_SIZE].evidence) == 1 and "excluded" in rows[Signal.COMPANY_SIZE].information_gap
    assert rows[Signal.SALES].evidence[0].value == "Regional enterprise sales"
    assert rows[Signal.HIRING].status == "missing" and rows[Signal.HIRING].evidence == []
    assert rows[Signal.HIRING].information_gap


def test_python_owns_scale_and_low_quality_ratings_do_not_change_score():
    data = research(
        [evidence(300, lower=300, upper=300)], sales_and_account_management=[evidence("Sales team")],
        customer_facing_operations=[evidence("On-site training")],
        internal_collaboration=[evidence("Offsites suspected", reliability="low")],
    )
    proposed = draft(extra=True)
    result = policy.compute_predictions(proposed, data).travel_spend_potential
    assert (result.company_scale, result.level, result.admitted_weight, result.weighted_score) == ("large", "medium", 6, "0.5")
    payload = proposed.model_dump(mode="json")
    payload["travel_spend_potential"].update(company_scale="very_large", level="very_high")
    with pytest.raises(ValidationError):
        LeadAssessmentOutput.model_validate_json(json.dumps(payload))


def test_independent_predictions_and_directional_evidence():
    data = research(
        [evidence(480, lower=480, upper=480), evidence(550, lower=550, upper=550)],
        business_growth=[evidence("Revenue rose", direction="increasing")],
        operational_complexity=[evidence("Separate billing entities")],
        industry_and_work_model=[evidence("Remote work with multiple regions")],
    )
    assessment = draft(growth=prediction("high", "high", [Signal.GROWTH]),
                       support=prediction("high", "high", [Signal.COMPLEXITY, Signal.INDUSTRY]))
    result = policy.compute_predictions(assessment, data)
    assert result.travel_spend_potential.level == "unknown"
    assert result.growth_potential.level == result.expected_support_needs.level == "high"

    mismatched = research([], hiring_activity=[evidence("Hiring up", direction="increasing", period="2026", key="1")],
                          funding=[evidence("Funding down", direction="decreasing", period="2026", key="2")])
    result = policy.compute_predictions(draft(growth=prediction("medium", "medium", [Signal.HIRING, Signal.FUNDING])), mismatched)
    assert result.growth_potential.level == "unknown"


def test_eligibility_requires_scoped_supported_bounds_and_preserves_input():
    def evaluate(rows):
        data = research(rows, sales_and_account_management=[evidence("Regional sales")],
                        customer_facing_operations=[evidence("Regular client visits")])
        proposed = draft()
        proposed.travel_spend_potential.intensity_ratings[0].rating = 2
        proposed.travel_spend_potential.intensity_ratings[1].rating = 2
        return policy.lead_targets(LEAD, 3, data, policy.compute_predictions(proposed, data))

    assert evaluate([evidence(220, lower=220, upper=220)]) == {2: ["company_size"]}
    assert evaluate([evidence("145–200", lower=145, upper=200)]) == {}
    assert evaluate([evidence(220, lower=220, upper=220, scope="unknown")]) == {}
    assert evaluate([evidence(220, lower=220, upper=220, approximation=True)]) == {}
    assert LEAD.employee_count is None
    activity = CustomerActivityInput("Demo", "demo.test", 3, 6, employee_count=220)
    assert policy.customer_targets(activity, None) == {2: ["employee_count"]}
    assert policy.customer_targets(activity, ProductAssessmentOutput(
        confidence="low", findings=[], rationale="Sparse", information_gaps=["Missing GBV"],
    )) == {}


def test_revised_renderer_uses_seeded_prompt_verbatim_except_placeholders():
    prompt_dir = SEED.parent / "prompts"
    task_template = (prompt_dir / "lead_submit_prediction.txt").read_text()
    templates = {
        "lead_submit_system": (prompt_dir / "lead_submit_system.txt").read_text(),
        Task.LEAD_PREDICTION: task_template,
    }
    rendered = PromptRenderer(None, ToolRegistry(), templates=templates).render(
        Task.LEAD_PREDICTION, policy=SEED.read_text(),
        inputs={"company_profile": LEAD, "validated_research_findings": research([])},
        settings=AISettings.from_env(Component.PREDICTION),
    )
    for line in task_template.splitlines():
        if line.strip() and "{{" not in line:
            assert line in rendered.instructions
    assert "{{" not in rendered.instructions
    assert json.loads(rendered.input_json)["submitted_data_provenance"].startswith("Entered by the user")


def test_compact_lead_evidence_preserves_observation_order_and_provenance():
    prompt_dir = SEED.parent / "prompts"
    renderer = PromptRenderer(None, ToolRegistry(), templates={
        "lead_submit_system": (prompt_dir / "lead_submit_system.txt").read_text(),
        Task.LEAD_PREDICTION: (prompt_dir / "lead_submit_prediction.txt").read_text(),
    })
    original = research([evidence(8000, lower=8000), evidence(9000, lower=9000)])
    rendered = renderer.render(Task.LEAD_PREDICTION, policy=SEED.read_text(), inputs={
        "company_profile": LEAD, "validated_research_findings": original,
    }, settings=AISettings.from_env(Component.PREDICTION))
    payload = json.loads(rendered.input_json)["validated_research_findings"]
    sources = payload.pop("sources")
    assert len(sources) == 1
    for signal in payload["signals"]:
        for observation in signal["evidence"]:
            observation["sources"] = [sources[index] for index in observation.pop("source_indices", [])]
    assert payload == original.model_dump(mode="json", exclude_none=True)
    assert rendered.context["validated_research_findings"] is original


@pytest.mark.parametrize("previous_title", [False, True])
def test_research_renderer_preserves_revised_schema_for_both_policy_editions(previous_title):
    prompt_dir = SEED.parent / "prompts"
    active_policy = SEED.read_text()
    if previous_title:
        lines = active_policy.splitlines(keepends=True)
        lines[1] = "Demo edition — revised evidence handling and validation (19 November 2026)\n"
        active_policy = "".join(lines)
    rendered = PromptRenderer(None, ToolRegistry(), templates={
        "lead_submit_system": (prompt_dir / "lead_submit_system.txt").read_text(),
        Task.LEAD_RESEARCH: (prompt_dir / "lead_submit_research.txt").read_text(),
    }).render(
        Task.LEAD_RESEARCH, policy=active_policy, inputs={"company_profile": LEAD},
        settings=AISettings.from_env(Component.RESEARCH),
    )
    assert rendered.output_schema is ScopedResearchOutput
    assert "TARGETED RESEARCH INSTRUCTIONS" in rendered.instructions


@pytest.mark.write_integration
@pytest.mark.parametrize("external_update", [False, True])
def test_revised_research_eligibility_persists_pending_and_preserves_approval(sessions, workspace_ids, external_update):
    seed_revised_policy(sessions)
    owner, _ = workspace_ids
    data = research([evidence(220, lower=220, upper=220)], sales_and_account_management=[evidence("Regional sales")],
                    customer_facing_operations=[evidence("Regular on-site visits")])
    proposed = draft()
    for rating in proposed.travel_spend_potential.intensity_ratings:
        rating.rating = 2
    decision = RecommendationOutput(
        outcome="upgrade", recommended_tier=2, confidence="high", key_findings=["Scoped headcount qualifies"],
        supporting_evidence=["company_size", "travel_spend_potential"], rationale="Scoped records establish headcount.",
        follow_up_actions=[], information_gaps=[],
    )
    client, requests = mock_client(
        [terminal([search_item(), message(data.model_dump_json())])],
        [terminal([message(proposed.model_dump_json())])],
        [terminal([message(decision.model_dump_json())])],
    )
    orchestrator = Orchestrator(sessions, ai_client=client)
    result = orchestrator.submit_lead(owner, LEAD)
    assert result.initial_tier == result.current_tier == 3 and result.approval_available
    assert json.loads(requests[2]["input"][1]["content"])["permissible_targets"] == {"2": ["company_size"]}
    with sessions() as session:
        assert session.get(Company, result.company_id).employee_count is None
        persisted = session.get(Evaluation, result.evaluation_id)
        assert persisted.decision_status == "pending"
        assert persisted.predictions["tier_alignment"] == "potential_underclassification"
    if external_update:
        orchestrator.submit_customer_activity(owner, CustomerActivityInput("Demo", "demo.test", 1, 6))
        with pytest.raises(RecommendationConflict):
            HumanApproval(sessions).accept(owner, result.evaluation_id)
        with sessions() as session:
            assert session.get(Company, result.company_id).current_tier == 1
            assert session.scalar(select(Evaluation.decision_status).where(Evaluation.id == result.evaluation_id)) == "superseded"
    else:
        assert HumanApproval(sessions).accept(owner, result.evaluation_id).current_tier == 2


def test_normalization_collects_independent_unresolved_provenance_errors():
    def with_conflicting_source_metadata(value):
        item = evidence(value)
        item.sources.append(item.sources[0].model_copy(update={"title": "Different title for same source"}))
        return item

    data = research(
        [with_conflicting_source_metadata("Unknown employee bounds")],
        sales_and_account_management=[with_conflicting_source_metadata("Regional account team")],
    )
    with pytest.raises(validation.AIOutputValidationError) as caught:
        policy.normalize_research(data)
    assert caught.value.diagnostic.rule_id == "research.duplicate_source_url"
    additional, = caught.value.additional_diagnostics
    assert additional.rule_id == "research.duplicate_source_url"
    assert additional.field_path != caught.value.diagnostic.field_path


@pytest.mark.write_integration
def test_revised_customer_qualifying_snapshot_key_is_available_even_if_assessment_omits_it(sessions, workspace_ids):
    seed_revised_policy(sessions)
    owner, _ = workspace_ids
    assessment = ProductAssessmentOutput(
        confidence="medium", findings=[{"evidence": [ActivityEvidence.MONTHS], "interpretation": "Six completed months."}],
        rationale="The recorded activity covers six completed months.", information_gaps=["Annual GBV is unverified."],
    )
    decision = RecommendationOutput(
        outcome="upgrade", recommended_tier=2, confidence="medium", key_findings=["Headcount qualifies for Tier 2."],
        supporting_evidence=["employee_count"], rationale="Headcount supports Tier 2 while Tier 1 GBV is unverified.",
        follow_up_actions=[], information_gaps=["Confirm annual GBV before considering Tier 1."],
    )
    client, requests = mock_client(
        [terminal([message(assessment.model_dump_json())])], [terminal([message(decision.model_dump_json())])],
    )
    result = Orchestrator(sessions, ai_client=client).submit_customer_reevaluation(
        owner, CustomerActivityInput("Demo", "demo.test", 3, 6, employee_count=220),
    )
    assert result.approval_available and result.recommendation.recommended_tier == 2
    data = json.loads(requests[1]["input"][1]["content"])
    assert data["eligible_tier_candidates"] == [1, 2]
    assert data["permissible_targets"] == {"2": ["employee_count"]}


def test_scoped_research_correction_cannot_invent_headcount_or_publication_date():
    from app.ai.client import _check_scoped_research_correction

    original = research([evidence(220, lower=220, upper=220)])
    _check_scoped_research_correction(original, research([]))
    corrected = research([evidence(620, lower=620, upper=620)])
    with pytest.raises(validation.AIOutputValidationError, match="preserve the original claim"):
        _check_scoped_research_correction(original, corrected)
    corrected = original.model_copy(deep=True)
    corrected.signals[0].evidence[0].sources[0].publication_date = "2026-01-01"
    with pytest.raises(validation.AIOutputValidationError, match="preserve the original claim"):
        _check_scoped_research_correction(original, corrected)
