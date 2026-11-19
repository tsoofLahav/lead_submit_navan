import json
from decimal import Decimal
from time import monotonic
from uuid import uuid4

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError as PydanticError

from app import create_app
from app.ai.budget import EvaluationBudget
from app.ai.client import AIClient, AIExecutionError, FailureKind
from app.ai.config import AISettings, Component
from app.ai.progress import ProgressKind
from app.ai.prompts import PromptRenderError, PromptRenderer, Task
from app.ai.schemas import (
    LeadAssessmentOutput, ProductAssessmentOutput, RecommendationOutput, ResearchOutput, ScopedResearchOutput, Signal, StructuredOutput,
    evaluation_output_fields,
)
from app.ai.tools import ReadOnlyFunction, ToolError, ToolRegistry, strict_schema
from app.reference_data import PromptNotFound
from app.reevaluation_gate import ReevaluationGate
from app.validation import (
    AIOutputValidationError, CustomerActivityInput, LeadInput,
    validate_product_assessment, validate_recommendation, validate_research,
)

POLICY = "Synthetic authoritative policy"
SYSTEM = "{{business_prediction_policy}} {{available_tools}} {{execution_limits}}"
LEAD = LeadInput("Acme {{business_prediction_policy}}", "example.com", Decimal(100_000))
ACTIVITY = CustomerActivityInput(
    "Acme", "example.com", 3, 4, total_gbv=Decimal(50_000), travel_spend_trend="increasing"
)


class ChatOutput(StructuredOutput):
    answer: str


class CompanyArgs(StructuredOutput):
    company_name: str


@pytest.fixture
def renderer(monkeypatch):
    templates = {
        "lead_submit_system": SYSTEM,
        "reevaluation_system": SYSTEM,
        "chat_assistant_system": SYSTEM,
        "lead_submit_research": "{{company_profile}} {{research_schema}}",
        "lead_submit_prediction": "{{company_profile}} {{validated_research_findings}} {{prediction_schema}}",
        "lead_submit_recommendation": "{{company_profile}} {{validated_predictions}} {{recommendation_schema}}",
        "reevaluation_prediction": (
            "{{validated_product_data}} {{derived_metrics}} {{trigger_results}} {{product_assessment_schema}}"
        ),
        "reevaluation_recommendation": "{{current_tier}} {{validated_product_assessment}} {{recommendation_schema}}",
    }

    def get_prompt(session, key):
        if key not in templates:
            raise PromptNotFound(key)
        return templates[key]

    monkeypatch.setattr("app.ai.prompts.get_prompt", get_prompt)
    monkeypatch.setattr("app.ai.prompts.get_business_policy", lambda _: POLICY)
    return PromptRenderer(None, ToolRegistry()), templates


def missing_research():
    return ResearchOutput.model_validate_json(json.dumps({"signals": [
        {"signal": signal, "status": "missing", "evidence": [], "information_gap": "No reliable details."}
        for signal in Signal
    ]}))


def scoped_evidence(value, **fields):
    return {"value": value, "estimated_range": None, "reliability": "high", "evidence_type": "direct",
            "sources": [], "context": None, "applicable": True, "approximation": False,
            "treatment": "usable", "employee_bounds": None, "observation_key": None,
            "observation_period": None, "growth_direction": None, **fields}


def render_research(renderer):
    prompts, _ = renderer
    return prompts.render(
        Task.LEAD_RESEARCH, policy=prompts.load_policy(), inputs={"company_profile": LEAD},
        settings=AISettings.from_env(Component.RESEARCH),
    )


def render_chat(renderer, tools=None):
    prompts, _ = renderer
    if tools is not None:
        prompts.tools = tools
    return prompts.render(
        Task.CHAT, policy=POLICY, inputs={"message": "Describe Acme."},
        settings=AISettings.from_env(Component.CHAT),
    )


def terminal(items, status="completed"):
    return {
        "type": f"response.{status}", "sequence_number": 10,
        "response": {
            "id": "resp_test", "object": "response", "created_at": 0,
            "status": status, "service_tier": "fast", "model": "gpt-6.2-astra",
            "output": items, "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        },
    }


def message(text=None, refusal=False):
    content = (
        {"type": "refusal", "refusal": "private refusal details"}
        if refusal else {"type": "output_text", "text": text, "annotations": []}
    )
    return {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed", "content": [content]}


def search_item():
    return {"type": "web_search_call", "id": "ws_test", "action": {"type": "search"}, "status": "completed"}


def mock_client(*batches, tools=None):
    requests = []
    remaining = iter(batches)

    def handler(request):
        requests.append(json.loads(request.content))
        batch = next(remaining)
        if isinstance(batch, Exception):
            raise batch
        if isinstance(batch, int):
            return httpx.Response(batch, json={"error": {"message": "private details"}})
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in batch) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    sdk = OpenAI(api_key="mock-only", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=2)
    return AIClient(tools, sdk=sdk), requests


@pytest.mark.parametrize("component", list(Component))
def test_component_defaults_and_overrides(monkeypatch, component):
    settings = AISettings.from_env(component)
    assert settings.model == "gpt-6.2-astra"
    assert settings.reasoning_effort == ("medium" if component == Component.RESEARCH else "low")
    assert settings.service_tier == "fast"
    assert settings.max_output_tokens == dict(research=8000, prediction=2000, decision=2000, chat=25000)[component]
    prefix = f"AI_{component.upper()}"
    monkeypatch.setenv(f"{prefix}_MODEL", "configured-model")
    monkeypatch.setenv(f"{prefix}_REASONING_EFFORT", "high")
    monkeypatch.setenv(f"{prefix}_SERVICE_TIER", "default")
    overrides = AISettings.from_env(component)
    assert (overrides.model, overrides.reasoning_effort, overrides.service_tier) == ("configured-model", "high", "default")
    options = ToolRegistry().request_options(component, overrides)
    if component == Component.RESEARCH:
        assert options["max_tool_calls"] == 5
        assert options["tool_choice"] == "required"
    else:
        assert options == {"tools": [], "tool_choice": "none"}


def test_prompt_context_is_explicit_and_role_separated(renderer):
    prompt = render_research(renderer)
    assert POLICY in prompt.instructions
    assert "Acme" not in prompt.instructions
    assert json.loads(prompt.input_json)["company_profile"]["company_name"] == LEAD.company_name
    assert "ResearchOutput" in prompt.instructions
    assert "{{research_schema}}" not in prompt.instructions


def test_missing_prompts_and_required_placeholders(renderer):
    prompts, templates = renderer
    templates["lead_submit_research"] = "{{company_profile}}"
    with pytest.raises(PromptRenderError, match="required placeholder"):
        render_research(renderer)
    templates["lead_submit_research"] = "{{company_profile}} {{research_schema}} {{unknown}}"
    with pytest.raises(PromptRenderError, match="unknown placeholder"):
        render_research(renderer)
    del templates["lead_submit_research"]
    with pytest.raises(PromptNotFound):
        render_research(renderer)
    with pytest.raises(PromptRenderError, match="required fields"):
        prompts.render(Task.LEAD_RESEARCH, policy=POLICY, inputs={}, settings=AISettings.from_env(Component.RESEARCH))


def test_only_explicit_activity_context_and_deterministic_metrics(renderer):
    prompts, _ = renderer
    gate = ReevaluationGate.assess(ACTIVITY)
    inputs = {"validated_product_data": ACTIVITY, "derived_metrics": gate.metrics, "trigger_results": gate}
    settings = AISettings.from_env(Component.PREDICTION)
    prompt = prompts.render(Task.REEVALUATION_PREDICTION, policy=POLICY, inputs=inputs, settings=settings)
    assert "submitted_annual_gbv" not in prompt.input_json
    payload = json.loads(prompt.input_json)
    assert payload["derived_metrics"]["annualized_gbv"] == "150000"
    assert "metrics" not in payload["trigger_results"]
    assert prompt.context["trigger_results"].metrics == gate.metrics
    with pytest.raises(PromptRenderError):
        prompts.render(Task.REEVALUATION_PREDICTION, policy=POLICY, inputs={**inputs, "derived_metrics": None}, settings=settings)


def test_sdk_request_stream_progress_and_budget(renderer, monkeypatch):
    monkeypatch.setenv("AI_RESEARCH_MAX_TOOL_CALLS", "1")
    prompt = render_research(renderer)
    started = {"type": "response.web_search_call.in_progress", "item_id": "ws_test", "output_index": 0, "sequence_number": 1}
    completed = {**started, "type": "response.web_search_call.completed", "sequence_number": 2}
    events = [started, started, completed, completed, terminal([search_item(), message(missing_research().model_dump_json())])]
    client, requests = mock_client(events)
    progress = []
    result = client.execute(prompt, on_progress=progress.append)
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "gpt-6.2-astra" and request["reasoning"] == {"effort": "medium"}
    assert request["service_tier"] == "fast" and request["max_tool_calls"] == 1
    assert request["tools"] == [{"type": "web_search", "search_context_size": "low"}]
    assert request["stream"] is True and request["store"] is False
    assert request["tool_choice"] == "required" and "previous_response_id" not in request
    assert request["text"]["format"]["strict"] is True
    assert [entry["role"] for entry in request["input"]] == ["developer", "user"]
    assert [event.kind for event in progress] == [ProgressKind.WEB_SEARCH_STARTED, ProgressKind.WEB_SEARCH_COMPLETED]
    assert result.metadata.completed_hosted_tool_calls == 1
    assert result.metadata.usage.total_tokens == 30 and result.metadata.validation_passed
    assert result.metadata.actual_service_tier == "fast" and result.metadata.duration_seconds >= 0
    assert result.output.model_dump() == missing_research().model_dump()


@pytest.mark.parametrize(
    ("status", "items", "failure"),
    [
        ("completed", [message(refusal=True)], FailureKind.REFUSAL),
        ("incomplete", [message("{")], FailureKind.INCOMPLETE),
        ("completed", [search_item(), message("{")], FailureKind.MALFORMED_OUTPUT),
        ("completed", [search_item(), message('{"signals": []}')], FailureKind.VALIDATION),
        ("completed", [message('{"signals": []}')], FailureKind.TOOL_ERROR),
    ],
)
def test_terminal_failures(renderer, status, items, failure):
    client, calls = mock_client([terminal(items, status)])
    with pytest.raises(AIExecutionError) as error:
        client.execute(render_research(renderer))
    assert error.value.kind == failure and len(calls) == 1
    assert error.value.metadata.failure == failure
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (401, FailureKind.AUTHENTICATION), (429, FailureKind.RATE_LIMIT), (400, FailureKind.CONFIGURATION),
        (500, FailureKind.API_ERROR), (httpx.ConnectError("private information"), FailureKind.CONNECTION),
        (httpx.ReadTimeout("private information"), FailureKind.TIMEOUT),
    ],
)
def test_transport_failures_have_no_retry(renderer, response, kind):
    client, requests = mock_client(response)
    with pytest.raises(AIExecutionError) as error:
        client.execute(render_research(renderer))
    assert error.value.kind == kind and len(requests) == 1
    assert "private" not in str(error.value)


def test_without_openai_key(renderer, monkeypatch):
    monkeypatch.setattr("app.load_dotenv", lambda *_: None)
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    assert create_app()
    with pytest.raises(AIExecutionError) as error:
        AIClient().execute(render_research(renderer))
    assert error.value.kind == FailureKind.CONFIGURATION


def test_chat_dispatches_authorized_function_using_server_workspace(renderer):
    workspace = uuid4()
    invoked = []
    tools = ToolRegistry([ReadOnlyFunction(
        "lookup_company", "Retrieve company.", CompanyArgs,
        lambda workspace_id, args: invoked.append((workspace_id, args.company_name)) or {"found": True},
    )])
    call = {
        "type": "function_call", "name": "lookup_company", "id": "fc_test", "call_id": "call_test",
        "arguments": '{"company_name":"Acme"}', "status": "completed",
    }
    client, requests = mock_client([terminal([call])], [terminal([message('{"answer":"Found."}')])], tools=tools)
    result = client.execute(render_chat(renderer, tools), chat_schema=ChatOutput, workspace_id=workspace)
    assert invoked == [(workspace, "Acme")]
    assert result.output.answer == "Found." and result.metadata.usage.total_tokens == 60
    assert requests[1]["input"][-1] == {"type": "function_call_output", "call_id": "call_test", "output": '{"found": true}'}
    assert requests[0]["parallel_tool_calls"] is False
    assert ToolRegistry().request_options(Component.DECISION, AISettings.from_env(Component.DECISION))["tools"] == []
    with pytest.raises(ToolError):
        tools.dispatch(Component.CHAT, workspace, "lookup_company", '{"company_name":"Acme","workspace_id":"fake"}')
    with pytest.raises(ToolError):
        tools.dispatch(Component.PREDICTION, workspace, "lookup_company", '{"company_name":"Acme"}')


def test_strict_schemas_and_conflicting_research():
    for model in (ScopedResearchOutput, LeadAssessmentOutput, ProductAssessmentOutput, RecommendationOutput):
        assert strict_schema(model)["additionalProperties"] is False
    with pytest.raises(PydanticError):
        RecommendationOutput.model_validate_json('{"outcome":"invalid"}')
    research = missing_research()
    validate_research(research)
    data = research.model_dump(mode="json")
    data["signals"][0].update(status="conflicting", evidence=[
        {
            "value": value, "estimated_range": None, "reliability": "medium", "evidence_type": "direct",
            "sources": [{"url": url, "title": "Public filing", "publication_date": None}], "context": None,
        }
        for value, url in ((100, "https://example.com/one"), (200, "https://example.com/two"))
    ])
    conflicting = ResearchOutput.model_validate_json(json.dumps(data))
    validate_research(conflicting)
    assert conflicting.model_dump(mode="json") == data
    conflicting.signals[0].evidence.pop()
    with pytest.raises(AIOutputValidationError, match="Conflicts"):
        validate_research(conflicting)


def test_snapshot_recommendation_is_eligible_and_mappable():
    gate = ReevaluationGate.assess(ACTIVITY)
    assessment = ProductAssessmentOutput.model_validate_json(json.dumps({
        "findings": [{"evidence": ["annualized_gbv"], "interpretation": "Tier 2 run-rate eligibility."}],
        "confidence": "medium", "material_eligibility_uncertainty": False, "rationale": "Evidence from the current snapshot.", "information_gaps": [],
    }))
    decision = RecommendationOutput(
        outcome="upgrade", recommended_tier=2, confidence="medium", key_findings=["Run rate"],
        supporting_evidence=["annualized_gbv"], rationale="Indicative annual GBV meets Tier 2.",
        follow_up_actions=[], information_gaps=[],
    )
    context = dict(current_tier=3, assessment=assessment, activity=ACTIVITY, metrics=gate.metrics, triggers=gate)
    validate_recommendation(decision, **context)
    fields = evaluation_output_fields(assessment, decision)
    assert fields["predictions"]["recommendation_evidence"]["outcome"] == "upgrade"
    assert json.loads(json.dumps(fields))["recommended_tier"] == 2
    assert not {"company_id", "decision_status", "decided_at"} & fields.keys()
    validate_recommendation(decision.model_copy(update={"recommended_tier": 1, "confidence": "low"}), **context)
    with pytest.raises(AIOutputValidationError, match="direction"):
        validate_recommendation(decision.model_copy(update={"outcome": "downgrade"}), **context)
    with pytest.raises(AIOutputValidationError, match="unavailable"):
        validate_recommendation(decision.model_copy(update={"supporting_evidence": ["employee_count"]}), **context)
    assessment.findings[0].evidence.append("employee_count")
    with pytest.raises(AIOutputValidationError, match="missing"):
        validate_product_assessment(assessment, ACTIVITY, gate.metrics, gate)


def test_recommendation_tier_rejects_json_boolean():
    with pytest.raises(PydanticError):
        RecommendationOutput.model_validate_json(json.dumps({
            "outcome": "retain", "recommended_tier": True, "confidence": "low", "key_findings": [],
            "supporting_evidence": [], "rationale": "Synthetic rationale.",
            "follow_up_actions": [], "information_gaps": [],
        }))


def test_stream_transport_timeout_is_typed(renderer):
    class StalledStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b": keepalive\n\n"
            raise httpx.ReadTimeout("private stalled connection")

    transport = httpx.MockTransport(lambda _: httpx.Response(
        200, stream=StalledStream(), headers={"content-type": "text/event-stream"},
    ))
    sdk = OpenAI(api_key="mock-only", http_client=httpx.Client(transport=transport))
    with pytest.raises(AIExecutionError) as error:
        AIClient(sdk=sdk).execute(render_research(renderer))
    assert error.value.kind == FailureKind.TIMEOUT


def test_unknown_chat_function_stops_without_continuation(renderer):
    tools = ToolRegistry([ReadOnlyFunction("approved", "Approved lookup.", CompanyArgs, lambda *_: {})])
    call = {
        "type": "function_call", "name": "unapproved", "id": "fc_test", "call_id": "call_test",
        "arguments": '{"company_name":"Acme"}', "status": "completed",
    }
    client, requests = mock_client([terminal([call])], tools=tools)
    with pytest.raises(AIExecutionError) as error:
        client.execute(render_chat(renderer, tools), chat_schema=ChatOutput, workspace_id=uuid4())
    assert error.value.kind == FailureKind.TOOL_ERROR and len(requests) == 1


def test_snapshot_downgrade_does_not_require_all_eligibility_references():
    activity = CustomerActivityInput(
        "Acme", "example.com", 2, 12, total_gbv=Decimal(10_000), employee_count=20,
        avg_monthly_active_users=Decimal(18), travel_spend_trend="decreasing",
        product_adoption_trend="stable",
    )
    gate = ReevaluationGate.assess(activity)
    assessment = ProductAssessmentOutput.model_validate_json(json.dumps({
        "confidence": "medium", "material_eligibility_uncertainty": False,
        "findings": [{"evidence": ["annualized_gbv", "adoption_rate_percent"], "interpretation": "Declining usage."}],
        "rationale": "Current activity assessment.", "information_gaps": [],
    }))
    decision = RecommendationOutput(
        outcome="downgrade", recommended_tier=3, confidence="medium", key_findings=["Run rate"],
        supporting_evidence=["annualized_gbv", "adoption_rate_percent"], rationale="Lower tier eligibility.",
        follow_up_actions=[], information_gaps=[],
    )
    context = dict(current_tier=2, assessment=assessment, activity=activity, metrics=gate.metrics, triggers=gate)
    validate_recommendation(decision, **context)
    validate_recommendation(decision.model_copy(update={
        "supporting_evidence": ["annualized_gbv", "employee_count"],
    }), **context)


def test_smoke_diagnostics_preserve_rule_safe_value_usage_and_phases(renderer, caplog):
    research = missing_research().model_dump(mode="json")
    research["signals"][0]["signal"] = Signal.EVENTS
    research["signals"][1]["information_gap"] = "private customer information"
    final = terminal([search_item(), message(json.dumps(research))])
    final["response"]["usage"].update(
        input_tokens_details={"cached_tokens": 4},
        output_tokens_details={"reasoning_tokens": 12},
    )
    started = {
        "type": "response.web_search_call.in_progress", "item_id": "ws_test",
        "output_index": 0, "sequence_number": 1,
    }
    completed = {**started, "type": "response.web_search_call.completed", "sequence_number": 2}
    delta = {
        "type": "response.output_text.delta", "item_id": "msg_test", "delta": "{",
        "output_index": 1, "content_index": 0, "sequence_number": 3,
    }
    client, requests = mock_client([started, started, completed, delta, final])
    with caplog.at_level("INFO"), pytest.raises(AIExecutionError) as error:
        client.execute(render_research(renderer))
    metadata = error.value.metadata
    assert error.value.kind == FailureKind.VALIDATION and len(requests) == 1
    assert metadata.validation_passed is False
    diagnostic = metadata.validation_diagnostic
    assert (diagnostic.rule_id, diagnostic.policy_section, diagnostic.field_path) == ("research.signal_coverage", "3.1", "signals")
    assert metadata.usage.cached_input_tokens == 4
    assert metadata.usage.reasoning_output_tokens == 12
    phases = metadata.phases
    search, = phases.searches
    assert phases.first_search_seconds == search.started_seconds
    assert search.started_seconds <= search.completed_seconds <= phases.first_output_delta_seconds <= phases.final_response_seconds
    assert phases.after_last_search_seconds == pytest.approx(phases.final_response_seconds - search.completed_seconds)
    assert phases.validation_seconds >= 0
    assert "private customer information" not in caplog.text


def test_rejected_source_reports_path_without_exposing_url(renderer, caplog):
    research = missing_research().model_dump(mode="json")
    research["signals"][0].update(status="missing", evidence=[scoped_evidence(
        10, sources=[{"url": "https://private-secret@example.com", "title": "private title", "publication_date": None}],
    )])
    client, _ = mock_client([terminal([search_item(), message(json.dumps(research))])])
    with caplog.at_level("INFO"), pytest.raises(AIExecutionError) as error:
        client.execute(render_research(renderer))
    metadata = error.value.metadata
    diagnostic = metadata.validation_diagnostic
    assert diagnostic.rule_id == "research.missing_evidence"
    assert diagnostic.field_path == "signals[0].evidence"
    assert diagnostic.observed == {"count": 1}
    assert metadata.usage.cached_input_tokens is None
    assert metadata.usage.reasoning_output_tokens is None
    assert metadata.phases.after_last_search_seconds is None
    assert "private-secret" not in caplog.text


def test_research_correction_cannot_add_sources_or_retry_again(renderer):
    original = missing_research().model_dump(mode="json")
    original["signals"][0].update(status="missing", evidence=[scoped_evidence(42)])
    corrected = json.loads(json.dumps(original))
    corrected["signals"][0].update(status="found", information_gap=None)
    corrected["signals"][0]["evidence"][0]["sources"] = [
        {"url": "https://example.com/invented", "title": None, "publication_date": None},
    ]
    client, requests = mock_client(
        [terminal([search_item(), message(json.dumps(original))])],
        [terminal([message(json.dumps(corrected))])],
    )
    with pytest.raises(AIExecutionError) as caught:
        client.execute(render_research(renderer), correct_validation=True)
    assert len(requests) == 2
    metadata = caught.value.metadata
    assert metadata.validation_diagnostic.rule_id == "correction.new_source"
    assert metadata.correction.initial_diagnostic.rule_id == "research.missing_evidence"
    assert not metadata.correction.succeeded


def test_output_limit_records_usage_without_accepting_or_correcting_partial_json(renderer):
    final = terminal([search_item(), message(missing_research().model_dump_json())], status="incomplete")
    final["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    client, requests = mock_client([final])
    budget = EvaluationBudget(monotonic() + 30, 4)
    with pytest.raises(AIExecutionError) as caught:
        client.execute(render_research(renderer), correct_validation=True, budget=budget)
    assert caught.value.kind == FailureKind.OUTPUT_TOKEN_LIMIT
    assert len(requests) == 1
    metadata = caught.value.metadata
    assert metadata.validation_passed is None and metadata.correction is None
    request, = metadata.requests
    assert request.incomplete_reason == "max_output_tokens"
    assert request.usage.output_tokens == 20 and request.max_output_tokens == 8000
    assert request.hosted_search_calls_observed == 1
    assert metadata.estimated_cost_usd == pytest.approx(0.01176)
    assert budget.accounted_usd == metadata.estimated_cost_usd


def test_budget_blocks_correction_using_cost_from_previous_stages(renderer):
    invalid = missing_research().model_dump(mode="json")
    invalid["signals"][0]["signal"] = Signal.EVENTS
    final = terminal([search_item(), message(json.dumps(invalid))])
    # High observed input leaves less than the correction reservation.
    final["response"]["usage"].update(input_tokens=95_000, total_tokens=95_020)
    client, requests = mock_client([final])
    budget = EvaluationBudget(monotonic() + 30, 2.5, accounted_usd=0.5)
    progress = []
    with pytest.raises(AIExecutionError) as caught:
        client.execute(render_research(renderer), correct_validation=True, budget=budget, on_progress=progress.append)
    assert caught.value.kind == FailureKind.EVALUATION_BUDGET
    assert len(requests) == 1 and budget.accounted_usd > 2
    assert all(event.kind != ProgressKind.OUTPUT_CORRECTION_STARTED for event in progress)
    assert caught.value.metadata.correction.initial_diagnostic.rule_id == "research.signal_coverage"


@pytest.mark.parametrize("seconds,amount,expected", [
    (-1, 4, FailureKind.TIMEOUT), (30, 0.01, FailureKind.EVALUATION_BUDGET),
])
def test_evaluation_preflight_blocks_request(renderer, seconds, amount, expected):
    client, requests = mock_client()
    with pytest.raises(AIExecutionError) as caught:
        client.execute(render_research(renderer), budget=EvaluationBudget(monotonic() + seconds, amount))
    assert caught.value.kind == expected
    assert not requests and not caught.value.metadata.requests


def test_hosted_usage_above_estimated_budget_rejects_result(renderer):
    final = terminal([search_item(), message(missing_research().model_dump_json())])
    final["response"]["usage"].update(input_tokens=150_000, total_tokens=150_020)
    client, requests = mock_client([final])
    budget = EvaluationBudget(monotonic() + 30, 2)
    with pytest.raises(AIExecutionError) as caught:
        client.execute(render_research(renderer), correct_validation=True, budget=budget)
    assert caught.value.kind == FailureKind.EVALUATION_BUDGET
    assert budget.accounted_usd > budget.max_estimated_usd
    assert len(requests) == 1 and caught.value.metadata.validation_passed is None


def test_evaluation_deadline_stops_stream_and_accounts_for_unreported_usage(renderer, monkeypatch):
    final = terminal([search_item(), message(missing_research().model_dump_json())])
    event = {
        "type": "response.web_search_call.in_progress", "item_id": "ws_test",
        "output_index": 0, "sequence_number": 1,
    }
    budget = EvaluationBudget(monotonic() + 30, 4)
    remaining_seconds = iter((30, 1, -1))
    monkeypatch.setattr(budget, "remaining_seconds", lambda: next(remaining_seconds))
    client, requests = mock_client([event, event, final])
    with pytest.raises(AIExecutionError) as caught:
        client.execute(render_research(renderer), correct_validation=True, budget=budget)
    assert caught.value.kind == FailureKind.TIMEOUT and len(requests) == 1
    request, = caught.value.metadata.requests
    assert request.hosted_search_calls_observed == 1
    assert request.usage is None and request.estimated_cost_usd is None
    assert budget.accounted_usd == request.reserved_cost_usd


def test_diagnostics_measure_context_and_observe_search_without_logging_query(renderer):
    prompt = render_research(renderer)
    query = "private email test@example.test and company research"
    search = search_item()
    search["action"].update(queries=[query, query], sources=[{"type": "url", "url": "https://private.test/path"}])
    started = {"type": "response.web_search_call.in_progress", "item_id": "ws_test",
               "output_index": 0, "sequence_number": 1}
    finished = {**started, "type": "response.web_search_call.completed", "sequence_number": 2}
    final = terminal([search, message(missing_research().model_dump_json())])
    client, _ = mock_client([started, finished, final])
    metadata = client.execute(prompt).metadata
    timing, = metadata.phases.searches
    configuration = metadata.configuration
    sizes = configuration["context_utf8_bytes"]
    assert sizes["policy"] == len(POLICY.encode())
    assert sizes["policy"] + sizes["instructions_excluding_policy_and_schema"] + sizes[
        "schema_in_instructions"
    ] == len(prompt.instructions.encode())
    assert sizes["other_task_input"] == len(prompt.input_json.encode())
    assert len(configuration["policy_sha256"]) == 64
    assert configuration["search_context_size"] == "low"
    assert timing.duration_seconds >= 0 and timing.action == "search"
    assert timing.exposed_sources_count == 1 and timing.retrieved_context_tokens is None
    assert len(timing.query_fingerprints) == 2
    assert timing.query_fingerprints[0] == timing.query_fingerprints[1]
    assert timing.query_fingerprints[0]["utf8_bytes"] == len(query.encode())
    assert query not in repr(metadata) and "private.test" not in repr(metadata)


@pytest.mark.parametrize(("status", "code", "error_type", "parameter", "kind"), [
    (400, "invalid_json_schema", "invalid_request_error", "text.format", FailureKind.CONFIGURATION),
    (429, "billing_hard_limit_reached", "insufficient_quota", None, FailureKind.RATE_LIMIT),
])
def test_api_status_diagnostics_allowlist_codes_without_echoing_error_body(
    renderer, status, code, error_type, parameter, kind,
):
    def fail(_):
        return httpx.Response(status, headers={"x-request-id": "req_ApiTest123"}, json={
            "error": {"code": code, "type": error_type, "param": parameter,
                      "message": "Sensitive customer text and credentials"},
        })

    sdk = OpenAI(api_key="mock-only", http_client=httpx.Client(transport=httpx.MockTransport(fail)))
    with pytest.raises(AIExecutionError) as caught:
        AIClient(sdk=sdk).execute(render_research(renderer))
    assert caught.value.kind == kind
    metadata = caught.value.metadata
    request, = metadata.requests
    assert request.api_error == metadata.api_error == {
        "origin": "http_error", "http_status": status, "code": code, "error_type": error_type,
        "parameter": parameter, "request_id": "req_ApiTest123",
    }
    assert "Sensitive customer" not in repr(metadata) and "mock-only" not in repr(metadata)
    assert request.usage is None and request.estimated_cost_usd is None


def test_terminal_summary_and_private_report_permissions(monkeypatch, tmp_path, capsys):
    from app.ai.diagnostics import enter_stage, observe_evaluation

    monkeypatch.setenv("AI_DIAGNOSTICS_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("AI_DIAGNOSTICS_REPORT_DIR", str(tmp_path))

    @observe_evaluation("customer")
    def gate_only():
        enter_stage("activity_persistence")

    gate_only()
    report_path, = tmp_path.glob("*.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == "no_ai" and report["totals"]["estimated_cost_usd"] == 0
    assert report_path.stat().st_mode & 0o777 == 0o600
    terminal_summary = capsys.readouterr().err
    assert "customer no_ai" in terminal_summary and report["correlation_id"] in terminal_summary


@pytest.mark.parametrize(("shape", "code", "parameter", "origin", "exception_type"), [
    ("nested", "credit_balance_exhausted", None, "sdk_stream_error", "APIError"),
    ("nested", "unsupported_value", "tools[0].search_context_size", "sdk_stream_error", "APIError"),
    ("flat", "project_spend_limit_exceeded", None, "stream_error_event", "_ExecutionFailure"),
    ("failed", "server_error", None, "response_failed", "_ExecutionFailure"),
    ("failed_without_error", None, None, "response_failed", "_ExecutionFailure"),
    ("nested_unsafe", None, None, "sdk_stream_error", "APIError"),
])
def test_stream_api_errors_survive_sdk_iteration_and_remain_safe(
    renderer, shape, code, parameter, origin, exception_type,
):
    secret = "sk-private-example"
    details = {"code": code, "param": parameter, "message": secret}
    if shape == "nested_unsafe":
        details.update(code=secret, param=secret, type=secret)
    if shape.startswith("nested"):
        event = {"error": details}
    elif shape == "flat":
        event = {"type": "error", "sequence_number": 1, **details}
    else:
        event = terminal([], status="failed")
        event["response"].update(error=details if code else None, usage=None)
    body = "data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
    transport = httpx.MockTransport(lambda _: httpx.Response(
        200, text=body, headers={"Content-Type": "text/event-stream", "x-request-id": "req_Stream123"},
    ))
    sdk = OpenAI(api_key="mock-only", http_client=httpx.Client(transport=transport))
    with pytest.raises(AIExecutionError) as caught:
        AIClient(sdk=sdk).execute(render_research(renderer), correct_validation=True)
    metadata = caught.value.metadata
    request, = metadata.requests
    assert metadata.exception_type == exception_type
    assert request.api_error == metadata.api_error == {
        "origin": origin, "http_status": 200, "code": code, "error_type": None,
        "parameter": parameter, "request_id": "req_Stream123",
    }
    assert metadata.completed_hosted_tool_calls == 0 and metadata.correction is None
    assert secret not in repr(metadata)
