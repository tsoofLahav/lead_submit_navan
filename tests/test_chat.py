import json
from decimal import Decimal
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.ai.client import FailureKind
from app.ai.config import Component
from app.ai.progress import ProgressKind
from app.ai.tools import ToolError
from app.chat import ChatAssistant, ChatFailure, ConversationNotFound, ConversationPersistenceError, MessageKind
from app.chat_tools import build_chat_tools
from app.human_approval import RecommendationNotFound
from app.models import BusinessPolicy, Company, Evaluation, ProductActivity, Prompt
from app.orchestrator import LeadWorkflowFailure, Orchestrator
from app.validation import CustomerActivityInput, LeadInput
from scripts.smoke_chat import seed
from test_ai import POLICY, SYSTEM, message, mock_client, terminal
from test_customer_reevaluation import UPGRADE, assessment, recommendation
from test_lead_submission import decision_data, prediction_data, research_data, response as lead_response

pytestmark = pytest.mark.write_integration


@pytest.fixture
def references(sessions):
    with sessions.begin() as session:
        session.add(BusinessPolicy(id=1, content=POLICY))
        templates = {
            "chat_assistant_system": SYSTEM, "lead_submit_system": SYSTEM, "reevaluation_system": SYSTEM,
            "lead_submit_research": "{{company_profile}} {{research_schema}}",
            "lead_submit_prediction": "{{company_profile}} {{validated_research_findings}} {{prediction_schema}}",
            "lead_submit_recommendation": "{{company_profile}} {{validated_predictions}} {{recommendation_schema}}",
            "reevaluation_prediction": (
                "{{validated_product_data}} {{derived_metrics}} {{trigger_results}} {{product_assessment_schema}}"
            ),
            "reevaluation_recommendation": "{{current_tier}} {{validated_product_assessment}} {{recommendation_schema}}",
        }
        session.add_all(Prompt(prompt_key=key, content=value) for key, value in templates.items())


def final(answer):
    return [terminal([message(json.dumps({"answer": answer}))])]


def call(name, args, index=1):
    return {
        "type": "function_call", "name": name, "id": f"fc_{index}", "call_id": f"call_{index}",
        "arguments": json.dumps(args), "status": "completed",
    }


def assistant_with_responses(sessions, *batches, **kwargs):
    client, requests = mock_client(*batches)
    return ChatAssistant(sessions, sdk=client._sdk, **kwargs), requests


def test_conversation_management_history_context_and_retry(sessions, workspace_ids, references):
    owner, outsider = workspace_ids
    chat, requests = assistant_with_responses(sessions, final("First answer"), 500, final("Recovered"), history_limit=3)
    other = chat.create_conversation(outsider)
    first = chat.send_message(owner, None, "   First    meaningful question   ")
    thread = first.user_message.conversation_id
    assert chat.get_conversation(owner, thread).title == "First meaningful question"
    assert [row.id for row in chat.list_conversations(owner)] == [thread]
    assert chat.get_conversation(outsider, other.id).id == other.id
    with pytest.raises(ConversationNotFound):
        chat.list_messages(outsider, thread)
    with pytest.raises(ConversationNotFound):
        chat.rename_conversation(outsider, thread, "Hidden")
    assert chat.rename_conversation(owner, thread, "Explicit title").title == "Explicit title"
    with pytest.raises(ChatFailure) as error:
        chat.send_message(owner, thread, "Second question")
    user = error.value.user_message
    assert len(chat.list_messages(owner, thread)) == 3
    with pytest.raises(ConversationNotFound):
        chat.retry_message(owner, other.id, user.id)
    confirmed = chat.retry_message(owner, thread, user.id)
    assert confirmed.user_message.id == user.id and confirmed.assistant_message.content == "Recovered"
    assert chat.retry_message(owner, thread, user.id).assistant_message.id == confirmed.assistant_message.id
    assert len(requests) == 3
    history = json.loads(requests[-1]["input"][1]["content"])["history"]
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert [row.content for row in chat.list_messages(owner, thread, 2, before_id=user.id)] == [
        "First    meaningful question", "First answer",
    ]
    assert POLICY not in requests[-1]["input"][0]["content"]
    assert "get_business_policy" in requests[-1]["input"][0]["content"]


def test_tool_reads_and_workspace_scoping(sessions, references):
    owner, company_id, evaluation_id = seed(sessions)
    outsider, foreign_id, foreign_evaluation_id = seed(sessions)
    registry = build_chat_tools(sessions, sessions)
    company = {"company_id": str(company_id), "company_name": None}
    history = {**company, "record_id": None, "limit": None}

    def invoke(name, args, workspace=owner):
        return json.loads(registry.dispatch(Component.CHAT, workspace, name, json.dumps(args)))

    result = invoke("list_companies", {"tier": 2, "is_customer": True, "has_pending_recommendation": True, "limit": 1})
    assert [row["company_id"] for row in result["companies"]] == [str(company_id)]
    details = invoke("get_company", company)
    assert details["submitted_annual_gbv"] == "520000.00" and details["submitted_annual_gbv_currency"] == "USD"
    assert invoke("get_company", {"company_id": str(foreign_id), "company_name": None}) == {"status": "not_found"}
    research = invoke("get_company_research", history)["records"][0]
    assert research["signals"]["signals"][0]["evidence"][0]["sources"][0]["url"].endswith("/operations")
    evaluation = invoke("get_company_evaluations", {**history, "record_id": str(evaluation_id)})["records"][0]
    assert evaluation["decision_status"] == "pending" and evaluation["recommended_tier"] == 1
    assert invoke("get_company_evaluations", {**history, "record_id": str(foreign_evaluation_id)})["status"] == "not_found"
    activity = invoke("get_customer_activity", history)["records"][0]
    assert Decimal(activity["derived_metrics"]["annualized_gbv"]) == 465000
    assert invoke("get_tier_history", history)["records"][0]["origin"] == "externally_reported"
    assert invoke("get_business_policy", {})["policy"] == POLICY
    assert invoke("get_company_research", history, outsider)["status"] == "not_found"
    with pytest.raises(ToolError):
        invoke("get_company", {**company, "workspace_id": str(outsider)})
    with pytest.raises(ToolError):
        invoke("get_company_evaluations", {**history, "limit": 100})
    for name in ("accept_recommendation", "web_search", "execute_sql"):
        with pytest.raises(ToolError):
            invoke(name, {})

    by_name = {"company_id": None, "company_name": "  northstar   demo  "}
    assert invoke("get_company", by_name)["company_id"] == str(company_id)
    with sessions.begin() as session:
        session.add(Company(
            workspace_id=owner, name="Northstar Demo", domain="another.test", current_tier=3, is_customer=False,
        ))
    options = invoke("get_company", by_name)
    assert options["status"] == "ambiguous" and len(options["matches"]) == 2
    assert invoke("get_company", {**by_name, "company_name": "No such company"}) == {"status": "not_found"}


def test_cumulative_budget_counts_invalid_and_parallel_requests(sessions, workspace_ids, references, monkeypatch):
    monkeypatch.setenv("AI_CHAT_TOOL_BUDGET", "2")
    owner, _ = workspace_ids
    batch = [
        call("get_business_policy", {"workspace_id": "forged"}, 1),
        call("get_business_policy", {}, 2),
        call("get_business_policy", {}, 3),
    ]
    chat, requests = assistant_with_responses(sessions, [terminal(batch)], final("Policy answer."))
    events = []
    turn = chat.send_message(owner, None, "Explain Tier 1.", on_progress=events.append)
    assert turn.execution.application_tool_attempts == 2 and turn.execution.application_tools == ("get_business_policy",)
    assert turn.execution.request_rounds == 2
    assert requests[1]["tool_choice"] == "none" and "max_tool_calls" not in requests[1]
    outputs = [json.loads(row["output"]) for row in requests[1]["input"] if row.get("type") == "function_call_output"]
    assert outputs == [{"status": "invalid_arguments"}, {"status": "ok", "policy": POLICY},
                       {"status": "tool_budget_exhausted"}]
    assert all(event.conversation_id == turn.user_message.conversation_id for event in events)
    assert [event.tool_status for event in events if event.kind == ProgressKind.CHAT_TOOL_COMPLETED] == [
        "invalid_arguments", "completed",
    ]


@pytest.mark.parametrize("tool", ["get_business_policy", "approve_recommendation"])
def test_round_limit_or_unauthorized_call_fails_safely(sessions, workspace_ids, references, monkeypatch, tool):
    monkeypatch.setenv("AI_CHAT_MAX_ROUNDS", "2")
    owner, _ = workspace_ids
    chat, requests = assistant_with_responses(
        sessions, [terminal([call(tool, {})])], [terminal([call(tool, {}, 2)])],
    )
    with pytest.raises(ChatFailure) as caught:
        chat.send_message(owner, None, "Change the current tier.")
    failure = caught.value
    assert failure.reason == FailureKind.TOOL_ERROR
    assert len(requests) == (2 if tool == "get_business_policy" else 1)
    assert len(chat.list_messages(owner, failure.user_message.conversation_id)) == 1


def test_lead_result_and_progress_stay_with_initiating_conversation(sessions, workspace_ids, references):
    owner, _ = workspace_ids
    business, business_requests = mock_client(
        lead_response(research_data(), research=True), lead_response(prediction_data()), lead_response(decision_data()),
    )
    chat, requests = assistant_with_responses(sessions, final("The recommendation was undetermined."),
                                             orchestrator=Orchestrator(sessions, ai_client=business))
    original = chat.create_conversation(owner)
    events = []
    switched = []

    def progress(event):
        events.append(event)
        if event.kind == ProgressKind.RESEARCH_STARTED:
            switched.append(chat.create_conversation(owner))

    result = chat.submit_lead(owner, original.id, LeadInput("Acme", "example.com", Decimal("100000")), on_progress=progress)
    assert result.metadata["kind"] == MessageKind.EVALUATION
    assert result.metadata["result"]["research"]["signals"]
    assert result.metadata["decision_status"] == "not_applicable"
    assert all(event.conversation_id == original.id for event in events)
    assert len(chat.list_messages(owner, original.id)) == 2 and not chat.list_messages(owner, switched[0].id)
    assert len(business_requests) == 3
    chat.send_message(owner, original.id, "What happened?")
    history = json.loads(requests[0]["input"][1]["content"])["history"]
    assert history[-1]["metadata"]["evaluation_id"] == result.metadata["evaluation_id"]
    assert history[-1]["metadata"]["decision_status"] == "not_applicable"
    assert "signals" not in history[-1]["metadata"]


@pytest.mark.parametrize("accept", [True, False])
def test_approval_confirmation_reads_current_status_and_stays_in_thread(sessions, workspace_ids, references, accept):
    owner, outsider = workspace_ids
    # Use actual customer workflow responses, with structured text in each mocked Response.
    client, _ = mock_client([terminal([message(json.dumps(assessment()))])],
                            [terminal([message(json.dumps(recommendation()))])])
    chat = ChatAssistant(sessions, orchestrator=Orchestrator(sessions, ai_client=client))
    result = chat.submit_customer_reevaluation(owner, None, UPGRADE)
    evaluation_id = UUID(result.metadata["evaluation_id"])
    assert result.metadata["decision_status"] == "pending" and result.metadata["approval_available"]
    other = chat.create_conversation(owner)
    with pytest.raises(RecommendationNotFound):
        chat.decide_recommendation(owner, other.id, evaluation_id, accept=accept)
    with pytest.raises(RecommendationNotFound):
        chat.decide_recommendation(outsider, result.conversation_id, evaluation_id, accept=accept)
    confirmation = chat.decide_recommendation(owner, result.conversation_id, evaluation_id, accept=accept)
    assert chat.record_decision_confirmation(owner, result.conversation_id, evaluation_id).id == confirmation.id
    rows = chat.list_messages(owner, result.conversation_id)
    assert len(rows) == 3 and rows[-1].metadata["kind"] == MessageKind.DECISION
    assert rows[-2].metadata["decision_status"] == ("accepted" if accept else "rejected")
    assert not rows[-2].metadata["approval_available"]


def test_message_write_failure_replays_only_prepared_result(sessions, workspace_ids, monkeypatch, references):
    owner, _ = workspace_ids
    from test_customer_reevaluation import uncertain_responses
    ai, _ = mock_client(*uncertain_responses())
    chat = ChatAssistant(sessions, orchestrator=Orchestrator(sessions, ai_client=ai))
    activity = CustomerActivityInput("Acme", "acme.test", 3, 1)
    store = chat.save_prepared_message

    def unavailable(*_):
        raise SQLAlchemyError("private connection data")

    monkeypatch.setattr(chat, "save_prepared_message", unavailable)
    with pytest.raises(ConversationPersistenceError) as caught:
        chat.submit_customer_reevaluation(owner, None, activity)
    failure = caught.value
    assert failure.business_result.activity_id is not None and "private" not in str(failure)
    monkeypatch.setattr(chat, "save_prepared_message", store)
    recovery = chat.save_prepared_message(owner, failure.draft)
    assert chat.save_prepared_message(owner, failure.draft).id == recovery.id
    assert recovery.metadata["kind"] == MessageKind.EVALUATION
    assert "undetermined" in recovery.content.lower()
    assert len(chat.list_messages(owner, recovery.conversation_id)) == 2
    with sessions() as session:
        assert session.scalar(select(func.count(ProductActivity.id))) == 1
        assert session.scalar(select(func.count(Evaluation.id))) == 1



def test_chat_stream_event_is_bound_to_conversation(sessions, workspace_ids, references):
    owner, _ = workspace_ids
    delta = {"type": "response.output_text.delta", "item_id": "msg_test", "delta": "{",
             "output_index": 0, "content_index": 0, "sequence_number": 1}
    chat, _ = assistant_with_responses(sessions, [delta, *final("Completed answer")])
    events = []
    turn = chat.send_message(owner, None, "Question", on_progress=events.append)
    event, = events
    assert event.kind == ProgressKind.CHAT_RESPONSE_STREAMING and event.response_delta == "{"
    assert event.conversation_id == turn.user_message.conversation_id
    assert chat.list_messages(owner, event.conversation_id)[-1].content == "Completed answer"


def test_business_failure_is_persisted_as_distinct_message(sessions, workspace_ids):
    owner, _ = workspace_ids
    chat = ChatAssistant(sessions)
    conversation = chat.create_conversation(owner)
    with pytest.raises(LeadWorkflowFailure):
        chat.submit_lead(owner, conversation.id, LeadInput("Northstar", "example.test", Decimal("-1")))
    request, failure = chat.list_messages(owner, conversation.id)
    assert failure.metadata["kind"] == MessageKind.WORKFLOW_FAILURE
    assert failure.metadata["reply_to"] == request.id
    assert failure.metadata["result"]["stage"] == "registration"



def test_workflow_without_selected_conversation_reuses_workspace_thread(sessions, workspace_ids, references):
    owner, outsider = workspace_ids
    from test_customer_reevaluation import uncertain_responses
    ai, _ = mock_client(*uncertain_responses())
    chat = ChatAssistant(sessions, orchestrator=Orchestrator(sessions, ai_client=ai))
    original = chat.create_conversation(owner)
    foreign = chat.create_conversation(outsider)
    result = chat.submit_customer_reevaluation(
        owner, None, CustomerActivityInput("Acme", "acme.test", 3, 1),
    )
    assert result.conversation_id == original.id
    assert len(chat.list_conversations(owner)) == 1 and not chat.list_messages(outsider, foreign.id)
