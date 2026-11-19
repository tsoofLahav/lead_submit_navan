from uuid import uuid4

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, inspect, select, text
from sqlalchemy.exc import DBAPIError

from app import create_app
from app.models import Base
from app.reference_data import PromptNotFound, get_business_policy, get_prompt, list_prompt_keys


pytestmark = pytest.mark.integration

EXPECTED_PROMPT_KEYS = (
    "lead_submit_system",
    "lead_submit_research",
    "lead_submit_prediction",
    "lead_submit_recommendation",
    "reevaluation_system",
    "reevaluation_prediction",
    "reevaluation_recommendation",
    "chat_assistant_system",
)


@pytest.fixture(scope="module")
def app():
    application = create_app()
    engine = application.extensions["db_engine"]
    engine.update_execution_options(postgresql_readonly=True)
    yield application
    engine.dispose()


@pytest.fixture(scope="module")
def connection(app):
    with app.extensions["db_engine"].connect() as connection:
        assert connection.scalar(text("SHOW transaction_read_only")) == "on"
        yield connection


@pytest.fixture(scope="module")
def schema(connection):
    return inspect(connection)


@pytest.fixture
def sessions(app, connection):
    return app.extensions["db_sessions"]


def test_database_connectivity(connection):
    assert connection.scalar(text("SELECT 1")) == 1


def test_healthy_response(app, connection):
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_failed_transaction_releases_connection(sessions):
    with pytest.raises(DBAPIError), sessions.begin() as session:
        session.execute(text("SELECT 1 / 0"))
    with sessions() as session:
        assert session.scalar(text("SELECT 1")) == 1


@pytest.mark.parametrize("table", list(Base.metadata.tables.values()), ids=lambda table: table.name)
def test_models_match_postgres(table, schema, connection):
    name = table.name
    options = {"schema": table.schema}
    reflected = {column["name"]: column for column in schema.get_columns(name, **options)}
    assert set(reflected) == set(table.columns.keys())
    for column in table.columns:
        actual = reflected[column.name]
        assert column.type.compile(connection.dialect) == actual["type"].compile(connection.dialect)
        assert column.nullable == actual["nullable"]
        if column.identity is not None:
            assert column.identity.always == actual["identity"]["always"]
        else:
            assert "identity" not in actual
            expected_default = str(column.server_default.arg) if column.server_default else None
            assert expected_default == actual["default"]

    primary_key = schema.get_pk_constraint(name, **options)
    assert table.primary_key.name == primary_key["name"]
    assert list(table.primary_key.columns.keys()) == primary_key["constrained_columns"]

    foreign_keys = {item["name"]: item for item in schema.get_foreign_keys(name, **options)}
    assert set(foreign_keys) == {key.name for key in table.foreign_key_constraints}
    for key in table.foreign_key_constraints:
        actual = foreign_keys[key.name]
        assert key.column_keys == actual["constrained_columns"]
        assert key.referred_table.name == actual["referred_table"]
        assert key.referred_table.schema == (actual["referred_schema"] or table.schema)
        assert [element.column.name for element in key.elements] == actual["referred_columns"]
        assert ({"ondelete": key.ondelete} if key.ondelete else {}) == actual["options"]

    assert {
        constraint.name: list(constraint.columns.keys())
        for constraint in table.constraints if isinstance(constraint, UniqueConstraint)
    } == {
        item["name"]: item["column_names"] for item in schema.get_unique_constraints(name, **options)
    }
    assert {
        constraint.name: str(constraint.sqltext)
        for constraint in table.constraints if isinstance(constraint, CheckConstraint)
    } == {
        item["name"]: item["sqltext"] for item in schema.get_check_constraints(name, **options)
    }

    indexes = {
        item["name"]: item for item in schema.get_indexes(name, **options)
        if not item.get("duplicates_constraint")
    }
    assert set(indexes) == {index.name for index in table.indexes}
    for index in table.indexes:
        actual = indexes[index.name]
        sorting = actual.get("column_sorting", {})
        actual_columns = [
            f"{column} DESC" if "desc" in sorting.get(column, ()) else column
            for column in actual["column_names"]
        ]
        assert [getattr(expr, "name", str(expr)) for expr in index.expressions] == actual_columns
        assert index.unique == actual["unique"]

    connection.execute(select(table).limit(1)).first()


def test_business_policy(sessions):
    with sessions() as session:
        assert bool(get_business_policy(session).strip()), "Business policy content is empty"


def test_prompt_keys(sessions):
    with sessions() as session:
        keys = list_prompt_keys(session)
        assert keys == sorted(keys)
        assert set(EXPECTED_PROMPT_KEYS) <= set(keys)


@pytest.mark.parametrize("prompt_key", EXPECTED_PROMPT_KEYS)
def test_expected_prompt(sessions, prompt_key):
    with sessions() as session:
        assert bool(get_prompt(session, prompt_key).strip()), "Prompt content is empty"


def test_missing_prompt(sessions):
    with sessions() as session, pytest.raises(PromptNotFound):
        get_prompt(session, f"missing-{uuid4()}")


def test_stored_ai_prompts_render_through_shared_renderer(connection):
    from decimal import Decimal

    from sqlalchemy.orm import Session

    from app.ai.config import AISettings, Component
    from app.ai.prompts import TASKS, PromptRenderer, Task
    from app.ai.schemas import LeadPredictionOutput, ProductAssessmentOutput, ResearchOutput, ResearchSignal, Signal
    from app.ai.tools import ToolRegistry
    from app.reevaluation_gate import ReevaluationGate
    from app.validation import CustomerActivityInput, LeadInput

    lead = LeadInput("Synthetic", "example.com", Decimal(100_000))
    activity = CustomerActivityInput("Synthetic", "example.com", 3, 4)
    gate = ReevaluationGate.assess(activity)
    observation = {
        "validated_product_data": activity, "derived_metrics": gate.metrics, "trigger_results": gate,
    }
    uncertain = {
        "level": "unknown", "confidence": "insufficient", "supporting_signals": [],
        "rationale": "Insufficient evidence.", "information_gaps": ["Evidence unavailable."],
    }
    prediction = LeadPredictionOutput.model_validate({
        "travel_spend_potential": {**uncertain, "company_scale": None, "intensity_ratings": []},
        "growth_potential": uncertain, "expected_support_needs": uncertain,
    })
    research = ResearchOutput(signals=[
        ResearchSignal(signal=signal, evidence=[], status="missing", information_gap="Evidence unavailable.")
        for signal in Signal
    ])
    cases = [
        (Task.LEAD_RESEARCH, Component.RESEARCH, {"company_profile": lead}),
        (Task.LEAD_PREDICTION, Component.PREDICTION,
         {"company_profile": lead, "validated_research_findings": research}),
        (Task.LEAD_RECOMMENDATION, Component.DECISION,
         {"company_profile": lead, "current_tier": 2, "validated_predictions": prediction,
          "validated_research_findings": research}),
        (Task.REEVALUATION_PREDICTION, Component.PREDICTION, observation),
        (Task.REEVALUATION_RECOMMENDATION, Component.DECISION,
         {**observation, "current_tier": 3, "validated_product_assessment": ProductAssessmentOutput(
             confidence="insufficient", material_eligibility_uncertainty=False, findings=[], rationale="No evidence.", information_gaps=["Activity data unavailable."])}),
        (Task.CHAT, Component.CHAT, {"message": "Synthetic message."}),
    ]
    with Session(connection) as session:
        renderer = PromptRenderer(session, ToolRegistry())
        policy = renderer.load_policy()
        for task, component, inputs in cases:
            rendered = renderer.render(task, policy=policy, inputs=inputs, settings=AISettings.from_env(component))
            if task == Task.CHAT:
                assert policy not in rendered.instructions
                assert "get_business_policy" in rendered.instructions
            else:
                assert policy in rendered.instructions
            assert "{{" not in rendered.instructions
            for key in {TASKS[task].system_key, task.value}:
                for line in get_prompt(session, key).splitlines():
                    if line.strip() and "{{" not in line:
                        assert line in rendered.instructions
