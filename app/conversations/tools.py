import json
from dataclasses import asdict
from functools import partial
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator
from pydantic_core import to_jsonable_python
from sqlalchemy.orm import Session, sessionmaker

from ..ai.schemas import StructuredOutput, Tier
from ..ai.tools import ReadOnlyFunction, ToolRegistry
from ..data.access import WorkspaceData
from ..data.models import Company, CompanyResearch, Evaluation, ProductActivity, TierChange
from ..business.customer_reevaluation.gate import ReevaluationGate
from ..data.reference_data import BusinessPolicyNotFound, get_business_policy
from ..business.rules.validation import CustomerActivityInput

MAX_COMPANIES = 20
MAX_HISTORY = 10
MAX_TOOL_RESULT_BYTES = 96_000


class CompanyFilters(StructuredOutput):
    tier: Tier | None
    is_customer: bool | None
    has_pending_recommendation: bool | None
    limit: Annotated[int, Field(ge=1, le=MAX_COMPANIES)] | None


class CompanySelector(StructuredOutput):
    company_id: UUID | None
    company_name: Annotated[str, Field(min_length=1, max_length=200)] | None

    @model_validator(mode="after")
    def identify_company(self):
        if (self.company_id is None) == (self.company_name is None) or (
            self.company_name is not None and not self.company_name.strip()
        ):
            raise ValueError("Provide one company ID or name.")
        return self


class HistorySelector(CompanySelector):
    record_id: UUID | None
    limit: Annotated[int, Field(ge=1, le=MAX_HISTORY)] | None


class NoArguments(StructuredOutput):
    pass


def _basic(company: Company) -> dict:
    return {
        "company_id": str(company.id), "name": company.name, "domain": company.domain,
        "current_tier": company.current_tier, "is_customer": company.is_customer,
    }


def _resolve(data: WorkspaceData, args: CompanySelector) -> Company | dict:
    if args.company_id is not None:
        company = data.get_company(args.company_id)
        return company if company is not None else {"status": "not_found"}
    matches = data.company_list(MAX_COMPANIES + 1, name=args.company_name)
    if not matches:
        return {"status": "not_found"}
    if len(matches) > 1:
        return {"status": "ambiguous", "matches": [_basic(company) for company in matches[:MAX_COMPANIES]],
                "more": len(matches) > MAX_COMPANIES}
    return matches[0]


def _company(data: WorkspaceData, args: CompanySelector) -> dict:
    company = _resolve(data, args)
    if isinstance(company, dict):
        return company
    return {"status": "ok", **_basic(company),
            "submitted_data_provenance": (
                "Stored company details may include submitted or prefilled sample data. "
                "Independently sourced evidence is available separately in company research."
            ),
            "submitted_annual_gbv": company.submitted_annual_gbv,
            "submitted_annual_gbv_currency": "USD" if company.submitted_annual_gbv is not None else None,
            "employee_count": company.employee_count}


ACTIVITY_FIELDS = (
    "months_since_signup", "total_gbv", "avg_monthly_bookings", "employee_count", "avg_monthly_active_users",
    "avg_monthly_support_requests", "travel_spend_trend", "company_size_trend", "product_adoption_trend",
)


def _history_record(company: Company, row) -> dict:
    value = {"record_id": str(row.id), "created_at": row.created_at}
    if isinstance(row, CompanyResearch):
        return {**value, "signals": row.signals}
    if isinstance(row, Evaluation):
        return {
            **value, "evaluation_type": row.evaluation_type, "research_id": row.research_id,
            "current_tier": row.current_tier, "recommended_tier": row.recommended_tier,
            "predictions": {key: value for key, value in row.predictions.items() if key != "policy_sha256"}, "confidence": row.confidence, "rationale": row.rationale,
            "information_gaps": row.information_gaps, "follow_up_actions": row.follow_up_actions,
            "decision_status": row.decision_status, "decided_at": row.decided_at,
        }
    if isinstance(row, ProductActivity):
        submitted = {key: getattr(row, key) for key in ACTIVITY_FIELDS}
        gate = ReevaluationGate.assess(CustomerActivityInput(
            company.name, company.domain, company.current_tier, **submitted,
        ))
        return {
            **value, "submitted": submitted, "derived_metrics": asdict(gate.metrics),
            "information_gaps": gate.information_gaps,
            "measurement_note": "User-reported aggregate since signup. Adoption compares historical active users with headcount at submission.",
        }
    return {
        **value, "previous_tier": row.previous_tier, "new_tier": row.new_tier, "evaluation_id": row.evaluation_id,
        "origin": "accepted_recommendation" if row.evaluation_id else "externally_reported",
    }


def _history(model, data: WorkspaceData, args: HistorySelector) -> dict:
    company = _resolve(data, args)
    if isinstance(company, dict):
        return company
    limit = args.limit or (1 if model in (CompanyResearch, ProductActivity) else 5)
    rows = data.company_records(model, company.id, limit, args.record_id)
    if not rows:
        return {"status": "not_found", "company_id": str(company.id)}
    return {"status": "ok", "company": _basic(company), "records": [_history_record(company, row) for row in rows]}


def _list(data: WorkspaceData, args: CompanyFilters) -> dict:
    limit = args.limit or MAX_COMPANIES
    companies = data.company_list(limit + 1, args.tier, args.is_customer, args.has_pending_recommendation)
    return {"status": "ok", "companies": [_basic(row) for row in companies[:limit]], "more": len(companies) > limit}


def _policy(data: WorkspaceData, _args: NoArguments) -> dict:
    try:
        return {"status": "ok", "policy": get_business_policy(data.session)}
    except BusinessPolicyNotFound:
        return {"status": "not_found"}


def build_chat_tools(sessions: sessionmaker[Session], reference_sessions: sessionmaker[Session]) -> ToolRegistry:
    def execute(handler, factory, workspace_id, args):
        with factory() as session:
            payload = to_jsonable_python(handler(WorkspaceData(session, workspace_id), args))
        if len(json.dumps(payload, allow_nan=False).encode()) > MAX_TOOL_RESULT_BYTES:
            return {"status": "response_too_large", "hint": "Request fewer history records or a specific record."}
        return payload

    definitions = (
        ("list_companies", "List workspace companies, optionally filtered by tier, customer status or pending recommendation.",
         CompanyFilters, _list),
        ("get_company", "Get current company details by exact name or ID. Ambiguous names return matching options.",
         CompanySelector, _company),
        ("get_company_research", "Get recent or specific public research, evidence reliability, sources and gaps.",
         HistorySelector, partial(_history, CompanyResearch)),
        ("get_company_evaluations", "Get recent or specific saved evaluations, evidence, rationale and current decision status.",
         HistorySelector, partial(_history, Evaluation)),
        ("get_customer_activity", "Get latest or historical user-reported customer activity snapshots and derived estimates.",
         HistorySelector, partial(_history, ProductActivity)),
        ("get_tier_history", "Get recorded tier changes and whether each was externally reported or explicitly approved.",
         HistorySelector, partial(_history, TierChange)),
        ("get_business_policy", "Read active eligibility, research, prediction and re-evaluation policy when policy is needed.",
         NoArguments, _policy),
    )
    return ToolRegistry(ReadOnlyFunction(
        name, description, arguments, partial(execute, handler, reference_sessions if handler is _policy else sessions),
    ) for name, description, arguments, handler in definitions)
