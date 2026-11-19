"""Opt-in lead evaluation with public company names and local-only write persistence."""

import argparse
import json
import os
from dataclasses import asdict
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from app import create_app
from app.ai.client import AIClient
from app.models import Base, Company, CompanyResearch, Evaluation, Workspace
from app.orchestrator import LeadEvaluationResult, LeadWorkflowFailure, Orchestrator
from app.validation import LeadInput
from scripts.smoke_database import open_local_database

PROFILES = {
    "google": LeadInput("Google", "google.com", Decimal("800000"), 45000),
    "mozilla": LeadInput("Mozilla", "mozilla.org", Decimal("100000")),
    "microsoft": LeadInput("Microsoft", "microsoft.com", Decimal("750000")),
    "gitlab": LeadInput("GitLab", "gitlab.com", Decimal("750000")),
    "python": LeadInput("Python Software Foundation", "python.org", Decimal("50000")),
    "wikimedia": LeadInput("Wikimedia Foundation", "wikimediafoundation.org", Decimal("50000")),
}


def encode_report_value(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def verify_persistence(sessions, result: LeadEvaluationResult) -> None:
    with sessions() as session:
        company = session.get(Company, result.company_id)
        research = session.get(CompanyResearch, result.research_id)
        evaluation = session.get(Evaluation, result.evaluation_id)
        if (
            company is None or research is None or evaluation is None
            or company.current_tier != result.initial_tier or research.company_id != company.id
            or evaluation.company_id != company.id or evaluation.research_id != research.id
            or evaluation.decision_status != ("pending" if result.approval_available else "not_applicable")
            or evaluation.recommended_tier != result.recommendation.recommended_tier
        ):
            raise SystemExit("The isolated database did not contain the expected workflow records.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    args = parser.parse_args()

    app = create_app()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run the live Lead Submission smoke test.")
    reference_engine = app.extensions["db_engine"]
    reference_engine.update_execution_options(postgresql_readonly=True)
    local_engine = open_local_database("LEAD_SMOKE_DATABASE_URL", "navan_stage4_smoke")
    try:
        Base.metadata.create_all(local_engine)
        sessions = sessionmaker(local_engine)
        with sessions.begin() as session:
            workspace = Workspace()
            session.add(workspace)
            session.flush()
            workspace_id = workspace.id
        lead = PROFILES[args.profile]
        orchestrator = Orchestrator(
            sessions, ai_client=AIClient(), reference_sessions=app.extensions["db_sessions"],
        )

        def report(event):
            print(f"{args.profile}: {event.kind.value}", flush=True)

        try:
            result = orchestrator.submit_lead(workspace_id, lead, on_progress=report)
        except LeadWorkflowFailure as failure:
            with sessions() as session:
                company = session.get(Company, failure.company_id) if failure.company_id is not None else None
                research = (
                    session.get(CompanyResearch, failure.research_id) if failure.research_id is not None else None
                )
                print(json.dumps({
                    "profile": args.profile, "stage": failure.stage, "reason": failure.reason,
                    "initial_tier": company.current_tier if company is not None else None,
                    "committed_research": research.signals if research is not None else None,
                    "predictions": failure.predictions.model_dump(mode="json") if failure.predictions else None,
                    "duration_seconds": failure.duration_seconds,
                    "executions": [asdict(item) for item in failure.executions],
                }, default=str))
            raise SystemExit(1) from None
        if not isinstance(result, LeadEvaluationResult):
            raise SystemExit("Unexpected duplicate in a newly created workspace.")
        verify_persistence(sessions, result)
        print(json.dumps(asdict(result), default=encode_report_value))
    finally:
        local_engine.dispose()
        reference_engine.dispose()


if __name__ == "__main__":
    main()
