"""Run four synthetic customer scenarios and persist only to local PostgreSQL."""

import argparse
import json
import os
from dataclasses import asdict
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app import create_app
from app.ai.client import AIClient
from app.human_approval import HumanApproval
from app.models import Base, Company, CompanyResearch, Evaluation, ProductActivity, TierChange, Workspace
from app.orchestrator import CustomerWorkflowFailure, Orchestrator
from app.reevaluation_gate import GateStatus
from app.validation import CustomerActivityInput
from scripts.smoke_database import open_local_database

D = Decimal
SCENARIOS = {
    "no_trigger": CustomerActivityInput(
        "Demo Stable Customer", "stable-customer.test", 2, 6, D("90000"), D("75"), 110, D("80"), D("1"),
        "stable", "stable", "stable",
    ),
    "insufficient": CustomerActivityInput("Demo Sparse Customer", "sparse-customer.test", 3, 1),
    "upgrade": CustomerActivityInput(
        "Demo Expansion Customer", "expansion-customer.test", 3, 6, D("360000"), D("180"), 220, D("110"), D("2"),
        "increasing", "stable", "increasing",
    ),
    "retain": CustomerActivityInput(
        "Demo Support Customer", "support-customer.test", 2, 6, D("90000"), D("85"), 100, D("120"), D("9"),
        "stable", "stable", "stable",
    ),
}
EXPECTED_GATES = {
    "no_trigger": GateStatus.NO_TRIGGER, "insufficient": GateStatus.INSUFFICIENT_DATA,
    "upgrade": GateStatus.TRIGGERED, "retain": GateStatus.TRIGGERED,
}


def encode(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def verify_persistence(sessions, result):
    with sessions() as session:
        snapshot = session.get(ProductActivity, result.activity_id)
        company = session.get(Company, result.company_id)
        evaluations = list(session.scalars(select(Evaluation).where(Evaluation.company_id == result.company_id)))
        research_count = session.scalar(
            select(func.count(CompanyResearch.id)).where(CompanyResearch.company_id == result.company_id)
        )
        if (
            not company or not company.is_customer or company.current_tier != result.customer.current_tier
            or not snapshot or snapshot.company_id != company.id or research_count
        ):
            raise RuntimeError("Unexpected company, activity, or research persistence.")
        if result.evaluation_id is None:
            raise RuntimeError("The first valid snapshot must produce a completed evaluation.")
        evaluation, = evaluations
        if (
            evaluation.id != result.evaluation_id or evaluation.evaluation_type != "re_evaluation"
            or evaluation.research_id is not None
            or evaluation.predictions["activity_snapshot_id"] != str(result.activity_id)
            or evaluation.decision_status != ("pending" if result.approval_available else "not_applicable")
        ):
            raise RuntimeError("Unexpected re-evaluation persistence.")
        return evaluation.decision_status


def run_scenario(sessions, orchestrator, scenario):
    with sessions.begin() as session:
        workspace = Workspace()
        session.add(workspace)
        session.flush()
        workspace_id = workspace.id

    def progress(event):
        print(f"{scenario}: {event.kind.value}", flush=True)

    result = orchestrator.submit_customer_reevaluation(workspace_id, SCENARIOS[scenario], on_progress=progress)
    if result.gate.status != EXPECTED_GATES[scenario]:
        raise RuntimeError("Unexpected deterministic gate outcome.")
    status_before_approval = verify_persistence(sessions, result)
    approval = None
    if result.approval_available:
        decision = HumanApproval(sessions).accept(workspace_id, result.evaluation_id)
        with sessions() as session:
            company = session.get(Company, result.company_id)
            history = session.scalar(select(TierChange).where(TierChange.evaluation_id == result.evaluation_id))
            evaluation = session.get(Evaluation, result.evaluation_id)
            if (
                decision.status != "accepted" or evaluation.decision_status != "accepted" or not history
                or history.company_id != result.company_id or history.previous_tier != result.customer.current_tier
                or history.new_tier != result.recommendation.recommended_tier or company.current_tier != history.new_tier
            ):
                raise RuntimeError("Acceptance did not persist the expected tier change.")
            approval = {**asdict(decision), "tier_change_id": history.id}

    return {"scenario": scenario, **asdict(result), "status_before_approval": status_before_approval, "approval": approval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", metavar="SCENARIO",
                        help="no_trigger, insufficient, upgrade, or retain. Defaults to all four.")
    args = parser.parse_args()
    scenarios = args.scenarios or list(SCENARIOS)
    if any(scenario not in SCENARIOS for scenario in scenarios):
        parser.error("Choose no_trigger, insufficient, upgrade, or retain.")
    app = create_app()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run the live evaluation scenarios.")
    reference_engine = app.extensions["db_engine"]
    reference_engine.update_execution_options(postgresql_readonly=True)
    local_engine = open_local_database("REEVALUATION_SMOKE_DATABASE_URL", "navan_stage5_smoke")
    failed = False
    try:
        Base.metadata.create_all(local_engine)
        sessions = sessionmaker(local_engine)
        orchestrator = Orchestrator(sessions, ai_client=AIClient(), reference_sessions=app.extensions["db_sessions"])
        for scenario in scenarios:
            try:
                report = run_scenario(sessions, orchestrator, scenario)
            except CustomerWorkflowFailure as error:
                report = {
                    "scenario": scenario, "stage": error.stage, "reason": error.reason,
                    "company_id": error.company_id, "activity_id": error.activity_id,
                    "gate": asdict(error.gate) if error.gate is not None else None,
                    "predictions": error.predictions,
                    "executions": [asdict(item) for item in error.executions],
                    "duration_seconds": error.duration_seconds,
                }
                failed = True
            print(json.dumps(report, default=encode), flush=True)
    finally:
        local_engine.dispose()
        reference_engine.dispose()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
