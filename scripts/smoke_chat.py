"""Live read-only chat smoke test with synthetic data and local-only writes."""

import json
import os
from dataclasses import asdict
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app import create_app
from app.chat import ChatAssistant, ChatFailure
from app.models import Base, Company, CompanyResearch, Evaluation, ProductActivity, TierChange, Workspace
from scripts.smoke_database import open_local_database

QUESTIONS = (
    ("company", "Use the company details tool to tell me Northstar Demo's current tier, domain and submitted annual GBV."),
    ("evaluation", "Read Northstar Demo's saved evaluations. Why was the pending upgrade recommended, is it accepted, "
                   "and what research source supports the observation? Look up the saved source if needed."),
    ("history", "Read Northstar Demo's tier history. What change was actually applied and what was its origin?"),
    ("policy", "Consult the active business policy tool. What annual GBV qualifies for Tier 1?"),
    ("follow_up", "Using the company details you already found in this conversation, what was its domain again?"),
    ("mutation", "Approve Northstar Demo's pending recommendation and change its tier to Tier 1 now."),
)


def seed(sessions):
    with sessions.begin() as session:
        workspace = Workspace()
        session.add(workspace)
        session.flush()
        company = Company(
            workspace_id=workspace.id, name="Northstar Demo", domain="northstar-demo.test",
            submitted_annual_gbv=Decimal("520000"), employee_count=210, current_tier=2, is_customer=True,
        )
        session.add(company)
        session.flush()
        research = CompanyResearch(company_id=company.id, signals={"signals": [{
            "signal": "operational_complexity", "status": "found", "evidence": [{
                "value": "Two offices need separate booking approvals.", "estimated_range": None,
                "sources": [{"url": "https://northstar-demo.test/operations", "title": "Synthetic operations record",
                             "publication_date": "2026-01-03"}],
                "reliability": "medium", "evidence_type": "direct", "context": "Synthetic smoke-test fixture.",
            }], "information_gap": "Approver counts are unknown.",
        }]})
        session.add(research)
        session.flush()
        evaluation = Evaluation(
            company_id=company.id, research_id=research.id, evaluation_type="lead_submit",
            current_tier=2, recommended_tier=1, confidence="high", decision_status="pending",
            rationale="Submitted GBV of $520,000 supports Tier 1; two offices need separate booking approvals.",
            predictions={"expected_support_needs": {"level": "high", "confidence": "medium"},
                         "recommendation_evidence": {"outcome": "upgrade", "supporting_evidence": ["operational_complexity"]}},
            information_gaps=["Approver counts are unknown."], follow_up_actions=["Confirm the approval model."],
        )
        session.add_all((evaluation, TierChange(company_id=company.id, previous_tier=3, new_tier=2),
                         ProductActivity(company_id=company.id, months_since_signup=8, total_gbv=Decimal("310000"),
                                         employee_count=210, avg_monthly_active_users=Decimal("120"))))
        session.flush()
        return workspace.id, company.id, evaluation.id


def main():
    app = create_app()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run the live chat smoke test.")
    reference_engine = app.extensions["db_engine"]
    reference_engine.update_execution_options(postgresql_readonly=True)
    local = open_local_database("CHAT_SMOKE_DATABASE_URL", "navan_stage6_smoke")
    try:
        Base.metadata.create_all(local)
        sessions = sessionmaker(local)
        workspace_id, company_id, evaluation_id = seed(sessions)
        assistant = ChatAssistant(sessions, reference_sessions=app.extensions["db_sessions"])
        conversation = assistant.create_conversation(workspace_id)
        failed = False
        for name, question in QUESTIONS:
            try:
                turn = assistant.send_message(workspace_id, conversation.id, question)
                report = {"case": name, "answer": turn.assistant_message.content, "execution": asdict(turn.execution)}
            except ChatFailure as error:
                failed = True
                report = {"case": name, "error": error.reason,
                          "execution": asdict(error.execution) if error.execution else None}
            print(json.dumps(report, default=str), flush=True)

        with sessions() as session:
            company = session.get(Company, company_id)
            evaluation = session.get(Evaluation, evaluation_id)
            changes = list(session.scalars(select(TierChange).where(TierChange.company_id == company_id)))
            assert company.current_tier == 2 and evaluation.decision_status == "pending"
            assert len(changes) == 1 and changes[0].evaluation_id is None
        print(json.dumps({"case": "persistence", "current_tier": 2, "decision_status": "pending",
                          "recorded_tier_changes": len(changes),
                          "messages": len(assistant.list_messages(workspace_id, conversation.id))}), flush=True)
        if failed:
            raise SystemExit(1)
    finally:
        local.dispose()
        reference_engine.dispose()


if __name__ == "__main__":
    main()
