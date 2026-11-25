"""Seed and smoke-test the revised policy only in a dedicated local database."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from app.ai.client import AIClient
from app.data.models import Base, BusinessPolicy, Prompt, Workspace
from app.business.lead_submission.workflow import LeadEvaluationResult, LeadSubmission
from scripts.smoke_database import open_local_database
from scripts.smoke_lead import PROFILES, encode_report_value, verify_persistence
from scripts.smoke_reevaluation import run_scenario

ROOT = Path(__file__).resolve().parents[1] / "seeds"


def seed_revised_policy(sessions):
    with sessions.begin() as session:
        policy = (ROOT / "business_prediction_policy.txt").read_text()
        existing = session.get(BusinessPolicy, 1)
        if existing is None:
            session.add(BusinessPolicy(id=1, content=policy))
        elif existing.content != policy:
            raise SystemExit("The local smoke database already contains a different policy.")
        for path in sorted((ROOT / "prompts").glob("*.txt")):
            content = path.read_text()
            row = session.get(Prompt, path.stem)
            if row is None:
                session.add(Prompt(prompt_key=path.stem, content=content))
            elif row.content != content:
                raise SystemExit(f"The local smoke database contains a different {path.stem} prompt.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("seed", "google", "gitlab", "customer"))
    scenario = parser.parse_args().scenario
    load_dotenv()
    engine = open_local_database("REVISED_POLICY_SMOKE_DATABASE_URL", "navan_policy_revised_smoke")
    try:
        Base.metadata.create_all(engine)
        sessions = sessionmaker(engine)
        seed_revised_policy(sessions)
        if scenario == "seed":
            return
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY for live smoke tests.")
        orchestrator = LeadSubmission(sessions, ai_client=AIClient())
        if scenario == "customer":
            report = run_scenario(sessions, orchestrator, "upgrade")
        else:
            with sessions.begin() as session:
                workspace = Workspace()
                session.add(workspace)
                session.flush()
                workspace_id = workspace.id
            result = orchestrator.submit_lead(
                workspace_id, PROFILES[scenario],
                on_progress=lambda event: print(f"{scenario}: {event.kind.value}", flush=True),
            )
            if not isinstance(result, LeadEvaluationResult):
                raise RuntimeError("Expected a completed lead evaluation.")
            verify_persistence(sessions, result)
            report = asdict(result)
        print(json.dumps(report, default=encode_report_value))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
