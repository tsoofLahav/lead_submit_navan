"""Opt-in live OpenAI smoke test using public company details and synthetic inputs."""

import json
import os
from dataclasses import asdict
from decimal import Decimal

from dotenv import load_dotenv

from app import create_app
from app.ai.client import AIClient, AIExecutionError
from app.ai.config import AISettings, Component
from app.ai.prompts import PromptRenderer, Task
from app.ai.tools import ToolRegistry
from app.business.rules.validation import LeadInput


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run the opt-in OpenAI smoke test.")

    app = create_app()
    engine = app.extensions["db_engine"]
    engine.update_execution_options(postgresql_readonly=True)
    settings = AISettings.from_env(Component.RESEARCH)
    tools = ToolRegistry()
    try:
        with app.extensions["db_sessions"]() as session:
            renderer = PromptRenderer(session, tools)
            prompt = renderer.render(
                Task.LEAD_RESEARCH,
                policy=renderer.load_policy(),
                inputs={"company_profile": LeadInput("Mozilla", "mozilla.org", Decimal("100000"))},
                settings=settings,
            )
    finally:
        engine.dispose()

    events = []

    def report(event):
        events.append(event)
        print(f"{event.kind.value} ({event.completed_tool_calls} completed)", flush=True)

    try:
        result = AIClient(tools).execute(prompt, on_progress=report)
    except AIExecutionError as error:
        print(json.dumps(asdict(error.metadata), default=str))
        raise SystemExit(str(error)) from None
    print(json.dumps(asdict(result.metadata), default=str))
    if not 1 <= result.metadata.completed_hosted_tool_calls <= settings.max_tool_calls:
        raise SystemExit("Expected completed hosted searches within the configured budget.")
    if not events or result.metadata.usage is None:
        raise SystemExit("Expected search progress and token usage.")


if __name__ == "__main__":
    main()
