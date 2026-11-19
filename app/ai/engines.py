from .budget import EvaluationBudget
from .client import AIClient, AIResult
from .config import AISettings, Component
from .progress import ProgressCallback
from .prompts import PromptRenderer, Task
from .schemas import LeadPredictionOutput, ProductAssessmentOutput, ResearchOutput
from ..reevaluation_gate import TriggerResult
from ..validation import CustomerActivityInput, LeadInput


class ResearchEngine:
    def __init__(self, client: AIClient, budget: EvaluationBudget | None = None):
        self.client = client
        self.budget = budget

    def research(
        self, renderer: PromptRenderer, policy: str, lead: LeadInput,
        on_progress: ProgressCallback | None = None,
    ) -> AIResult:
        prompt = renderer.render(
            Task.LEAD_RESEARCH, policy=policy, inputs={"company_profile": lead},
            settings=AISettings.from_env(Component.RESEARCH),
        )
        return self.client.execute(prompt, on_progress=on_progress, correct_validation=True, budget=self.budget)


class PredictionEngine:
    def __init__(self, client: AIClient, budget: EvaluationBudget | None = None):
        self.client = client
        self.budget = budget

    def predict(
        self, renderer: PromptRenderer, policy: str, lead: LeadInput, research: ResearchOutput,
        on_progress: ProgressCallback | None = None,
    ) -> AIResult:
        prompt = renderer.render(
            Task.LEAD_PREDICTION, policy=policy,
            inputs={"company_profile": lead, "validated_research_findings": research},
            settings=AISettings.from_env(Component.PREDICTION),
        )
        return self.client.execute(prompt, on_progress=on_progress, correct_validation=True, budget=self.budget)


    def assess_customer(
        self, renderer: PromptRenderer, policy: str, activity: CustomerActivityInput, gate: TriggerResult,
        on_progress: ProgressCallback | None = None,
    ) -> AIResult:
        prompt = renderer.render(
            Task.REEVALUATION_PREDICTION, policy=policy,
            inputs={
                "validated_product_data": activity, "derived_metrics": gate.metrics, "trigger_results": gate,
            },
            settings=AISettings.from_env(Component.PREDICTION),
        )
        return self.client.execute(prompt, on_progress=on_progress, correct_validation=True, budget=self.budget)


class DecisionEngine:
    def __init__(self, client: AIClient, budget: EvaluationBudget | None = None):
        self.client = client
        self.budget = budget

    def recommend(
        self, renderer: PromptRenderer, policy: str, lead: LeadInput, current_tier: int,
        research: ResearchOutput, predictions: LeadPredictionOutput,
        on_progress: ProgressCallback | None = None,
    ) -> AIResult:
        prompt = renderer.render(
            Task.LEAD_RECOMMENDATION, policy=policy,
            inputs={
                "company_profile": lead, "current_tier": current_tier,
                "validated_research_findings": research, "validated_predictions": predictions,
            },
            settings=AISettings.from_env(Component.DECISION),
        )
        return self.client.execute(prompt, on_progress=on_progress, correct_validation=True, budget=self.budget)


    def recommend_customer(
        self, renderer: PromptRenderer, policy: str, activity: CustomerActivityInput,
        gate: TriggerResult, assessment: ProductAssessmentOutput,
        on_progress: ProgressCallback | None = None,
    ) -> AIResult:
        prompt = renderer.render(
            Task.REEVALUATION_RECOMMENDATION, policy=policy,
            inputs={
                "current_tier": activity.current_tier, "validated_product_assessment": assessment,
                "validated_product_data": activity, "derived_metrics": gate.metrics, "trigger_results": gate,
            },
            settings=AISettings.from_env(Component.DECISION),
        )
        return self.client.execute(prompt, on_progress=on_progress, correct_validation=True, budget=self.budget)
