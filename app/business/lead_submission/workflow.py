from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import cast
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from ...ai.budget import EvaluationBudget
from ...ai.client import AIClient, AIExecutionError, ExecutionMetadata
from ...ai.config import AIConfigurationError, Component
from ...ai.diagnostics import current_evaluation, enter_stage, observe_evaluation
from ...ai.engines import DecisionEngine, PredictionEngine, ResearchEngine
from ...ai.progress import ProgressCallback, ProgressKind
from ...ai.prompts import PromptRenderError, Task
from ...ai.schemas import LeadPredictionOutput, RecommendationOutput, ResearchOutput, evaluation_output_fields
from ...ai.tools import ToolError
from ...data.access import CompanyConflict, WorkspaceData
from ...data.company_lock import company_lock
from ...data.models import Company, CompanyResearch, Evaluation
from ...data.reference_data import BusinessPolicyNotFound, PromptNotFound
from ..rules.tier_eligibility import initial_tier, is_actionable_tier_change
from ..rules.validation import LeadInput, ValidationError, validate_lead
from .._evaluation import EvaluationWorkflow


@dataclass(frozen=True)
class LeadResult:
    company_id: UUID
    current_tier: int
    created: bool


class LeadStage(StrEnum):
    REGISTRATION = "registration"
    CONTEXT = "context"
    RESEARCH = "research"
    RESEARCH_PERSISTENCE = "research_persistence"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    EVALUATION_PERSISTENCE = "evaluation_persistence"


@dataclass(frozen=True)
class LeadEvaluationResult:
    company_id: UUID
    company: LeadInput
    is_customer: bool
    initial_tier: int
    current_tier: int
    research_id: UUID
    research: ResearchOutput
    predictions: LeadPredictionOutput
    recommendation: RecommendationOutput
    evaluation_id: UUID
    approval_available: bool
    executions: tuple[ExecutionMetadata, ...]
    duration_seconds: float


class LeadWorkflowFailure(RuntimeError):
    def __init__(
        self, stage: LeadStage, reason: str, company_id: UUID | None, research_id: UUID | None,
        executions: tuple[ExecutionMetadata, ...], duration_seconds: float,
        predictions: LeadPredictionOutput | None = None, *, exception_type: str | None = None,
    ):
        self.stage = stage
        self.exception_type = exception_type
        self.reason = reason
        self.company_id = company_id
        self.research_id = research_id
        self.executions = executions
        self.duration_seconds = duration_seconds
        self.predictions = predictions
        super().__init__(f"Lead evaluation failed during {stage.value} ({reason}).")


class _LeadStateError(ValueError):
    pass


class LeadSubmission(EvaluationWorkflow):
    """Register and evaluate leads within a workspace."""

    def register_lead(self, workspace_id: UUID, lead: LeadInput) -> LeadResult:
        lead = validate_lead(lead)
        with company_lock(self.sessions, workspace_id, lead.company_name):
            return self._register_lead(workspace_id, lead)

    def _register_lead(self, workspace_id: UUID, lead: LeadInput) -> LeadResult:
        with self.sessions.begin() as session:
            data = WorkspaceData(session, workspace_id)
            company = data.find_company(lead.company_name, lead.company_domain)
            if company is not None:
                return LeadResult(company.id, company.current_tier, created=False)
            company = Company(
                workspace_id=workspace_id,
                name=lead.company_name,
                domain=lead.company_domain,
                submitted_annual_gbv=lead.submitted_annual_gbv,
                employee_count=lead.employee_count,
                current_tier=initial_tier(lead.submitted_annual_gbv, lead.employee_count),
                is_customer=False,
            )
            session.add(company)
            data.flush_company_changes()
            return LeadResult(company.id, company.current_tier, created=True)

    @observe_evaluation("lead")
    def submit_lead(
        self, workspace_id: UUID, lead: LeadInput, *, on_progress: ProgressCallback | None = None,
        correlation_id: str | None = None,
    ) -> LeadEvaluationResult | LeadResult:
        """Register and evaluate a new lead. A duplicate returns its existing registration."""
        start = monotonic()
        enter_stage(LeadStage.REGISTRATION)
        try:
            lead = validate_lead(lead)
            with company_lock(self.sessions, workspace_id, lead.company_name):
                registration = self._register_lead(workspace_id, lead)
                if not registration.created:
                    return registration
                self._emit(on_progress, ProgressKind.COMPANY_REGISTERED)
                self._emit(on_progress, ProgressKind.INITIAL_TIER_ASSIGNED, tier=registration.current_tier)
                return self._evaluate_lead(workspace_id, registration.company_id, start, on_progress)
        except (ValidationError, CompanyConflict, SQLAlchemyError) as error:
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=LeadStage.REGISTRATION)
            raise LeadWorkflowFailure(
                LeadStage.REGISTRATION, type(error).__name__, None, None, (), monotonic() - start,
                exception_type=type(error).__name__,
            ) from None

    @observe_evaluation("lead", retry=True)
    def retry_lead_evaluation(
        self, workspace_id: UUID, company_id: UUID, *, on_progress: ProgressCallback | None = None,
        correlation_id: str | None = None,
    ) -> LeadEvaluationResult:
        """Start a fresh evaluation using the stored lead, without re-registering it."""
        start = monotonic()
        with self.sessions() as session:
            company = WorkspaceData(session, workspace_id).get_company(company_id)
            name = company.name if company is not None else None
        if name is None:
            enter_stage(LeadStage.CONTEXT)
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=LeadStage.CONTEXT)
            raise LeadWorkflowFailure(
                LeadStage.CONTEXT, "lead_not_found", None, None, (), monotonic() - start,
                exception_type="_LeadStateError",
            )
        with company_lock(self.sessions, workspace_id, name):
            return self._evaluate_lead(workspace_id, company_id, start, on_progress)

    def _evaluate_lead(
        self, workspace_id: UUID, company_id: UUID, start: float, on_progress: ProgressCallback | None,
    ) -> LeadEvaluationResult:
        stage = LeadStage.CONTEXT
        enter_stage(stage)
        research_id = None
        owned_company_id = None
        executions = []
        predictions = None
        try:
            with self.sessions() as session:
                data = WorkspaceData(session, workspace_id)
                company = data.get_company(company_id)
                if company is None or company.submitted_annual_gbv is None:
                    raise _LeadStateError("lead_not_found")
                owned_company_id = company.id
                lead = validate_lead(LeadInput(
                    company.name, company.domain, company.submitted_annual_gbv, company.employee_count,
                ))
                recorded_tier, is_customer = company.current_tier, company.is_customer
                original_tier = data.original_tier(company)

            client = self.ai_client if self.ai_client is not None else AIClient()
            budget = EvaluationBudget.from_env(start)
            diagnostics = current_evaluation.get()
            if diagnostics is not None:
                diagnostics.configuration = {
                    "deadline_seconds": budget.deadline - start, "max_estimated_usd": budget.max_estimated_usd,
                }
            policy, renderer = self._ai_context(
                client, "lead_submit_system", Task.LEAD_RESEARCH, Task.LEAD_PREDICTION, Task.LEAD_RECOMMENDATION,
            )

            stage = LeadStage.RESEARCH
            enter_stage(stage)
            self._emit(on_progress, ProgressKind.RESEARCH_STARTED, Component.RESEARCH)
            execution = ResearchEngine(client, budget).research(renderer, policy, lead, on_progress)
            executions.append(execution.metadata)
            research = cast(ResearchOutput, execution.output)

            stage = LeadStage.RESEARCH_PERSISTENCE
            enter_stage(stage)
            with self.sessions.begin() as session:
                research_record = CompanyResearch(company_id=company_id, signals=research.model_dump(mode="json"))
                session.add(research_record)
                session.flush()
                pending_research_id = research_record.id
            research_id = pending_research_id
            self._emit(on_progress, ProgressKind.RESEARCH_COMPLETED, Component.RESEARCH)

            stage = LeadStage.PREDICTION
            enter_stage(stage)
            self._emit(on_progress, ProgressKind.PREDICTION_STARTED, Component.PREDICTION)
            execution = PredictionEngine(client, budget).predict(renderer, policy, lead, research, on_progress)
            executions.append(execution.metadata)
            predictions = cast(LeadPredictionOutput, execution.output)
            self._emit(on_progress, ProgressKind.PREDICTION_COMPLETED, Component.PREDICTION)

            stage = LeadStage.RECOMMENDATION
            enter_stage(stage)
            self._emit(on_progress, ProgressKind.RECOMMENDATION_STARTED, Component.DECISION)
            execution = DecisionEngine(client, budget).recommend(renderer, policy, lead, recorded_tier, research, predictions, on_progress)
            executions.append(execution.metadata)
            recommendation = cast(RecommendationOutput, execution.output)
            self._emit(on_progress, ProgressKind.RECOMMENDATION_COMPLETED, Component.DECISION)

            stage = LeadStage.EVALUATION_PERSISTENCE
            enter_stage(stage)
            actionable = is_actionable_tier_change(recorded_tier, recommendation.recommended_tier)
            with self.sessions.begin() as session:
                evaluation = Evaluation(
                    company_id=company_id, research_id=research_id, evaluation_type="lead_submit",
                    created_at=func.clock_timestamp(),
                    current_tier=recorded_tier, decision_status="pending" if actionable else "not_applicable",
                    **evaluation_output_fields(predictions, recommendation),
                )
                session.add(evaluation)
                session.flush()
                evaluation_id = evaluation.id
            self._emit(on_progress, ProgressKind.EVALUATION_COMPLETED)
            return LeadEvaluationResult(
                company_id, lead, is_customer, original_tier, recorded_tier, research_id, research, predictions,
                recommendation, evaluation_id, actionable, tuple(executions), monotonic() - start,
            )
        except (
            AIExecutionError, AIConfigurationError, PromptRenderError, PromptNotFound,
            BusinessPolicyNotFound, ToolError, ValidationError, SQLAlchemyError, _LeadStateError,
        ) as error:
            reason = type(error).__name__
            if isinstance(error, AIExecutionError):
                reason = error.kind.value
                executions.append(error.metadata)
            elif isinstance(error, _LeadStateError):
                reason = str(error)
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=stage)
            raise LeadWorkflowFailure(
                stage, reason, owned_company_id, research_id, tuple(executions), monotonic() - start, predictions,
                exception_type=type(error).__name__,
            ) from None
