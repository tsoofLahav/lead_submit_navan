from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from time import monotonic
from types import MappingProxyType
from typing import cast
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .ai.budget import EvaluationBudget
from .ai.client import AIClient, AIExecutionError, ExecutionMetadata
from .ai.config import AIConfigurationError, Component
from .ai.diagnostics import current_evaluation, enter_stage, observe_evaluation
from .ai.engines import DecisionEngine, PredictionEngine, ResearchEngine
from .ai.progress import ProgressCallback, ProgressEvent, ProgressKind
from .ai.prompts import PromptRenderError, PromptRenderer, Task
from .ai.schemas import LeadPredictionOutput, ProductAssessmentOutput, RecommendationOutput, ResearchOutput, evaluation_output_fields
from .ai.tools import ToolError
from .data_access import CompanyConflict, WorkspaceData
from .human_approval import HumanApproval
from .models import Company, CompanyResearch, Evaluation, ProductActivity, TierChange
from .reevaluation_gate import GateStatus, ReevaluationGate, TriggerResult
from .reference_data import BusinessPolicyNotFound, PromptNotFound, get_prompt
from .tier_eligibility import initial_tier
from .validation import CustomerActivityInput, LeadInput, ValidationError, validate_activity, validate_lead


@dataclass(frozen=True)
class LeadResult:
    company_id: UUID
    current_tier: int
    created: bool


@dataclass(frozen=True)
class CustomerActivityResult:
    company_id: UUID
    activity_id: UUID
    created: bool
    external_tier_changed: bool
    superseded_count: int
    gate: TriggerResult


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


class CustomerStage(StrEnum):
    ACTIVITY_PERSISTENCE = "activity_persistence"
    CONTEXT = "context"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    EVALUATION_PERSISTENCE = "evaluation_persistence"


@dataclass(frozen=True)
class CustomerReevaluationResult:
    company_id: UUID
    activity_id: UUID
    customer: CustomerActivityInput
    gate: TriggerResult
    duration_seconds: float
    predictions: ProductAssessmentOutput | None = None
    recommendation: RecommendationOutput | None = None
    evaluation_id: UUID | None = None
    approval_available: bool = False
    executions: tuple[ExecutionMetadata, ...] = ()


class CustomerWorkflowFailure(RuntimeError):
    def __init__(
        self, stage: CustomerStage, reason: str, company_id: UUID | None, activity_id: UUID | None,
        executions: tuple[ExecutionMetadata, ...], duration_seconds: float,
        gate: TriggerResult | None = None, predictions: ProductAssessmentOutput | None = None,
        *, exception_type: str | None = None,
    ):
        self.stage = stage
        self.exception_type = exception_type
        self.reason = reason
        self.company_id = company_id
        self.activity_id = activity_id
        self.executions = executions
        self.duration_seconds = duration_seconds
        self.gate = gate
        self.predictions = predictions
        super().__init__(f"Customer re-evaluation failed during {stage.value} ({reason}).")


class _CustomerStateError(ValueError):
    pass


class Orchestrator:
    def __init__(
        self, sessions: sessionmaker[Session], *, ai_client: AIClient | None = None,
        reference_sessions: sessionmaker[Session] | None = None,
    ):
        self.sessions = sessions
        self.reference_sessions = reference_sessions if reference_sessions is not None else sessions
        self.ai_client = ai_client

    def register_lead(self, workspace_id: UUID, lead: LeadInput) -> LeadResult:
        lead = validate_lead(lead)
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
            registration = self.register_lead(workspace_id, lead)
        except (ValidationError, CompanyConflict, SQLAlchemyError) as error:
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=LeadStage.REGISTRATION)
            raise LeadWorkflowFailure(
                LeadStage.REGISTRATION, type(error).__name__, None, None, (), monotonic() - start,
                exception_type=type(error).__name__,
            ) from None
        if not registration.created:
            return registration
        self._emit(on_progress, ProgressKind.COMPANY_REGISTERED)
        self._emit(on_progress, ProgressKind.INITIAL_TIER_ASSIGNED, tier=registration.current_tier)
        return self._evaluate_lead(workspace_id, registration.company_id, start, on_progress)

    @observe_evaluation("lead", retry=True)
    def retry_lead_evaluation(
        self, workspace_id: UUID, company_id: UUID, *, on_progress: ProgressCallback | None = None,
        correlation_id: str | None = None,
    ) -> LeadEvaluationResult:
        """Start a fresh evaluation using the stored lead, without re-registering it."""
        return self._evaluate_lead(workspace_id, company_id, monotonic(), on_progress)

    @staticmethod
    def _emit(
        callback: ProgressCallback | None, kind: ProgressKind, component: Component | None = None,
        *, stage: LeadStage | CustomerStage | None = None, tier: int | None = None, gate_status: GateStatus | None = None,
    ) -> None:
        if callback is not None:
            callback(ProgressEvent(
                kind, component, datetime.now(UTC), workflow_stage=stage, initial_tier=tier, gate_status=gate_status,
            ))

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
                recorded_tier = company.current_tier
                original_tier, activity_count, change_count = data.lead_history(company)

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
                if WorkspaceData(session, workspace_id).get_company(company_id) is None:
                    raise _LeadStateError("lead_not_found")
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
            actionable = HumanApproval.is_actionable_tier_change(recorded_tier, recommendation.recommended_tier)
            with self.sessions.begin() as session:
                data = WorkspaceData(session, workspace_id)
                company = data.get_company(company_id)
                if company is None:
                    raise _LeadStateError("lead_not_found")
                if company.current_tier != recorded_tier or (
                    actionable and data.lead_history(company)[1:] != (activity_count, change_count)
                ):
                    raise _LeadStateError("stale_company_state")
                current_tier, is_customer = company.current_tier, company.is_customer
                evaluation = Evaluation(
                    company_id=company_id, research_id=research_id, evaluation_type="lead_submit",
                    current_tier=recorded_tier, decision_status="pending" if actionable else "not_applicable",
                    **evaluation_output_fields(predictions, recommendation),
                )
                session.add(evaluation)
                session.flush()
                evaluation_id = evaluation.id
            self._emit(on_progress, ProgressKind.EVALUATION_COMPLETED)
            return LeadEvaluationResult(
                company_id, lead, is_customer, original_tier, current_tier, research_id, research, predictions,
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

    def _ai_context(self, client: AIClient, system_key: str, *tasks: Task) -> tuple[str, PromptRenderer]:
        with self.reference_sessions() as session:
            policy = PromptRenderer(session, client.tools).load_policy()
            templates = MappingProxyType({key: get_prompt(session, key) for key in (system_key, *tasks)})
        return policy, PromptRenderer(None, client.tools, templates=templates)

    @observe_evaluation("customer")
    def submit_customer_reevaluation(
        self, workspace_id: UUID, activity: CustomerActivityInput, *, on_progress: ProgressCallback | None = None,
        correlation_id: str | None = None,
    ) -> CustomerReevaluationResult:
        """Save customer activity and run AI only if the existing gate triggers."""
        start = monotonic()
        enter_stage(CustomerStage.ACTIVITY_PERSISTENCE)
        try:
            saved = self.submit_customer_activity(workspace_id, activity)
        except (ValidationError, CompanyConflict, SQLAlchemyError) as error:
            stage = CustomerStage.ACTIVITY_PERSISTENCE
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=stage)
            raise CustomerWorkflowFailure(
                stage, type(error).__name__, None, None, (), monotonic() - start,
                exception_type=type(error).__name__,
            ) from None
        self._emit(on_progress, ProgressKind.CUSTOMER_REGISTERED if saved.created else ProgressKind.CUSTOMER_UPDATED)
        self._emit(on_progress, ProgressKind.CUSTOMER_ACTIVITY_PERSISTED)
        self._emit(on_progress, ProgressKind.REEVALUATION_GATE_ASSESSED, gate_status=saved.gate.status)
        if saved.gate.status != GateStatus.TRIGGERED:
            return CustomerReevaluationResult(
                saved.company_id, saved.activity_id, validate_activity(activity), saved.gate, monotonic() - start,
            )
        return self._evaluate_customer(workspace_id, saved.activity_id, start, on_progress, saved=saved,
                                       submitted_tier=activity.current_tier)

    @observe_evaluation("customer", retry=True)
    def retry_customer_reevaluation(
        self, workspace_id: UUID, activity_id: UUID, *, on_progress: ProgressCallback | None = None,
        correlation_id: str | None = None,
    ) -> CustomerReevaluationResult:
        """Restart AI against a still-current saved snapshot without resubmitting it."""
        return self._evaluate_customer(workspace_id, activity_id, monotonic(), on_progress)

    @staticmethod
    def _check_customer_snapshot(data: WorkspaceData, snapshot: ProductActivity) -> None:
        newer_activity, later_tier_change, completed = data.customer_snapshot_state(snapshot)
        if newer_activity:
            raise _CustomerStateError("stale_activity")
        if later_tier_change:
            raise _CustomerStateError("stale_tier")
        if completed:
            raise _CustomerStateError("evaluation_already_completed")

    def _evaluate_customer(
        self, workspace_id: UUID, activity_id: UUID, start: float, on_progress: ProgressCallback | None,
        *, saved: CustomerActivityResult | None = None, submitted_tier: int | None = None,
    ) -> CustomerReevaluationResult:
        stage = CustomerStage.CONTEXT
        enter_stage(stage)
        owned_company_id = saved.company_id if saved else None
        owned_activity_id = saved.activity_id if saved else None
        gate = saved.gate if saved else None
        predictions = None
        executions = []
        try:
            with self.sessions() as session:
                data = WorkspaceData(session, workspace_id)
                snapshot = data.get_company_record(ProductActivity, activity_id)
                if snapshot is None:
                    raise _CustomerStateError("activity_not_found")
                company = data.get_company(snapshot.company_id)
                if company is None or not company.is_customer:
                    raise _CustomerStateError("activity_not_found")
                owned_company_id, owned_activity_id = company.id, snapshot.id
                self._check_customer_snapshot(data, snapshot)
                if submitted_tier is not None and company.current_tier != submitted_tier:
                    raise _CustomerStateError("stale_tier")
                activity = validate_activity(CustomerActivityInput(
                    company.name, company.domain, company.current_tier,
                    **{
                        field.name: getattr(snapshot, field.name) for field in fields(CustomerActivityInput)
                        if field.name not in {"company_name", "company_domain", "current_tier"}
                    },
                ))
            if gate is None:
                gate = ReevaluationGate.assess(activity)
                self._emit(on_progress, ProgressKind.REEVALUATION_GATE_ASSESSED, gate_status=gate.status)
                if gate.status != GateStatus.TRIGGERED:
                    raise _CustomerStateError("activity_not_triggered")

            client = self.ai_client if self.ai_client is not None else AIClient()
            budget = EvaluationBudget.from_env(start)
            diagnostics = current_evaluation.get()
            if diagnostics is not None:
                diagnostics.configuration = {
                    "deadline_seconds": budget.deadline - start, "max_estimated_usd": budget.max_estimated_usd,
                }
            policy, renderer = self._ai_context(
                client, "reevaluation_system", Task.REEVALUATION_PREDICTION, Task.REEVALUATION_RECOMMENDATION,
            )
            stage = CustomerStage.PREDICTION
            enter_stage(stage)
            self._emit(on_progress, ProgressKind.PREDICTION_STARTED, Component.PREDICTION)
            execution = PredictionEngine(client, budget).assess_customer(renderer, policy, activity, gate, on_progress)
            executions.append(execution.metadata)
            predictions = cast(ProductAssessmentOutput, execution.output)
            self._emit(on_progress, ProgressKind.PREDICTION_COMPLETED, Component.PREDICTION)

            stage = CustomerStage.RECOMMENDATION
            enter_stage(stage)
            self._emit(on_progress, ProgressKind.RECOMMENDATION_STARTED, Component.DECISION)
            execution = DecisionEngine(client, budget).recommend_customer(renderer, policy, activity, gate, predictions, on_progress)
            executions.append(execution.metadata)
            recommendation = cast(RecommendationOutput, execution.output)
            self._emit(on_progress, ProgressKind.RECOMMENDATION_COMPLETED, Component.DECISION)

            stage = CustomerStage.EVALUATION_PERSISTENCE
            enter_stage(stage)
            actionable = HumanApproval.is_actionable_tier_change(activity.current_tier, recommendation.recommended_tier)
            with self.sessions.begin() as session:
                data = WorkspaceData(session, workspace_id)
                snapshot = data.get_company_record(ProductActivity, activity_id)
                company = data.get_company(owned_company_id)
                if snapshot is None or company is None or snapshot.company_id != company.id or not company.is_customer:
                    raise _CustomerStateError("activity_not_found")
                if company.current_tier != activity.current_tier:
                    raise _CustomerStateError("stale_tier")
                self._check_customer_snapshot(data, snapshot)
                output = evaluation_output_fields(predictions, recommendation)
                output["predictions"]["activity_snapshot_id"] = str(activity_id)
                evaluation = Evaluation(
                    company_id=company.id, research_id=None, evaluation_type="re_evaluation",
                    current_tier=activity.current_tier, decision_status="pending" if actionable else "not_applicable",
                    **output,
                )
                session.add(evaluation)
                session.flush()
                evaluation_id = evaluation.id
            self._emit(on_progress, ProgressKind.EVALUATION_COMPLETED)
            return CustomerReevaluationResult(
                owned_company_id, activity_id, activity, gate, monotonic() - start, predictions,
                recommendation, evaluation_id, actionable, tuple(executions),
            )
        except (
            AIExecutionError, AIConfigurationError, PromptRenderError, PromptNotFound,
            BusinessPolicyNotFound, ToolError, ValidationError, SQLAlchemyError, _CustomerStateError,
        ) as error:
            reason = type(error).__name__
            if isinstance(error, AIExecutionError):
                reason = error.kind.value
                executions.append(error.metadata)
            elif isinstance(error, _CustomerStateError):
                reason = str(error)
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=stage)
            raise CustomerWorkflowFailure(
                stage, reason, owned_company_id, owned_activity_id, tuple(executions),
                monotonic() - start, gate, predictions, exception_type=type(error).__name__,
            ) from None

    def submit_customer_activity(
        self, workspace_id: UUID, activity: CustomerActivityInput
    ) -> CustomerActivityResult:
        activity = validate_activity(activity)
        gate = ReevaluationGate.assess(activity)
        with self.sessions.begin() as session:
            data = WorkspaceData(session, workspace_id)
            company = data.find_company(activity.company_name, activity.company_domain)
            created = company is None
            tier_changed = False
            if created:
                company = Company(
                    workspace_id=workspace_id,
                    name=activity.company_name,
                    domain=activity.company_domain,
                    is_customer=True,
                    current_tier=activity.current_tier,
                    employee_count=activity.employee_count,
                )
                session.add(company)
            else:
                tier_changed = activity.current_tier != company.current_tier
                if tier_changed:
                    session.add(
                        TierChange(
                            company_id=company.id,
                            previous_tier=company.current_tier,
                            new_tier=activity.current_tier,
                        )
                    )
                company.current_tier = activity.current_tier
                company.is_customer = True
                company.domain = activity.company_domain
                if activity.employee_count is not None:
                    company.employee_count = activity.employee_count
            data.flush_company_changes()

            values = asdict(activity)
            for key in ("company_name", "company_domain", "current_tier"):
                del values[key]
            snapshot = ProductActivity(company_id=company.id, **values)
            session.add(snapshot)
            session.flush()
            superseded = HumanApproval.supersede_pending(data, company.id)
            return CustomerActivityResult(company.id, snapshot.id, created, tier_changed, superseded, gate)
