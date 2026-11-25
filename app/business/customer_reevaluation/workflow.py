from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from time import monotonic
from typing import cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from ...ai.budget import EvaluationBudget
from ...ai.client import AIClient, AIExecutionError, ExecutionMetadata
from ...ai.config import AIConfigurationError, Component
from ...ai.diagnostics import current_evaluation, digest, enter_stage, observe_evaluation
from ...ai.engines import DecisionEngine, PredictionEngine
from ...ai.progress import ProgressCallback, ProgressKind
from ...ai.prompts import PromptRenderError, Task
from ...ai.schemas import ProductAssessmentOutput, RecommendationOutput, evaluation_output_fields
from ...ai.tools import ToolError
from ...data.access import CompanyConflict, WorkspaceData
from ...data.company_lock import company_lock
from ...data.models import Company, Evaluation, ProductActivity, TierChange
from .gate import ReevaluationGate, TriggerResult
from ...data.reference_data import BusinessPolicyNotFound, PromptNotFound, get_business_policy
from ..rules.tier_eligibility import is_actionable_tier_change
from ..rules.validation import CustomerActivityInput, ValidationError, validate_activity
from .._evaluation import EvaluationWorkflow


@dataclass(frozen=True)
class CustomerActivityResult:
    company_id: UUID
    activity_id: UUID
    created: bool
    external_tier_changed: bool
    superseded_count: int
    gate: TriggerResult
    unchanged: bool = False


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
    unchanged: bool = False


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


class CustomerReevaluation(EvaluationWorkflow):
    """Record and evaluate customer activity within a workspace."""

    @observe_evaluation("customer")
    def submit_customer_reevaluation(
        self, workspace_id: UUID, activity: CustomerActivityInput, *, on_progress: ProgressCallback | None = None,
        correlation_id: str | None = None,
    ) -> CustomerReevaluationResult:
        """Evaluate new or changed activity under the current policy."""
        start = monotonic()
        enter_stage(CustomerStage.ACTIVITY_PERSISTENCE)
        try:
            activity = validate_activity(activity)
            with company_lock(self.sessions, workspace_id, activity.company_name):
                try:
                    with self.reference_sessions() as session:
                        policy = get_business_policy(session)
                except (BusinessPolicyNotFound, SQLAlchemyError) as error:
                    saved = self._submit_customer_activity(workspace_id, activity)
                    self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=CustomerStage.CONTEXT)
                    raise CustomerWorkflowFailure(
                        CustomerStage.CONTEXT, type(error).__name__, saved.company_id, saved.activity_id,
                        (), monotonic() - start, saved.gate, exception_type=type(error).__name__,
                    ) from None
                saved = self._submit_customer_activity(workspace_id, activity, policy_sha256=digest(policy))
                if saved.unchanged:
                    return CustomerReevaluationResult(
                        saved.company_id, saved.activity_id, activity, saved.gate, monotonic() - start,
                        unchanged=True,
                    )
                self._emit(on_progress, ProgressKind.CUSTOMER_REGISTERED if saved.created else ProgressKind.CUSTOMER_UPDATED)
                self._emit(on_progress, ProgressKind.CUSTOMER_ACTIVITY_PERSISTED)
                self._emit(on_progress, ProgressKind.REEVALUATION_GATE_ASSESSED, gate_status=saved.gate.status)
                return self._evaluate_customer(
                    workspace_id, saved.activity_id, start, on_progress, saved=saved, policy=policy,
                )
        except (ValidationError, CompanyConflict, SQLAlchemyError, BusinessPolicyNotFound, _CustomerStateError) as error:
            stage = CustomerStage.ACTIVITY_PERSISTENCE
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=stage)
            reason = str(error) if isinstance(error, _CustomerStateError) else type(error).__name__
            raise CustomerWorkflowFailure(
                stage, reason, None, None, (), monotonic() - start, exception_type=type(error).__name__,
            ) from None

    @observe_evaluation("customer", retry=True)
    def retry_customer_reevaluation(
        self, workspace_id: UUID, activity_id: UUID, *, on_progress: ProgressCallback | None = None,
        correlation_id: str | None = None,
    ) -> CustomerReevaluationResult:
        """Restart AI against a still-current saved snapshot without resubmitting it."""
        start = monotonic()
        with self.sessions() as session:
            data = WorkspaceData(session, workspace_id)
            snapshot = data.get_company_record(ProductActivity, activity_id)
            company = data.get_company(snapshot.company_id) if snapshot is not None else None
            name = company.name if company is not None else None
        if name is None:
            enter_stage(CustomerStage.CONTEXT)
            self._emit(on_progress, ProgressKind.EVALUATION_FAILED, stage=CustomerStage.CONTEXT)
            raise CustomerWorkflowFailure(
                CustomerStage.CONTEXT, "activity_not_found", None, None, (), monotonic() - start,
                exception_type="_CustomerStateError",
            )
        with company_lock(self.sessions, workspace_id, name):
            return self._evaluate_customer(workspace_id, activity_id, start, on_progress)

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
        *, saved: CustomerActivityResult | None = None, policy: str | None = None,
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

            client = self.ai_client if self.ai_client is not None else AIClient()
            budget = EvaluationBudget.from_env(start)
            diagnostics = current_evaluation.get()
            if diagnostics is not None:
                diagnostics.configuration = {
                    "deadline_seconds": budget.deadline - start, "max_estimated_usd": budget.max_estimated_usd,
                }
            policy, renderer = self._ai_context(
                client, "reevaluation_system", Task.REEVALUATION_PREDICTION, Task.REEVALUATION_RECOMMENDATION, policy=policy,
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
            actionable = is_actionable_tier_change(activity.current_tier, recommendation.recommended_tier)
            with self.sessions.begin() as session:
                output = evaluation_output_fields(predictions, recommendation)
                output["predictions"]["activity_snapshot_id"] = str(activity_id)
                output["predictions"]["policy_sha256"] = digest(policy)
                evaluation = Evaluation(
                    company_id=owned_company_id, research_id=None, evaluation_type="re_evaluation",
                    created_at=func.clock_timestamp(),
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
        self, workspace_id: UUID, activity: CustomerActivityInput, *, policy_sha256: str | None = None,
    ) -> CustomerActivityResult:
        activity = validate_activity(activity)
        with company_lock(self.sessions, workspace_id, activity.company_name):
            return self._submit_customer_activity(workspace_id, activity, policy_sha256=policy_sha256)

    def _submit_customer_activity(
        self, workspace_id: UUID, activity: CustomerActivityInput, *, policy_sha256: str | None = None,
    ) -> CustomerActivityResult:
        gate = ReevaluationGate.assess(activity)
        with self.sessions.begin() as session:
            data = WorkspaceData(session, workspace_id)
            company = data.find_company(activity.company_name, activity.company_domain)
            created = company is None
            values = asdict(activity)
            for key in ("company_name", "company_domain", "current_tier"):
                del values[key]
            if policy_sha256 is not None and company is not None and company.current_tier == activity.current_tier:
                latest, = data.company_records(ProductActivity, company.id, limit=1) or (None,)
                if latest is not None and company.domain == activity.company_domain and all(
                    getattr(latest, key) == value for key, value in values.items()
                ):
                    completed = session.scalar(select(Evaluation.id).where(
                        Evaluation.company_id == company.id, Evaluation.evaluation_type == "re_evaluation",
                        Evaluation.current_tier == activity.current_tier,
                        Evaluation.predictions["activity_snapshot_id"].as_string() == str(latest.id),
                        Evaluation.predictions["policy_sha256"].as_string() == policy_sha256,
                    ).limit(1))
                    if completed is not None:
                        return CustomerActivityResult(company.id, latest.id, False, False, 0, gate, unchanged=True)
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
                            created_at=func.clock_timestamp(),
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

            snapshot = ProductActivity(company_id=company.id, created_at=func.clock_timestamp(), **values)
            session.add(snapshot)
            session.flush()
            superseded = data.supersede_pending_evaluations(company.id)
            return CustomerActivityResult(company.id, snapshot.id, created, tier_changed, superseded, gate)
