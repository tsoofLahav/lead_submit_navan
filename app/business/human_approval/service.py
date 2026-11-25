from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import exists, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from ...data.access import WorkspaceData
from ...data.company_lock import acquire_company_lock
from ...data.models import Company, Evaluation, ProductActivity, TierChange
from ..rules.tier_eligibility import is_actionable_tier_change


class RecommendationNotFound(LookupError):
    pass


class RecommendationConflict(ValueError):
    pass


@dataclass(frozen=True)
class DecisionResult:
    evaluation_id: UUID
    company_id: UUID
    status: str
    current_tier: int


class HumanApproval:
    def __init__(self, sessions: sessionmaker[Session]):
        self.sessions = sessions

    def accept(self, workspace_id: UUID, evaluation_id: UUID) -> DecisionResult:
        return self._decide(workspace_id, evaluation_id, accept=True)

    def reject(self, workspace_id: UUID, evaluation_id: UUID) -> DecisionResult:
        return self._decide(workspace_id, evaluation_id, accept=False)

    def _decide(self, workspace_id: UUID, evaluation_id: UUID, *, accept: bool) -> DecisionResult:
        with self.sessions.begin() as session:
            company_name = session.scalar(
                select(Company.name).join(Evaluation, Evaluation.company_id == Company.id)
                .where(Evaluation.id == evaluation_id, Company.workspace_id == workspace_id)
            )
            if company_name is None:
                raise RecommendationNotFound("Recommendation not found in this workspace.")
            acquire_company_lock(session, workspace_id, company_name)
            data = WorkspaceData(session, workspace_id)
            evaluation = data.get_company_record(Evaluation, evaluation_id)
            if evaluation is None:
                raise RecommendationNotFound("Recommendation not found in this workspace.")
            company = data.get_company(evaluation.company_id)
            if company is None:
                raise RecommendationNotFound("Recommendation not found in this workspace.")
            previous_tier = evaluation.current_tier
            target_tier = evaluation.recommended_tier
            if (
                evaluation.decision_status != "pending"
                or not is_actionable_tier_change(previous_tier, target_tier)
            ):
                raise RecommendationConflict("The recommendation is no longer pending or actionable.")

            status = "accepted" if accept else "rejected"
            owned_company = select(Company.id).where(Company.workspace_id == workspace_id)
            transition = update(Evaluation).where(
                Evaluation.id == evaluation_id,
                Evaluation.decision_status == "pending",
                Evaluation.company_id.in_(owned_company),
                Evaluation.current_tier == previous_tier,
                Evaluation.recommended_tier == target_tier,
            )
            if accept:
                transition = transition.where(
                    Evaluation.company_id.in_(owned_company.where(Company.current_tier == previous_tier)),
                    ~exists().where(
                        ProductActivity.company_id == Evaluation.company_id,
                        ProductActivity.created_at > Evaluation.created_at,
                    ),
                )
            changed = session.execute(
                transition.values(decision_status=status, decided_at=func.now())
            )
            if changed.rowcount != 1:
                raise RecommendationConflict("The recommendation is stale or has already been decided.")

            if accept:
                updated = session.execute(
                    update(Company)
                    .where(
                        Company.id == company.id,
                        Company.workspace_id == workspace_id,
                        Company.current_tier == previous_tier,
                    )
                    .values(current_tier=target_tier)
                )
                if updated.rowcount != 1:
                    raise RecommendationConflict("The company's current tier has changed.")
                session.add(
                    TierChange(
                        company_id=company.id,
                        evaluation_id=evaluation_id,
                        previous_tier=previous_tier,
                        new_tier=target_tier,
                    )
                )
                data.supersede_pending_evaluations(company.id)
            return DecisionResult(
                evaluation_id, company.id, status, target_tier if accept else company.current_tier
            )
