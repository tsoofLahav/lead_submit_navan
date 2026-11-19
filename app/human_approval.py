from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import exists, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from .data_access import WorkspaceData
from .models import Company, Evaluation, ProductActivity, TierChange
from .validation import TIERS


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

    @staticmethod
    def is_actionable_tier_change(current_tier: int, recommended_tier: int | None) -> bool:
        return recommended_tier in TIERS and recommended_tier != current_tier

    @staticmethod
    def supersede_pending(data: WorkspaceData, company_id: UUID) -> int:
        """Invalidate pending evaluations inside the caller's transaction."""
        result = data.session.execute(
            update(Evaluation)
            .where(
                Evaluation.company_id == company_id,
                Evaluation.decision_status == "pending",
                Evaluation.company_id.in_(
                    select(Company.id).where(Company.workspace_id == data.workspace_id)
                ),
            )
            .values(decision_status="superseded", decided_at=func.now())
        )
        return result.rowcount

    def _decide(self, workspace_id: UUID, evaluation_id: UUID, *, accept: bool) -> DecisionResult:
        with self.sessions.begin() as session:
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
                or not self.is_actionable_tier_change(previous_tier, target_tier)
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
                self.supersede_pending(data, company.id)
            return DecisionResult(
                evaluation_id, company.id, status, target_tier if accept else company.current_tier
            )
