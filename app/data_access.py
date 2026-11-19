from typing import TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import Company, CompanyResearch, Conversation, Evaluation, Message, ProductActivity, TierChange
from .validation import company_name_key, domain_key

CompanyRecord = TypeVar("CompanyRecord", CompanyResearch, ProductActivity, Evaluation, TierChange)


class CompanyConflict(ValueError):
    pass


class WorkspaceData:
    """Read records through the workspace resolved for the current request."""

    def __init__(self, session: Session, workspace_id: UUID):
        self.session = session
        self.workspace_id = workspace_id

    def find_company(self, name: str, domain: str) -> Company | None:
        name_key = company_name_key(name)
        matches = []
        domain_ids = set()
        companies = self.session.scalars(select(Company).where(Company.workspace_id == self.workspace_id))
        for company in companies:
            if company_name_key(company.name) == name_key:
                matches.append(company)
            if domain_key(company.domain) == domain:
                domain_ids.add(company.id)
        if len(matches) > 1:
            raise CompanyConflict("Multiple companies match the normalized name in this workspace.")
        company = matches[0] if matches else None
        if domain_ids - ({company.id} if company is not None else set()):
            raise CompanyConflict("The domain belongs to a different company in this workspace.")
        return company

    def get_company(self, company_id: UUID) -> Company | None:
        return self.session.scalar(
            select(Company).where(Company.id == company_id, Company.workspace_id == self.workspace_id)
        )

    def get_company_record(self, model: type[CompanyRecord], record_id: UUID) -> CompanyRecord | None:
        if model not in (CompanyResearch, ProductActivity, Evaluation, TierChange):
            raise ValueError("Unsupported company record.")
        return self.session.scalar(
            select(model).join(Company, Company.id == model.company_id)
            .where(model.id == record_id, Company.workspace_id == self.workspace_id)
        )

    def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        return self.session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.workspace_id == self.workspace_id
            )
        )

    def get_message(self, message_id: int) -> Message | None:
        return self.session.scalar(
            select(Message).join(Conversation, Conversation.id == Message.conversation_id)
            .where(Message.id == message_id, Conversation.workspace_id == self.workspace_id)
        )

    def list_conversations(self, limit: int) -> list[Conversation]:
        return list(self.session.scalars(
            select(Conversation).where(Conversation.workspace_id == self.workspace_id)
            .order_by(Conversation.created_at.desc(), Conversation.id).limit(limit)
        ))

    def conversation_messages(self, conversation_id: UUID, limit: int, before_id: int | None = None) -> list[Message]:
        query = (
            select(Message).join(Conversation, Conversation.id == Message.conversation_id)
            .where(Message.conversation_id == conversation_id, Conversation.workspace_id == self.workspace_id)
            .order_by(Message.id.desc()).limit(limit)
        )
        if before_id is not None:
            query = query.where(Message.id < before_id)
        return list(reversed(list(self.session.scalars(query))))

    def conversation_message_by_metadata(self, conversation_id: UUID, key: str, value: str) -> Message | None:
        return self.session.scalar(
            select(Message).join(Conversation, Conversation.id == Message.conversation_id)
            .where(Message.conversation_id == conversation_id, Conversation.workspace_id == self.workspace_id,
                   Message.role == "assistant", Message.message_metadata[key].as_string() == value)
            .order_by(Message.id.desc()).limit(1)
        )

    def company_list(self, limit: int, tier: int | None = None, customer: bool | None = None,
                     pending: bool | None = None, name: str | None = None) -> list[Company]:
        query = select(Company).where(Company.workspace_id == self.workspace_id)
        if tier is not None:
            query = query.where(Company.current_tier == tier)
        if customer is not None:
            query = query.where(Company.is_customer == customer)
        if pending is not None:
            has_pending = select(Evaluation.id).where(
                Evaluation.company_id == Company.id, Evaluation.decision_status == "pending",
            ).exists()
            query = query.where(has_pending if pending else ~has_pending)
        if name is not None:
            normalized = func.lower(func.btrim(func.regexp_replace(Company.name, r"\s+", " ", "g")))
            query = query.where(normalized == " ".join(name.split()).lower())
        return list(self.session.scalars(query.order_by(Company.name, Company.id).limit(limit)))

    def company_records(self, model: type[CompanyRecord], company_id: UUID, limit: int,
                        record_id: UUID | None = None) -> list[CompanyRecord]:
        if model not in (CompanyResearch, ProductActivity, Evaluation, TierChange):
            raise ValueError("Unsupported company record.")
        query = (
            select(model).join(Company, Company.id == model.company_id)
            .where(Company.id == company_id, Company.workspace_id == self.workspace_id)
            .order_by(model.created_at.desc(), model.id.desc()).limit(limit)
        )
        if record_id is not None:
            query = query.where(model.id == record_id)
        return list(self.session.scalars(query))

    def evaluations_by_id(self, ids: set[UUID]) -> dict[UUID, Evaluation]:
        if not ids:
            return {}
        return {item.id: item for item in self.session.scalars(
            select(Evaluation).join(Company, Company.id == Evaluation.company_id)
            .where(Company.workspace_id == self.workspace_id, Evaluation.id.in_(ids))
        )}

    def lead_history(self, company: Company) -> tuple[int, int, int]:
        """Return original tier and counts of append-only activity and tier records."""
        scope = (Company.id == company.id, Company.workspace_id == self.workspace_id)
        changes = select(TierChange).join(Company, Company.id == TierChange.company_id).where(*scope)
        first_tier = self.session.scalar(
            changes.with_only_columns(TierChange.previous_tier).order_by(TierChange.created_at, TierChange.id).limit(1)
        )
        activity_count = self.session.scalar(
            select(func.count(ProductActivity.id)).join(Company, Company.id == ProductActivity.company_id).where(*scope)
        )
        change_count = self.session.scalar(select(func.count()).select_from(changes.subquery()))
        return first_tier if first_tier is not None else company.current_tier, activity_count, change_count

    def customer_snapshot_state(self, snapshot: ProductActivity) -> tuple[bool, bool, bool]:
        """Check for superseding activity, later tier history, and a completed evaluation."""
        scope = (Company.id == snapshot.company_id, Company.workspace_id == self.workspace_id)
        newer_activity = (
            select(ProductActivity.id).join(Company, Company.id == ProductActivity.company_id)
            .where(*scope, ProductActivity.id != snapshot.id, ProductActivity.created_at >= snapshot.created_at)
        )
        later_tier_change = (
            select(TierChange.id).join(Company, Company.id == TierChange.company_id)
            .where(*scope, TierChange.created_at > snapshot.created_at)
        )
        completed_evaluation = (
            select(Evaluation.id).join(Company, Company.id == Evaluation.company_id)
            .where(*scope, Evaluation.evaluation_type == "re_evaluation",
                   Evaluation.predictions["activity_snapshot_id"].as_string() == str(snapshot.id))
        )
        return tuple(self.session.execute(select(
            newer_activity.exists(), later_tier_change.exists(), completed_evaluation.exists(),
        )).one())

    def flush_company_changes(self) -> None:
        try:
            self.session.flush()
        except IntegrityError as error:
            constraint = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            if constraint == "companies_workspace_id_domain_key":
                raise CompanyConflict("The domain belongs to a different company in this workspace.") from None
            raise
