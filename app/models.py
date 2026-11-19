from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    metadata = MetaData(schema="public", naming_convention={"pk": "%(table_name)s_pkey"})


class UUIDRecord:
    id: Mapped[UUID] = mapped_column(
        PG_UUID, primary_key=True, server_default=text("gen_random_uuid()")
    )


class CreatedRecord:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class Workspace(UUIDRecord, CreatedRecord, Base):
    __tablename__ = "workspaces"


class DemoCompany(UUIDRecord, Base):
    """Global templates read during workspace creation."""

    __tablename__ = "demo_companies"
    __table_args__ = (
        UniqueConstraint("domain", name="demo_companies_domain_key"),
        CheckConstraint("current_tier >= 1 AND current_tier <= 3", name="demo_companies_current_tier_check"),
    )

    name: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(Text)
    submitted_annual_gbv: Mapped[Decimal] = mapped_column(Numeric(15, 2))
    employee_count: Mapped[int | None] = mapped_column(Integer)
    current_tier: Mapped[int] = mapped_column(Integer)


class Company(UUIDRecord, CreatedRecord, Base):
    __tablename__ = "companies"
    __table_args__ = (
        UniqueConstraint("workspace_id", "domain", name="companies_workspace_id_domain_key"),
        CheckConstraint("current_tier = ANY (ARRAY[1, 2, 3])", name="companies_current_tier_check"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("workspaces.id", name="companies_workspace_id_fkey", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(Text)
    submitted_annual_gbv: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    employee_count: Mapped[int | None] = mapped_column(Integer)
    current_tier: Mapped[int] = mapped_column(SmallInteger)
    is_customer: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))

    workspace: Mapped[Workspace] = relationship()


class CompanyResearch(UUIDRecord, CreatedRecord, Base):
    __tablename__ = "company_research"
    __table_args__ = (
        UniqueConstraint("company_id", "id", name="company_research_company_id_id_key"),
        Index("idx_research_company", "company_id"),
    )

    company_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("companies.id", name="company_research_company_id_fkey", ondelete="CASCADE")
    )
    signals: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=False)

    company: Mapped[Company] = relationship()


class ProductActivity(UUIDRecord, CreatedRecord, Base):
    """An aggregate customer activity snapshot."""

    __tablename__ = "product_activity"
    __table_args__ = (
        Index("idx_product_activity_company", "company_id", text("created_at DESC")),
        CheckConstraint("months_since_signup > 0", name="product_activity_months_since_signup_check"),
        CheckConstraint("total_gbv >= 0::numeric", name="product_activity_total_gbv_check"),
        CheckConstraint("avg_monthly_bookings >= 0::numeric", name="product_activity_avg_monthly_bookings_check"),
        CheckConstraint("employee_count > 0", name="product_activity_employee_count_check"),
        CheckConstraint("avg_monthly_active_users >= 0::numeric", name="product_activity_avg_monthly_active_users_check"),
        CheckConstraint(
            "avg_monthly_support_requests >= 0::numeric",
            name="product_activity_avg_monthly_support_requests_check",
        ),
        *(
            CheckConstraint(
                f"{column} = ANY (ARRAY['increasing'::text, 'stable'::text, 'decreasing'::text, 'unknown'::text])",
                name=f"product_activity_{column}_check",
            )
            for column in ("travel_spend_trend", "company_size_trend", "product_adoption_trend")
        ),
    )

    company_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("companies.id", name="product_activity_company_id_fkey", ondelete="CASCADE")
    )
    months_since_signup: Mapped[int] = mapped_column(Integer)
    total_gbv: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    avg_monthly_bookings: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    employee_count: Mapped[int | None] = mapped_column(Integer)
    avg_monthly_active_users: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    avg_monthly_support_requests: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    travel_spend_trend: Mapped[str] = mapped_column(Text, server_default=text("'unknown'::text"))
    company_size_trend: Mapped[str] = mapped_column(Text, server_default=text("'unknown'::text"))
    product_adoption_trend: Mapped[str] = mapped_column(Text, server_default=text("'unknown'::text"))

    company: Mapped[Company] = relationship()


class Evaluation(UUIDRecord, CreatedRecord, Base):
    __tablename__ = "evaluations"
    __table_args__ = (
        UniqueConstraint("company_id", "id", name="evaluations_company_id_id_key"),
        ForeignKeyConstraint(
            ["company_id", "research_id"],
            ["company_research.company_id", "company_research.id"],
            name="evaluations_company_id_research_id_fkey",
        ),
        Index("idx_evaluations_company", "company_id"),
        CheckConstraint(
            "evaluation_type = ANY (ARRAY['lead_submit'::text, 're_evaluation'::text])",
            name="evaluations_evaluation_type_check",
        ),
        CheckConstraint("current_tier = ANY (ARRAY[1, 2, 3])", name="evaluations_current_tier_check"),
        CheckConstraint("recommended_tier = ANY (ARRAY[1, 2, 3])", name="evaluations_recommended_tier_check"),
        CheckConstraint(
            "confidence = ANY (ARRAY['high'::text, 'medium'::text, 'low'::text, 'insufficient'::text])",
            name="evaluations_confidence_check",
        ),
        CheckConstraint(
            "decision_status = ANY (ARRAY['pending'::text, 'accepted'::text, 'rejected'::text,"
            " 'superseded'::text, 'not_applicable'::text])",
            name="evaluations_decision_status_check",
        ),
    )

    company_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("companies.id", name="evaluations_company_id_fkey", ondelete="CASCADE")
    )
    research_id: Mapped[UUID | None] = mapped_column(PG_UUID)
    evaluation_type: Mapped[str] = mapped_column(Text)
    predictions: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=False)
    current_tier: Mapped[int] = mapped_column(SmallInteger)
    recommended_tier: Mapped[int | None] = mapped_column(SmallInteger)
    confidence: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    follow_up_actions: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)
    information_gaps: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)
    decision_status: Mapped[str] = mapped_column(Text, server_default=text("'pending'::text"))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    company: Mapped[Company] = relationship()
    research: Mapped[CompanyResearch | None] = relationship(viewonly=True)


class TierChange(UUIDRecord, CreatedRecord, Base):
    __tablename__ = "tier_changes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id", "evaluation_id"],
            ["evaluations.company_id", "evaluations.id"],
            name="tier_changes_company_id_evaluation_id_fkey",
        ),
        Index("idx_tier_changes_company", "company_id"),
        CheckConstraint("previous_tier = ANY (ARRAY[1, 2, 3])", name="tier_changes_previous_tier_check"),
        CheckConstraint("new_tier = ANY (ARRAY[1, 2, 3])", name="tier_changes_new_tier_check"),
        CheckConstraint("previous_tier <> new_tier", name="tier_changes_check"),
    )

    company_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("companies.id", name="tier_changes_company_id_fkey", ondelete="CASCADE")
    )
    evaluation_id: Mapped[UUID | None] = mapped_column(PG_UUID)
    previous_tier: Mapped[int] = mapped_column(SmallInteger)
    new_tier: Mapped[int] = mapped_column(SmallInteger)

    company: Mapped[Company] = relationship()
    evaluation: Mapped[Evaluation | None] = relationship(viewonly=True)


class BusinessPolicy(Base):
    __tablename__ = "business_policy"
    __table_args__ = (CheckConstraint("id = 1", name="business_policy_id_check"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, server_default=text("1"), autoincrement=False)
    content: Mapped[str] = mapped_column(Text)


class Prompt(Base):
    __tablename__ = "prompts"

    prompt_key: Mapped[str] = mapped_column(Text, primary_key=True)
    content: Mapped[str] = mapped_column(Text)


class Conversation(UUIDRecord, CreatedRecord, Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_conversations_workspace", "workspace_id"),)

    workspace_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("workspaces.id", name="conversations_workspace_id_fkey", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(Text, server_default=text("'New conversation'::text"))

    workspace: Mapped[Workspace] = relationship()


class Message(CreatedRecord, Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_conversation", "conversation_id"),
        CheckConstraint("role = ANY (ARRAY['user'::text, 'assistant'::text])", name="messages_role_check"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("conversations.id", name="messages_conversation_id_fkey", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    message_metadata: Mapped[Any] = mapped_column("metadata", JSONB(none_as_null=True), nullable=True)

    conversation: Mapped[Conversation] = relationship()
