"""Non-blocking PostgreSQL locks for workspace-scoped company writes."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ..business.rules.validation import company_name_key


class CompanyBusy(RuntimeError):
    def __init__(self):
        super().__init__("An operation for this company is already running.")


def acquire_company_lock(session: Session, workspace_id: UUID, name: str) -> None:
    """Acquire a lock held until the caller's transaction ends."""
    key = f"company-write:{workspace_id}:{company_name_key(name)}"
    if not session.scalar(select(func.pg_try_advisory_xact_lock(func.hashtextextended(key, 0)))):
        raise CompanyBusy()


@contextmanager
def company_lock(sessions: sessionmaker[Session], workspace_id: UUID, name: str) -> Iterator[None]:
    """Hold a lock across a workflow's independently committed writes."""
    with sessions() as lock_session:
        acquire_company_lock(lock_session, workspace_id, name)
        try:
            yield
        finally:
            lock_session.commit()
