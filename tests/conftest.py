import os

import pytest
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.orm import sessionmaker

from app.models import Base, Workspace


@pytest.fixture(scope="module")
def engine():
    raw_url = os.environ.get("TEST_DATABASE_URL")
    if not raw_url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated local PostgreSQL test database.")
    url = make_url(raw_url)
    host = url.query.get("host", url.host)
    if (
        url.drivername not in {"postgresql", "postgresql+psycopg"}
        or url.database != "navan_stage2_test"
        or not isinstance(host, str)
        or (host not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("/"))
        or any(option in url.query for option in ("hostaddr", "service", "dbname"))
    ):
        pytest.fail("TEST_DATABASE_URL must explicitly target the local navan_stage2_test database.")
    engine = create_engine(url.set(drivername="postgresql+psycopg"), hide_parameters=True)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT current_database()")) == "navan_stage2_test"
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def sessions(engine):
    with engine.connect() as connection, connection.begin() as transaction:
        yield sessionmaker(connection, join_transaction_mode="create_savepoint")
        transaction.rollback()


@pytest.fixture
def workspace_ids(sessions):
    with sessions.begin() as session:
        first, second = Workspace(), Workspace()
        session.add_all((first, second))
        session.flush()
        return first.id, second.id
