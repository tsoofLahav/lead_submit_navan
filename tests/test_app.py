import pytest
from sqlalchemy import URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import configure_mappers

from app import create_app
from app.data.models import Base
from app.data.reference_data import BusinessPolicyNotFound, PromptNotFound, get_business_policy, get_prompt


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    application = create_app()
    yield application
    application.extensions["db_engine"].dispose()


def test_create_app(app):
    configure_mappers()
    assert len(Base.registry.mappers) == 11
    assert app.extensions["db_engine"].url.drivername == "postgresql+psycopg"
    assert app.extensions["db_engine"].url.query["sslmode"] == "require"


@pytest.mark.parametrize("url", [None, "", "invalid", "sqlite:///test", "postgresql://localhost/"])
def test_invalid_configuration(monkeypatch, url):
    monkeypatch.setattr("app.load_dotenv", lambda *_: None)
    if url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", url)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        create_app()


def test_invalid_url_does_not_expose_secret(monkeypatch):
    url = URL.create("postgresql", username="test", password="fake-secret", host="localhost")
    monkeypatch.setenv("DATABASE_URL", url.render_as_string(hide_password=False) + ":invalid-port/test")
    with pytest.raises(RuntimeError) as error:
        create_app()
    assert "fake-secret" not in str(error.value)
    assert error.value.__suppress_context__


def test_unhealthy_response(app, monkeypatch, caplog):
    def unavailable():
        raise OperationalError(None, None, RuntimeError("private connection detail"))

    monkeypatch.setitem(app.extensions, "db_sessions", unavailable)
    response = app.test_client().get("/health")
    assert response.status_code == 503
    assert response.json == {"status": "unavailable"}
    assert "private connection detail" not in caplog.text


class EmptySession:
    from contextlib import nullcontext

    no_autoflush = nullcontext()

    def get(self, model, key):
        return None


def test_missing_reference_data():
    with pytest.raises(BusinessPolicyNotFound):
        get_business_policy(EmptySession())
    with pytest.raises(PromptNotFound):
        get_prompt(EmptySession(), "missing")
