import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .data.database import configure_database
from .orchestrator.routes import configure_http


def create_app() -> Flask:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    app = Flask(__name__)
    configure_database(app, os.environ.get("DATABASE_URL"))
    secure_cookie = os.environ.get("SESSION_COOKIE_SECURE")
    if secure_cookie is not None and secure_cookie.lower() not in {"true", "false"}:
        raise RuntimeError("SESSION_COOKIE_SECURE must be true or false.")
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY"),
        SESSION_COOKIE_NAME="navan_workspace",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=(
            secure_cookie.lower() == "true" if secure_cookie is not None
            else os.environ.get("RENDER", "").lower() == "true"
        ),
    )

    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        try:
            with app.extensions["db_sessions"]() as session:
                session.execute(text("SELECT 1")).scalar_one()
        except SQLAlchemyError:
            app.logger.warning("Database health check failed.")
            return {"status": "unavailable"}, 503
        return {"status": "ok"}, 200

    configure_http(app)
    return app
