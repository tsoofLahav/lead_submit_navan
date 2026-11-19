from flask import Flask
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import sessionmaker


def configure_database(app: Flask, database_url: str | None) -> None:
    if not database_url:
        raise RuntimeError("DATABASE_URL is required.")

    try:
        url = make_url(database_url)
    except (ArgumentError, ValueError):
        raise RuntimeError("DATABASE_URL must be a valid PostgreSQL URL.") from None

    if url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"} or not url.database:
        raise RuntimeError("DATABASE_URL must be a PostgreSQL URL that names a database.")

    # Require TLS by default while allowing explicit libpq options in the URL.
    if "sslmode" not in url.query:
        url = url.update_query_dict({"sslmode": "require"})

    engine = create_engine(
        url.set(drivername="postgresql+psycopg"),
        pool_pre_ping=True,
        pool_timeout=5,
        connect_args={"connect_timeout": 5},
        hide_parameters=True,
    )
    app.extensions["db_engine"] = engine
    app.extensions["db_sessions"] = sessionmaker(engine)
