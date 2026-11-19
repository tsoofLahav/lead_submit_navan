"""Guard the disposable local databases used by live smoke scripts."""

import os
from ipaddress import ip_address

from sqlalchemy import create_engine, make_url, text
from sqlalchemy.engine import Engine


def open_local_database(variable: str, database_name: str) -> Engine:
    raw = os.getenv(variable)
    if not raw:
        raise SystemExit(f"Set {variable} to a dedicated local {database_name} database.")
    url = make_url(raw)
    host = url.query.get("host", url.host)
    if (
        url.drivername not in {"postgresql", "postgresql+psycopg"}
        or url.database != database_name
        or not isinstance(host, str)
        or (host not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("/"))
        or set(url.query) - {"host", "port", "sslmode"}
    ):
        raise SystemExit(f"{variable} must target the local {database_name} database.")
    engine = create_engine(url.set(drivername="postgresql+psycopg"), hide_parameters=True)
    try:
        with engine.connect() as connection:
            database, server_address = connection.execute(text("SELECT current_database(), inet_server_addr()")).one()
            if database != database_name or (
                server_address is not None and not ip_address(server_address).is_loopback
            ):
                raise SystemExit("Unexpected local smoke database.")
    except (Exception, SystemExit):
        engine.dispose()
        raise
    return engine
