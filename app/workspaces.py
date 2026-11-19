from uuid import UUID

from flask import current_app, session as browser_session

from .models import Workspace

COOKIE_WORKSPACE_KEY = "workspace_id"


def resolve_workspace() -> UUID:
    """Resolve the current request's signed cookie and persist a workspace when needed."""
    if not current_app.secret_key:
        raise RuntimeError("SECRET_KEY is required for workspace resolution.")

    raw_id = browser_session.get(COOKIE_WORKSPACE_KEY)
    try:
        workspace_id = UUID(raw_id) if isinstance(raw_id, str) else None
    except ValueError:
        workspace_id = None

    sessions = current_app.extensions["db_sessions"]
    with sessions.begin() as db_session:
        if workspace_id is None or db_session.get(Workspace, workspace_id) is None:
            workspace = Workspace()
            db_session.add(workspace)
            db_session.flush()
            workspace_id = workspace.id

    if raw_id != str(workspace_id):
        browser_session[COOKIE_WORKSPACE_KEY] = str(workspace_id)
    return workspace_id
