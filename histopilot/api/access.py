"""Agent access: who a request acts as, scoped tokens, and each project's AI-exposure level.

Creating and revoking tokens, changing the exposure level and deciding agents' requests are
admin routes, reachable with the service's session token only; see histopilot/api/security.py.

A person approves a parked request in three steps: claim it (pending → approving, once, and
only before it expires), replay it with the session, then record how the replay went:
approved, failed (the service refused it; nothing changed) or unknown (no answer came, so it
may have run). Declining needs no claim.
"""

import threading
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from sqlalchemy import select

from histopilot.api.route_classes import CLASSES
from histopilot.api.scopes import REQUEST_KIND
from histopilot.application.access_tokens import AccessTokens
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.schemas.access import CreateAccessToken, ResolveAgentRequest, SetExposure
from histopilot.storage.database import Database, Record
from histopilot.storage.project_lock import StorageError

# How a claimed request's replay ended.
REPLAYED = ("approved", "failed", "unknown")


def grant(request: Request) -> dict:
    """What the request's credential allows; the session token allows everything."""
    return getattr(request.state, "access", None) or {"kind": "session"}


def token_project(request: Request) -> str | None:
    """The one project a scoped token reaches; None for the session token."""
    current = grant(request)
    return current["token"]["projectId"] if current["kind"] == "token" else None


def _audited(request: Request, project: str) -> None:
    """Name the project an admin request outside its routes acted on, for its audit log."""
    request.state.audit_project = project


def _stamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require(entry: dict, state: str) -> None:
    if entry["state"] == state:
        return
    if state == "approving":
        raise StorageError(
            f"This agent request is {entry['state']}; claim it before replaying it.",
            "AGENT_REQUEST_NOT_CLAIMED",
            409,
        )
    raise StorageError(
        f"This agent request is already {entry['state']}.", "AGENT_REQUEST_SETTLED", 409
    )


def _presented(entry: dict) -> dict:
    expired = entry["state"] == "pending" and datetime.fromisoformat(
        entry["expiresAt"]
    ) <= datetime.now(UTC)
    return {**entry, "state": "expired" if expired else entry["state"]}


def access_router(
    projects: ProjectWorkspace, tokens: AccessTokens, database: Database
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    # Claims and decisions read and write one request at a time, so each settles once.
    decisions = threading.Lock()

    def requests_of(request: Request) -> list[dict]:
        with database.sessions.begin() as session:
            rows = [
                dict(row.payload)
                for row in session.scalars(select(Record).where(Record.kind == REQUEST_KIND))
            ]
        own = token_project(request)
        if own is not None:
            rows = [row for row in rows if row["projectId"] == own]
        return sorted(
            (_presented(row) for row in rows), key=lambda row: row["createdAt"], reverse=True
        )

    @router.get("/agent-requests")
    def agent_requests(request: Request, project: str | None = None, state: str | None = None):
        """Changes agents asked for, waiting for a person or already decided."""
        rows = requests_of(request)
        rows = [
            row
            for row in rows
            if (project is None or row["projectId"] == project)
            and (state is None or row["state"] == state)
        ]
        return {"requests": rows}

    @router.get("/agent-requests/{request_id}")
    def agent_request(request: Request, request_id: str):
        for row in requests_of(request):
            if row["id"] == request_id:
                return row
        raise StorageError("This agent request does not exist.", "AGENT_REQUEST_NOT_FOUND", 404)

    def settle(request: Request, request_id: str, change) -> dict:
        """Apply ``change(entry)`` to one request under the decisions lock."""
        with decisions, database.sessions.begin() as session:
            record = session.get(Record, (REQUEST_KIND, request_id))
            if record is None:
                raise StorageError(
                    "This agent request does not exist.", "AGENT_REQUEST_NOT_FOUND", 404
                )
            entry = _presented(dict(record.payload))
            _audited(request, entry["projectId"])
            entry = change(entry)
            record.payload = entry
        return entry

    @router.post("/agent-requests/{request_id}/claim")
    def claim_agent_request(request: Request, request_id: str):
        """Take a pending request for one replay. A decided or expired request, or one
        another person already claimed, is refused before anything runs."""

        def claim(entry: dict) -> dict:
            _require(entry, "pending")
            return {**entry, "state": "approving", "claimedAt": _stamp()}

        return settle(request, request_id, claim)

    @router.post("/agent-requests/{request_id}/resolve")
    def resolve_agent_request(request: Request, request_id: str, payload: ResolveAgentRequest):
        """Record a decision: declined for a pending request, or how the replay of a claimed
        one ended."""

        def resolve(entry: dict) -> dict:
            _require(entry, "approving" if payload.outcome in REPLAYED else "pending")
            return {
                **entry,
                "state": payload.outcome,
                "resolvedAt": _stamp(),
                "status": payload.status,
            }

        return settle(request, request_id, resolve)

    @router.get("/access")
    def access(request: Request):
        current = grant(request)
        if current["kind"] == "session":
            return {"kind": "session", "scopes": CLASSES, "projectId": None, "exposure": None}
        token = current["token"]
        return {
            "kind": "token",
            "tokenId": token["id"],
            "name": token["name"],
            "projectId": token["projectId"],
            "projectName": projects.name(token["projectId"]),
            "scopes": token["scopes"],
            "exposure": projects.exposure(token["projectId"]),
            "expiresAt": token["expiresAt"],
        }

    @router.get("/tokens")
    def list_tokens(project: str | None = None):
        return {"tokens": tokens.list(project)}

    @router.post("/tokens", status_code=201)
    def create_token(request: Request, payload: CreateAccessToken):
        _audited(request, payload.projectId)
        if projects.exposure(payload.projectId) == "none":
            raise StorageError(
                "Agents see nothing of this project: its AI-exposure level is none. A person "
                "raises it with `histopilot project exposure PROJECT --set metadata` first.",
                "TOKEN_EXPOSURE_REQUIRED",
                409,
            )
        return tokens.create(
            project_id=payload.projectId,
            scopes=payload.scopes,
            name=payload.name,
            days=payload.days,
        )

    @router.post("/tokens/{token_id}/revoke")
    def revoke_token(request: Request, token_id: str):
        revoked = tokens.revoke(token_id)
        _audited(request, revoked["projectId"])
        return revoked

    @router.get("/projects/{identity}/ai-exposure")
    def exposure(identity: str):
        return {"projectId": identity, "level": projects.exposure(identity)}

    @router.put("/projects/{identity}/ai-exposure")
    def set_exposure(identity: str, payload: SetExposure):
        return projects.set_exposure(identity, payload.level, payload.expectedLevel)

    return router
