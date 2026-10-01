"""Scoped access tokens for AI agents: one project, some route classes, hashed at rest.

A token reads `hpt_<id>_<secret>`. The workspace database keeps its SHA-256, never the
secret, so a stolen database grants nothing. Scopes nest: preview includes read, and
commit includes both. Admin routes are never reachable with a token; they need the
service's own session token. See docs/agents.md.
"""

import hashlib
import secrets
import threading
from datetime import UTC, datetime, timedelta
from secrets import compare_digest

from sqlalchemy import select

from histopilot.api.route_classes import (
    DEFAULT_TOKEN_SCOPES,
    MAX_TOKEN_DAYS,
    TOKEN_DAYS,
    TOKEN_PREFIX,
    TOKEN_SCOPES,
)
from histopilot.storage.database import Database, Record
from histopilot.storage.project_lock import StorageError

KIND = "access-token"
# Last-use times are written at most this often, so reads never become writes.
TOUCH_SECONDS = 60


def normalize_scopes(scopes) -> list[str]:
    requested = {
        str(scope).strip() for scope in scopes or DEFAULT_TOKEN_SCOPES if str(scope).strip()
    }
    unknown = requested - set(TOKEN_SCOPES)
    if unknown:
        raise StorageError(
            f"Unknown scope {', '.join(sorted(unknown))}; choose from {', '.join(TOKEN_SCOPES)}.",
            "TOKEN_SCOPE_INVALID",
            422,
        )
    highest = max((TOKEN_SCOPES.index(scope) for scope in requested), default=0)
    return list(TOKEN_SCOPES[: highest + 1])


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


class AccessTokens:
    def __init__(self, database: Database, *, clock=lambda: datetime.now(UTC)):
        self.database = database
        self.clock = clock
        self._touched: dict[str, datetime] = {}
        self._lock = threading.Lock()

    @staticmethod
    def public(payload: dict, now: datetime | None = None) -> dict:
        state = "active"
        if payload.get("revokedAt"):
            state = "revoked"
        elif now is not None and datetime.fromisoformat(payload["expiresAt"]) <= now:
            state = "expired"
        shown = {key: value for key, value in payload.items() if key != "secretHash"}
        return {**shown, "state": state}

    def create(self, *, project_id: str, scopes=None, name: str = "", days: int = TOKEN_DAYS):
        """A new token; its secret is returned once and never stored."""
        if not isinstance(days, int) or not 1 <= days <= MAX_TOKEN_DAYS:
            raise StorageError(
                f"A token lasts 1 to {MAX_TOKEN_DAYS} days.", "TOKEN_LIFETIME_INVALID", 422
            )
        name = " ".join(str(name or "").split())[:80]
        identity = secrets.token_hex(8)
        secret = secrets.token_urlsafe(32)
        now = self.clock()
        payload = {
            "id": identity,
            "name": name,
            "projectId": project_id,
            "scopes": normalize_scopes(scopes),
            "createdAt": _iso(now),
            "expiresAt": _iso(now + timedelta(days=days)),
            "revokedAt": None,
            "lastUsedAt": None,
            "secretHash": _digest(secret),
        }
        with self.database.sessions.begin() as session:
            session.add(Record(kind=KIND, id=identity, payload=payload))
        return {**self.public(payload, now), "token": f"{TOKEN_PREFIX}{identity}_{secret}"}

    def list(self, project_id: str | None = None) -> list[dict]:
        now = self.clock()
        with self.database.sessions.begin() as session:
            rows = [
                dict(record.payload)
                for record in session.scalars(select(Record).where(Record.kind == KIND))
            ]
        rows = [row for row in rows if project_id is None or row["projectId"] == project_id]
        rows.sort(key=lambda row: row["createdAt"], reverse=True)
        return [self.public(row, now) for row in rows]

    def revoke(self, identity: str) -> dict:
        with self.database.sessions.begin() as session:
            record = session.get(Record, (KIND, identity))
            if record is None:
                raise StorageError("This access token does not exist.", "TOKEN_NOT_FOUND", 404)
            payload = dict(record.payload)
            if not payload.get("revokedAt"):
                payload["revokedAt"] = _iso(self.clock())
                record.payload = payload
        return self.public(payload, self.clock())

    def verify(self, presented: str) -> dict | None:
        """The token's public record when ``presented`` is a live token, else None."""
        if not isinstance(presented, str) or not presented.startswith(TOKEN_PREFIX):
            return None
        parts = presented.split("_", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            return None
        identity, secret = parts[1], parts[2]
        with self.database.sessions.begin() as session:
            record = session.get(Record, (KIND, identity))
            payload = dict(record.payload) if record is not None else None
        if payload is None or not compare_digest(payload["secretHash"], _digest(secret)):
            return None
        now = self.clock()
        public = self.public(payload, now)
        if public["state"] != "active":
            return None
        self._touch(identity, now)
        return public

    def _touch(self, identity: str, now: datetime) -> None:
        with self._lock:
            last = self._touched.get(identity)
            if last is not None and (now - last).total_seconds() < TOUCH_SECONDS:
                return
            self._touched[identity] = now
        with self.database.sessions.begin() as session:
            record = session.get(Record, (KIND, identity))
            if record is not None:
                record.payload = {**record.payload, "lastUsedAt": _iso(now)}
