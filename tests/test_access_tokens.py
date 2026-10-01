"""Scoped access tokens: hashed at rest, nested scopes, expiry and revocation."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from histopilot.application.access_tokens import KIND, AccessTokens, normalize_scopes
from histopilot.storage.database import Database, Record
from histopilot.storage.project_lock import StorageError


@pytest.fixture
def store(tmp_path):
    database = Database(tmp_path / "workspace")
    database.initialize()
    now = [datetime(2026, 9, 30, tzinfo=UTC)]
    tokens = AccessTokens(database, clock=lambda: now[0])
    yield tokens, database, now
    database.close()


def test_the_database_keeps_only_a_hash_and_listing_never_shows_a_secret(store):
    tokens, database, _ = store
    created = tokens.create(project_id="project-a", name="claude desktop")
    secret = created["token"]
    assert secret.startswith(f"hpt_{created['id']}_") and created["scopes"] == ["read", "preview"]
    with database.sessions.begin() as session:
        stored = [row.payload for row in session.scalars(select(Record).where(Record.kind == KIND))]
    assert secret.split("_", 2)[2] not in repr(stored)
    listed = tokens.list("project-a")
    assert [row["id"] for row in listed] == [created["id"]]
    assert "token" not in listed[0] and "secretHash" not in listed[0]
    assert tokens.list("project-b") == []


def test_only_the_exact_live_token_verifies(store):
    tokens, _, now = store
    created = tokens.create(project_id="project-a", days=7)
    assert tokens.verify(created["token"])["projectId"] == "project-a"
    for wrong in ("", "hpt_", created["token"] + "x", created["token"].replace("hpt_", "hpx_")):
        assert tokens.verify(wrong) is None
    now[0] += timedelta(days=7)
    assert tokens.verify(created["token"]) is None
    assert tokens.list()[0]["state"] == "expired"


def test_revocation_acts_on_the_next_request(store):
    tokens, _, _ = store
    created = tokens.create(project_id="project-a")
    assert tokens.verify(created["token"]) is not None
    assert tokens.revoke(created["id"])["state"] == "revoked"
    assert tokens.verify(created["token"]) is None
    with pytest.raises(StorageError) as raised:
        tokens.revoke("missing")
    assert raised.value.code == "TOKEN_NOT_FOUND"


def test_scopes_nest_and_lifetimes_are_bounded(store):
    tokens, _, _ = store
    assert normalize_scopes(["commit"]) == ["read", "preview", "commit"]
    assert normalize_scopes(["read"]) == ["read"]
    assert normalize_scopes(None) == ["read", "preview"]
    with pytest.raises(StorageError):
        normalize_scopes(["admin"])
    for days in (0, 91):
        with pytest.raises(StorageError):
            tokens.create(project_id="project-a", days=days)


def test_last_use_is_recorded_at_most_once_a_minute(store):
    tokens, _, now = store
    created = tokens.create(project_id="project-a")
    tokens.verify(created["token"])
    first = tokens.list()[0]["lastUsedAt"]
    now[0] += timedelta(seconds=30)
    tokens.verify(created["token"])
    assert tokens.list()[0]["lastUsedAt"] == first
    now[0] += timedelta(seconds=31)
    tokens.verify(created["token"])
    assert tokens.list()[0]["lastUsedAt"] != first
