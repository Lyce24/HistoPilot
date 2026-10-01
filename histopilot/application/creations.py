"""Creations that an operation ID makes safe to retry.

A client that lost the answer to a create sends the same request with the same operation ID
again, and gets the record the first attempt made instead of a second one. The service's own
database remembers which record each operation created; the ID is optional, so older clients
create as before (docs/cli-contract.md#confirmation).
"""

from hashlib import sha256
from threading import RLock

from histopilot.storage.database import Record
from histopilot.storage.io import content_hash
from histopilot.storage.project_lock import StorageError

KIND = "creation"
# One create at a time per process, so a retry never races its first attempt.
_LOCK = RLock()


def _key(scope: str, operation_id: str) -> str:
    return sha256(f"{scope}\0{operation_id}".encode()).hexdigest()


def _replayed(database, scope: str, operation_id: str | None, request: dict) -> str | None:
    """The record an earlier request with this operation ID created, or None."""
    if not operation_id:
        return None
    with database.sessions() as session:
        found = session.get(Record, (KIND, _key(scope, operation_id)))
        payload = None if found is None else dict(found.payload)
    if payload is None:
        return None
    if payload["requestHash"] != content_hash(request):
        raise StorageError(
            "This operation ID belongs to a different request.", "OPERATION_CONFLICT", 409
        )
    return payload["recordId"]


def _remember(database, scope: str, operation_id: str | None, request: dict, record: str) -> None:
    if not operation_id:
        return
    with database.sessions.begin() as session:
        session.merge(
            Record(
                kind=KIND,
                id=_key(scope, operation_id),
                payload={"scope": scope, "recordId": record, "requestHash": content_hash(request)},
            )
        )


def once(database, scope: str, operation_id: str | None, request: dict, create, existing):
    """``create()`` once per operation ID; a retry of the same request gets
    ``existing(record ID)`` instead. The created record must carry its `id`."""
    with _LOCK:
        found = _replayed(database, scope, operation_id, request)
        if found is not None:
            return existing(found)
        created = create()
        _remember(database, scope, operation_id, request, created["id"])
        return created
