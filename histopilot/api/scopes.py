"""What a scoped agent token may do: its route classes, its one project, its exposure level.

The session token is unaffected. A scoped token reaches read and preview routes of its own
project, within its scopes; admin routes and routes this table does not know are refused.
Its commits are parked for a person to approve (docs/cli-contract.md, docs/agents.md). On a
project shared at the `metadata` level, JSON answers are redacted and files are refused.
Every scoped request, and every commit or admin request from anyone, is audited.
"""

import json
import secrets
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from histopilot.api import audit
from histopilot.api.redaction import Redactor
from histopilot.api.responses import coded_response, refusal
from histopilot.api.route_classes import ADMIN, COMMIT, route_class, route_template
from histopilot.application.project_workspace import WorkspaceError
from histopilot.storage.database import Record
from histopilot.storage.filesystem import SCOPED_ROOTS, FilesystemError
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

PROJECTS = "/api/v1/projects/"
TASK_CENTER = "/api/v1/task-center/"
# The service's own full-access session; a token never trades itself up for it.
SESSION = "/api/v1/session"
# Machine-level routes a scoped token may use; their handlers narrow them to its project.
MACHINE = {
    ("GET", "/api/v1/access"),
    ("GET", "/api/v1/version"),
    ("GET", "/api/v1/templates"),
    ("GET", "/api/v1/projects"),
    ("GET", "/api/v1/agent-requests"),
    ("GET", "/api/v1/task-center/summary"),
    ("GET", "/api/v1/task-center/tasks"),
    ("GET", "/api/v1/task-center/history"),
    ("GET", "/api/v1/task-center/rollup"),
    ("GET", "/api/v1/task-center/owners"),
}
MACHINE_PREFIXES = (
    ("GET", "/api/v1/agent-requests/"),
    ("GET", "/api/v1/task-center/tasks/"),
    ("GET", "/api/v1/task-center/owners/"),
    ("POST", "/api/v1/task-center/tasks/"),
    ("POST", "/api/v1/task-center/owners/"),
)
# Routes that address one case, return its files, or read raw tables; refused on metadata
# projects, where only published datasets' identifiers are known to the redactor.
CASE_ROUTES = (
    "/slides/",
    "/slide-reviews",
    "/morphology/",
    "/oof/",
    "/artifacts/",
    "/cases/",
    "/inference/export",
    "/imports/",
    "/log",
)
REQUEST_KIND = "agent-request"
REQUEST_HOURS = 24
# The service's own coded errors, answered as its exception handlers answer them.
CODED = (StorageError, WorkspaceError, FilesystemError)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def missing_work(what: str) -> StorageError:
    """Another project's task or owner looks the same as a missing one to a scoped token."""
    if what == "owner":
        return StorageError("This task owner does not exist.", "TASK_OWNER_NOT_FOUND", 404)
    return StorageError("This task does not exist.", "TASK_NOT_FOUND", 404)


class ScopeGate:
    def __init__(self, projects, tokens, database, *, work_project=None):
        self.projects = projects
        self.tokens = tokens
        self.database = database
        # (what, key) -> the project that owns a Task Center task or owner, or None.
        self.work_project = work_project
        # project -> (its datasets, the redactor built from their identifiers)
        self._redactors: dict[str, tuple[tuple, Redactor]] = {}
        self._lock = threading.Lock()

    # The gate --------------------------------------------------------------------------

    async def __call__(self, request, presented: str, call_next):
        token = await run_in_threadpool(self.tokens.verify, presented)
        if token is None:
            return refusal(
                "This access token is unknown, expired or revoked.", "TOKEN_INVALID", 401
            )
        method, path = request.method, request.url.path
        kind = route_class(method, path)
        project = token["projectId"]
        document = folder = exposure = response = None
        parked = False
        try:
            response = self._refusal(method, path, kind, token)
            if response is None:
                document, folder = await run_in_threadpool(self.projects.locate, project)
                exposure = document.get("aiExposure") or "none"
                response = _exposure_refusal(exposure, path)
            if response is None and kind == COMMIT:
                await run_in_threadpool(self._own_work, path, project)
                response = await self._park(request, token)
                parked = response.status_code == 202
            elif response is None:
                response = await self._forward(
                    request, call_next, token, exposure, document, folder
                )
        except CODED as error:
            response = coded_response(error)
        finally:
            status = 500 if response is None else response.status_code
            await run_in_threadpool(
                self._audit, project, folder, token, method, path, kind, status, exposure, parked
            )
        return response

    async def _forward(self, request, call_next, token, exposure, document, folder):
        """Serve the request within the project's folders, redacted at the metadata level."""
        request.state.access = {"kind": "token", "token": token, "exposure": exposure}
        store = ScientificStore(folder, document["id"])
        roots = await run_in_threadpool(self._roots, document, folder, store)
        scope = SCOPED_ROOTS.set(roots)
        try:
            response = await call_next(request)
        finally:
            SCOPED_ROOTS.reset(scope)
        if exposure == "metadata":
            response = await self._redact(store, response)
        return response

    @staticmethod
    def _refusal(method: str, path: str, kind, token):
        if path == SESSION:
            return refusal(
                "An agent token cannot fetch the service's session; it reaches only its own "
                "project.",
                "SESSION_NOT_FOR_TOKENS",
            )
        if kind is None or kind == ADMIN or kind not in token["scopes"]:
            return refusal(
                f"This token's scopes ({', '.join(token['scopes'])}) do not reach this route.",
                "SCOPE_MISSING",
            )
        if path.startswith(PROJECTS):
            if path[len(PROJECTS) :].split("/", 1)[0] != token["projectId"]:
                return refusal(
                    f"This token reaches one project only: {token['projectId']}.",
                    "PROJECT_OUT_OF_SCOPE",
                )
            return None
        if (method, path) in MACHINE or any(
            method == prefix_method and path.startswith(prefix)
            for prefix_method, prefix in MACHINE_PREFIXES
        ):
            return None
        return refusal("Machine-wide routes need the service's own session.", "SCOPE_MISSING")

    def _own_work(self, path: str, project: str) -> None:
        """A parked Task Center change must name this project's own work: the person who
        approves it acts with the full session, which reaches every project's tasks."""
        key = _work_addressed(path)
        if key is None:
            return
        what = "task" if path.startswith(f"{TASK_CENTER}tasks/") else "owner"
        if self.work_project is None or self.work_project(what, key) != project:
            raise missing_work(what)

    def _roots(self, document: dict, folder: Path, store: ScientificStore) -> tuple[Path, ...]:
        """The project's own folder, its registered source folders, and the folders its
        frozen feature versions and bundles read, such as a pack attached from elsewhere."""
        folders = [folder] + [source.get("path") for source in document.get("sources") or []]
        folders += self._frozen_folders(store)
        return tuple(dict.fromkeys(Path(item).resolve() for item in folders if item))

    @staticmethod
    def _frozen_folders(store) -> list[str]:
        """Folders a person froze into the project; an agent's own drafts add none."""
        folders: list = []
        for item in store.list_configurations("feature"):
            spec = (item.get("manifest") or {}).get("spec") or {}
            folders += [spec.get("path"), spec.get("coordinatesPath")]
        for item in store.list_configurations("feature-bundle"):
            packs = (item.get("manifest") or {}).get("packs") or []
            folders += [pack.get("outputPath") for pack in packs if isinstance(pack, dict)]
        return [folder for folder in folders if isinstance(folder, str) and folder]

    # Parked commits --------------------------------------------------------------------

    async def _park(self, request, token) -> JSONResponse:
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            return refusal("The request body is not JSON.", "REQUEST_INVALID", 422)
        now = _now()
        entry = {
            "id": f"request-{uuid.uuid4().hex}",
            "projectId": token["projectId"],
            "tokenId": token["id"],
            "tokenName": token["name"],
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query,
            "body": body,
            "state": "pending",
            "createdAt": _iso(now),
            "expiresAt": _iso(now + timedelta(hours=REQUEST_HOURS)),
        }
        await run_in_threadpool(self._store, entry)
        return JSONResponse(
            {
                "code": "CONFIRMATION_PENDING",
                "detail": "A person must approve this change: "
                f"`histopilot confirm show {entry['id']}`, then `histopilot confirm approve "
                f"{entry['id']}`.",
                "requestId": entry["id"],
                "expiresAt": entry["expiresAt"],
            },
            status_code=202,
        )

    def _store(self, entry: dict) -> None:
        with self.database.sessions.begin() as session:
            session.add(Record(kind=REQUEST_KIND, id=entry["id"], payload=entry))

    # Redaction -------------------------------------------------------------------------

    async def _redact(self, store: ScientificStore, response):
        if "application/json" not in response.headers.get("content-type", ""):
            return refusal(
                "Files, images and logs of a project shared at the metadata level stay with "
                "the service.",
                "EXPOSURE_METADATA",
            )
        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            document = json.loads(body)
        except ValueError:
            return refusal("The answer could not be redacted.", "EXPOSURE_METADATA")
        redactor = await run_in_threadpool(self.redactor, store)
        redacted = redactor.document(document)
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        return JSONResponse(redacted, status_code=response.status_code, headers=headers)

    def redactor(self, store: ScientificStore) -> Redactor:
        """The project's redactor, rebuilt when its datasets change. Archived and trashed
        datasets count: their identifiers can still appear in names and records."""
        datasets = store.list_datasets(include_inactive=True)
        stamp = tuple(sorted(item["id"] for item in datasets))
        with self._lock:
            cached = self._redactors.get(store.project_id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        built = Redactor(self._key(store.project_id), _identifiers(store, datasets))
        with self._lock:
            self._redactors[store.project_id] = (stamp, built)
        return built

    def _key(self, project: str) -> bytes:
        with self.database.sessions.begin() as session:
            record = session.get(Record, ("redaction-key", project))
            if record is None:
                record = Record(
                    kind="redaction-key", id=project, payload={"key": secrets.token_hex(32)}
                )
                session.add(record)
            return bytes.fromhex(record.payload["key"])

    # Audit -----------------------------------------------------------------------------

    def _audit(self, project, folder, token, method, path, kind, status, exposure, parked) -> None:
        """One line in the token's project log; `exposure` is None when the request was
        refused before the level was read, and `parked` marks a commit filed for a person
        (a 202 that changed nothing)."""
        if folder is None:
            try:
                _, folder = self.projects.locate(project)
            except Exception:  # noqa: BLE001 - a missing project has no log to write
                return
        audit.append(
            folder,
            {
                "actor": {"kind": "agent", "tokenId": token["id"], "tokenName": token["name"]},
                "method": method,
                "route": route_template(method, path),
                # The project a request named, if any: evidence that a token stayed in its own.
                "projectAddressed": path[len(PROJECTS) :].split("/", 1)[0]
                if path.startswith(PROJECTS)
                else None,
                # Likewise the Task Center task or owner it named, which carries no project.
                "workAddressed": _work_addressed(path),
                "class": kind,
                "status": status,
                "exposure": exposure,
                **({"parked": True} if parked else {}),
            },
        )


def _work_addressed(path: str) -> str | None:
    if not path.startswith(TASK_CENTER):
        return None
    parts = path[len(TASK_CENTER) :].split("/")
    return parts[1] if len(parts) >= 2 and parts[0] in ("tasks", "owners") and parts[1] else None


def _exposure_refusal(exposure: str, path: str):
    if exposure == "none":
        return refusal(
            "Agents see nothing of this project: its AI-exposure level is none.", "EXPOSURE_NONE"
        )
    if exposure == "metadata" and any(part in path for part in CASE_ROUTES):
        return refusal(
            "Files, images, logs, raw tables and single cases of a project shared at the "
            "metadata level stay with the service.",
            "EXPOSURE_METADATA",
        )
    return None


def _identifiers(store: ScientificStore, datasets: list[dict]) -> dict[str, str]:
    """Every patient and slide identifier of the project's datasets, with its prefix. A
    dataset whose records cannot be read refuses the answer rather than let one through."""
    known: dict[str, str] = {}
    for item in datasets:
        try:
            content = store.read_artifact(item["id"], "records.json", include_inactive=True)
        except StorageError as error:
            if error.code == "ARTIFACT_NOT_FOUND":
                continue  # a dataset without records adds no identifiers
            raise _unredactable() from error
        except OSError as error:
            raise _unredactable() from error
        try:
            rows = json.loads(content)
        except ValueError as error:
            raise _unredactable() from error
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            # A slide that stands in for its own patient keeps its slide pseudonym.
            if row.get("patientId"):
                known.setdefault(str(row["patientId"]), "P")
            if row.get("slideId"):
                known[str(row["slideId"])] = "S"
    return known


def _unredactable() -> StorageError:
    return StorageError(
        "A dataset's records could not be read, so the answer could not be redacted.",
        "EXPOSURE_METADATA",
        403,
    )


def audit_session(
    projects, method: str, path: str, status: int, client: str | None, project=None
) -> None:
    """Commit and admin requests made with the session token, for the same log: under the
    project's routes, or for the project a handler named, such as a token's."""
    kind = route_class(method, path)
    if kind not in (COMMIT, ADMIN):
        return
    if path.startswith(PROJECTS):
        project = path[len(PROJECTS) :].split("/", 1)[0]
    if not project:
        return
    try:
        _, folder = projects.locate(project)
    except Exception:  # noqa: BLE001 - a missing project has no log to write
        return
    audit.append(
        folder,
        {
            "actor": {"kind": "person", "client": client or "unknown"},
            "method": method,
            "route": route_template(method, path),
            "class": kind,
            "status": status,
        },
    )
