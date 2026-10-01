"""Workspace-level Task Center: queue, capacity and runner control for this machine."""

import os
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import Field, StrictBool, StrictInt

from histopilot.api.access import token_project
from histopilot.api.scopes import missing_work
from histopilot.schemas.workspace import RequestModel
from histopilot.taskcenter.service import TaskCenterService

LOG_CHUNK_BYTES = 256 * 1024
# One read of a followed log; a client asks again from the offset it was given.
LOG_READ_BYTES = 4 * 1024 * 1024
GpuIndex = Annotated[str, Field(pattern=r"^[0-9]{1,3}$")]
GpuSlotCount = Annotated[StrictInt, Field(ge=1, le=16)]


class TaskCenterAction(RequestModel):
    operationId: str = Field(min_length=1, max_length=200)


class OwnerAction(TaskCenterAction):
    position: Literal["top", "up", "down", "bottom"] | None = None


class CapacityDefaults(RequestModel):
    cpuThreadsPerRun: Annotated[StrictInt, Field(ge=1, le=32)] | None = None
    dataLoaderWorkers: Annotated[StrictInt, Field(ge=0, le=16)] | None = None


class CapacityUpdate(TaskCenterAction):
    parallelGpuTasks: GpuSlotCount | None = None
    gpuSlots: dict[GpuIndex, GpuSlotCount | None] | None = Field(default=None, max_length=64)
    cpuTaskSlots: Annotated[StrictInt, Field(ge=1, le=128)] | None = None
    paused: StrictBool | None = None
    autoResume: StrictBool | None = None
    defaults: CapacityDefaults | None = None


def task_center_router(center: TaskCenterService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/task-center")

    def require_own(project: str | None, owner: dict, what: str) -> None:
        if project is not None and (owner or {}).get("projectId") != project:
            raise missing_work(what)

    # Polled reads: task store, workspace registry and read-only leases only.
    @router.get("/summary")
    def summary(request: Request):
        project = token_project(request)
        if project is None:
            return center.summary()
        # The machine's queue, capacity and workspace stay with the service.
        rows = center.tasks(project=project, limit=2000)["tasks"]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        runner = center.summary()["runner"]
        return {"runner": {key: runner.get(key) for key in ("alive", "state")}, "counts": counts}

    @router.get("/snapshot")
    def snapshot():
        """Summary, running tasks and the live queue in one read, for the Task Center page."""
        return center.snapshot()

    @router.get("/rollup")
    def rollup(
        owner: str | None = Query(None, max_length=200),
        ownerKind: str | None = Query(None, max_length=100),  # noqa: N803 - query name
        ownerId: str | None = Query(None, max_length=500),  # noqa: N803
        recordKind: str | None = Query(None, max_length=100),  # noqa: N803
        recordId: str | None = Query(None, max_length=500),  # noqa: N803
        recordIds: str | None = Query(None, max_length=40_000),  # noqa: N803
        project: str | None = Query(None, max_length=4096),
        kinds: str | None = Query(None, max_length=500),
        batchIds: str | None = Query(None, max_length=40_000),  # noqa: N803
        request: Request = None,
    ):
        """A stage page's run status: counts, state, queue place, ETA and last failure."""
        scoped = token_project(request) if request is not None else None
        if scoped is not None:
            return center.rollup(project=scoped, kinds=kinds)
        return center.rollup(
            owner=owner,
            owner_kind=ownerKind,
            owner_id=ownerId,
            record_kind=recordKind,
            record_id=recordId,
            record_ids=recordIds,
            project=project,
            kinds=kinds,
            batch_ids=batchIds,
        )

    @router.get("/history")
    def history(
        project: str | None = Query(None, max_length=4096),
        kind: str | None = Query(None, max_length=500),
        state: str | None = Query(None, max_length=500),
        limit: int = Query(25, ge=1, le=200),
        offset: int = Query(0, ge=0, le=1_000_000),
        request: Request = None,
    ):
        """Finished tasks grouped by owner, newest first."""
        project = (token_project(request) if request is not None else None) or project
        return center.history(project=project, kind=kind, state=state, limit=limit, offset=offset)

    @router.get("/tasks")
    def tasks(
        state: str | None = Query(None, max_length=500),
        owner: str | None = Query(None, max_length=200),
        project: str | None = Query(None, max_length=4096),
        kind: str | None = Query(None, max_length=500),
        limit: int = Query(200, ge=1, le=2000),
        offset: int | None = Query(None, ge=0, le=1_000_000),
        request: Request = None,
    ):
        project = (token_project(request) if request is not None else None) or project
        return center.tasks(
            state=state, owner=owner, project=project, kind=kind, limit=limit, offset=offset
        )

    @router.get("/tasks/{task_id}")
    def task(task_id: str, request: Request):
        found = center.task(task_id)
        require_own(token_project(request), found.get("owner"), "task")
        return found

    @router.get("/tasks/{task_id}/log")
    def task_log(
        task_id: str,
        download: bool = False,
        offset: int | None = Query(None, ge=0),
        limit: int = Query(LOG_READ_BYTES, ge=1, le=LOG_READ_BYTES),
        request: Request = None,
    ):
        """The task's whole log as text, streamed (the detail view carries only its tail).

        With ``offset``, up to ``limit`` bytes from that byte offset, and the log's size and
        the next offset in headers, so a client can follow a growing log without rereading.
        """
        scoped = token_project(request) if request is not None else None
        if scoped is not None:
            require_own(scoped, center.task(task_id).get("owner"), "task")
        stream, name = center.log_file(task_id)
        if offset is not None:
            with stream:
                size = os.fstat(stream.fileno()).st_size
                stream.seek(min(offset, size))
                data = stream.read(limit) if offset < size else b""
            return Response(
                content=data,
                media_type="text/plain; charset=utf-8",
                headers={
                    "X-HistoPilot-Log-Size": str(size),
                    "X-HistoPilot-Log-Next-Offset": str(min(offset, size) + len(data)),
                },
            )

        def chunks():
            with stream:
                while True:
                    data = stream.read(LOG_CHUNK_BYTES)
                    if not data:
                        return
                    yield data

        disposition = "attachment" if download else "inline"
        return StreamingResponse(
            chunks(),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'{disposition}; filename="{quote(name)}"'},
        )

    @router.post("/tasks/{task_id}/{action}")
    def task_action(
        request: Request,
        task_id: str,
        action: Literal["cancel", "retry"],
        payload: TaskCenterAction,
    ):
        # The audit log of the project whose work this is.
        request.state.audit_project = (center.task(task_id).get("owner") or {}).get("projectId")
        return center.task_action(task_id, action, payload.operationId)

    @router.get("/owners")
    def owners(request: Request, scope: Literal["live", "all"] = "live"):
        listed = center.owners(scope=scope)
        project = token_project(request)
        if project is not None:
            listed = {
                "owners": [row for row in listed["owners"] if row.get("projectId") == project]
            }
        return listed

    @router.get("/owners/{key}")
    def owner(key: str, request: Request):
        found = center.owner(key)
        require_own(token_project(request), found, "owner")
        return found

    @router.post("/owners/{key}/{action}")
    def owner_action(
        request: Request,
        key: str,
        action: Literal["hold", "release", "stop", "cancel", "retry", "move"],
        payload: OwnerAction,
    ):
        request.state.audit_project = center.owner(key).get("projectId")
        return center.owner_action(key, action, payload.operationId, position=payload.position)

    @router.get("/capacity")
    def capacity():
        return center.capacity()

    @router.put("/capacity")
    def update_capacity(payload: CapacityUpdate):
        changes = payload.model_dump(exclude={"operationId"}, exclude_unset=True)
        if changes.get("defaults") is not None:
            changes["defaults"] = payload.defaults.model_dump(exclude_unset=True)
        return center.update_capacity(changes, payload.operationId)

    @router.post("/runner/start")
    def start_runner(payload: TaskCenterAction):
        return center.start_runner(payload.operationId)

    @router.post("/runner/restart")
    def restart_runner(payload: TaskCenterAction):
        return center.restart_runner(payload.operationId)

    return router
