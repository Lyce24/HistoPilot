"""Workspace-level Task Center: queue, capacity and runner control for this machine."""

from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import Field, StrictBool, StrictInt

from histopilot.schemas.workspace import RequestModel
from histopilot.taskcenter.service import TaskCenterService

LOG_CHUNK_BYTES = 256 * 1024
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

    # Polled reads: task store, workspace registry and read-only leases only.
    @router.get("/summary")
    def summary():
        return center.summary()

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
    ):
        """A stage page's run status: counts, state, queue place, ETA and last failure."""
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
    ):
        """Finished tasks grouped by owner, newest first."""
        return center.history(project=project, kind=kind, state=state, limit=limit, offset=offset)

    @router.get("/tasks")
    def tasks(
        state: str | None = Query(None, max_length=500),
        owner: str | None = Query(None, max_length=200),
        project: str | None = Query(None, max_length=4096),
        kind: str | None = Query(None, max_length=500),
        limit: int = Query(200, ge=1, le=2000),
        offset: int | None = Query(None, ge=0, le=1_000_000),
    ):
        return center.tasks(
            state=state, owner=owner, project=project, kind=kind, limit=limit, offset=offset
        )

    @router.get("/tasks/{task_id}")
    def task(task_id: str):
        return center.task(task_id)

    @router.get("/tasks/{task_id}/log")
    def task_log(task_id: str, download: bool = False):
        """The task's whole log as text, streamed (the detail view carries only its tail)."""
        stream, name = center.log_file(task_id)

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
    def task_action(task_id: str, action: Literal["cancel", "retry"], payload: TaskCenterAction):
        return center.task_action(task_id, action, payload.operationId)

    @router.get("/owners")
    def owners(scope: Literal["live", "all"] = "live"):
        return center.owners(scope=scope)

    @router.get("/owners/{key}")
    def owner(key: str):
        return center.owner(key)

    @router.post("/owners/{key}/{action}")
    def owner_action(
        key: str,
        action: Literal["hold", "release", "stop", "cancel", "retry", "move"],
        payload: OwnerAction,
    ):
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
