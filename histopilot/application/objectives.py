"""Objectives: the scientific questions a project's experiments answer, and where each
experiment sits under them.

The project file histopilot-objectives.json holds the objectives (a name, a rank and a
description), each experiment's filing (its objective, its role there, and its series and
member) and the experiment a copy came from. Experiment records are never rewritten to file
them: their revisions, update times and frozen designs stay as they were, training workers
never read this file, and lineage is not a reference that cleanup follows, so trashing an
old version never requires trashing its copies. Study archives carry the file with the rest
of the project folder.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from histopilot.schemas.objectives import (
    EXPERIMENT_ROLES,
    OBJECTIVE_ID_PATTERN,
    OBJECTIVE_RANKS,
    CreateObjective,
    ExperimentFiling,
    ExperimentRole,
    FileExperiment,
    ObjectiveRank,
    RemoveObjective,
    UpdateObjective,
)
from histopilot.storage.io import (
    content_hash,
    decode_json_object,
    read_file_bounded,
    regular_file,
    utc_now,
    write_json_atomic,
)
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError

OBJECTIVES_FILE = "histopilot-objectives.json"
MAX_BYTES = 4 * 1024 * 1024
MAX_OBJECTIVES = 100
MAX_FILED = 20000
# Recent removals, kept so a removal whose answer was lost replays instead of failing.
MAX_REMOVALS = 200
FILING_FIELDS = ("objectiveId", "role", "series", "seriesMember")


def _error(message: str, code: str, status: int = 409) -> StorageError:
    return StorageError(message, code, status)


def _folded(name: str) -> str:
    return " ".join(name.split()).casefold()


def _natural(text: str) -> list:
    """Sort key that reads digits as numbers, so "Aim 10" follows "Aim 9"."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text)]


class _Stored(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Objective(_Stored):
    id: str = Field(pattern=OBJECTIVE_ID_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    rank: ObjectiveRank
    description: str = Field(max_length=2000)
    revision: int = Field(ge=1)
    createdAt: str = Field(max_length=64)
    updatedAt: str = Field(max_length=64)
    creationOperationId: str = Field(min_length=1, max_length=200)
    creationRequestHash: str = Field(min_length=64, max_length=64)


class _Lineage(_Stored):
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(max_length=200)


class _Filing(_Stored):
    objectiveId: str | None = Field(default=None, pattern=OBJECTIVE_ID_PATTERN)
    role: ExperimentRole | None = None
    series: str | None = Field(default=None, min_length=1, max_length=80)
    seriesMember: str | None = Field(default=None, min_length=1, max_length=80)
    copiedFrom: _Lineage | None = None
    updatedAt: str = Field(max_length=64)


class _Removal(_Stored):
    objectiveId: str = Field(pattern=OBJECTIVE_ID_PATTERN)
    name: str = Field(max_length=120)
    unfiled: int = Field(ge=0)
    requestHash: str = Field(min_length=64, max_length=64)
    removedAt: str = Field(max_length=64)


class _Document(_Stored):
    version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    objectives: list[_Objective] = Field(default_factory=list)
    experiments: dict[str, _Filing] = Field(default_factory=dict)
    removals: dict[str, _Removal] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded_and_distinct(self):
        if len(self.objectives) > MAX_OBJECTIVES or len(self.experiments) > MAX_FILED:
            raise ValueError("The objectives file holds more entries than it may.")
        if len(self.removals) > MAX_REMOVALS:
            raise ValueError("The objectives file keeps too many removals.")
        if any(not key or len(key) > 256 for key in self.experiments):
            raise ValueError("Experiment IDs must be short, non-empty strings.")
        ids = [item.id for item in self.objectives]
        names = [_folded(item.name) for item in self.objectives]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("Objective IDs and names must be distinct.")
        return self


def _ordered(objectives: list[_Objective]) -> list[_Objective]:
    """Primary objectives first, then secondary, then exploratory; by name within a rank."""
    return sorted(
        objectives,
        key=lambda item: (OBJECTIVE_RANKS.index(item.rank), _natural(item.name), item.id),
    )


def _public(item: _Objective) -> dict:
    return item.model_dump(exclude={"creationOperationId", "creationRequestHash"})


def _values(filing: _Filing | ExperimentFiling | None) -> dict:
    """The four filing fields; none for an experiment never filed."""
    return {key: getattr(filing, key, None) for key in FILING_FIELDS}


class ObjectiveService:
    """Objectives and filings of one project's experiments, kept in its objectives file.

    Writes hold the project's lifecycle guard (reentrant, so experiment creation can file
    the new experiment inside its own guard) and replace the file atomically; reads take no
    lock and see the previous or the next version whole."""

    def __init__(self, store):
        self.store = store
        self.folder = Path(store.folder)
        self.path = self.folder / OBJECTIVES_FILE

    # Reading ----------------------------------------------------------------------------

    def _read(self) -> _Document:
        if regular_file(self.path, missing_ok=True) is None:
            return _Document()
        content = read_file_bounded(self.path, MAX_BYTES)
        try:
            return _Document.model_validate(decode_json_object(content))
        except (ValueError, UnicodeError, RecursionError, ValidationError) as error:
            raise _error(
                "The project's objectives file is invalid; restore its last intact copy.",
                "OBJECTIVES_CORRUPT",
            ) from error

    def list(self) -> dict:
        """The objectives, primary first, and every filed experiment's filing by experiment
        ID. Predictors, runs, batches and interpretations name their experiment, so any list
        can be grouped by objective from this answer alone."""
        document = self._read()
        known = {item.id for item in document.objectives}
        filings = {}
        for experiment, filing in document.experiments.items():
            values = _values(filing)
            if values["objectiveId"] not in known:
                values["objectiveId"] = None
            if any(values.values()):
                filings[experiment] = values
        return {
            "revision": document.revision,
            "items": [_public(item) for item in _ordered(document.objectives)],
            "filings": filings,
            "ranks": list(OBJECTIVE_RANKS),
            "roles": list(EXPERIMENT_ROLES),
        }

    def get(self, identity: str) -> dict:
        return _public(self._objective(self._read(), identity))

    @staticmethod
    def _objective(document: _Document, identity: str) -> _Objective:
        for item in document.objectives:
            if item.id == identity:
                return item
        raise _error("This objective does not exist in this project.", "OBJECTIVE_NOT_FOUND", 404)

    def annotate(self, items: Iterable[dict]) -> str | None:
        """Add each listed experiment's filing to it, in place: objectiveId, objective
        (its ID, name and rank), role, series, seriesMember and copiedFrom.

        An unreadable objectives file must not hide the experiments: they are listed
        unfiled and the error's message is returned for the caller to show."""
        try:
            document = self._read()
        except StorageError as error:
            document, problem = _Document(), str(error)
        else:
            problem = None
        objectives = {item.id: item for item in document.objectives}
        for item in items:
            filing = document.experiments.get(item["id"])
            objective = objectives.get(filing.objectiveId) if filing else None
            item.update(
                {
                    "objectiveId": objective.id if objective else None,
                    "objective": {
                        "id": objective.id,
                        "name": objective.name,
                        "rank": objective.rank,
                    }
                    if objective
                    else None,
                    "role": filing.role if filing else None,
                    "series": filing.series if filing else None,
                    "seriesMember": filing.seriesMember if filing else None,
                    "copiedFrom": filing.copiedFrom.model_dump()
                    if filing and filing.copiedFrom
                    else None,
                }
            )
        return problem

    # Writing ----------------------------------------------------------------------------

    def _save(self, document: _Document) -> None:
        document.revision += 1
        try:
            write_json_atomic(self.path, document.model_dump(mode="json"), limit=MAX_BYTES)
        except OSError as error:
            raise _error(
                "The objectives could not be saved. Reload before retrying.",
                "STORAGE_WRITE_FAILED",
                403,
            ) from error

    def _writable(self) -> _Document:
        """The current document, once the project is known to accept changes."""
        self.store.lifecycle.assert_usable(())
        return self._read()

    @staticmethod
    def _require_free_name(document: _Document, name: str, *, own: str | None = None) -> None:
        for item in document.objectives:
            if item.id != own and _folded(item.name) == _folded(name):
                raise _error(
                    f"An objective named {item.name!r} already exists. Choose another name.",
                    "OBJECTIVE_EXISTS",
                )

    def create(self, request: CreateObjective) -> dict:
        digest = content_hash(request.model_dump(exclude={"operationId"}))
        with lifecycle_guard(self.folder):
            document = self._writable()
            for item in document.objectives:
                if item.creationOperationId == request.operationId:
                    if item.creationRequestHash != digest:
                        raise _error(
                            "This operation ID belongs to another objective request.",
                            "OPERATION_CONFLICT",
                        )
                    # Retrying a creation whose answer was lost returns the same objective.
                    return _public(item)
            if len(document.objectives) >= MAX_OBJECTIVES:
                raise _error(
                    "A project holds at most 100 objectives. Remove one before adding another.",
                    "OBJECTIVE_LIMIT",
                )
            self._require_free_name(document, request.name)
            now = utc_now(zulu=True)
            item = _Objective(
                id=f"objective-{uuid4().hex}",
                name=request.name,
                rank=request.rank,
                description=request.description,
                revision=1,
                createdAt=now,
                updatedAt=now,
                creationOperationId=request.operationId,
                creationRequestHash=digest,
            )
            document.objectives.append(item)
            self._save(document)
            return _public(item)

    def update(self, identity: str, request: UpdateObjective) -> dict:
        changes = request.model_dump(exclude={"expectedRevision"}, exclude_none=True)
        with lifecycle_guard(self.folder):
            document = self._writable()
            item = self._objective(document, identity)
            if item.model_dump(include=set(changes)) == changes:
                # Nothing to change: also how a retry whose answer was lost ends.
                return _public(item)
            if item.revision != request.expectedRevision:
                raise _error(
                    "The objective changed since it was read. Reload it before saving.",
                    "REVISION_CONFLICT",
                )
            if "name" in changes:
                self._require_free_name(document, changes["name"], own=identity)
            for key, value in changes.items():
                setattr(item, key, value)
            item.revision += 1
            item.updatedAt = utc_now(zulu=True)
            self._save(document)
            return _public(item)

    def remove(self, identity: str, request: RemoveObjective) -> dict:
        """Remove an objective. Its experiments stay as they are, unfiled from it: their
        roles and series are kept for whichever objective they join next."""
        digest = content_hash(
            {"objectiveId": identity, "expectedRevision": request.expectedRevision}
        )
        with lifecycle_guard(self.folder):
            document = self._writable()
            done = document.removals.get(request.operationId)
            if done is not None:
                if done.requestHash != digest:
                    raise _error(
                        "This operation ID belongs to another removal.", "OPERATION_CONFLICT"
                    )
                return self._removal(done)
            item = self._objective(document, identity)
            if item.revision != request.expectedRevision:
                raise _error(
                    "The objective changed since it was read. Review it before removing it.",
                    "REVISION_CONFLICT",
                )
            document.objectives = [entry for entry in document.objectives if entry.id != identity]
            now = utc_now(zulu=True)
            unfiled = 0
            for experiment, filing in list(document.experiments.items()):
                if filing.objectiveId != identity:
                    continue
                unfiled += 1
                filing.objectiveId = None
                filing.updatedAt = now
                if not any(_values(filing).values()) and filing.copiedFrom is None:
                    del document.experiments[experiment]
            removal = _Removal(
                objectiveId=identity,
                name=item.name,
                unfiled=unfiled,
                requestHash=digest,
                removedAt=now,
            )
            document.removals[request.operationId] = removal
            while len(document.removals) > MAX_REMOVALS:
                del document.removals[next(iter(document.removals))]
            self._save(document)
            return self._removal(removal)

    @staticmethod
    def _removal(removal: _Removal) -> dict:
        return {
            "objectiveId": removal.objectiveId,
            "name": removal.name,
            "unfiled": removal.unfiled,
        }

    def file(self, experiment_id: str, request: FileExperiment) -> None:
        """Replace an experiment's filing, if it is still the one the caller read.

        The caller checks that the experiment exists. Sending the saved filing again
        changes nothing, which is also how a retry whose answer was lost ends."""
        wanted = _values(request)
        with lifecycle_guard(self.folder):
            document = self._writable()
            current = document.experiments.get(experiment_id)
            if _values(current) == wanted:
                return
            if _values(current) != _values(request.expected):
                raise _error(
                    "The experiment was filed differently since it was read. Reload it, "
                    "then file it again.",
                    "EXPERIMENT_FILING_CONFLICT",
                )
            self._place(document, experiment_id, wanted, copied_from=None)
            self._save(document)

    def file_new(
        self,
        experiment_id: str,
        filing: ExperimentFiling | None,
        copied_from: dict | None,
    ) -> None:
        """File a new experiment and note the experiment it was copied from, once.

        Experiment creation calls this inside its own guard after saving the record, and
        again when a lost creation is replayed, so an interrupted first attempt still ends
        filed; an experiment that already has a filing keeps it."""
        if filing is None and copied_from is None:
            return
        with lifecycle_guard(self.folder):
            document = self._writable()
            if experiment_id in document.experiments:
                return
            self._place(document, experiment_id, _values(filing), copied_from=copied_from)
            self._save(document)

    def require_objective(self, identity: str | None) -> None:
        """Refuse an unreadable objectives file, or an objective ID this project does not
        have; no objective is fine."""
        document = self._read()
        if identity is not None:
            self._objective(document, identity)

    def _place(
        self, document: _Document, experiment_id: str, values: dict, *, copied_from: dict | None
    ) -> None:
        if values["objectiveId"] is not None:
            self._objective(document, values["objectiveId"])
        current = document.experiments.get(experiment_id)
        lineage = (
            _Lineage.model_validate(copied_from)
            if copied_from is not None
            else current.copiedFrom
            if current
            else None
        )
        if not any(values.values()) and lineage is None:
            document.experiments.pop(experiment_id, None)
            return
        if current is None and len(document.experiments) >= MAX_FILED:
            raise _error("This project has filed as many experiments as it may.", "OBJECTIVE_LIMIT")
        document.experiments[experiment_id] = _Filing(
            **values, copiedFrom=lineage, updatedAt=utc_now(zulu=True)
        )
