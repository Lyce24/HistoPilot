"""Service-owned synthetic workspace, immutable cohort snapshots, and drafts.

This small implemented application path is separate from the unimplemented
scientific ingestion/training services. Original source directories are only
referenced; registered paths do not imply dataset ingestion or readiness.
"""

import json
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from histopilot.contracts.experiment import ExperimentSpec
from histopilot.schemas.workspace import CohortRequest, ExperimentRequest
from histopilot.storage.database import Database, Record
from histopilot.storage.filesystem import LocalFilesystem


class WorkspaceError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identity(prefix: str, value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}-{sha256(canonical.encode()).hexdigest()[:20]}"


class LocalWorkspace:
    def __init__(self, database: Database, filesystem: LocalFilesystem):
        self.database = database
        self.filesystem = filesystem

    def initialize(self) -> None:
        with self.database.sessions.begin() as session:
            if session.get(Record, ("workspace", "synthetic-v1")) is None:
                fixture = json.loads(
                    files("histopilot.resources").joinpath("demo_workspace.json").read_text()
                )
                fixture["exampleManifests"] = fixture.pop("manifests")
                fixture["mode"] = "synthetic-demo"
                fixture["executionEnabled"] = False
                session.add(Record(kind="workspace", id="synthetic-v1", payload=fixture))

    @staticmethod
    def _base(session: Session) -> dict:
        return deepcopy(session.get(Record, ("workspace", "synthetic-v1")).payload)

    @staticmethod
    def _records(session: Session, kind: str) -> list[dict]:
        records = session.scalars(select(Record).where(Record.kind == kind).order_by(Record.id))
        return sorted(
            (deepcopy(record.payload) for record in records), key=lambda item: item["createdAt"]
        )

    @staticmethod
    def _immutable(session: Session, kind: str, identity: str, payload: dict) -> dict:
        session.execute(
            insert(Record)
            .values(kind=kind, id=identity, payload=payload)
            .on_conflict_do_nothing(index_elements=["kind", "id"])
        )
        return deepcopy(session.get(Record, (kind, identity)).payload)

    def workspace(self) -> dict:
        with self.database.sessions.begin() as session:
            state = self._base(session)
            state["cohortSnapshots"] = self._records(session, "cohort")
            state["drafts"] = self._records(session, "experiment")
            state["sources"] = self._records(session, "source")
            return state

    def models(self, kind: str) -> list[dict]:
        with self.database.sessions.begin() as session:
            return self._base(session)[kind]

    def save_cohort(self, request: CohortRequest) -> dict:
        with self.database.sessions.begin() as session:
            base = self._base(session)
            if request.datasetId != base["dataset"]["id"]:
                raise WorkspaceError("The dataset version does not exist.", 404)
            filters = {
                "specimenType": request.specimenType,
                "msi": request.msi,
                "braf": request.braf,
            }
            patients = [
                patient
                for patient in base["patients"]
                if all(value == "Any" or patient[key] == value for key, value in filters.items())
            ]
            if not patients:
                raise WorkspaceError("The selected filters produce an empty cohort.")
            patient_ids = sorted(patient["id"] for patient in patients)
            slide_ids = sorted(
                slide["id"] for slide in base["slides"] if slide["patientId"] in patient_ids
            )
            content = {
                "datasetId": request.datasetId,
                "target": "kras",
                "filters": filters,
                "patientIds": patient_ids,
                "slideIds": slide_ids,
                "splitId": base["split"]["id"],
            }
            identity = _identity("cohort", content)
            specimen = "All specimens" if request.specimenType == "Any" else request.specimenType
            payload = {"id": identity, "name": f"{specimen} · KRAS", **content, "createdAt": _now()}
            return self._immutable(session, "cohort", identity, payload)

    def save_experiments(self, request: ExperimentRequest) -> list[dict]:
        with self.database.sessions.begin() as session:
            base = self._base(session)
            record = session.get(Record, ("cohort", request.cohortId))
            if record is None:
                raise WorkspaceError("Freeze a stored cohort before creating an experiment.", 404)
            cohort = deepcopy(record.payload)
            if request.folds > len(cohort["patientIds"]):
                raise WorkspaceError("The fold count exceeds the number of patients in the cohort.")
            encoders = {model["id"] for model in base["encoders"]}
            models = {model["id"] for model in base["milModels"]}
            features = {feature["encoderId"]: feature["id"] for feature in base["featureSets"]}
            resolved = []
            for pair in sorted(set(request.pairs)):
                components = pair.split(":")
                if (
                    len(components) != 2
                    or components[0] not in encoders
                    or components[1] not in models
                ):
                    raise WorkspaceError(
                        "Select an encoder and MIL model from the service registry."
                    )
                encoder_id, mil_id = components
                if encoder_id not in features:
                    raise WorkspaceError("The selected encoder has no registered feature set.")
                resolved.append((encoder_id, mil_id))
            drafts = []
            for encoder_id, mil_id in resolved:
                spec = ExperimentSpec(
                    dataset_id=cohort["datasetId"],
                    cohort_id=cohort["id"],
                    split_id=cohort["splitId"],
                    feature_set_id=features[encoder_id],
                    encoder_id=encoder_id,
                    mil_model=mil_id,
                    seeds=tuple(request.seeds),
                    folds=request.folds,
                    aggregation=request.aggregation,
                ).model_dump(mode="json")
                identity = _identity("draft", spec)
                payload = {
                    "id": identity,
                    "status": "Draft",
                    "datasetId": cohort["datasetId"],
                    "cohortId": cohort["id"],
                    "cohortSnapshot": cohort,
                    "splitId": cohort["splitId"],
                    "encoderId": encoder_id,
                    "milId": mil_id,
                    "featureSetId": features[encoder_id],
                    "seeds": spec["seeds"],
                    "folds": spec["folds"],
                    "aggregation": spec["aggregation"],
                    "manifest": spec,
                    "createdAt": _now(),
                }
                drafts.append(self._immutable(session, "experiment", identity, payload))
            return drafts

    def experiment_manifest(self, identity: str) -> dict:
        with self.database.sessions.begin() as session:
            record = session.get(Record, ("experiment", identity))
            if record is None:
                raise WorkspaceError("The experiment draft does not exist.", 404)
            return deepcopy(record.payload["manifest"])

    def delete_experiment(self, identity: str) -> None:
        with self.database.sessions.begin() as session:
            record = session.get(Record, ("experiment", identity))
            if record is None:
                raise WorkspaceError("The experiment draft does not exist.", 404)
            if record.payload["status"] != "Draft":
                raise WorkspaceError("Only an unexecuted draft can be removed.", 409)
            session.delete(record)

    def add_source(self, value: str) -> dict:
        path = self.filesystem.directory(value)
        identity = _identity("source", str(path))
        payload = {
            "id": identity,
            "path": str(path),
            "name": path.name or str(path),
            "kind": "directory",
            "readOnly": True,
            "importStatus": "not-imported",
            "createdAt": _now(),
        }
        with self.database.sessions.begin() as session:
            return self._immutable(session, "source", identity, payload)
