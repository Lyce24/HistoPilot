"""Dataset ingestion and freezing entrypoints to implement next."""

from histopilot.domain import DatasetVersion


class DatasetService:
    def import_dataset(
        self, *, project_id: str, slides_uri: str, table_uris: tuple[str, ...]
    ) -> DatasetVersion:
        raise NotImplementedError(
            "Dataset ingestion and patient/specimen/slide resolution are not implemented."
        )

    def freeze(self, dataset_version_id: str) -> DatasetVersion:
        raise NotImplementedError(
            "Dataset snapshot persistence and content hashing are not implemented."
        )
