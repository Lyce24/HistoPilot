"""Versioned SQLite metadata store with short SQLAlchemy transactions."""

from pathlib import Path
from typing import Any

from sqlalchemy import JSON, URL, Integer, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = "schema_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class Record(Base):
    """Small metadata documents, never feature arrays or original WSI contents."""

    __tablename__ = "records"

    kind: Mapped[str] = mapped_column(String(32), primary_key=True)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class Database:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.path = workspace / "histopilot.db"
        self.engine = create_engine(
            URL.create("sqlite", database=str(self.path)),
            connect_args={"check_same_thread": False, "timeout": 10},
        )
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

        @event.listens_for(self.engine, "connect")
        def configure_connection(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.close()

    def initialize(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        with self.engine.connect() as connection:
            mode = connection.exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one()
            if mode.lower() != "wal":
                raise RuntimeError("The workspace database does not support SQLite WAL mode.")
        Base.metadata.create_all(self.engine)
        with self.sessions.begin() as session:
            current = session.get(SchemaVersion, 1)
            if current is None:
                session.add(SchemaVersion(id=1, version=SCHEMA_VERSION))
            elif current.version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported workspace schema {current.version}; expected {SCHEMA_VERSION}. "
                    "Automatic migration from other versions is not implemented."
                )

    def close(self) -> None:
        self.engine.dispose()
