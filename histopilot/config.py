"""Local control-service settings. Reading configuration has no filesystem side effects."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    workspace: Path = field(default_factory=lambda: Path.home() / ".histopilot" / "workspace")
    data_roots: tuple[Path, ...] = ()
    host: str = "127.0.0.1"
    port: int = 8787
    dev: bool = False
    static_dir: Path = field(default_factory=lambda: Path(__file__).parent / "static")

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or self.host not in {"127.0.0.1", "localhost"}:
            raise ValueError(
                "Only loopback hosting is supported. Use SSH forwarding for remote access."
            )
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be an integer between 1 and 65535")
        object.__setattr__(self, "workspace", Path(self.workspace).expanduser().resolve())
        object.__setattr__(self, "static_dir", Path(self.static_dir).expanduser().resolve())
        object.__setattr__(
            self,
            "data_roots",
            tuple(dict.fromkeys(Path(root).expanduser().resolve() for root in self.data_roots)),
        )


def load_settings(
    config: Path | None = None,
    *,
    workspace: Path | None = None,
    data_roots: tuple[Path, ...] | None = None,
    host: str | None = None,
    port: int | None = None,
    dev: bool = False,
) -> Settings:
    """Explicit CLI values override TOML; relative TOML paths use its directory."""
    path = (config or Path.home() / ".histopilot" / "config.toml").expanduser().resolve()
    if config is not None and not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    values = tomllib.loads(path.read_text()) if path.is_file() else {}
    unknown = set(values) - {"server", "storage"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
    server, storage = values.get("server", {}), values.get("storage", {})
    if not isinstance(server, dict) or not isinstance(storage, dict):
        raise ValueError("server and storage must be TOML tables")
    if set(server) - {"host", "port"} or set(storage) - {"workspace", "data_roots"}:
        raise ValueError("Unknown server/storage setting; see examples/config.toml")

    def configured_path(value: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Configured filesystem paths must be nonempty strings")
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else path.parent / candidate

    roots = storage.get("data_roots", [])
    if not isinstance(roots, list):
        raise ValueError("storage.data_roots must be an array of paths")
    return Settings(
        workspace=workspace
        if workspace is not None
        else configured_path(storage["workspace"])
        if "workspace" in storage
        else Path.home() / ".histopilot" / "workspace",
        data_roots=data_roots if data_roots is not None else tuple(map(configured_path, roots)),
        host=host if host is not None else server.get("host", "127.0.0.1"),
        port=port if port is not None else server.get("port", 8787),
        dev=dev,
    )
