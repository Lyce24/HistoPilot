"""FastAPI control service. Compute backends are never imported here."""

__all__ = ["create_app"]


def __getattr__(name):
    # Lazy, so the CLI can read the route and error tables without building the app.
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
