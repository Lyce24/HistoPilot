"""FastAPI control service. Compute backends are never imported here."""

from .app import create_app

__all__ = ["create_app"]
