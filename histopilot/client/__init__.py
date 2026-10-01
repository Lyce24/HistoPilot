"""Python client for the local HistoPilot service: the core of the CLI and the agent tools.

Stdlib HTTP only. It never builds the service in-process, so importing it stays cheap.
"""

from .api import Client
from .errors import ClientError
from .transport import DEFAULT_URL

__all__ = ["DEFAULT_URL", "Client", "ClientError"]
