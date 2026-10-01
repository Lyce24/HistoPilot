"""Opt-in sign-in for the service's session, as Jupyter's launch token does.

Without it, any process that can reach the loopback port, such as another OS account or,
under WSL2, a Windows program, can fetch the full session token. With login on,
`/api/v1/session` answers only a browser that opened the printed link once, which leaves a
cookie, or a client that presents the secret. The secret lives in a file only this user
can read, in the Task Center's state directory.
"""

import hashlib
import hmac
import os
import secrets
from pathlib import Path

HEADER = "x-histopilot-login"


def cookie_name(port: int) -> str:
    # Cookies ignore ports, and several services can share 127.0.0.1.
    return f"histopilot-login-{port}"


def cookie_value(secret: str) -> str:
    return hmac.new(secret.encode(), b"histopilot-login", hashlib.sha256).hexdigest()


def secret_path(port: int) -> Path:
    from histopilot.taskcenter.paths import state_dir

    return state_dir() / f"login-{port}.secret"


def read_secret(port: int) -> str | None:
    try:
        return secret_path(port).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def ensure_secret(port: int) -> str:
    """The port's secret, created once and kept across restarts until rotated."""
    existing = read_secret(port)
    if existing:
        return existing
    return rotate_secret(port)


def rotate_secret(port: int) -> str:
    """A new secret; every browser signed in with the old one must open the new link."""
    path = secret_path(port)
    secret = secrets.token_urlsafe(32)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(secret + "\n")
    os.replace(temporary, path)
    return secret
