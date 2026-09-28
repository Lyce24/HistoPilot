"""On-demand thread stacks for a running service or runner: ``kill -USR1 <pid>``."""

import faulthandler
import signal
from pathlib import Path

_HANDLES: dict[int, object] = {}


def enable_stack_dumps(path: Path) -> Path | None:
    """Append every thread's Python stack to ``path`` whenever the process gets SIGUSR1.

    A hung process can then be diagnosed without a debugger or ptrace permissions.
    """
    number = getattr(signal, "SIGUSR1", None)
    if number is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a", encoding="utf-8")  # noqa: SIM115 - kept open for faulthandler
        faulthandler.register(number, file=handle, all_threads=True)
    except (OSError, RuntimeError, ValueError):
        return None
    previous = _HANDLES.get(number)
    _HANDLES[number] = handle  # faulthandler writes to the descriptor; keep it open
    if previous is not None:
        previous.close()
    return path
