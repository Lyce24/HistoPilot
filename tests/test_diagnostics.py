import faulthandler
import os
import signal
import time

import pytest

from histopilot.diagnostics import enable_stack_dumps


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals only")
def test_sigusr1_appends_every_threads_stack(tmp_path):
    path = enable_stack_dumps(tmp_path / "logs" / "stacks.log")
    try:
        assert path == tmp_path / "logs" / "stacks.log"
        os.kill(os.getpid(), signal.SIGUSR1)
        deadline = time.monotonic() + 5
        while "test_sigusr1_appends_every_threads_stack" not in path.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        faulthandler.unregister(signal.SIGUSR1)
