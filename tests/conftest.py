"""Suite-wide isolation from the user's machine-level HistoPilot state.

- Registries under the temp directory (training leases, packing claims) are private to
  the session, and subprocess workers inherit the same ``TMPDIR``.
- Every test gets its own Task Center state directory, and the runner never autostarts.
"""

import os
import shutil
import tempfile

import pytest

AUTOSTART_ENV = "HISTOPILOT_TASK_CENTER_AUTOSTART"


@pytest.fixture(scope="session", autouse=True)
def _private_tempdir(request):
    previous = {name: os.environ.get(name) for name in ("TMPDIR", AUTOSTART_ENV)}
    previous_tempdir = tempfile.tempdir
    private = tempfile.mkdtemp(prefix="hp-tests-")
    os.environ["TMPDIR"] = private
    tempfile.tempdir = private
    os.environ[AUTOSTART_ENV] = "0"
    yield private
    tempfile.tempdir = previous_tempdir
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if not request.session.testsfailed:
        shutil.rmtree(private, ignore_errors=True)


@pytest.fixture(autouse=True)
def _task_center_state(tmp_path):
    from histopilot.taskcenter.client import _reset_default_client

    # Its own patch, not the test's `monkeypatch`: a test that calls `monkeypatch.undo()`
    # must not fall back to the person's real Task Center state.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HISTOPILOT_STATE_DIR", str(tmp_path / "histopilot-state"))
        _reset_default_client()
        yield
    _reset_default_client()


@pytest.fixture
def task_center(_task_center_state):
    """This test's private Task Center; see ``support/task_center.py``."""
    from support.task_center import Center

    center = Center()
    yield center
    center.cleanup()
