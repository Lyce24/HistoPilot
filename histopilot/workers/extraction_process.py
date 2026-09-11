"""Small tmux process boundary; scientific state belongs to the project folder."""

import shlex
import shutil
import subprocess
import sys
from pathlib import Path


class TmuxExtractionExecutor:
    def available(self) -> bool:
        return shutil.which("tmux") is not None

    def running(self, session: str) -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", f"={session}"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode and b"Operation not permitted" in result.stderr:
            raise RuntimeError("Cannot inspect tmux sessions: permission denied.")
        return result.returncode == 0

    def launch(self, session: str, runner: Path, plan: Path) -> None:
        # Inspect existing sessions first, including after an interrupted submission.
        subprocess.run(["tmux", "ls"], capture_output=True, timeout=10)
        if self.running(session):
            raise RuntimeError("This extraction session already exists.")
        command = shlex.join([sys.executable, "-u", str(runner), str(plan)])
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, command],
            capture_output=True,
            check=True,
            timeout=15,
        )

    def cancel(self, session: str) -> None:
        if self.running(session):
            subprocess.run(
                ["tmux", "kill-session", "-t", f"={session}"],
                capture_output=True,
                check=True,
                timeout=10,
            )
