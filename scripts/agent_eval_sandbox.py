"""A sandbox service for the agent evaluation suite (tests/skill_evals/histopilot.md).

Builds the canary study of `histopilot.agent.evals` in FOLDER, which must be new or empty:
its own workspace, data root and Task Center state, never a real one. Writes the canaries
and three scoped tokens to FOLDER/sandbox.json (readable by you only), then serves the
study on a loopback port until Ctrl-C.

python scripts/agent_eval_sandbox.py FOLDER [--port 8799] [--no-serve]
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="A new or empty folder for the sandbox.")
    parser.add_argument("--port", type=int, default=8799, help="Loopback port (default: 8799).")
    parser.add_argument("--no-serve", action="store_true", help="Build the study, then exit.")
    options = parser.parse_args(argv)
    folder = options.folder.expanduser().resolve()
    if folder.exists() and any(folder.iterdir()):
        print(f"{folder} is not empty; choose a new folder.", file=sys.stderr)
        return 2
    # The sandbox holds tokens: yours only. The Task Center creates its own state folder.
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    folder.chmod(0o700)
    for child in ("data", "workspace"):
        (folder / child).mkdir(mode=0o700, exist_ok=True)
    # Before HistoPilot is imported: the sandbox's own Task Center, never the person's.
    os.environ["HISTOPILOT_STATE_DIR"] = str(folder / "state")
    os.environ["HISTOPILOT_TASK_CENTER_AUTOSTART"] = "0"

    from fastapi.testclient import TestClient

    from histopilot.agent import evals
    from histopilot.api.app import create_app
    from histopilot.config import Settings

    settings = Settings(
        workspace=folder / "workspace", data_roots=(folder / "data",), port=options.port
    )
    with TestClient(create_app(settings), base_url=f"http://127.0.0.1:{options.port}") as http:
        http.headers["X-HistoPilot-Token"] = http.get("/api/v1/session").json()["token"]
        study = evals.seed(http, settings.workspace)
        study["url"] = f"http://127.0.0.1:{options.port}"
        study["tokens"] = {
            "public": evals.token(http, study["projects"]["public"])["token"],
            "shared": evals.token(http, study["projects"]["shared"])["token"],
            "public-commit": evals.token(
                http, study["projects"]["public"], ("read", "preview", "commit")
            )["token"],
        }
    target = folder / "sandbox.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(study, handle, indent=2)
    print(f"Wrote {target}: canaries and tokens for the graders.")
    if options.no_serve:
        return 0
    import uvicorn

    print(f"Serving the sandbox at {study['url']}; Ctrl-C stops it.")
    uvicorn.run(create_app(settings), host="127.0.0.1", port=options.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
