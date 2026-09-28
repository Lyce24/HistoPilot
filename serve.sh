#!/usr/bin/env bash
# Start the local service in this terminal. A frontend that changed since its last bundle
# is rebuilt first, so every start serves this checkout's current UI and code.
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

show_help() {
  cat <<'EOF'
Usage: bash serve.sh [--no-build] [histopilot serve options]

Start HistoPilot in the foreground at http://127.0.0.1:8787.
If web/ changed since the last bundle, the frontend is rebuilt and bundled first
(a few seconds; skipped when nothing changed). The service loads this checkout's
Python code, so every start runs its current version. Dependencies are never
installed. Press Ctrl+C to stop the service. Existing servers are never stopped
or restarted.

Launcher option:
  --no-build          Start with the existing bundle even if web/ changed

Options are forwarded to histopilot serve, including:
  --workspace PATH    Project workspace
  --data-root PATH    Allow a source directory (repeatable)
  --config PATH       Alternate TOML configuration
  --port PORT         Local service port
  --host HOST         Loopback host (127.0.0.1 or localhost)
  --dev               API reload; start Vite separately in another terminal
  --browser           Open a browser (default: --no-browser)
  --help, -h          Show this help without starting the service

Run from the repository root, or use an absolute path to this script.
Relative option paths resolve from the repository root.

First-time setup, from the repository root (repeat after dependency updates):
  uv sync --locked
  npm --prefix web ci

Examples:
  bash serve.sh
  bash serve.sh --data-root /path/to/research-data --port 8788
  bash serve.sh --dev

See docs/deployment.md for setup, Vite and SSH forwarding.
EOF
}

dev_mode=false
build_web=true
forwarded=()
for option in "$@"; do
  case "$option" in
    --help|-h) show_help; exit 0 ;;
    --no-build) build_web=false; continue ;;
    --dev) dev_mode=true ;;
  esac
  forwarded+=("$option")
done

cd -- "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  printf 'Cannot start HistoPilot: uv is not on PATH. Install uv, then follow the setup in bash serve.sh --help.\n' >&2
  exit 1
fi

project_environment="${UV_PROJECT_ENVIRONMENT:-$repo_root/.venv}"
if [[ ! -x "$project_environment/bin/python" || ! -x "$project_environment/bin/histopilot" ]] \
  || ! "$project_environment/bin/python" -B -c 'import histopilot.cli, histopilot.api.app, uvicorn' >/dev/null 2>&1; then
  printf 'Cannot start HistoPilot: the project Python environment is missing or incomplete.\n' >&2
  printf 'Prepare it manually:\n  cd -- %q\n  uv sync --locked\n' "$repo_root" >&2
  exit 1
fi

if [[ "$dev_mode" == true ]]; then
  : # Vite serves the UI during development.
elif [[ "$build_web" == false ]]; then
  if [[ ! -f histopilot/static/index.html ]]; then
    printf 'Cannot start HistoPilot: the frontend bundle is missing.\n' >&2
    printf 'Start without --no-build to build it, or build it manually:\n  cd -- %q\n' "$repo_root" >&2
    printf '  npm --prefix web run build\n  uv run --locked python scripts/bundle_web.py\n' >&2
    exit 1
  fi
elif ! "$project_environment/bin/python" -B scripts/bundle_web.py --check; then
  # Every service started from this checkout serves histopilot/static: never swap it under one.
  services="$(ps -axo command= 2>/dev/null || true)"
  if grep -qF -- "$project_environment/bin/histopilot serve" <<<"$services"; then
    printf 'Cannot start HistoPilot: the frontend must be rebuilt, but a service from this checkout is running and serves the current bundle.\n' >&2
    printf 'Stop it (Ctrl+C in its terminal), then run bash serve.sh again.\n' >&2
    printf 'To start another service with the existing bundle anyway, add --no-build.\n' >&2
    exit 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    printf 'Cannot start HistoPilot: the frontend must be rebuilt, but npm is not on PATH.\n' >&2
    printf 'Install Node.js (see web/package.json engines), or start with --no-build to serve the existing bundle.\n' >&2
    exit 1
  fi
  if [[ ! -d web/node_modules ]]; then
    printf 'Cannot start HistoPilot: the frontend must be rebuilt, but web/node_modules is missing.\n' >&2
    printf 'Install it once:\n  cd -- %q\n  npm --prefix web ci\n' "$repo_root" >&2
    exit 1
  fi
  # A failed build stops here; the bundle is replaced only after a successful one.
  if ! "$project_environment/bin/python" -B scripts/bundle_web.py --build; then
    printf 'Cannot start HistoPilot: the frontend build failed (see above). The existing bundle is unchanged.\n' >&2
    printf 'Fix the build, or start with --no-build to serve the existing bundle.\n' >&2
    exit 1
  fi
fi

exec uv run --locked --no-sync --offline --no-python-downloads histopilot serve --no-browser ${forwarded[@]+"${forwarded[@]}"}
