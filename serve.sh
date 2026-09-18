#!/usr/bin/env bash
# Start the local service in this terminal; setup and restarts remain manual.
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

show_help() {
  cat <<'EOF'
Usage: bash serve.sh [histopilot serve options]

Start HistoPilot in the foreground at http://127.0.0.1:8787.
The existing uv environment and built UI are used without installing or building.
Press Ctrl+C to stop the service. Existing servers are never stopped or restarted.

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

First-time setup, from the repository root:
  uv sync --locked
  npm --prefix web ci
  npm --prefix web run build
  uv run --locked python scripts/bundle_web.py

Examples:
  bash serve.sh
  bash serve.sh --data-root /path/to/research-data --port 8788
  bash serve.sh --dev

See docs/deployment.md for setup, Vite and SSH forwarding.
EOF
}

dev_mode=false
for option in "$@"; do
  case "$option" in
    --help|-h) show_help; exit 0 ;;
    --dev) dev_mode=true ;;
  esac
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

if [[ "$dev_mode" == false && ! -f histopilot/static/index.html ]]; then
  printf 'Cannot start HistoPilot: the frontend bundle is missing.\n' >&2
  printf 'Build it manually:\n  cd -- %q\n' "$repo_root" >&2
  printf '  npm --prefix web ci\n  npm --prefix web run build\n  uv run --locked python scripts/bundle_web.py\n' >&2
  printf 'For Vite development, use bash serve.sh --dev and start Vite separately.\n' >&2
  exit 1
fi

exec uv run --locked --no-sync --offline --no-python-downloads histopilot serve --no-browser "$@"
