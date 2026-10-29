# syntax=docker/dockerfile:1
#
# HistoPilot on CPU: the service, the browser UI and the CLI, without Torch.
# It opens projects, views slides and reads results, and serves the synthetic demo; training
# and predictor runs need the CUDA image (docker/Dockerfile.worker). See
# docs/deployment.md#containers.
#
#   docker build -f docker/Dockerfile -t histopilot .
#   docker run --init -p 127.0.0.1:8787:8787 -v /path/to/data:/data:ro histopilot

ARG PYTHON_IMAGE=python:3.11-slim
ARG NODE_IMAGE=node:24-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10.2

FROM ${UV_IMAGE} AS uv

# The browser UI, built as checks.yml builds it for the wheel.
FROM ${NODE_IMAGE} AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# The locked environment, with HistoPilot installed as a wheel that carries the UI.
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/histopilot/.venv
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-editable --no-install-project --extra imaging
COPY . .
COPY --from=web /src/web/dist web/dist
RUN python scripts/bundle_web.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra imaging
# The installed package carries what checks.yml checks in the wheel.
RUN /opt/histopilot/.venv/bin/python - <<'PY'
from importlib.resources import files
package = files("histopilot")
assert package.joinpath("static/index.html").is_file()
assert package.joinpath("resources/blca_demo_workspace.json").is_file()
for page in ("agents", "cli-contract", "error-codes", "methods", "user-guide"):
    assert package.joinpath(f"resources/docs/{page}.md").is_file(), page
PY

FROM ${PYTHON_IMAGE}
ARG HISTOPILOT_UID=1000
ARG HISTOPILOT_GID=1000
LABEL org.opencontainers.image.title="HistoPilot" \
      org.opencontainers.image.description="HistoPilot service and browser UI (CPU)" \
      org.opencontainers.image.source="https://github.com/Lyce24/HistoPilot" \
      org.opencontainers.image.licenses="MIT"
# State, workspace and temporary files live under /var/lib/histopilot, owned by the service
# user: a new named volume mounted there takes this ownership. /data is the data root.
RUN groupadd --gid "${HISTOPILOT_GID}" histopilot \
 && useradd --uid "${HISTOPILOT_UID}" --gid histopilot --create-home --shell /usr/sbin/nologin histopilot \
 && install -d -o histopilot -g histopilot -m 0700 \
      /var/lib/histopilot /var/lib/histopilot/state /var/lib/histopilot/workspace /var/lib/histopilot/tmp \
 && install -d -o histopilot -g histopilot -m 0755 /data
COPY --from=build /opt/histopilot/.venv /opt/histopilot/.venv
# HISTOPILOT_CONTAINER=1 lets the service listen on the container's interfaces; Host, Origin,
# session and sign-in checks are unchanged. The runner never starts here: no tmux, no compute.
ENV PATH=/opt/histopilot/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HISTOPILOT_CONTAINER=1 \
    HISTOPILOT_PORT=8787 \
    HISTOPILOT_URL=http://127.0.0.1:8787 \
    HISTOPILOT_STATE_DIR=/var/lib/histopilot/state \
    HISTOPILOT_TASK_CENTER_AUTOSTART=0 \
    TMPDIR=/var/lib/histopilot/tmp
USER histopilot
WORKDIR /home/histopilot
VOLUME ["/var/lib/histopilot/state", "/var/lib/histopilot/workspace"]
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request as r; r.build_opener(r.ProxyHandler({})).open('http://127.0.0.1:%s/api/v1/health' % os.environ['HISTOPILOT_PORT'], timeout=4)"]
ENTRYPOINT ["histopilot"]
CMD ["serve", "--no-runner", "--no-browser", "--login", "--port", "8787", \
     "--workspace", "/var/lib/histopilot/workspace", "--data-root", "/data"]
