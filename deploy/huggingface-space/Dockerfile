# HistoPilot's synthetic BLCA demo as a Hugging Face Docker Space: the browser UI and the
# demo's reads, read-only, from a released wheel. PUBLISH.md lists the steps; see
# docs/deployment.md, "Public read-only demo", for what demo mode serves and refuses.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

ARG HISTOPILOT_VERSION=0.1.0

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    HISTOPILOT_DEMO=1
WORKDIR /home/user

# The release wheel already contains the built browser UI; the demo needs no extras.
RUN uv venv /home/user/venv \
 && uv pip install --python /home/user/venv/bin/python \
      "https://github.com/Lyce24/HistoPilot/releases/download/v${HISTOPILOT_VERSION}/histopilot-${HISTOPILOT_VERSION}-py3-none-any.whl"

EXPOSE 7860
CMD ["histopilot", "serve", "--demo", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
