# HistoPilot's public demo as a Hugging Face Docker Space: the browser UI, the synthetic BLCA
# walkthrough and the packaged real studies, all read-only. PUBLISH.md lists the steps; see
# docs/deployment.md, "Public read-only demo", for what demo mode serves and refuses.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    HISTOPILOT_DEMO=1 \
    HISTOPILOT_DEMO_PROJECTS=/home/user/demo-projects
WORKDIR /home/user

# The wheel already contains the built browser UI; the demo needs no extras.
COPY --chown=user histopilot-*-py3-none-any.whl /home/user/wheels/
RUN uv venv /home/user/venv \
 && uv pip install --python /home/user/venv/bin/python /home/user/wheels/histopilot-*-py3-none-any.whl

# Real studies, packaged by package_demo.py: records and results, no checkpoints or features.
COPY --chown=user demo-projects.tar.gz /home/user/
RUN tar -xzf demo-projects.tar.gz && rm demo-projects.tar.gz

EXPOSE 7860
CMD ["histopilot", "serve", "--demo", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
