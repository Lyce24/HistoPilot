# Publish the demo as a Hugging Face Space

This folder is a Docker Space. `README.md` holds the Space's settings in its front matter. `Dockerfile` installs a HistoPilot wheel, unpacks the packaged studies and runs `histopilot serve --demo` on port 7860. The Space serves the synthetic BLCA walkthrough and the packaged real studies read-only, and refuses everything else; see [public read-only demo](../../docs/deployment.md#public-read-only-demo).

## Before you start

- **A wheel** with the built UI. Download a release's `histopilot-X.Y.Z-py3-none-any.whl` from GitHub, or build one: `cd web && npm ci && npm run build && cd .. && python scripts/bundle_web.py && uv build --wheel` (see [build a wheel](../../docs/deployment.md#build-a-wheel)). Check that it contains `histopilot/static/index.html`.
- **The studies.** Stop the service that owns them, then package their folders:

  ```bash
  python deploy/huggingface-space/package_demo.py demo-projects.tar.gz /path/to/projects/tcga-nsclc
  ```

  It leaves out model checkpoints, feature packs, per-slide feature embeddings and slide images. A visitor's reads never open them, and a public page must not carry features whose license forbids redistribution. Read the license of every feature set you package: the TCGA-NSCLC study's UNI2-h features are CC-BY-NC-ND 4.0, so the Space shows results derived from them and none of the features.
- **A Hugging Face access token that can write** (Settings → Access Tokens), and the Space's owner and name, for example `histopilot`.

## Create and push the Space

Put the four files in one folder and upload it:

```bash
mkdir space && cp deploy/huggingface-space/{README.md,Dockerfile} space/
cp dist/histopilot-*-py3-none-any.whl demo-projects.tar.gz space/
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("OWNER/histopilot", repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path="space", repo_id="OWNER/histopilot", repo_type="space")
PY
```

The front matter declares `license: mit` for HistoPilot itself. The Space card's last paragraph credits the feature release; keep it.

## Watch it start

The Space builds the image, then starts it. Follow both in **Logs** (Build, then Container) on the Space's page. It is up when the container log shows:

```text
HistoPilot · public demo, read-only
Serves     the synthetic BLCA demo and 1 packaged project to anyone who reaches this port; …
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
```

## Check it

The start page offers each packaged study, then the BLCA walkthrough and the code. From a terminal:

```bash
SPACE=https://OWNER-histopilot.hf.space
curl -s $SPACE/api/v1/health                                  # {"status":"ok",...}
TOKEN=$(curl -s $SPACE/api/v1/session | python -c 'import json,sys; print(json.load(sys.stdin)["token"])')
curl -s -H "X-HistoPilot-Token: $TOKEN" $SPACE/api/v1/projects   # the studies and the demo
curl -s -X POST $SPACE/api/v1/projects                        # 403 DEMO_MODE_READ_ONLY
curl -s $SPACE/api/v1/filesystem/roots                        # 403 DEMO_MODE_ROUTE_UNAVAILABLE
```

## Later

- **A new version or new results**: rebuild the wheel or the archive, replace the file in the Space folder and upload again; the Space rebuilds.
- **Sleep**: a Space on free hardware sleeps after a period without visitors and wakes on the next visit, which takes about a minute.
- **Taking it down**: the Space's **Settings** can pause or delete it.
