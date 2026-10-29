# Publish the demo as a Hugging Face Space

This folder is a Docker Space: `README.md` holds the Space's settings in its front matter, and `Dockerfile` installs a released HistoPilot wheel and runs `histopilot serve --demo` on port 7860. The Space serves the synthetic BLCA demo read-only and refuses everything else; see [public read-only demo](../../docs/deployment.md#public-read-only-demo).

## Before you start

- A GitHub Release `v0.1.0` with the asset `histopilot-0.1.0-py3-none-any.whl`. The wheel must contain the built UI (`histopilot/static/index.html`); [build a wheel](../../docs/deployment.md#build-a-wheel) runs `scripts/bundle_web.py` before `uv build` for that. Check the asset from any machine:

  ```bash
  curl -fsSLo /tmp/histopilot.whl \
    https://github.com/Lyce24/HistoPilot/releases/download/v0.1.0/histopilot-0.1.0-py3-none-any.whl
  unzip -l /tmp/histopilot.whl | grep 'histopilot/static/index.html'
  ```

  For another version, change `ARG HISTOPILOT_VERSION` in the `Dockerfile` to match the tag without its `v`.
- The front matter declares `license: mit`. Add the matching license to the repository first, or change the field.
- A Hugging Face account with an access token that can write (huggingface.co, **Settings → Access Tokens**). Git asks for it as the password when you push.

## Create the Space

1. Open <https://huggingface.co/new-space>.
2. Choose the owner and a name, for example `histopilot`. The app's own address is then usually `https://OWNER-histopilot.hf.space`; the Space's **Embed this Space** menu shows the exact one.
3. **License**: MIT. **Space SDK**: Docker, with the **Blank** template. **Hardware**: CPU basic, the free tier, is enough. **Visibility**: Public.
4. Select **Create Space**. It stays empty until the first push.

## Push the files

From the root of the HistoPilot repository, replacing `OWNER`:

```bash
git clone https://huggingface.co/spaces/OWNER/histopilot /tmp/histopilot-space
cp deploy/huggingface-space/README.md deploy/huggingface-space/Dockerfile /tmp/histopilot-space/
cd /tmp/histopilot-space
git add README.md Dockerfile
git commit -m "Serve the HistoPilot demo"
git push
```

The push replaces the template's `README.md`; its front matter carries the Space's settings, so keep it at the top of the file.

## Watch it start

The Space builds the image, then starts it. Follow both in **Logs** (Build, then Container) on the Space's page. It is up when the container log shows:

```text
HistoPilot · public demo, read-only
...
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
```

## Check it

The Space's page shows the start page with **Open the BLCA demo** and no way to create or load a project. From a terminal:

```bash
SPACE=https://OWNER-histopilot.hf.space
curl -s $SPACE/api/v1/health                      # {"status":"ok",...}
curl -s -X POST $SPACE/api/v1/projects            # 403 DEMO_MODE_READ_ONLY
curl -s $SPACE/api/v1/filesystem/roots            # 403 DEMO_MODE_ROUTE_UNAVAILABLE
```

## Later

- **A new release**: change `ARG HISTOPILOT_VERSION` in both copies of the `Dockerfile` and push to the Space. A rebuild without a change (**Settings → Factory rebuild**) also picks up newer versions of the wheel's dependencies within its declared ranges.
- **Sleep**: a Space on free hardware sleeps after a period without visitors and wakes on the next visit, which takes about a minute.
- **Taking it down**: the Space's **Settings** can pause or delete it.
