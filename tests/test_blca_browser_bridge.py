"""The browser verifier must not submit jobs or touch another project."""

import pytest

from scripts.blca_browser_bridge import permitted

PROJECT = "project-" + "1" * 32
BASE = "http://127.0.0.1:8787"
SCOPE = f"{BASE}/api/v1/projects/{PROJECT}"


@pytest.mark.parametrize(
    "method,url",
    [
        ("GET", BASE + "/"),
        ("GET", BASE + "/assets/app.js"),
        ("GET", SCOPE + "/workspace"),
        ("POST", SCOPE + "/datasets/data/query"),
        ("POST", SCOPE + "/evaluation-runs/eval/cases/query"),
        ("POST", SCOPE + "/interpretations/gallery"),
        ("POST", SCOPE + "/protocols/explore"),
    ],
)
def test_read_only_browser_requests(method, url):
    assert permitted(method, url, PROJECT)


@pytest.mark.parametrize(
    "method,url",
    [
        ("POST", SCOPE + "/model-experiments/train/submit"),
        ("POST", SCOPE + "/interpretations/visualize"),
        ("POST", SCOPE + "/interpretations/study/launch"),
        ("PUT", SCOPE + "/datasets/data/slide-reviews/slide"),
        ("DELETE", SCOPE + "/drafts/draft"),
        ("GET", BASE + "/api/v1/projects/project-" + "0" * 32 + "/workspace"),
        ("GET", "https://example.com/"),
        ("GET", SCOPE + "/%2e%2e/other"),
        ("GET", SCOPE + "/%252e%252e/other"),
        ("POST", SCOPE + "/morphology/index"),
        ("POST", SCOPE + "/feature-packs"),
    ],
)
def test_browser_mutations_and_other_projects_never_reach_asgi(method, url):
    assert not permitted(method, url, PROJECT)
