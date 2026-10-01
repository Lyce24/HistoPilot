"""The roadmap's stage status, ported: shared cases and `histopilot project roadmap`.

`web/src/lib/roadmapCases.json` also runs against the browser's `buildRoadmap` in
`web/src/lib/roadmap.cases.test.ts`.
"""

import json
from pathlib import Path

from support.cli import Service

from histopilot import roadmap

CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "web/src/lib/roadmapCases.json").read_text("utf-8")
)
COMPUTED = (
    "id",
    "status",
    "unlocked",
    "blockers",
    "evidence",
    "artifactCount",
    "compatibilityIssue",
    "retainedWork",
)


def test_the_stages_match_the_browser():
    assert roadmap.MODULES == CASES["modules"] and roadmap.STEPS == CASES["steps"]
    for scenario in CASES["scenarios"]:
        stages = roadmap.build(scenario["workspace"], scenario["evidence"])
        computed = [{key: stage.get(key) for key in COMPUTED} for stage in stages]
        assert computed == scenario["expected"], scenario["name"]
        chosen = roadmap.suggested(stages)
        assert (chosen["id"] if chosen else None) == scenario["suggested"], scenario["name"]


def test_the_cli_reads_a_new_projects_roadmap(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        outcome = service.cli("project", "roadmap", "--project", project, "--json")
        text = service.cli("project", "roadmap", "--project", project)
    assert outcome.code == 0, outcome.stdout
    data = outcome.envelope["data"]
    assert [stage["id"] for stage in data["stages"]] == [item["id"] for item in roadmap.MODULES]
    assert data["stages"][0]["status"] == "not-started" and data["stages"][0]["step"] == "01"
    assert data["next"] == "dataset"
    assert "Next: 01 Datasets" in text.stdout
