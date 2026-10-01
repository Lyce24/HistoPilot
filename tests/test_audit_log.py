"""The agent audit log: appended lines, newest first, rotation."""

from histopilot.api import audit


def test_lines_are_appended_and_read_newest_first(tmp_path):
    for status in (200, 403):
        audit.append(
            tmp_path, {"route": "GET /api/v1/projects/{identity}/workspace", "status": status}
        )
    entries = audit.read(tmp_path)
    assert [entry["status"] for entry in entries] == [403, 200]
    assert all(entry["time"].endswith("Z") for entry in entries)
    assert oct(audit.audit_path(tmp_path).stat().st_mode & 0o777) == "0o600"


def test_the_log_rotates_and_keeps_a_few_generations(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROTATE_BYTES", 200)
    for index in range(40):
        audit.append(tmp_path, {"index": index})
    rotated = sorted(path.name for path in (tmp_path / "audit").iterdir())
    assert rotated == ["actions.1.jsonl", "actions.2.jsonl", "actions.3.jsonl", "actions.jsonl"]
    assert audit.read(tmp_path, limit=1)[0]["index"] == 39
