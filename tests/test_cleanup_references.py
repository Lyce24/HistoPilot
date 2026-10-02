"""Cleanup finds a document's references by ID shape exactly as a walk of every string does."""

import json
import os

import pytest

from histopilot.application import lifecycle
from histopilot.storage.project_lock import StorageError

DRAFT = "draft-" + "a" * 32
CONFIGURATION = "configuration-" + "b" * 64
PACK = "pack-" + "c" * 64
DATASET = "dataset-" + "d" * 64


def walked(document):
    parts = document if isinstance(document, list) else [document]
    return {value for part in parts for value in lifecycle._strings(part)}


@pytest.mark.parametrize(
    "document",
    [
        {"experimentId": DRAFT, "nested": [{"inputs": [CONFIGURATION, 3, None, True]}]},
        {DATASET: {"as": "a key"}, "pack": {"id": PACK}},
        # Text around an ID, or an escaped one inside other text, names no record.
        {"path": f"/compute-jobs/{CONFIGURATION}/plan.json", "note": f'see "{DRAFT}"'},
        {"unicode": "é" + DRAFT, "upper": DRAFT.upper(), "short": "draft-abc"},
        [{"first": DRAFT}, {"second": [PACK, PACK]}],
    ],
)
def test_ids_by_shape_match_a_walk_of_every_string(document):
    shaped = {value for value in walked(document) if lifecycle._ID_SHAPE.fullmatch(value)}
    assert lifecycle._references(document, exact=False) == shaped
    assert lifecycle._references(document, exact=True) == walked(document)


def test_plan_files_are_read_once_per_version_and_bad_evidence_still_blocks(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"runs": [{"evaluationId": CONFIGURATION}]}))
    assert lifecycle._plan_ids(path) == {CONFIGURATION}
    reads = []
    original = lifecycle._read_optional
    lifecycle._read_optional = lambda target: reads.append(target) or original(target)
    try:
        assert lifecycle._plan_ids(path) == {CONFIGURATION} and not reads
        replacement = tmp_path / "plan.next"
        replacement.write_text(json.dumps({"runs": [{"evaluationId": DRAFT}]}))
        os.replace(replacement, path)
        assert lifecycle._plan_ids(path) == {DRAFT} and reads == [path]
    finally:
        lifecycle._read_optional = original
    assert lifecycle._plan_ids(tmp_path / "missing.json") == set()
    path.write_text("{not json")
    with pytest.raises(StorageError) as error:
        lifecycle._plan_ids(path)
    assert error.value.code == "CLEANUP_EVIDENCE_INVALID"
    link = tmp_path / "linked.json"
    link.symlink_to(tmp_path / "plan.next")
    with pytest.raises(StorageError):
        lifecycle._plan_ids(link)
