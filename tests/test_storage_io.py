"""The storage I/O kernel: canonical encodings, timestamps and bounded JSON files.

Content hashes and ids are persisted in project stores and records. Each helper that
``storage.io`` replaced hashed one of three byte encodings; the digests below were produced
by those former helpers on the fixed inputs, so a change here means stored identities moved.
"""

import hashlib
import json
import os
import re

import pytest

from histopilot.clinical_features import fit_clinical_preprocessor
from histopilot.storage import scientific
from histopilot.storage.io import (
    MAX_JSON_DEPTH,
    canonical_json,
    content_hash,
    decode_json_object,
    read_json_bounded,
    utc_now,
    write_json_atomic,
)
from histopilot.storage.project_lock import StorageError

DOCUMENT = {
    "slideId": "副本",
    "label": "é",
    "counts": [1, 2.5, None, True, False],
    "nested": {"b": "slide-1", "a": {"z": 0.1}},
}
SEQUENCE = ["histopilot-patient-evaluation-v2", 42, "outer:0", "副本"]

# Digests the former helpers produced, per encoding and input.
COMPACT_ASCII = {
    "DOCUMENT": "c26b315350ea82e3f1cc6bfc0034d896228b45e5a0743dbeb20ddd7fdd29641f",
    "SEQUENCE": "cd996c899aa0fa0ba10d46bff561756a07bcb111d347befff691f231537291f9",
}
DEFAULT_SEPARATORS = {
    "DOCUMENT": "0ea5bec2d21889f745f9abc8fd6e68d73ee2c415590d50529c3523b2fc0cfc7c",
    "SEQUENCE": "45b1916d412410a83f9b6ef3c966bd7b2fd4a11a6b6a84cc8c3bc8b6c29bef9f",
}
COMPACT_UTF8 = {
    "DOCUMENT": "a60a3f543717758670f75c66ce62d358ab5bf2be542c1307aa727719bd13e4cb",
    "SEQUENCE": "ce74d66eddc4df137c303168cddf466dfba5b3a3cd2ea7e8b6b59d924890d8e1",
}
INPUTS = {"DOCUMENT": DOCUMENT, "SEQUENCE": SEQUENCE}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# The former helper, and the kernel call that now stands at its call sites.
FORMER_CALL_SITES = [
    # Compact ASCII: evidence, plan, receipt and preview hashes.
    ("application.feature_bundles._hash", content_hash, COMPACT_ASCII),
    ("application.features._hash", content_hash, COMPACT_ASCII),
    ("application.lifecycle._hash", content_hash, COMPACT_ASCII),
    ("application.target_splits._hash", content_hash, COMPACT_ASCII),
    ("application.protocols._json (preview hash)", content_hash, COMPACT_ASCII),
    (
        "application.modern_splits._json (group order)",
        lambda value: sha256(canonical_json(value, ascii=True, compact=True)),
        COMPACT_ASCII,
    ),
    ("schemas.nnmil._hash", content_hash, COMPACT_ASCII),
    ("storage.packed._digest", content_hash, COMPACT_ASCII),
    ("taskcenter.adapters.compute.plan_hash", content_hash, COMPACT_ASCII),
    ("training.fold._receipt_hash", content_hash, COMPACT_ASCII),
    # Default separators: extraction and feature-pack job ids and preview hashes.
    (
        "application.extractions._hash",
        lambda value: content_hash(value, compact=False),
        DEFAULT_SEPARATORS,
    ),
    (
        "application.feature_packs._hash",
        lambda value: content_hash(value, compact=False),
        DEFAULT_SEPARATORS,
    ),
    # Compact UTF-8: import previews and artifacts.
    (
        "application.imports._json (preview hash)",
        lambda value: content_hash(value, ascii=False),
        COMPACT_UTF8,
    ),
]


@pytest.mark.parametrize(("former", "call", "digests"), FORMER_CALL_SITES)
@pytest.mark.parametrize("name", sorted(INPUTS))
def test_each_former_call_site_keeps_its_digest(former, call, digests, name):
    assert call(INPUTS[name]) == digests[name], former


def test_configuration_ids_keep_the_compact_utf8_encoding():
    # Every configuration id is the SHA-256 of this encoding of its manifest.
    assert sha256(scientific.encode_document(DOCUMENT)) == COMPACT_UTF8["DOCUMENT"]
    assert (
        scientific._hash_content(DOCUMENT, {"records.json": {"sha256": "0" * 64, "sizeBytes": 1}})
        == "751fd6b7e3183c51e52eec880dbcad7b83fe99272da8df81b52ed70a7a240f4e"
    )


def test_clinical_preprocessing_keeps_its_training_values_digest():
    memberships = [
        {"slideId": "s1", "patientId": "副本", "partition": "train"},
        {"slideId": "s2", "patientId": "p2", "partition": "train"},
    ]
    values = {"s1": {"age": 61, "stage": "IIA"}, "s2": {"age": None, "stage": "副本"}}
    fields = [{"field": "age", "kind": "numeric"}, {"field": "stage", "kind": "categorical"}]
    preprocessor = fit_clinical_preprocessor(memberships, values, fields)
    assert (
        preprocessor["trainingValuesSha256"]
        == "1d41a9ae50f3ee9f29fb4f2b62f058c3c0d7403252642b90dc6704946f44075e"
    )


def test_the_three_encodings_differ_only_in_separators_and_escaping():
    value = {"b": "副本", "a": [1, None]}
    assert canonical_json(value, ascii=True, compact=True) == (
        b'{"a":[1,null],"b":"\\u526f\\u672c"}'
    )
    assert canonical_json(value, ascii=True, compact=False) == (
        b'{"a": [1, null], "b": "\\u526f\\u672c"}'
    )
    assert canonical_json(value, ascii=False, compact=True) == (
        '{"a":[1,null],"b":"副本"}'.encode()
    )
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")}, ascii=True, compact=True)


def test_timestamps_keep_each_record_family_suffix():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00", utc_now())
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", utc_now(zulu=True))


def test_atomic_json_keeps_the_indented_format_and_its_limit(tmp_path):
    path = tmp_path / "state.json"
    write_json_atomic(path, {"b": 1, "a": "é"})
    assert path.read_bytes() == json.dumps({"b": 1, "a": "é"}, indent=2).encode() + b"\n"
    write_json_atomic(path, {"b": 2}, sync=False)
    assert json.loads(path.read_bytes()) == {"b": 2}
    with pytest.raises(StorageError) as error:
        write_json_atomic(path, {"text": "x" * 100}, limit=64)
    assert (error.value.code, error.value.status_code) == ("METADATA_LIMIT", 413)
    with pytest.raises(ValueError):
        write_json_atomic(path, {"value": float("inf")})
    assert json.loads(path.read_bytes()) == {"b": 2}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["state.json"]


def test_atomic_json_refuses_symbolic_link_components(tmp_path):
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "link")
    with pytest.raises(StorageError) as error:
        write_json_atomic(tmp_path / "link" / "state.json", {})
    assert error.value.code == "STORAGE_UNSAFE_PATH"


@pytest.mark.parametrize(
    "content",
    [b"[1, 2]", b'{"a": NaN}', b'{"a": Infinity}', b'{"a": 1e999}', b"{", b"\xff"],
)
def test_bounded_json_accepts_only_finite_objects(tmp_path, content):
    path = tmp_path / "state.json"
    path.write_bytes(content)
    with pytest.raises(StorageError) as error:
        read_json_bounded(path)
    assert error.value.code == "TRAINING_STATE_INVALID"


def test_bounded_json_limits_size_and_nesting(tmp_path):
    path = tmp_path / "state.json"
    path.write_bytes(b'{"a": [1, 2.5, {"b": null}]}')
    assert read_json_bounded(path) == {"a": [1, 2.5, {"b": None}]}
    with pytest.raises(StorageError) as error:
        read_json_bounded(path, 8)
    assert error.value.code == "STORAGE_CORRUPT"
    with pytest.raises(StorageError) as error:
        read_json_bounded(tmp_path / "missing.json")
    assert error.value.code == "STORAGE_CORRUPT"
    deep = {"a": 1}
    for _ in range(MAX_JSON_DEPTH):
        deep = {"a": deep}
    with pytest.raises(ValueError, match="nesting"):
        decode_json_object(json.dumps(deep))
