"""Redaction for `metadata` projects: pseudonyms, dropped paths and text, intact hashes."""

import json

from histopilot.api.redaction import Redactor

CANARY_PATIENT = "CANARY-PT-7F3Q"
CANARY_SLIDE = "CANARY-SL-9K2W"


def redactor():
    return Redactor(b"project-key", {CANARY_PATIENT: "P", CANARY_SLIDE: "S", "s1": "S"})


def test_identifiers_become_stable_keyed_pseudonyms():
    first, second = redactor(), redactor()
    row = {"patientId": CANARY_PATIENT, "slideIds": [CANARY_SLIDE], "label": "high"}
    shown = first.document(row)
    assert shown["patientId"].startswith("P-") and len(shown["patientId"]) == 12
    assert shown["slideIds"][0].startswith("S-") and shown["label"] == "high"
    assert second.document(row) == shown
    assert Redactor(b"other-key").document(row)["patientId"] != shown["patientId"]


def test_paths_free_text_and_case_attributes_are_dropped():
    record = {
        "id": "run-1",
        "name": f"Attention · {CANARY_SLIDE}",
        "notes": "Discussed with Dr. Example",
        "storagePath": "/data/private/study",
        "command": {"argv": ["python", "-m", "x"], "cwd": "/data/private", "env": {"A": "1"}},
        "attributes": {"age": 71, "site": "Hospital A"},
        "error": "Could not read /data/private/study/slides/abc.svs",
        "findings": [
            {"code": "X", "message": f"Slide {CANARY_SLIDE} is missing", "severity": "warning"}
        ],
    }
    shown = redactor().document(record)
    text = json.dumps(shown)
    for secret in (CANARY_SLIDE, "Dr. Example", "/data/private", "Hospital A", "argv"):
        assert secret not in text
    assert shown["name"].startswith("Attention · S-")
    assert shown["error"] == "Could not read <path>"
    assert shown["command"] == {}


def test_hashes_and_record_ids_are_untouched():
    preview = {
        "previewHash": "ab" + CANARY_SLIDE + "cd",
        "contentHash": "0123456789abcdef",
        "datasetId": "dataset-0123",
        "id": "configuration-9f",
    }
    assert redactor().document(preview) == preview


def test_short_identifiers_are_found_by_key_in_case_rows_and_in_text():
    # Real slide IDs are often short: T1 to T138, or plain numbers.
    known = {"T1": "S", "T12": "S", "T94": "S", "12": "S"}
    shown = Redactor(b"project-key", known).document(
        {
            "coverage": {"selectedSlideIds": ["T1", "T12", "12"], "missingPackSlideIds": []},
            "progress": {"currentSlide": "T94", "percent": 97.2},
            "rows": [{"id": "T12", "label": "high"}, {"id": "12"}],
            "configuration": {"wsi_rel_paths": ["T1.tiff"], "job_dir": "features/blca"},
            "files": [{"name": "T12.h5"}],
            "message": "T1 of T10; class T1x; 12 slides",
        }
    )
    text = json.dumps(shown)
    for raw in ('"T1"', '"T12"', '"T94"', '"12"', "T1.tiff", "T12.h5", "features/blca"):
        assert raw not in text, raw
    assert shown["configuration"] == {}
    assert shown["files"][0]["name"].endswith(".h5")
    # Unknown IDs and words stay; a short all-digit ID in text reads as a count.
    assert shown["message"].startswith("S-")
    assert shown["message"].endswith(" of T10; class T1x; 12 slides")
    assert shown["rows"][0]["label"] == "high" and shown["progress"]["percent"] == 97.2


def test_per_case_lists_and_maps_are_replaced_even_for_short_numbers():
    known = {str(number): "S" for number in range(1, 11)}
    shown = Redactor(b"project-key", known).document(
        {
            "slides": [str(number) for number in range(1, 11)],
            "bySlide": {str(number): 0.5 for number in range(1, 11)},
            "classCounts": {"1": 3, "2": 4},
            "folds": ["1", "2"],
        }
    )
    assert all(item.startswith("S-") for item in shown["slides"])
    # A map keyed by cases holds per-case values: only its size is shown.
    assert shown["bySlide"] == {"withheld": "per-case values", "count": 10}
    # A handful of values is not a per-case list: class names and folds stay.
    assert shown["classCounts"] == {"1": 3, "2": 4} and shown["folds"] == ["1", "2"]


def test_per_case_records_person_written_text_and_containers_under_hashes_are_withheld():
    known = {f"CANARY-SL-{number}": "S" for number in range(6)}
    shown = Redactor(b"project-key", known).document(
        {
            "predictions": [
                {"slideId": slide, "probability": 0.9, "label": "high"} for slide in known
            ],
            "versionLabel": {"tag": "v1", "note": "Discussed with Dr. Example", "revision": 2},
            "comments": "Reviewed with Dr. Example",
            "sha256": {"/data/private/CANARY-SL-1.h5": "ab"},
            "previewHash": "ab" * 32,
            "intervals": {"note": "Slides from one patient are not independent."},
        }
    )
    text = json.dumps(shown)
    for secret in ("CANARY-SL-", "Dr. Example", "/data/private", "0.9"):
        assert secret not in text, secret
    assert shown["predictions"] == {"withheld": "per-case values", "count": 6}
    assert shown["versionLabel"] == {"tag": "v1", "revision": 2}
    assert shown["previewHash"] == "ab" * 32
    # The service's own notes stay.
    assert shown["intervals"]["note"].startswith("Slides from one patient")


def test_a_slide_standing_in_for_its_patient_keeps_one_pseudonym():
    shown = Redactor(b"project-key", {"T1": "S"}).document(
        {"slideId": "T1", "patientId": "T1", "selectedSlideIds": ["T1"], "name": "T1"}
    )
    names = {shown["slideId"], shown["patientId"], shown["selectedSlideIds"][0], shown["name"]}
    assert len(names) == 1 and names.pop().startswith("S-")


def test_keys_that_name_cases_are_swept_too():
    shown = redactor().document({"selections": {CANARY_PATIENT: True}})
    assert CANARY_PATIENT not in json.dumps(shown)


def test_file_names_with_spaces_never_leave_part_of_an_identifier():
    # Slide IDs may hold a space ("Case 100 B1"). A path ends at whitespace, so replacing the
    # path before the identifier left "<path> 100 B1.h5" as a key, beside that slide's
    # pseudonym.
    slides = [f"Case {number} B1" for number in range(100, 105)]
    numbers = [str(number) for number in range(7, 12)]
    known = dict.fromkeys(slides + numbers, "S")
    stamps = {
        f"/data/features/encoder/{slide}.h5": {"slideId": slide, "patchCount": 9313}
        for slide in slides
    }
    shown = Redactor(b"project-key", known).document(
        {
            "sourceStamps": stamps,
            "packStamps": {"features.bin": {"sizeBytes": 1}, "coords.bin": {"sizeBytes": 2}},
            # Short all-digit IDs are not swept from text; a file's stem still names them.
            "byFile": {f"/data/features/{number}.h5": 1 for number in numbers},
            "error": "Could not read /data/features/encoder/Case 100 B1.h5 today",
            "one": {"/data/features/encoder/Case 101 B1.h5": 1},
            "files": ["Case 102 B1.h5", "Case 103 B1.h5"],
        }
    )
    text = json.dumps(shown)
    assert "B1" not in text and "/data" not in text
    # A map keyed by the cases' files holds per-case values: only its size is shown.
    assert shown["sourceStamps"] == {"withheld": "per-case values", "count": 5}
    assert shown["byFile"] == {"withheld": "per-case values", "count": 5}
    assert shown["packStamps"] == {"features.bin": {"sizeBytes": 1}, "coords.bin": {"sizeBytes": 2}}
    assert shown["error"] == "Could not read <path> today"
    assert shown["one"] == {"<path>": 1}
    assert all(name.startswith("S-") and name.endswith(".h5") for name in shown["files"])
