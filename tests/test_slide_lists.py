"""Choosing slides: a source, then an optional cohort filter."""

import pytest

from histopilot.application.slide_lists import (
    SlideListError,
    list_slide_ids,
    resolve_slide_selection,
)


@pytest.fixture
def cohort(tmp_path):
    """Per-cohort subfolders of one slide root, as a multi-cohort study is stored."""
    root = tmp_path / "slides" / "colon"
    for folder, name in (
        ("rih", "SL-1.svs"),
        ("rih", "SL-2.svs"),
        ("TCGA", "TCGA-A6.svs"),
        ("SURGEN", "SR386.tiff"),
    ):
        path = root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture slide")
    (root / "notes.txt").write_bytes(b"not a slide")
    records = [{"slideId": name} for name in ("SL-1", "SL-2", "TCGA-A6", "absent-from-disk")]
    return root, records


def test_the_folder_is_the_default_selection(cohort):
    root, _records = cohort
    selection = resolve_slide_selection(root)
    assert selection["source"] == "folder"
    assert [entry.wsi for entry in selection["slides"]] == [
        "rih/SL-1.svs",
        "rih/SL-2.svs",
        "SURGEN/SR386.tiff",
        "TCGA/TCGA-A6.svs",
    ]
    # Non-slide files are not slides, and nothing declares an MPP without a list.
    assert selection["initialCount"] == 4
    assert selection["declaresMpp"] is False
    assert selection["datasetFiltered"] is False
    assert selection["sha256"] is None


def test_a_list_replaces_the_folder_scan_and_pins_each_source_mpp(cohort):
    root, _records = cohort
    selection = resolve_slide_selection(
        root, list_content=b"wsi,mpp\nrih/SL-1.svs,0.5016\nTCGA/TCGA-A6.svs,0.252\n"
    )
    assert selection["source"] == "list"
    assert [(entry.slideId, entry.mpp) for entry in selection["slides"]] == [
        ("SL-1", 0.5016),
        ("TCGA-A6", 0.252),
    ]
    assert selection["declaresMpp"] is True


def test_a_dataset_narrows_the_selection_it_never_widens_it(cohort):
    root, records = cohort
    selection = resolve_slide_selection(root, records=records)
    assert [entry.slideId for entry in selection["slides"]] == ["SL-1", "SL-2", "TCGA-A6"]
    assert selection["datasetFiltered"] is True
    # A slide on disk the cohort does not claim, and a cohort row with no slide on disk.
    assert selection["outside"] == ["SURGEN/SR386.tiff"]
    assert selection["unlisted"] == ["absent-from-disk"]


def test_a_list_and_a_dataset_compose_in_that_order(cohort):
    root, records = cohort
    selection = resolve_slide_selection(
        root,
        list_content=b"wsi,mpp,cohort\nrih/SL-1.svs,0.5016,RIH\nSURGEN/SR386.tiff,0.25,SurGen\n",
        records=records,
    )
    assert [entry.slideId for entry in selection["slides"]] == ["SL-1"]
    assert selection["initialCount"] == 2 and selection["selectedCount"] == 1
    assert selection["outside"] == ["SURGEN/SR386.tiff"]


def test_a_folder_whose_slides_would_overwrite_each_other_is_refused(tmp_path):
    root = tmp_path / "slides"
    for folder in ("current", "superseded", "quarantine"):
        path = root / folder / "SL-1.svs"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture slide")
    with pytest.raises(SlideListError, match="2 pairs of selected slides share a file name"):
        resolve_slide_selection(root)
    # Naming one of each pair is the way out, and it is what a slide list is for.
    selection = resolve_slide_selection(root, list_content=b"wsi\ncurrent/SL-1.svs\n")
    assert [entry.wsi for entry in selection["slides"]] == ["current/SL-1.svs"]


def test_a_narrower_folder_avoids_the_collision_entirely(tmp_path):
    root = tmp_path / "slides"
    for folder in ("current", "superseded"):
        path = root / folder / "SL-1.svs"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture slide")
    assert resolve_slide_selection(root / "current")["initialCount"] == 1
    # A non-recursive scan of the parent sees no slides at all, and says so.
    with pytest.raises(SlideListError, match="No slide files were found"):
        resolve_slide_selection(root, recursive=False)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (b"wsi,mpp\nrih/SL-1.svs,0.5016\nrih/SL-2.svs,\n", "row 3 has no mpp"),
        (b"wsi,mpp\nrih/SL-1.svs,scanner\n", "non-numeric mpp"),
        (b"wsi,mpp\nrih/SL-1.svs,0\n", "positive finite mpp"),
        (b"wsi\nrih/SL-1.svs\nrih/SL-1.svs\n", "selected twice"),
        (b"wsi\n../outside.svs\n", "inside the slide folder"),
        (b"wsi\n/absolute/SL-1.svs\n", "relative to the slide folder"),
        (b"wsi\nrih/absent.svs\n", "is not under"),
        (b"wsi\nnotes.txt\n", "not a supported whole-slide image"),
        (b"wsi,cohort\n,RIH\n", "row 2 has no wsi"),
        (b"slide\nrih/SL-1.svs\n", "needs a wsi column"),
        (b"wsi,wsi\nrih/SL-1.svs,x\n", "repeats a column name"),
        (b"wsi,mpp\n", "contains no rows"),
    ],
)
def test_unusable_lists_are_refused(cohort, table, message):
    root, _records = cohort
    with pytest.raises(SlideListError, match=message):
        resolve_slide_selection(root, list_content=table)


def test_a_selection_that_the_dataset_excludes_entirely_is_refused(cohort):
    root, _records = cohort
    with pytest.raises(SlideListError, match="No selected slide belongs to this dataset"):
        resolve_slide_selection(root, records=[{"slideId": "somewhere-else"}])


def test_an_empty_folder_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SlideListError, match="No slide files were found"):
        resolve_slide_selection(tmp_path / "empty")


def test_extensions_narrow_what_counts_as_a_slide(cohort):
    root, _records = cohort
    selection = resolve_slide_selection(root, wsi_ext=[".tiff"])
    assert [entry.wsi for entry in selection["slides"]] == ["SURGEN/SR386.tiff"]
    with pytest.raises(SlideListError, match="begin with a dot"):
        resolve_slide_selection(root, wsi_ext=["tiff"])


def test_reading_identities_from_a_list_never_touches_slide_storage():
    """Registering features needs the names only; the slides may be offline or elsewhere."""
    identities, has_mpp, digest = list_slide_ids(
        b"wsi,mpp\nrih/SL-1.svs,0.5016\nTCGA/TCGA-A6.svs,0.252\n"
    )
    assert identities == {"SL-1", "TCGA-A6"}
    assert has_mpp is True and len(digest) == 64
    assert list_slide_ids(b"wsi\nrih/SL-1.svs\n")[1] is False
    with pytest.raises(SlideListError, match="both name slide 'SL-1'"):
        list_slide_ids(b"wsi\nrih/SL-1.svs\nother/SL-1.svs\n")


def test_dataset_filter_ignores_name_collisions_in_unselected_slides(tmp_path):
    root = tmp_path / "slides"
    for name in ("eligible/A.svs", "archive/B.svs", "rejected/B.svs"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"slide")
    selection = resolve_slide_selection(root, records=[{"slideId": "A"}])
    assert [entry.slideId for entry in selection["slides"]] == ["A"]
    assert selection["outside"] == ["archive/B.svs", "rejected/B.svs"]
    with pytest.raises(SlideListError, match="share a file name"):
        resolve_slide_selection(root, records=[{"slideId": "B"}])
