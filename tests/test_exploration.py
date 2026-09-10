import pytest

from histopilot.application.exploration import ExploreRequest, explore
from histopilot.storage.project_lock import StorageError


def test_aggregates_and_filters_cover_all_rows_before_pagination():
    rows = [
        {
            "slideId": str(index),
            "patientId": str(index // 2),
            "attributes": {"grade": "low" if index < 250 else "high", "sex": "F"},
        }
        for index in range(400)
    ]
    dictionary = [
        {"key": "grade", "owner": "slide", "type": "categorical"},
        {"key": "sex", "owner": "patient", "type": "categorical"},
    ]
    report = explore(rows, dictionary, ExploreRequest(field="grade", limit=10))
    assert len(report["records"]) == 10
    assert report["distribution"]["counts"] == [
        {"value": "low", "count": 250},
        {"value": "high", "count": 150},
    ]
    filtered = explore(
        rows,
        dictionary,
        ExploreRequest(
            field="sex", filters=[{"field": "grade", "values": ["high"]}], offset=20, limit=10
        ),
    )
    assert filtered["total"] == 150
    assert filtered["summary"]["mappedPatientCount"] == 75
    assert filtered["distribution"]["counts"] == [{"value": "F", "count": 75}]
    assert filtered["distribution"]["unit"] == "patient"
    assert filtered["records"][0]["slideId"] == "270"


def test_same_slide_conjunction_missing_literal_and_numeric_histogram():
    rows = [
        {"slideId": "1", "patientId": "P1", "attributes": {"a": "yes", "b": "no", "n": "1"}},
        {"slideId": "2", "patientId": "P1", "attributes": {"a": "no", "b": "yes", "n": "2"}},
        {"slideId": "3", "patientId": None, "attributes": {"a": None, "b": "(missing)", "n": None}},
    ]
    dictionary = [{"key": key, "owner": "slide", "type": "categorical"} for key in ["a", "b"]] + [
        {"key": "n", "owner": "slide", "type": "integer"}
    ]
    report = explore(
        rows,
        dictionary,
        ExploreRequest(
            filters=[{"field": "a", "values": ["yes"]}, {"field": "b", "values": ["yes"]}]
        ),
    )
    assert report["total"] == 0
    report = explore(rows, dictionary, ExploreRequest(filters=[{"field": "b", "values": [None]}]))
    assert report["total"] == 0
    report = explore(rows, dictionary, ExploreRequest(field="n"))
    assert report["distribution"]["kind"] == "numeric"
    assert sum(item["count"] for item in report["distribution"]["counts"]) == 2
    assert report["distribution"]["missingCount"] == 1


@pytest.mark.parametrize(
    "values",
    [
        ["-1e308", "0", "1e308"],
        ["5e-324", "1e-323"],
        ["9007199254740992", "9007199254740993"],
        ["7", "7"],
    ],
)
def test_numeric_histogram_handles_extremes_subnormals_large_integers_and_constant_values(values):
    rows = [
        {"slideId": str(index), "patientId": None, "attributes": {"n": value}}
        for index, value in enumerate(values)
    ]
    report = explore(
        rows, [{"key": "n", "owner": "slide", "type": "decimal"}], ExploreRequest(field="n")
    )
    assert sum(item["count"] for item in report["distribution"]["counts"]) == len(values)
    assert len(report["distribution"]["counts"]) == min(10, len(set(values)))


@pytest.mark.parametrize("value", ["not-a-number", "NaN", "Infinity"])
def test_bad_numeric_chart_values_return_controlled_validation_error(value):
    with pytest.raises(StorageError) as error:
        explore(
            [{"slideId": "1", "patientId": None, "attributes": {"n": value}}],
            [{"key": "n", "owner": "slide", "type": "decimal"}],
            ExploreRequest(field="n"),
        )
    assert error.value.code == "INVALID_ATTRIBUTE"


def test_unlinked_patient_chart_and_full_population_target_values_are_explicit():
    rows = [
        {
            "slideId": str(index),
            "patientId": None,
            "attributes": {"label": "common" if index < 200 else "rare"},
        }
        for index in range(201)
    ]
    report = explore(
        rows,
        [{"key": "label", "owner": "patient", "type": "text"}],
        ExploreRequest(field="label", limit=1),
    )
    assert len(report["records"]) == 1
    assert report["distribution"]["total"] == 0
    assert report["distribution"]["unlinkedSlideCount"] == 201
    assert report["valueCounts"] == [
        {"value": "common", "count": 200},
        {"value": "rare", "count": 1},
    ]
    assert report["valueCountsUnit"] == "slide"
    assert not report["valuesTruncated"]


def test_category_and_cross_tab_truncation_account_for_the_whole_population():
    rows = [
        {
            "slideId": str(index),
            "patientId": str(index),
            "attributes": {"a": f"a{index:03}", "b": f"b{index:03}"},
        }
        for index in range(205)
    ]
    dictionary = [{"key": key, "owner": "slide", "type": "categorical"} for key in ("a", "b")]
    report = explore(rows, dictionary, ExploreRequest(field="a", compare="b", limit=1))
    distribution = report["distribution"]
    assert distribution["truncated"]
    assert distribution["otherCount"] == distribution["omittedCategoryCount"] == 105
    assert (
        sum(item["count"] for item in distribution["counts"]) + distribution["otherCount"]
        == distribution["total"]
    )
    assert report["crossTab"]["displayedCount"] + report["crossTab"]["omittedCount"] == 205
    assert report["crossTab"]["truncated"]
    assert len(report["valueCounts"]) == 200
    assert report["valuesTruncated"]


def test_patient_attribute_charts_distinguish_fallback_groups_from_known_patients():
    rows = [
        {
            "slideId": slide,
            "patientId": patient,
            "patientIdSource": source,
            "attributes": {"sex": sex, "age": age},
        }
        for slide, patient, source, sex, age in [
            ("s1", "p1", "source", "F", "60"),
            ("s2", "p1", "crosswalk", "F", "60"),
            ("s3", "s3", "slide_fallback", "F", "60"),
            ("s4", "s4", "slide_fallback", "M", "70"),
            ("s5", None, "unresolved", "M", "70"),
        ]
    ]
    dictionary = [{"key": key, "owner": "patient", "type": "categorical"} for key in ("sex", "age")]
    report = explore(rows, dictionary, ExploreRequest(field="sex", compare="age", limit=1))
    assert len(report["records"]) == 1
    assert report["summary"] == {
        "slideCount": 5,
        "mappedPatientCount": 3,
        "verifiedPatientCount": 1,
        "fallbackSlideCount": 2,
        "unlinkedSlideCount": 1,
        "groupCount": 3,
    }
    assert report["distribution"]["unit"] == "group"
    assert report["distribution"]["total"] == 3
    assert report["distribution"]["unlinkedSlideCount"] == 1
    assert report["distribution"]["counts"] == [
        {"value": "F", "count": 2},
        {"value": "M", "count": 1},
    ]
    assert report["crossTab"]["unit"] == "group"
    assert report["crossTab"]["total"] == 3
    assert report["crossTab"]["unlinkedSlideCount"] == 1
    assert report["crossTab"]["counts"] == [[2, 0], [0, 1]]

    known_only = explore(rows, dictionary, ExploreRequest(field="sex", compare="age", search="p1"))
    assert known_only["summary"]["verifiedPatientCount"] == 1
    assert known_only["summary"]["fallbackSlideCount"] == 0
    assert known_only["distribution"]["unit"] == "patient"
    assert known_only["crossTab"]["unit"] == "patient"
    assert known_only["distribution"]["total"] == 1


def test_slide_owned_charts_keep_slide_units_when_patient_ids_use_fallback():
    rows = [
        {
            "slideId": "s1",
            "patientId": "s1",
            "patientIdSource": "slide_fallback",
            "attributes": {"grade": "high", "sex": "F"},
        }
    ]
    dictionary = [
        {"key": "grade", "owner": "slide", "type": "categorical"},
        {"key": "sex", "owner": "patient", "type": "categorical"},
    ]
    report = explore(rows, dictionary, ExploreRequest(field="grade", compare="sex"))
    assert report["summary"]["verifiedPatientCount"] == 0
    assert report["summary"]["fallbackSlideCount"] == 1
    assert report["distribution"]["unit"] == "slide"
    assert report["crossTab"]["unit"] == "slide"
