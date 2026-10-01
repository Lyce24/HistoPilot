"""The browser's default choices, ported: shared cases, composition, and the CLI's use of them.

`web/src/lib/resolverCases.json` is also run by `web/src/lib/resolvers.test.ts` against the
browser's own functions, so both sides propose the same defaults.
"""

import json
from pathlib import Path

import pytest
from support import projects as fixtures
from support.cli import Service
from typer.testing import CliRunner

from histopilot import resolvers, templates
from histopilot.cli import app
from histopilot.client import ClientError
from histopilot.client.resolve import configuration_of
from histopilot.client.transport import Response
from histopilot.storage.scientific import ScientificStore

CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "web/src/lib/resolverCases.json").read_text("utf-8")
)
PREDICTORS = CASES["predictors"]


# Shared cases ------------------------------------------------------------------------------


def test_predictors_and_methods_match_the_browser():
    for case in CASES["experimentPredictors"]:
        chosen = resolvers.experiment_predictors(PREDICTORS, case["experimentIds"], case["method"])
        assert [item["id"] for item in chosen] == case["expected"], case
    for case in CASES["defaultMethod"]:
        method = resolvers.default_method(
            PREDICTORS, case["experimentIds"], case["linkedPredictor"] or None
        )
        assert method == case["expected"], case


def test_cohorts_and_inputs_match_the_browser():
    for case in CASES["reservedTestingCohort"]:
        found = resolvers.reserved_testing_cohort(
            case["protocolIds"], CASES["protocols"], CASES["cohorts"]
        )
        assert (found["id"] if found else None) == case["expected"], case
    for case in CASES["initialInputs"]:
        inputs = resolvers.initial_inputs(case["cohort"])
        assert inputs == case["expected"]["inputs"]
        assert resolvers.execution_selection(inputs) == case["expected"]["selection"]
        assert resolvers.cohort_labeled(case["cohort"]) is case["expected"]["labeled"]


def test_configuration_choices_match_the_browser():
    for case in CASES["configurationChoice"]:
        plan = resolvers.configuration_choice(
            CASES["seedEnsembleChoices"],
            CASES["configurationPredictors"],
            case["experiment"],
            case["batchId"],
            case["candidateId"],
        )
        assert plan == case["expected"], case


def test_model_descriptions_match_the_browser():
    shared = CASES["modelDescription"]
    name = resolvers.batch_names(shared["experiments"])
    for case in shared["cases"]:
        assert resolvers.model_description(case["manifest"], name) == case["expected"], case


def test_targets_and_labels_match_the_browser():
    for case in CASES["inferTarget"]:
        assert resolvers.infer_target(case["values"], case["truncated"]) == case["expected"]
    # Cases with a linked or named reference exercise browser-only arguments.
    for case in CASES["labelSources"]:
        if case["linked"] is not None:
            continue
        sources = resolvers.label_sources(case["run"], CASES["standards"])
        assert sources == case["expected"], case
        if case["reference"] is None:
            chosen = resolvers.label_source(sources)
            assert (chosen["id"] if chosen else "none") == case["expectedSource"], case


def test_comparison_arms_and_feature_specs_match_the_browser():
    for case in CASES["withArmModel"]:
        assert resolvers.with_arm_model(case["base"], case["model"]) == case["expected"]
    for case in CASES["ablationArms"]:
        arms = resolvers.ablation_arms(case["base"], case["models"], case["inputs"])
        assert arms == case["expected"], case
    for case in CASES["featureSpecFromExtraction"]:
        spec = resolvers.feature_spec_from_extraction(case["job"])
        assert spec == case["expected"], case


# Composition -------------------------------------------------------------------------------


def test_apply_selection_proposes_what_apply_models_does():
    cohorts = [
        {
            "id": "cohort-1",
            "manifest": {
                "spec": {
                    "sourceTargetSplitId": "split-1",
                    "featureBundleId": "bundle-1",
                    "inference": {"batchSize": 8},
                }
            },
        },
        {"id": "unlabeled", "manifest": {"spec": {"purpose": "inference"}}},
    ]
    proposed = resolvers.apply_selection(PREDICTORS, CASES["protocols"], cohorts, ["exp-a"])
    assert proposed["method"] == "seed_ensemble" and proposed["methodSource"] == "default"
    assert proposed["selection"]["predictorIds"] == ["p-seed-1"]
    assert (proposed["cohortSource"], proposed["selection"]["cohortId"]) == ("reserved", "cohort-1")
    selection = proposed["selection"]
    assert selection["featureBundleId"] == "bundle-1" and selection["inference"]["batchSize"] == 8
    assert selection["inference"]["patientAggregation"] == "predictor"
    assert (selection["scope"], selection["namePrefix"]) == ("selected", "Evaluation")
    assert proposed["findings"] == []

    both = resolvers.apply_selection(
        PREDICTORS, CASES["protocols"], cohorts, ["exp-a"], method="both", cohort_id="unlabeled"
    )
    assert both["selection"]["predictorIds"] == ["p-ens-1", "p-legacy", "p-refit-1"]
    assert both["selection"]["namePrefix"] == "Inference" and both["cohortLabeled"] is False
    assert "featureBundleId" not in both["selection"]

    linked = resolvers.apply_selection(
        PREDICTORS, CASES["protocols"], cohorts, ["exp-a"], linked_predictor="p-refit-1"
    )
    assert (linked["method"], linked["selection"]["predictorIds"]) == ("refit", ["p-refit-1"])

    empty = resolvers.apply_selection(PREDICTORS, CASES["protocols"], [], ["exp-b"])
    codes = [item["code"] for item in empty["findings"]]
    assert codes == ["COHORT_REQUIRED"] and empty["selection"]["cohortId"] == ""
    missing = resolvers.apply_selection(PREDICTORS, [], cohorts, [], cohort_id="cohort-1")
    assert [item["code"] for item in missing["findings"]] == ["EXPERIMENTS_REQUIRED"]

    many = [
        {"id": f"p{i}", "lifecycleState": "active", "manifest": {"experimentId": "big"}}
        for i in range(resolvers.MAX_APPLY_PREDICTORS + 1)
    ]
    too_many = resolvers.apply_selection(many, [], cohorts, ["big"], cohort_id="cohort-1")
    assert [item["code"] for item in too_many["findings"]] == ["TOO_MANY_PREDICTORS"]


def test_comparisons_targets_and_partitions():
    recipe = templates.recipe()
    arms = resolvers.ablation_arms(recipe, ["nnmil"], ["image"])
    batch = resolvers.comparison_batch({"batchName": "B"}, arms)
    assert batch["mode"] == "explicit" and batch["candidateSelection"] == "all"
    assert batch["comparison"] == {"reference": 1, "primaryMetric": "auroc"}
    with pytest.raises(ValueError, match="2 to 8"):
        resolvers.comparison_batch({}, [recipe])
    with pytest.raises(ValueError, match="clinical fields"):
        resolvers.matched_input_recipes(recipe)

    spec = {
        "datasetId": "ds",
        "splitUnit": "slide",
        "eligibility": [],
        "split": {"method": "random"},
        "target": {"field": "label", "task": "", "classes": []},
        "testTarget": {"field": "other", "labels": {"x": "y"}, "task": ""},
    }
    assert resolvers.partition_request(spec) == {
        "datasetId": "ds",
        "splitUnit": "slide",
        "eligibility": [],
        "split": {"method": "random"},
        "targetFields": {"train": "label", "test": "other"},
    }
    assert resolvers.partition_request({**spec, "testTarget": None})["testTarget"] is None
    inherited = {key: value for key, value in spec.items() if key != "testTarget"}
    assert resolvers.partition_request(inherited)["targetFields"] == {
        "train": "label",
        "test": "label",
    }
    changed = resolvers.training_target(
        spec,
        {"field": "label", "task": "binary_classification", "unit": "slide", "classes": ["a", "b"]},
    )
    assert changed["testTarget"]["classes"] == ["a", "b"]
    assert changed["testTarget"]["labels"] == {"x": "y"}


# The CLI ------------------------------------------------------------------------------------

PROJECT = "project-1"


class Answers:
    """A fake service: answers by method and path (query ignored); records what was sent."""

    def __init__(self, table):
        self.table = table
        self.sent = []

    def send(self, method, path, *, headers, body, timeout):
        if path.endswith("/session"):
            return Response(200, json.dumps({"token": "t"}).encode(), {})
        route = path.split("?")[0].removeprefix(f"/api/v1/projects/{PROJECT}")
        self.sent.append((method, route, json.loads(body) if body else None))
        if (method, route) in self.table:
            answer = self.table[(method, route)]
            return Response(200, json.dumps(answer).encode(), {"content-type": "application/json"})
        return Response(404, json.dumps({"detail": "No.", "code": "NOT_HERE"}).encode(), {})


@pytest.fixture
def answers(monkeypatch):
    from histopilot.commands import common

    table: dict = {}
    fake = Answers(table)
    monkeypatch.setattr(common, "TRANSPORT_FACTORY", lambda _url: fake)
    return fake


def cli(*arguments):
    result = CliRunner().invoke(app, [*arguments, "--project", PROJECT, "--json"])
    return result.exit_code, json.loads(result.stdout)


def test_apply_template_is_filled_in_as_apply_models_proposes(answers):
    answers.table.update(
        {
            ("GET", "/predictors"): {"items": PREDICTORS},
            ("GET", "/configurations"): {"configurations": CASES["protocols"]},
            ("GET", "/evaluation-cohorts"): {"items": CASES["cohorts"]},
        }
    )
    code, envelope = cli("apply", "template", "--experiment", "exp-a")
    assert code == 0, envelope
    spec = envelope["data"]
    assert spec["predictorIds"] == ["p-seed-1"] and spec["cohortId"] == "cohort-1"
    notes = {item["code"]: item["message"] for item in envelope["warnings"]}
    assert "seed ensembles are ready" in notes["APPLY_METHOD"]
    assert "reserved" in notes["APPLY_COHORT"]
    code, envelope = cli("apply", "template", "--predictor", "p-refit-1")
    assert code == 0 and envelope["data"]["predictorIds"] == ["p-refit-1"]
    assert cli("apply", "template", "--method", "refit")[0] == 2
    assert cli("apply", "template", "--experiment", "exp-a", "--method", "best")[0] == 2
    # Without a choice, the starter keeps its placeholders.
    assert cli("apply", "template")[1]["data"]["cohortId"] == "@cohort-tag"


def test_apply_config_writes_the_spec_or_builds_the_seed_ensemble_first(answers, tmp_path):
    answers.table.update(
        {
            ("GET", "/model-experiments/exp-a"): {"id": "exp-a", "name": "Exp A"},
            ("GET", "/predictors/seed-ensembles"): {"items": CASES["seedEnsembleChoices"]},
            ("GET", "/predictors"): {"items": CASES["configurationPredictors"]},
            ("GET", "/configurations"): {"configurations": []},
            ("GET", "/evaluation-cohorts"): {"items": CASES["cohorts"]},
            ("POST", "/predictors/seed-ensembles/preview"): {
                "canFreeze": True,
                "previewHash": "h" * 64,
                "findings": [],
            },
            ("POST", "/predictors/seed-ensembles"): {
                "id": "seed-new",
                "lifecycleState": "active",
                "manifest": {"experimentId": "exp-a", "method": "seed_ensemble"},
            },
        }
    )
    code, envelope = cli(
        "experiment", "apply-config", "exp-a", "--batch", "b1", "--candidate", "c3"
    )
    assert code == 0, envelope
    assert envelope["data"]["predictorId"] == "seed-built"
    assert envelope["data"]["spec"]["predictorIds"] == ["seed-built"]

    code, envelope = cli(
        "experiment", "apply-config", "exp-a", "--batch", "b1", "--candidate", "c5"
    )
    assert code == 7 and envelope["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert not any(
        route == "/predictors/seed-ensembles" and method == "POST"
        for method, route, _ in answers.sent
    )
    # Once built, the new predictor is applied; its registry entry arrives with the next read.
    answers.table[("GET", "/predictors")] = {
        "items": [
            *CASES["configurationPredictors"],
            {
                "id": "seed-new",
                "lifecycleState": "active",
                "manifest": {"experimentId": "exp-a", "method": "seed_ensemble"},
            },
        ]
    }
    target = tmp_path / "apply.yaml"
    code, envelope = cli(
        "experiment",
        "apply-config",
        "exp-a",
        "--batch",
        "b1",
        "--candidate",
        "c5",
        "--yes",
        "-o",
        str(target),
    )
    assert code == 0, envelope
    assert envelope["data"]["builtPredictor"]["id"] == "seed-new"
    assert envelope["data"]["spec"]["predictorIds"] == ["seed-new"]
    assert target.read_text().startswith("# ") and "predictorIds:\n- seed-new" in target.read_text()
    built = [
        body
        for method, route, body in answers.sent
        if (method, route) == ("POST", "/predictors/seed-ensembles")
    ]
    assert built[0]["name"] == "Exp A · configuration 5 · seed ensemble"
    assert built[0]["previewHash"] == "h" * 64

    code, envelope = cli(
        "experiment", "apply-config", "exp-a", "--batch", "b1", "--candidate", "c2"
    )
    assert code == 3 and envelope["error"]["code"] == "CONFIGURATION_AMBIGUOUS"
    assert envelope["data"]["plan"]["predictorIds"] == ["fold-2", "fold-3"]
    code, envelope = cli(
        "experiment", "apply-config", "exp-a", "--batch", "b1", "--candidate", "c6"
    )
    assert code == 3 and envelope["error"]["code"] == "CONFIGURATION_NOT_READY"


def test_unlabeled_runs_are_scored_against_the_first_fitting_reference(answers):
    unlabeled = CASES["labelSources"][3]["run"]
    answers.table.update(
        {
            ("GET", f"/evaluation-runs/{unlabeled['id']}"): unlabeled,
            ("GET", "/reference-standards"): {"items": CASES["standards"]},
            ("POST", f"/evaluation-runs/{unlabeled['id']}/scores"): {"auroc": 0.8},
        }
    )
    code, envelope = cli("run", "metrics", unlabeled["id"])
    assert code == 0, envelope
    scored = [body for method, route, body in answers.sent if route.endswith("/scores")]
    assert scored == [{"referenceId": "ref-a"}]
    assert envelope["warnings"][0]["code"] == "REFERENCE_DEFAULT"
    code, envelope = cli("run", "label-sources", unlabeled["id"])
    rows = envelope["data"]
    assert [row["id"] for row in rows] == ["ref-a", "ref-same-name", "ref-b"]
    assert [row["default"] for row in rows] == [True, False, False]


def test_features_template_from_an_extraction(answers):
    job = CASES["featureSpecFromExtraction"][0]["job"]
    answers.table[("GET", f"/extractions/{job['id']}")] = job
    answers.table[("GET", "/extractions/extract-running")] = CASES["featureSpecFromExtraction"][3][
        "job"
    ]
    code, envelope = cli("features", "template", "--from-extraction", job["id"])
    assert code == 0, envelope
    spec = envelope["data"]
    assert spec["path"] == "/out/features_uni_v2" and spec["recursive"] is False
    assert spec["encoderId"] == "uni_v2" and spec["sourceExtractionJobId"] == job["id"]
    code, envelope = cli("features", "template", "--from-extraction", "extract-running")
    assert code == 3 and envelope["error"]["code"] == "EXTRACTION_NOT_READY"


def test_experiment_template_builds_a_controlled_comparison(answers):
    code, envelope = cli(
        "experiment",
        "template",
        "--compare-model",
        "abmil",
        "--compare-model",
        "nnmil",
        "--compare-input",
        "image",
        "--compare-input",
        "clinical",
        "--clinical-field",
        "age:numeric",
    )
    assert code == 0, envelope
    batch = envelope["data"]["batches"][0]
    # The service omits a default input mode, image, from saved recipes.
    arms = [(arm["model"], arm.get("inputMode", "image")) for arm in batch["configurations"]]
    assert arms == [
        ("abmil", "multimodal"),
        ("abmil", "image"),
        ("nnmil", "image"),
        ("abmil", "clinical"),
    ]
    assert batch["recipe"] == batch["configurations"][0]
    assert batch["comparison"] == {"reference": 1, "primaryMetric": "auroc"}
    assert cli("experiment", "template", "--compare-model", "resnet")[0] == 2
    assert cli("experiment", "template", "--clinical-field", "age")[0] == 2
    assert cli("experiment", "template", "--compare-input", "image")[0] == 2


def test_targets_infer_reads_the_fields_training_values(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project("Targets")
        store = ScientificStore(service.settings.workspace / "projects" / "Targets", project)
        dataset, _ = fixtures.dataset(store)
        written = tmp_path / "targets.yaml"
        service.cli("targets", "template", "-o", str(written), "--project", project)
        text = (
            written.read_text()
            .replace("'@dataset-tag'", dataset["id"])
            .replace("@dataset-tag", dataset["id"])
        )
        written.write_text(text)
        outcome = service.cli(
            "targets", "infer", str(written), "--field", "label", "--project", project, "--json"
        )
    assert outcome.code == 0, outcome.stdout
    target = outcome.envelope["data"]["target"]
    assert target["field"] == "label" and target["task"] == "binary_classification"
    assert sorted(target["classes"]) == ["0", "1"] and target["labels"] == {"0": "0", "1": "1"}
    assert target["positiveClass"] is None
    assert outcome.envelope["warnings"][0]["code"] == "POSITIVE_CLASS_REQUIRED"
    assert outcome.envelope["data"]["datasetId"] == dataset["id"]


# Exports -----------------------------------------------------------------------------------


def test_the_results_csv_matches_the_browser_byte_for_byte():
    from histopilot import exports

    for case in CASES["resultsCsv"]:
        assert exports.csv_text(exports.result_rows(case["results"])) == case["expected"]
    samples = {
        1.0: "1",
        0.5: "0.5",
        5e-05: "0.00005",
        0.000001: "0.000001",
        1e-07: "1e-7",
        1e21: "1e+21",
        1e20: "100000000000000000000",
        123.45: "123.45",
        -2.5e-08: "-2.5e-8",
        0.1 + 0.2: "0.30000000000000004",
        float("nan"): "NaN",
    }
    assert {value: exports.js_number(value) for value in samples} == samples


def test_export_results_writes_the_csv(answers, tmp_path):
    answers.table[("GET", "/model-experiments/exp-a/results")] = CASES["resultsCsv"][0]["results"]
    target = tmp_path / "folds.csv"
    code, envelope = cli("experiment", "export-results", "exp-a", "-o", str(target))
    assert code == 0, envelope
    assert target.read_text(encoding="utf-8") == CASES["resultsCsv"][0]["expected"]
    assert envelope["data"]["rows"] == 4
    assert cli("experiment", "export-results", "exp-a", "-o", str(target))[0] == 2


def test_configurations_are_named_by_batch_name_and_number():
    def batch(identity, name, *candidates):
        rows = [{"id": candidate, "number": number} for number, candidate in candidates]
        return {"id": identity, "name": name, "manifest": {"configurations": rows}}

    experiment = {
        "name": "Study",
        "batches": [batch("b-1", "Baseline", (1, "c-1"), (2, "c-2")), batch("b-2", "Other")],
    }
    assert configuration_of(experiment, "b-1", "c-2") == ("b-1", "c-2")
    assert configuration_of(experiment, "baseline", "2") == ("b-1", "c-2")
    # A name or number the experiment lacks is refused with the ones it has.
    with pytest.raises(ClientError) as batch_missing:
        configuration_of(experiment, "Base", "1")
    assert batch_missing.value.code == "BATCH_NOT_IN_EXPERIMENT"
    assert "Baseline (b-1)" in batch_missing.value.message
    with pytest.raises(ClientError) as number_missing:
        configuration_of(experiment, "Baseline", "3")
    assert number_missing.value.code == "CONFIGURATION_NOT_IN_BATCH"
    assert "2 (c-2)" in number_missing.value.message
    shared = {**experiment, "batches": [*experiment["batches"], batch("b-3", "BASELINE")]}
    with pytest.raises(ClientError) as ambiguous:
        configuration_of(shared, "Baseline", "1")
    assert ambiguous.value.code == "BATCH_AMBIGUOUS"
