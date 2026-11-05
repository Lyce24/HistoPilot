"""scripts/ci_test_report.py summarizes each CI test leg and fails the training leg when a test
skipped for want of Torch, Lightning or scikit-learn."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
script = runpy.run_path(str(ROOT / "scripts" / "ci_test_report.py"))

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="7">
  <testcase classname="tests.test_mil_parity" name="test_clam[sb]" time="0.1"/>
  <testcase classname="tests.test_mil_parity" name="test_dsmil" time="0.1"/>
  <testcase classname="tests.test_oceanpath_parity" name="test_forward[small]" time="0.1"/>
  <testcase classname="tests.test_api" name="test_health" time="0.1"/>
  <testcase classname="tests.test_api" name="test_broken" time="0.1"><failure message="assert 1 == 2"/></testcase>
  <testcase classname="tests.test_gpu" name="test_amp" time="0"><skipped type="pytest.skip" message="CUDA unavailable"/></testcase>
  {extra}
</testsuite></testsuites>
"""
TORCH_SKIP = (
    '<testcase classname="tests.test_models" name="tests/test_models.py" time="0">'
    "<skipped type=\"pytest.skip\" message=\"could not import 'torch': No module named 'torch'\"/>"
    "</testcase>"
)
MARKER_SKIP = (
    '<testcase classname="tests.test_task_center_training" name="test_fold" time="0">'
    '<skipped type="pytest.skip" message="runs real training workers: needs the training extra"/>'
    "</testcase>"
)


def test_the_summary_counts_outcomes_skip_reasons_and_parity_tests(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text(JUNIT.format(extra=""))
    text, missing = script["report"](script["cases"](str(report)), "training", "with Torch, on CPU")
    assert missing == []
    assert text.splitlines()[:3] == [
        "### Tests, with Torch, on CPU",
        "",
        "4 passed, 1 failed, 1 skipped.",
    ]
    assert "| CUDA unavailable | 1 |" in text
    assert "- 2 passed: MIL models against the authors' implementations" in text
    assert "- 1 passed: Mamba-2 MIL against OceanPath's fused kernels" in text


def test_the_training_leg_fails_on_a_test_skipped_for_want_of_torch(tmp_path, capsys):
    report = tmp_path / "pytest.xml"
    report.write_text(JUNIT.format(extra=TORCH_SKIP + MARKER_SKIP))
    assert script["main"]([str(report), "--leg", "training"]) == 1
    out = capsys.readouterr().out
    assert "**2 test(s) skipped for want of the training extra, which this leg installs:**" in out
    assert "`tests.test_models::tests/test_models.py`: could not import 'torch'" in out
    assert "`tests.test_task_center_training::test_fold`: runs real training workers" in out


def test_the_lean_leg_only_reports_those_skips(tmp_path, capsys):
    report = tmp_path / "pytest.xml"
    report.write_text(JUNIT.format(extra=TORCH_SKIP + MARKER_SKIP))
    assert script["main"]([str(report), "--leg", "lean"]) == 0
    assert "skipped for want of the training extra" not in capsys.readouterr().out


def test_a_cuda_skip_never_counts_as_wanting_the_extra():
    case = script["Case"]("tests.test_gpu", "test_amp", "skipped", "torch: CUDA unavailable")
    assert not script["wants_training"](case)
