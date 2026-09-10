"""Concurrent browser reads must not hang SQLite connection lifecycle operations."""

import subprocess
import sys
from pathlib import Path


def test_concurrent_scientific_reads_finish_and_queries_can_overlap(tmp_path):
    # A native SQLite deadlock cannot be interrupted by a Future timeout. Keep
    # this regression in a bounded child so a failure cannot hang the test suite.
    program = r"""
import faulthandler
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
import sys

from histopilot.storage.scientific import ScientificStore

faulthandler.dump_traceback_later(15, exit=True)
folder = Path(sys.argv[1])
folder.mkdir()
store = ScientificStore(folder, "project-concurrency")
store.initialize()
draft = store.create_draft("import", "Dataset", {})
dataset = store.publish_dataset(
    draft["id"], expected_revision=1, manifest={"kind": "dataset"},
    artifacts={"records.json": b'[{"slideId":"slide-1"}]'}, operation_id="dataset",
)
configuration = store.publish_configuration(
    manifest={
        "kind": "protocol", "datasetId": dataset["id"],
        "memberships": [
            {"slideId": f"slide-{i}", "patientId": f"patient-{i}",
             "partition": "train", "planId": "seed:42/outer:0/inner:0", "phase": "inner"}
            for i in range(1500)
        ],
    }, operation_id="protocol",
)

# Each request resolves a separate store instance, as the real API does.
def read(index):
    current = ScientificStore(folder, "project-concurrency")
    operation = index % 5
    if operation == 0:
        assert current.list_configurations()[0] == configuration
    elif operation == 1:
        assert current.list_datasets() == [dataset]
    elif operation == 2:
        assert current.status()["configurationCount"] == 1
    elif operation == 3:
        assert current.read_artifact(dataset["id"], "records.json") == b'[{"slideId":"slide-1"}]'
    else:
        assert current.get_dataset(dataset["id"]) == dataset

with ThreadPoolExecutor(max_workers=12) as pool:
    list(pool.map(read, range(200)))

# The lifecycle guard must not serialize query bodies or whole transactions.
barrier = Barrier(2, timeout=3)
def overlapping_query(_):
    current = ScientificStore(folder, "project-concurrency")
    with current._connection() as connection:
        barrier.wait()
        assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 1
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(overlapping_query, range(2)))
faulthandler.cancel_dump_traceback_later()
print("concurrent reads and overlapping queries passed")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "experiment")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "concurrent reads and overlapping queries passed" in completed.stdout
