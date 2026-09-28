"""Feature jobs that evaluation fixtures need, run through this test's Task Center.

A feature validation or pack is queued as a ``packing`` task, as in production, and
``support.features.run_pack`` then takes it through the runner's steps with the worker
running in this process. The job reads exactly as one the runner finished.
"""

from histopilot.application.feature_packs import FeaturePackService
from support.features import run_pack
from support.task_center import Center


def packing_service(store, filesystem):
    """A feature job service that queues into this test's private Task Center."""
    return FeaturePackService(store, filesystem, task_center=Center().client)


def run_packing(service, job):
    """Run a queued feature job to its conclusion and return the worker's receipt."""
    return run_pack(service.tasks.client.store, job)


def compute_tasks(center):
    """The compute jobs (evaluations, inference, refits) queued in ``center``.

    Fixtures validate their features through the Task Center too, so a test counting its
    launches counts only these.
    """
    return center.tasks(kind="compute-job")
