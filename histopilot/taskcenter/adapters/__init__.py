"""Adapter registry. Adapter modules are imported lazily and cached per name."""

import importlib
import sys
import threading

from histopilot.taskcenter.adapters.base import Adapter, AdapterError, RunnerContext
from histopilot.taskcenter.adapters.generic import GenericAdapter

__all__ = [
    "ADAPTERS",
    "Adapter",
    "AdapterError",
    "GenericAdapter",
    "RunnerContext",
    "adapter",
    "preload",
]

ADAPTERS = {
    "generic": "histopilot.taskcenter.adapters.generic:GenericAdapter",
    "mil-fold": "histopilot.taskcenter.adapters.mil:MilFoldAdapter",
    "mil-collect": "histopilot.taskcenter.adapters.mil:MilCollectAdapter",
    "compute-job": "histopilot.taskcenter.adapters.compute:ComputeJobAdapter",
    "predictor-coordinator": "histopilot.taskcenter.adapters.coordinator:CoordinatorAdapter",
    "bulk-submit": "histopilot.taskcenter.adapters.generic:GenericAdapter",
    # Phase 3: preparation and archive records (Area D).
    "extraction": "histopilot.taskcenter.adapters.extraction:ExtractionAdapter",
    "extraction-validation": "histopilot.taskcenter.adapters.extraction:ExtractionValidationAdapter",
    "packing": "histopilot.taskcenter.adapters.packing:PackingAdapter",
    "archive": "histopilot.taskcenter.adapters.archive:ArchiveAdapter",
}

_CACHE: dict[str, Adapter] = {}
_LOCK = threading.Lock()


class UnavailableAdapter(Adapter):
    """Stands in for an adapter module that exists but cannot be imported.

    Classifying such tasks generically could misreport their outcome (a fold that exits 0
    has not necessarily succeeded), so every hook reports a transient failure instead:
    queued tasks wait and exits are retried once the module is fixed and the runner restarts.
    """

    def __init__(self, name: str, error: BaseException):
        self.message = f"The {name} task adapter cannot be loaded: {error}"

    def prepare(self, task, ctx):
        raise AdapterError(self.message)

    def on_started(self, task, identity, gpu, ctx):
        raise AdapterError(self.message)

    def on_exit(self, task, exit, ctx):
        raise AdapterError(self.message)

    def on_requeue(self, task, ctx):
        raise AdapterError(self.message)


def _warn(message: str) -> None:
    print(f"histopilot task center: {message}", file=sys.stderr, flush=True)


def _load(name: str) -> Adapter:
    target = ADAPTERS.get(name)
    if target is None:
        _warn(f"unknown task adapter {name!r}; using the generic adapter.")
        return GenericAdapter()
    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            _warn(f"task adapter {name!r} failed to import: {error}")
            return UnavailableAdapter(name, error)
        _warn(f"task adapter module {module_name} is missing; using the generic adapter.")
        return GenericAdapter()
    except Exception as error:  # any import failure must not stop the runner
        _warn(f"task adapter {name!r} failed to import: {error}")
        return UnavailableAdapter(name, error)
    factory = getattr(module, attribute, None)
    if factory is None:
        _warn(f"task adapter {target} is missing; using the generic adapter.")
        return GenericAdapter()
    return factory()


def adapter(name: str) -> Adapter:
    with _LOCK:
        instance = _CACHE.get(name)
        if instance is None:
            instance = _CACHE[name] = _load(name)
        return instance


def preload() -> dict[str, Adapter]:
    """Import every registered adapter so a running runner never picks up edited code."""
    return {name: adapter(name) for name in ADAPTERS}


def _reset() -> None:
    with _LOCK:
        _CACHE.clear()
