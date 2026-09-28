"""Adapter for opaque commands that own no HistoPilot record."""

from histopilot.taskcenter.adapters.base import Adapter


class GenericAdapter(Adapter):
    def can_requeue(self, task, ctx) -> bool:
        # Only commands that declare themselves safe to run again resume automatically.
        return bool((task.get("adapterData") or {}).get("requeueSafe"))
