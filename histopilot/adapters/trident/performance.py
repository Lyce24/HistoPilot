"""Bounded extraction worker defaults without importing model dependencies."""

import os


def execution_device_count(options: dict) -> int:
    """Match TRIDENT's distinct GPUs and optional repeated CPU workers."""
    devices = options.get("gpus") or [options.get("gpu", 0)]
    return max(1, len({device for device in devices if device >= 0}) + devices.count(-1))


def usable_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_max_workers(options: dict) -> int:
    """Cap automatic loaders at eight total, sharing the budget across devices.

    Each device needs at least one worker because TRIDENT's CSV slide discovery
    also uses this setting for a thread pool. Explicit overrides remain intact.
    """
    explicit = options.get("max_workers")
    if explicit is not None:
        return explicit
    cpus = usable_cpu_count()
    devices = execution_device_count(options)
    total = min(8, max(1, int(cpus * 0.75)))
    # Preparation shares the training scheduler's conservative two-pool CPU
    # accounting. Fit that lease too, so automatic jobs can launch on small hosts.
    reservation_limit = max(1, (cpus - devices) // (2 * devices))
    return max(1, min(total // devices, reservation_limit))
