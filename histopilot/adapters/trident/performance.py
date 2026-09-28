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


# Task Center sizing. Measured on an RTX A5000 (2026-09-25): HEST segmentation at batch 64
# reserves about 9.3 GiB. Feature extraction with a ViT-L class encoder at batch 64 stays
# below that; ViT-H/g class encoders (and 224-px resize of large patches) need more. These
# defaults only apply until the Task Center has measured an extraction with the same
# workload (see ``workload_key``); measured peaks then replace them.
DEFAULT_VRAM_GB = 10.0
SEGMENTER_VRAM_GB = {"hest": 9.5, "grandqc": 6.0, "otsu": 0.0}
ARTIFACT_REMOVAL_VRAM_GB = 3.0
LARGE_ENCODERS = frozenset(
    {
        "uni_v2",
        "virchow",
        "virchow2",
        "virchow2-cls",
        "gigapath",
        "gigapath-flash",
        "hoptimus0",
        "hoptimus1",
        "musk",
        "midnight12k",
        "openmidnight",
        "genbio-pathfm",
        "gemma4-e4b",
        "gemma4-26b",
    }
)
ENCODER_VRAM_GB = {"large": 9.0, "default": 5.0}
REFERENCE_BATCH = 64
MIN_VRAM_GB = 2.0
# Feature widths used for the output-size estimate; unknown encoders count as 1536.
ENCODER_DIMENSIONS = {
    "conch_v1": 512,
    "conch_v15": 768,
    "uni_v1": 1024,
    "uni_v2": 1536,
    "ctranspath": 768,
    "phikon": 768,
    "phikon_v2": 1024,
    "resnet50": 1024,
    "keep": 768,
    "gigapath": 1536,
    "virchow": 2560,
    "virchow2": 2560,
    "virchow2-cls": 1280,
    "hoptimus0": 1536,
    "hoptimus1": 1536,
    "h0-mini": 768,
    "musk": 2048,
    "hibou_l": 1024,
    "lunit-vits8": 384,
    "kaiko-vits8": 384,
    "kaiko-vits16": 384,
    "kaiko-vitb8": 768,
    "kaiko-vitb16": 768,
    "kaiko-vitl14": 1024,
}
DEFAULT_DIMENSION = 1536
# Bytes of float32 patch features per byte of compressed slide input at 20x / 256 px with a
# 1024-wide encoder. Measured on gej_v2 (1,111 SDPC slides, UNI v1): 32.6 GB of features
# from 155.8 GiB of slides, about 0.2; TCGA SVS runs land near 0.1. 0.25 keeps a margin.
FEATURE_BYTES_PER_INPUT_BYTE = 0.25
PER_SLIDE_OVERHEAD_BYTES = 2 * 1024 * 1024  # contours, geojson, thumbnail, coordinates


def _batch(options: dict, key: str) -> int:
    value = options.get(key) or options.get("batch_size") or REFERENCE_BATCH
    return max(1, int(value))


def uses_gpu(options: dict) -> bool:
    devices = options.get("gpus") or [options.get("gpu", 0)]
    return any(device >= 0 for device in devices)


def estimate_vram_gb(options: dict) -> float:
    """Default GPU memory request for one TRIDENT run on one GPU (stages run in turn)."""
    if not uses_gpu(options):
        return 0.0
    task = options.get("task") or "all"
    needs = []
    if task in {"seg", "all"}:
        segmenter = SEGMENTER_VRAM_GB.get(options.get("segmenter") or "hest", DEFAULT_VRAM_GB)
        segmentation = segmenter * _batch(options, "seg_batch_size") / REFERENCE_BATCH
        if options.get("remove_artifacts") or options.get("remove_penmarks"):
            segmentation = max(segmentation, ARTIFACT_REMOVAL_VRAM_GB)
        needs.append(segmentation)
    if task in {"feat", "all"}:
        encoder = options.get("slide_encoder") or options.get("patch_encoder") or ""
        base = ENCODER_VRAM_GB["large" if encoder in LARGE_ENCODERS else "default"]
        size = options.get("patch_encoder_img_size")
        scale = (size / 224) ** 2 if size else 1.0
        needs.append(base * scale * _batch(options, "feat_batch_size") / REFERENCE_BATCH)
    return round(max(MIN_VRAM_GB, *needs) if needs else MIN_VRAM_GB, 2)


def estimate_ram_gb(options: dict) -> float:
    """Private RAM for one TRIDENT run: the model process plus its slide-reading workers."""
    workers = resolve_max_workers(options) or 0
    return round(4.0 + 0.5 * workers, 2)


def workload_key(options: dict) -> str:
    """Task Center measurement key: settings that change peak GPU memory, nothing else."""
    keys = (
        "task",
        "segmenter",
        "remove_artifacts",
        "remove_penmarks",
        "seg_batch_size",
        "patch_encoder",
        "patch_encoder_img_size",
        "slide_encoder",
        "feat_batch_size",
        "batch_size",
        "patch_size",
    )
    return "trident:" + ",".join(f"{key}={options.get(key)}" for key in keys)


def estimate_output_bytes(options: dict, input_bytes: int, slide_count: int) -> int:
    """Approximate bytes a run writes for ``slide_count`` slides of ``input_bytes`` in total."""
    task = options.get("task") or "all"
    total = slide_count * PER_SLIDE_OVERHEAD_BYTES
    if task in {"feat", "all"}:
        encoder = options.get("slide_encoder")
        if encoder:
            total += slide_count * 64 * 1024  # one embedding per slide
        else:
            encoder = options.get("patch_encoder") or ""
            dimension = ENCODER_DIMENSIONS.get(encoder, DEFAULT_DIMENSION)
            patch = max(1, int(options.get("patch_size") or 256))
            stride = max(1, patch - int(options.get("overlap") or 0))
            density = (float(options.get("mag") or 20.0) / 20.0) ** 2 * (256 / stride) ** 2
            total += int(input_bytes * FEATURE_BYTES_PER_INPUT_BYTE * dimension / 1024 * density)
    return int(total)
