"""Typed batch CLI settings, checked against official TRIDENT on 2026-09-10."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SOURCE = "https://github.com/mahmoodlab/TRIDENT/blob/main/run_batch_of_slides.py"
PATCH_ENCODERS = (
    "conch_v1",
    "conch_v15",
    "uni_v1",
    "uni_v2",
    "ctranspath",
    "phikon",
    "phikon_v2",
    "resnet50",
    "keep",
    "gigapath",
    "gigapath-flash",
    "virchow",
    "virchow2",
    "virchow2-cls",
    "hoptimus0",
    "hoptimus1",
    "h0-mini",
    "musk",
    "openmidnight",
    "gpfm",
    "hibou_l",
    "kaiko-vitb8",
    "kaiko-vitb16",
    "kaiko-vits8",
    "kaiko-vits16",
    "kaiko-vitl14",
    "lunit-vits8",
    "midnight12k",
    "phaet",
    "mascaret",
    "genbio-pathfm",
    "gemma4-e4b",
    "gemma4-26b",
)
SLIDE_ENCODERS = (
    "threads",
    "titan",
    "prism",
    "prism2",
    "chief",
    "gigapath",
    "gigapath-flash",
    "madeleine",
    "feather",
    "feather_uni_v2",
    "care",
    "abmil",
    "mean-conch_v1",
    "mean-conch_v15",
    "mean-uni_v1",
    "mean-uni_v2",
    "mean-ctranspath",
    "mean-phikon",
    "mean-resnet50",
    "mean-gigapath",
    "mean-gigapath-flash",
    "mean-virchow",
    "mean-virchow2",
    "mean-virchow2-cls",
    "mean-hoptimus0",
    "mean-phikon_v2",
    "mean-phaet",
    "mean-mascaret",
    "mean-musk",
    "mean-hibou_l",
    "mean-kaiko-vit8s",
    "mean-kaiko-vit16s",
    "mean-kaiko-vit8b",
    "mean-kaiko-vit16b",
    "mean-kaiko-vit14l",
)
RESIZE_PATCH_STRIDES = {
    "uni_v1": 16,
    "uni_v2": 14,
    "virchow": 14,
    "virchow2": 14,
    "virchow2-cls": 14,
    "kaiko-vitb8": 8,
    "kaiko-vitb16": 16,
    "kaiko-vits8": 8,
    "kaiko-vits16": 16,
    "kaiko-vitl14": 14,
    "gigapath": 16,
    "gigapath-flash": 16,
    "hoptimus0": 14,
    "hoptimus1": 14,
    "gpfm": 14,
    "lunit-vits8": 8,
    "h0-mini": 14,
}


def option(default, *, group, description, advanced=True, **kwargs):
    return Field(
        default=default,
        description=description,
        json_schema_extra={"group": group, "advanced": advanced},
        **kwargs,
    )


class TridentOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    task: Literal["seg", "coords", "feat", "all"] = option(
        "all",
        group="execution",
        advanced=False,
        description="Run segmentation, coordinates, features, or all three in order.",
    )
    gpu: int = option(
        0,
        group="execution",
        ge=-1,
        description="Legacy single GPU index; -1 uses CPU. A GPU list takes precedence.",
    )
    gpus: list[Annotated[int, Field(ge=-1)]] | None = option(
        None,
        group="execution",
        min_length=1,
        max_length=64,
        description="GPU IDs; use -1 for CPU. Multiple IDs split slides between workers.",
    )
    skip_errors: bool = option(
        False,
        group="execution",
        description="Continue after slide errors. Completion still reports missing outputs.",
    )
    clear_dead_locks: bool = option(
        False,
        group="execution",
        description="Remove stale TRIDENT lock files in this output directory before running.",
    )
    dead_lock_max_age_hours: float = option(
        24.0, group="execution", gt=0, description="Age threshold in hours for stale output locks."
    )
    max_workers: int | None = option(
        None,
        group="execution",
        ge=0,
        description="Maximum data-loader workers; 0 uses the main process.",
    )
    batch_size: int = option(
        64,
        group="execution",
        gt=0,
        description="Shared segmentation and feature batch size unless overridden below.",
    )
    wsi_cache: str | None = option(
        None,
        group="slides",
        min_length=1,
        max_length=4096,
        description="Local SSD cache parent. Each run uses its own disposable subdirectory.",
    )
    cache_batch_size: int = option(
        32, group="slides", gt=0, description="Maximum slides copied into the local cache at once."
    )
    wsi_ext: list[str] | None = option(
        None,
        group="slides",
        min_length=1,
        max_length=100,
        description="Allowed WSI extensions, such as .svs and .tiff.",
    )
    custom_mpp_keys: list[str] | None = option(
        None,
        group="slides",
        min_length=1,
        max_length=100,
        description="Slide metadata keys containing microns per pixel.",
    )
    custom_list_of_wsis: str | None = option(
        None,
        group="slides",
        min_length=1,
        max_length=4096,
        description="Optional TRIDENT CSV with wsi and optional mpp columns; restricted to the selected dataset.",
    )
    reader_type: Literal["openslide", "image", "cucim", "sdpc", "omezarr", "czi"] | None = option(
        None, group="slides", description="Force a reader, or leave blank for automatic detection."
    )
    search_nested: bool = option(
        False,
        group="slides",
        description="Allow nested slide folders. The frozen dataset still defines the slide list.",
    )
    segmenter: Literal["hest", "grandqc", "otsu"] = option(
        "hest",
        group="segmentation",
        advanced=False,
        description="Tissue segmentation model; Otsu is the classical CPU option.",
    )
    seg_conf_thresh: float = option(
        0.5,
        group="segmentation",
        ge=0,
        le=1,
        description="Binarization confidence threshold; lower values retain more tissue.",
    )
    remove_holes: bool = option(
        False,
        group="segmentation",
        description="Exclude holes inside tissue contours from tissue regions.",
    )
    remove_artifacts: bool = option(
        False,
        group="segmentation",
        description="Run GrandQC artifact removal for pen marks, blur, stains and other artifacts.",
    )
    remove_penmarks: bool = option(
        False,
        group="segmentation",
        description="Remove pen marks only unless full artifact removal is also enabled.",
    )
    seg_batch_size: int | None = option(
        None, group="segmentation", gt=0, description="Segmentation batch size override."
    )
    mag: float = option(
        20.0,
        group="patching",
        advanced=False,
        gt=0,
        description="Target magnification; fractional magnifications are supported.",
    )
    patch_size: int = option(
        256,
        group="patching",
        advanced=False,
        gt=0,
        description="Square patch width in pixels at the target magnification.",
    )
    overlap: int = option(
        0,
        group="patching",
        advanced=False,
        ge=0,
        description="Patch overlap in pixels; must be smaller than patch size.",
    )
    min_tissue_proportion: float = option(
        0.0,
        group="patching",
        ge=0,
        le=1,
        description="Minimum fraction of a patch that must contain tissue.",
    )
    coords_dir: str | None = option(
        None,
        group="patching",
        min_length=1,
        max_length=4096,
        description="Coordinate configuration directory under the output root; default uses magnification, patch size and overlap.",
    )
    dump_patches: bool = option(
        False, group="patching", description="Also save patch images during coordinate extraction."
    )
    dump_patches_max: int = option(
        0,
        group="patching",
        ge=0,
        description="Maximum patch images per slide; 0 saves all patches.",
    )
    dump_patches_format: Literal["png", "jpg"] = option(
        "png", group="patching", description="Patch image format."
    )
    dump_patches_jpeg_quality: int = option(
        90, group="patching", ge=1, le=100, description="JPEG quality when patch format is jpg."
    )
    patch_encoder: str = option(
        "uni_v1",
        group="features",
        advanced=False,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
        description="TRIDENT patch encoder.",
    )
    patch_encoder_ckpt_path: str | None = option(
        None,
        group="features",
        min_length=1,
        max_length=4096,
        description="Local encoder checkpoint for offline use; otherwise TRIDENT uses its model registry or Hugging Face.",
    )
    patch_encoder_img_size: int | None = option(
        None,
        group="features",
        gt=0,
        description="Custom model input resolution for supported ViT encoders; must match the model patch stride.",
    )
    slide_encoder: str | None = option(
        None,
        group="features",
        advanced=False,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
        description="Optional slide encoder; TRIDENT selects its required patch encoder automatically.",
    )
    feat_batch_size: int | None = option(
        None, group="features", gt=0, description="Feature extraction batch size override."
    )

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.overlap >= self.patch_size:
            raise ValueError("overlap must be smaller than patch_size")
        if self.coords_dir is not None:
            path = Path(self.coords_dir)
            if path.is_absolute() or ".." in path.parts or self.coords_dir in {".", ""}:
                raise ValueError("coords_dir must be a relative directory under the output root")
        for values in (self.wsi_ext, self.custom_mpp_keys):
            if values and any(
                not value.strip() or value.startswith("-") or "\x00" in value for value in values
            ):
                raise ValueError("List options must contain nonempty values, not command flags")
        if self.wsi_ext and any(
            not value.startswith(".") or "/" in value for value in self.wsi_ext
        ):
            raise ValueError("wsi_ext values must be file extensions beginning with a dot")
        if self.patch_encoder_img_size is not None:
            if self.slide_encoder is not None:
                raise ValueError("patch_encoder_img_size is ignored by slide encoders")
            stride = RESIZE_PATCH_STRIDES.get(self.patch_encoder)
            if stride is None:
                raise ValueError("This patch encoder does not support a custom input resolution")
            if self.patch_encoder_img_size % stride:
                raise ValueError(f"patch_encoder_img_size must be a multiple of {stride}")
        if self.slide_encoder and self.patch_encoder_ckpt_path:
            raise ValueError(
                "Slide encoders use TRIDENT's checkpoint registry, not patch_encoder_ckpt_path"
            )
        return self


def option_catalog() -> dict:
    schema = TridentOptions.model_json_schema()
    items = []
    for name, field in schema["properties"].items():
        concrete = next(
            (part for part in field.get("anyOf", []) if part.get("type") != "null"), field
        )
        kind = concrete.get("type", "string")
        if kind == "array":
            kind = concrete["items"]["type"] + "[]"
        item = {
            "name": name,
            "flag": f"--{name}",
            "label": name.replace("_", " ").capitalize(),
            "group": field["group"],
            "advanced": field["advanced"],
            "type": kind,
            "default": field.get("default"),
            "description": field["description"],
            "nullable": any(part.get("type") == "null" for part in field.get("anyOf", [])),
        }
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in concrete:
                item[key] = concrete[key]
        if "enum" in concrete:
            item["choices"] = concrete["enum"]
        if name == "patch_encoder":
            item["choices"] = list(PATCH_ENCODERS)
        elif name == "slide_encoder":
            item["choices"] = list(SLIDE_ENCODERS)
        items.append(item)
    return {
        "schemaVersion": 1,
        "source": SOURCE,
        "checkedAt": "2026-09-10",
        "options": items,
        "defaults": TridentOptions().model_dump(mode="json"),
        "patchEncoders": list(PATCH_ENCODERS),
        "slideEncoders": list(SLIDE_ENCODERS),
        "managedOptions": ["wsi_dir", "job_dir"],
    }


def output_layout(options: TridentOptions, job_dir: str | Path) -> dict:
    root = Path(job_dir)
    coords = root / (
        options.coords_dir or f"{options.mag:g}x_{options.patch_size}px_{options.overlap}px_overlap"
    )
    encoder = options.slide_encoder or options.patch_encoder
    features = coords / f"{'slide_features' if options.slide_encoder else 'features'}_{encoder}"
    return {
        "jobDir": str(root),
        "coordsDir": str(coords),
        "patchesDir": str(coords / "patches"),
        "featuresDir": str(features),
        "contoursDir": str(root / "contours"),
        "geojsonDir": str(root / "contours_geojson"),
        "thumbnailsDir": str(root / "thumbnails"),
        "coordinatePattern": str(coords / "patches" / "{slide}_patches.h5"),
        "featurePattern": str(features / "{slide}.h5"),
        "featureKind": "slide" if options.slide_encoder else "patch",
    }
