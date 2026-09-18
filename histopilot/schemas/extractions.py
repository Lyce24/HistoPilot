"""User intent for TRIDENT extraction with an optional dataset selection."""

from typing import Any

from pydantic import Field, model_validator

from histopilot.schemas.slide_lists import SlideListSpec


class ExtractionSpec(SlideListSpec):
    # The slide selection has one source and one optional filter. The source is the slide
    # list when options name one, the slide folder when one is chosen, and otherwise the
    # dataset's own linked slides. A dataset, when given, then narrows that selection.
    datasetId: str | None = Field(default=None, min_length=1, max_length=128)
    slideRoot: str | None = Field(default=None, min_length=1, max_length=4096)
    recursive: bool = True
    outputPath: str = Field(min_length=1, max_length=4096)
    options: dict[str, Any] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def needs_a_slide_source(self):
        if self.slideList is not None and self.options.get("custom_list_of_wsis"):
            raise ValueError("Choose one slide list.")
        if self.datasetId is None and self.slideRoot is None:
            raise ValueError("Choose a slide folder, a dataset, or both.")
        return self


class SubmitExtractionRequest(ExtractionSpec):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
