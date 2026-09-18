"""A CSV slide selection supplied from the server or the user's computer."""

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from histopilot.schemas.workspace import RequestModel

MAX_SLIDE_LIST_BYTES = 8 * 1024 * 1024
MAX_SLIDE_UPLOAD_BYTES = 1024 * 1024


class SlideListSource(RequestModel):
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    contentBase64: str | None = Field(
        default=None, min_length=1, max_length=4 * ((MAX_SLIDE_UPLOAD_BYTES + 2) // 3)
    )
    filename: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def one_source(self):
        if (self.path is None) == (self.contentBase64 is None):
            raise ValueError("Choose one server slide list or upload one CSV file.")
        if self.contentBase64 is not None and self.filename is None:
            raise ValueError("An uploaded slide list needs its filename.")
        return self


class SlideListSpec(RequestModel):
    slideList: SlideListSource | None = None

    @model_serializer(mode="wrap")
    def preserve_existing_selection(self, handler: SerializerFunctionWrapHandler) -> dict:
        """An unused new input must not change frozen identities or retry hashes."""
        serialized = handler(self)
        if self.slideList is None:
            serialized.pop("slideList", None)
        return serialized
