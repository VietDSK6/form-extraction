from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


FieldType = Literal[
    "text",
    "textarea",
    "number",
    "date",
    "select",
    "multiselect",
    "checkbox",
    "radio",
    "switch",
    "list",
]
ScalarValue = StrictStr | StrictInt | StrictFloat | StrictBool
ExtractionValue = ScalarValue | list[ScalarValue] | None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldOption(StrictModel):
    value: ScalarValue
    label: str = Field(min_length=1, max_length=200)

    @field_validator("label")
    @classmethod
    def strip_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label must not be blank")
        return value


class FormFieldDefinition(StrictModel):
    field_code: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=200)
    type: FieldType
    required: bool = False
    options: list[FieldOption] = Field(default_factory=list, max_length=100)
    extraction_hint: str | None = Field(default=None, max_length=500)

    @field_validator("field_code", "label")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("extraction_hint")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_unique_options(self) -> "FormFieldDefinition":
        seen: set[tuple[type[Any], Any]] = set()
        for option in self.options:
            marker = (type(option.value), option.value)
            if marker in seen:
                raise ValueError(f"duplicate option value for field '{self.field_code}'")
            seen.add(marker)
        return self


class ExtractFormRequest(StrictModel):
    transcript: str = Field(min_length=1, max_length=10_000)
    fields: Annotated[list[FormFieldDefinition], Field(min_length=1, max_length=60)]
    current_values: dict[str, Any] = Field(default_factory=dict)
    reference_date: date | None = None

    @field_validator("transcript")
    @classmethod
    def strip_transcript(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("transcript must not be blank")
        return value

    @model_validator(mode="after")
    def validate_unique_field_codes(self) -> "ExtractFormRequest":
        codes = [field.field_code for field in self.fields]
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        if duplicates:
            raise ValueError(f"duplicate field_code: {', '.join(duplicates)}")
        return self


class ExtractionItem(StrictModel):
    field_code: str
    value: ExtractionValue
    status: Literal["extracted", "ambiguous"]
    evidence: str | None = None


class ExtractionWarning(StrictModel):
    code: str
    message: str
    field_code: str | None = None


class UsageMetadata(StrictModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float


class ExtractFormResponse(StrictModel):
    request_id: str
    extractions: list[ExtractionItem]
    missing_required: list[str]
    skipped_existing: list[str]
    warnings: list[ExtractionWarning]
    usage: UsageMetadata


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorDetail


class LiveHealthResponse(StrictModel):
    status: Literal["ok"] = "ok"


class ReadyHealthResponse(StrictModel):
    status: Literal["ready", "not_ready"]
    openai_configured: bool
    model: str

