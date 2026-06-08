"""
Upload request schema.

Pydantic model representing the form fields for POST /upload-doc.
File binary is handled separately via FastAPI's UploadFile mechanism.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class UploadRequest(BaseModel):
    """
    Form fields accompanying the uploaded file.

    In FastAPI, multipart/form-data endpoints receive non-file fields
    as Form() parameters. This schema documents and validates those fields.
    The file binary itself is received as a FastAPI UploadFile -- it is
    not part of this schema.

    Attributes:
        file_type:   Declared type of the upload. Must be "pdf" or "csv".
                     Validated against the file's magic bytes and MIME type
                     in DocumentService before parsing begins.
        description: Optional human-readable description. Stored in document
                     metadata. Sanitised (HTML stripped) before storage.
        tags:        Optional list of tags. Each tag is 1-30 chars,
                     alphanumeric plus hyphens. Maximum 10 tags.
    """

    file_type: Literal["pdf", "csv"] = Field(
        description="Declared file type. Must match the uploaded file's actual type.",
    )
    description: Annotated[str, Field(max_length=500)] = Field(
        default="",
        description="Optional document description. Max 500 characters.",
    )
    tags: list[Annotated[str, Field(min_length=1, max_length=30)]] = Field(
        default_factory=list,
        max_length=10,
        description="Optional tags. Max 10. Each tag: 1-30 alphanumeric+hyphen chars.",
    )

    @field_validator("tags")
    @classmethod
    def validate_tag_format(cls, tags: list[str]) -> list[str]:
        """Validate each tag is alphanumeric plus hyphens only."""
        import re

        pattern = re.compile(r"^[a-zA-Z0-9\-]+$")
        for tag in tags:
            if not pattern.match(tag):
                raise ValueError(
                    f"Tag '{tag}' contains invalid characters. "
                    "Tags must contain only letters, numbers, and hyphens."
                )
        return tags

    @field_validator("description")
    @classmethod
    def strip_html(cls, value: str) -> str:
        """Strip HTML tags from description to prevent injection."""
        import re

        return re.sub(r"<[^>]+>", "", value).strip()
