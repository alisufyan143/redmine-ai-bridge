"""State management module for System Failure Context Bridge.

Provides immutable ledger models with strict validation, character limits,
timezone enforcement, and rejection of unrecognized attributes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StateValidationError(Exception):
    """Custom exception raised for state validation errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ImmutableBaseState(BaseModel):
    """Base immutable state model enforcing strict typing and size constraints."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )

    @field_validator("*", mode="after")
    @classmethod
    def validate_field_constraints(cls, value: Any) -> Any:
        """Enforces 10,000 character limit on strings and strict UTC timezone awareness on datetimes."""
        if isinstance(value, str):
            if len(value) > 10_000:
                raise ValueError(
                    f"String field exceeds maximum limit of 10,000 characters (current length: {len(value)})."
                )
        elif isinstance(value, datetime):
            if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                raise ValueError(
                    "Timestamp must be a timezone-aware UTC datetime object, received naive datetime."
                )
            if value.utcoffset() != timezone.utc.utcoffset(value):
                raise ValueError(
                    f"Timestamp must be in UTC timezone (offset zero), received offset: {value.utcoffset()}."
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str) and len(item) > 10_000:
                    raise ValueError(
                        f"List element at index {index} exceeds 10,000 character limit (length: {len(item)})."
                    )
        elif isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, str) and len(val) > 10_000:
                    raise ValueError(
                        f"Dictionary value for key '{key}' exceeds 10,000 character limit (length: {len(val)})."
                    )
        return value


class IssueContext(ImmutableBaseState):
    """Represents an ingested bug ticket and initial metadata."""

    ticket_id: str
    title: str
    description: str
    reporter: str
    created_at: datetime
    priority: str = Field(default="normal")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualTriageResult(ImmutableBaseState):
    """Represents visual triage results from screenshot/UI analysis."""

    ticket_id: str
    processed_at: datetime
    screenshot_urls: list[str] = Field(default_factory=list)
    ui_elements_detected: list[str] = Field(default_factory=list)
    anomaly_detected: bool
    triage_summary: str
    visual_confidence_score: float = Field(ge=0.0, le=1.0)


class GraphContext(ImmutableBaseState):
    """Represents retrieved Graph RAG knowledge and entity relationships."""

    ticket_id: str
    extracted_at: datetime
    query: str
    relevant_nodes: list[str] = Field(default_factory=list)
    entity_relationships: dict[str, list[str]] = Field(default_factory=dict)
    retrieved_code_snippets: list[str] = Field(default_factory=list)
    subgraph_depth: int = Field(default=1, ge=1)


class CodePatch(ImmutableBaseState):
    """Represents a generated code patch and associated metadata."""

    ticket_id: str
    generated_at: datetime
    target_file: str
    diff: str
    commit_message: str
    rationale: str
    applied_branch: str


class ValidationReport(ImmutableBaseState):
    """Represents the automated validation report for generated patches."""

    ticket_id: str
    validated_at: datetime
    is_valid: bool
    syntax_check_passed: bool
    unit_tests_passed: bool
    coverage_score: float = Field(ge=0.0, le=1.0)
    error_logs: list[str] = Field(default_factory=list)
    summary: str
