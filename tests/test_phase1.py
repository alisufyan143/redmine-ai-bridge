"""Unit tests for Phase 1: MLOps Foundation & Strict State Management."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Any

from pydantic import ValidationError
import pytest

from state import (
    CodePatch,
    GraphContext,
    IssueContext,
    StateValidationError,
    ValidationReport,
    VisualTriageResult,
)
from utils.telemetry import (
    AgentTimeoutError,
    JSONFormatter,
    circuit_breaker,
    get_json_logger,
)


# ==========================================
# 1. State Management & Model Validation Tests
# ==========================================

def test_valid_issue_context() -> None:
    now = datetime.now(timezone.utc)
    issue = IssueContext(
        ticket_id="TICK-101",
        title="Payment gateway timeout",
        description="Users report HTTP 504 on checkout.",
        reporter="alice@example.com",
        created_at=now,
        priority="high",
        tags=["payment", "critical"],
        metadata={"service": "billing"},
    )
    assert issue.ticket_id == "TICK-101"
    assert issue.created_at == now
    assert issue.priority == "high"


def test_issue_context_rejects_long_string() -> None:
    now = datetime.now(timezone.utc)
    long_description = "A" * 10_001
    with pytest.raises(ValidationError) as exc_info:
        IssueContext(
            ticket_id="TICK-102",
            title="Valid title",
            description=long_description,
            reporter="bob@example.com",
            created_at=now,
        )
    assert "exceeds maximum limit of 10,000 characters" in str(exc_info.value)


def test_issue_context_rejects_naive_timestamp() -> None:
    naive_datetime = datetime(2026, 8, 26, 12, 0, 0)
    with pytest.raises(ValidationError) as exc_info:
        IssueContext(
            ticket_id="TICK-103",
            title="Valid title",
            description="Valid description",
            reporter="charlie@example.com",
            created_at=naive_datetime,
        )
    assert "Timestamp must be a timezone-aware UTC datetime object" in str(exc_info.value)


def test_issue_context_rejects_non_utc_timestamp() -> None:
    non_utc_datetime = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    with pytest.raises(ValidationError) as exc_info:
        IssueContext(
            ticket_id="TICK-104",
            title="Valid title",
            description="Valid description",
            reporter="david@example.com",
            created_at=non_utc_datetime,
        )
    assert "Timestamp must be in UTC timezone" in str(exc_info.value)


def test_issue_context_forbids_extra_fields() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError) as exc_info:
        IssueContext.model_validate(
            {
                "ticket_id": "TICK-105",
                "title": "Valid title",
                "description": "Valid description",
                "reporter": "eva@example.com",
                "created_at": now.isoformat(),
                "unexpected_field": "disallowed_value",
            }
        )
    assert "extra_forbidden" in str(exc_info.value)


def test_model_immutability() -> None:
    now = datetime.now(timezone.utc)
    issue = IssueContext(
        ticket_id="TICK-106",
        title="Original Title",
        description="Original Description",
        reporter="frank@example.com",
        created_at=now,
    )
    with pytest.raises(ValidationError):
        issue.title = "Modified Title"  # type: ignore[misc]


def test_visual_triage_result_validation() -> None:
    now = datetime.now(timezone.utc)
    result = VisualTriageResult(
        ticket_id="TICK-201",
        processed_at=now,
        screenshot_urls=["https://storage.example.com/shot1.png"],
        ui_elements_detected=["submit_button", "error_banner"],
        anomaly_detected=True,
        triage_summary="Red error toast displayed.",
        visual_confidence_score=0.95,
    )
    assert result.anomaly_detected is True
    assert result.visual_confidence_score == 0.95

    # Confidence score out of bounds
    with pytest.raises(ValidationError):
        VisualTriageResult(
            ticket_id="TICK-201",
            processed_at=now,
            anomaly_detected=False,
            triage_summary="Summary",
            visual_confidence_score=1.5,
        )


def test_graph_context_validation() -> None:
    now = datetime.now(timezone.utc)
    graph = GraphContext(
        ticket_id="TICK-301",
        extracted_at=now,
        query="payment timeout error handler",
        relevant_nodes=["PaymentGateway", "TimeoutHandler"],
        entity_relationships={"PaymentGateway": ["TimeoutHandler", "MetricsClient"]},
        retrieved_code_snippets=["def handle_timeout(): pass"],
        subgraph_depth=2,
    )
    assert graph.subgraph_depth == 2
    assert "PaymentGateway" in graph.relevant_nodes


def test_code_patch_validation() -> None:
    now = datetime.now(timezone.utc)
    patch = CodePatch(
        ticket_id="TICK-401",
        generated_at=now,
        target_file="services/payment.py",
        diff="--- a/services/payment.py\n+++ b/services/payment.py\n@@ -1 +1 @@\n-timeout=5\n+timeout=15",
        commit_message="fix(payment): increase gateway timeout threshold",
        rationale="Prevents 504 errors on slow networks.",
        applied_branch="fix/ticket-401-payment-timeout",
    )
    assert patch.target_file == "services/payment.py"
    assert patch.applied_branch.startswith("fix/")


def test_validation_report_validation() -> None:
    now = datetime.now(timezone.utc)
    report = ValidationReport(
        ticket_id="TICK-501",
        validated_at=now,
        is_valid=True,
        syntax_check_passed=True,
        unit_tests_passed=True,
        coverage_score=0.92,
        error_logs=[],
        summary="All syntax checks and unit tests passed successfully.",
    )
    assert report.is_valid is True
    assert report.coverage_score == 0.92


def test_state_validation_error_instantiation() -> None:
    err = StateValidationError("Ledger validation failed", details={"field": "ticket_id"})
    assert "Ledger validation failed" in str(err)
    assert err.details == {"field": "ticket_id"}


# ==========================================
# 2. Telemetry & Circuit Breaker Tests
# ==========================================

def test_json_formatter_outputs_valid_json() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=100,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    formatted_output = formatter.format(record)
    parsed = json.loads(formatted_output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test log message"
    assert "timestamp" in parsed


@pytest.mark.asyncio
async def test_circuit_breaker_success() -> None:
    @circuit_breaker(timeout_seconds=2)
    async def fast_task(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    result = await fast_task(21)
    assert result == 42


@pytest.mark.asyncio
async def test_circuit_breaker_timeout_raises_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    @circuit_breaker(timeout_seconds=1)
    async def slow_task(duration: float, keyword_arg: str = "default") -> str:
        await asyncio.sleep(duration)
        return "completed"

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(AgentTimeoutError) as exc_info:
            await slow_task(5.0, keyword_arg="custom_value")

    err = exc_info.value
    assert err.function_name == "test_circuit_breaker_timeout_raises_and_logs.<locals>.slow_task"
    assert err.timeout_seconds == 1.0
    assert err.args == (5.0,)
    assert err.kwargs == {"keyword_arg": "custom_value"}

    # Assert critical log was captured and contains JSON formatted circuit breaker payload
    critical_logs = [record for record in caplog.records if record.levelno == logging.CRITICAL]
    assert len(critical_logs) >= 1
    record = critical_logs[0]
    assert hasattr(record, "extra_payload")
    payload = record.extra_payload
    assert payload["event"] == "CIRCUIT_BREAKER_TIMEOUT"
    assert payload["error_type"] == "AgentTimeoutError"
