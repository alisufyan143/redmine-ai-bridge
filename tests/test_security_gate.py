"""Unit tests for Phase 7: Security Gate Agent."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.security_gate import SecurityGateAgent
from state import CodePatch, ValidationReport


@pytest.fixture
def sample_patch() -> CodePatch:
    """Fixture providing a standard CodePatch instance."""
    return CodePatch(
        ticket_id="TICK-10001",
        generated_at=datetime.now(timezone.utc),
        target_file="app/models/issue.rb",
        diff="--- a/app/models/issue.rb\n+++ b/app/models/issue.rb\n@@ -10,3 +10,3 @@\n-  where(\"status_id = #{params[:status]}\")\n+  where(status_id: params[:status])",
        commit_message="fix(security): sanitize status query parameter",
        rationale="Eliminates raw string interpolation in ActiveRecord query.",
        applied_branch="fix/tick-10001-sanitize-status-query",
    )


# ==========================================
# Test 1: Passed Security Audit
# ==========================================

@pytest.mark.asyncio
async def test_audit_patch_passed(sample_patch: CodePatch) -> None:
    """Test 1: Assert clean patch returns ValidationReport with is_valid=True."""
    now = datetime.now(timezone.utc)
    expected_report = ValidationReport(
        ticket_id="TICK-10001",
        validated_at=now,
        is_valid=True,
        syntax_check_passed=True,
        unit_tests_passed=True,
        coverage_score=0.95,
        error_logs=[],
        summary="Security audit passed. No SQL injection or command injection patterns found.",
    )

    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = expected_report
    mock_client.models.generate_content = AsyncMock(return_value=mock_response)

    agent = SecurityGateAgent(client=mock_client, model="gemini-3.5-flash")

    result = await agent.audit_patch(
        patch=sample_patch,
        syntax_valid=True,
        syntax_message="Syntax OK",
    )

    mock_client.models.generate_content.assert_awaited_once()
    call_kwargs = mock_client.models.generate_content.call_args[1]

    assert call_kwargs["model"] == "gemini-3.5-flash"
    assert "TICK-10001" in call_kwargs["contents"][0]
    assert call_kwargs["config"].response_schema == ValidationReport

    assert isinstance(result, ValidationReport)
    assert result.is_valid is True
    assert result.syntax_check_passed is True
    assert len(result.error_logs) == 0


# ==========================================
# Test 2: Failed Security Audit (Vulnerability Detected)
# ==========================================

@pytest.mark.asyncio
async def test_audit_patch_vulnerability_detected(sample_patch: CodePatch) -> None:
    """Test 2: Assert insecure patch returns ValidationReport with is_valid=False and error logs."""
    now = datetime.now(timezone.utc)
    vulnerable_report = ValidationReport(
        ticket_id="TICK-10001",
        validated_at=now,
        is_valid=False,
        syntax_check_passed=True,
        unit_tests_passed=False,
        coverage_score=0.40,
        error_logs=["CRITICAL: SQL Injection vulnerability detected via raw string interpolation in query."],
        summary="Security audit failed due to unescaped parameter usage.",
    )

    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = vulnerable_report
    mock_client.models.generate_content = AsyncMock(return_value=mock_response)

    agent = SecurityGateAgent(client=mock_client, model="gemini-3.5-flash")

    result = await agent.audit_patch(
        patch=sample_patch,
        syntax_valid=True,
        syntax_message="Syntax OK",
    )

    assert isinstance(result, ValidationReport)
    assert result.is_valid is False
    assert len(result.error_logs) == 1
    assert "SQL Injection" in result.error_logs[0]
