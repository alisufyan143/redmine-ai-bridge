"""Unit tests for Phase 4: Visual Triage Agent."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.visual_triage import VisualTriageAgent
from state import VisualTriageResult


@pytest.fixture
def mock_genai_client() -> MagicMock:
    """Fixture providing a mocked GenAI async client."""
    client = MagicMock()
    client.models = MagicMock()
    client.models.generate_content = AsyncMock()
    return client


# ==========================================
# Test 1: Structured Parsing & Model Return
# ==========================================

@pytest.mark.asyncio
async def test_analyze_issue_structured_parsing(mock_genai_client: MagicMock) -> None:
    """Test 1: Assert agent calls Gemini with multimodal parts and returns validated VisualTriageResult."""
    expected_result = VisualTriageResult(
        ticket_id="TICK-7001",
        processed_at=datetime.now(timezone.utc),
        screenshot_urls=["https://storage.example.com/login_crash.png"],
        ui_elements_detected=["UsernameInput", "PasswordInput", "SubmitButton", "ErrorModal"],
        anomaly_detected=True,
        triage_summary="HTTP 500 Internal Server Error modal is visible blocking authentication flow.",
        visual_confidence_score=0.98,
    )

    mock_response = MagicMock()
    mock_response.parsed = expected_result
    mock_genai_client.models.generate_content.return_value = mock_response

    agent = VisualTriageAgent(client=mock_genai_client, model="gemini-3.5-flash")

    result = await agent.analyze_issue(
        issue_title="Login modal throws 500",
        issue_description="Users unable to sign in, generic 500 crash modal displayed.",
        image_bytes=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 20,
        mime_type="image/png",
    )

    # Assert generate_content was awaited once with correct model and structured output config
    mock_genai_client.models.generate_content.assert_awaited_once()
    call_kwargs = mock_genai_client.models.generate_content.call_args[1]

    assert call_kwargs["model"] == "gemini-3.5-flash"
    assert len(call_kwargs["contents"]) == 2
    assert "Issue Title: Login modal throws 500" in call_kwargs["contents"][0]
    assert call_kwargs["config"].response_mime_type == "application/json"
    assert call_kwargs["config"].response_schema == VisualTriageResult

    # Assert validated object return
    assert isinstance(result, VisualTriageResult)
    assert result.ticket_id == "TICK-7001"
    assert result.anomaly_detected is True
    assert result.visual_confidence_score == 0.98
    assert "ErrorModal" in result.ui_elements_detected


# ==========================================
# Test 2: Error Handling & Logging
# ==========================================

@pytest.mark.asyncio
async def test_analyze_issue_error_handling_and_logging(
    mock_genai_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test 2: Assert API failure logs an error and propagates cleanly."""
    mock_genai_client.models.generate_content.side_effect = RuntimeError("Gemini Quota Exceeded")

    agent = VisualTriageAgent(client=mock_genai_client, model="gemini-3.5-flash")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as exc_info:
            await agent.analyze_issue(
                issue_title="Failing test issue",
                issue_description="Description text",
                image_bytes=b"fake_image_bytes",
                mime_type="image/jpeg",
            )

    assert "Gemini Quota Exceeded" in str(exc_info.value)

    # Assert error was logged in telemetry
    error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_logs) >= 1
    assert any("Visual triage API request failed" in r.message for r in error_logs)
