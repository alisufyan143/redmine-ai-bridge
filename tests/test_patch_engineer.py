"""Unit tests for Phase 6: Patch Engineer Agent."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.patch_engineer import PatchEngineerAgent
from state import CodePatch, GraphContext, IssueContext, VisualTriageResult
from utils.telemetry import AgentTimeoutError


@pytest.fixture
def sample_contexts() -> tuple[IssueContext, VisualTriageResult, GraphContext]:
    """Fixture providing integrated pipeline state instances."""
    now = datetime.now(timezone.utc)
    issue = IssueContext(
        ticket_id="TICK-9001",
        title="Null pointer exception in IssuesController#update",
        description="Editing custom field without assigning tracker throws 500.",
        reporter="tester@example.com",
        created_at=now,
        priority="urgent",
    )
    triage = VisualTriageResult(
        ticket_id="TICK-9001",
        processed_at=now,
        screenshot_urls=["https://storage.example.com/null_pointer.png"],
        ui_elements_detected=["IssuesForm", "FlashError"],
        anomaly_detected=True,
        triage_summary="Crash occurs on unassigned custom field lookup.",
        visual_confidence_score=0.97,
    )
    graph = GraphContext(
        ticket_id="TICK-9001",
        extracted_at=now,
        query="IssuesController update custom_field null pointer",
        relevant_nodes=["IssuesController", "IssueCustomField", "Tracker"],
        entity_relationships={"IssuesController": ["IssueCustomField"], "IssueCustomField": ["Tracker"]},
        retrieved_code_snippets=[],
        subgraph_depth=2,
    )
    return issue, triage, graph


# ==========================================
# Test 1: Successful Patch Generation
# ==========================================

@pytest.mark.asyncio
async def test_generate_patch_success(
    sample_contexts: tuple[IssueContext, VisualTriageResult, GraphContext],
) -> None:
    """Test 1: Assert agent compiles multimodal context and returns a valid CodePatch schema."""
    issue, triage, graph = sample_contexts
    now = datetime.now(timezone.utc)

    expected_patch = CodePatch(
        ticket_id="TICK-9001",
        generated_at=now,
        target_file="app/controllers/issues_controller.rb",
        diff="--- a/app/controllers/issues_controller.rb\n+++ b/app/controllers/issues_controller.rb\n@@ -45,3 +45,3 @@\n-  @issue.tracker.custom_fields\n+  @issue.tracker&.custom_fields || []",
        commit_message="fix(issues): safe navigate tracker custom fields on update",
        rationale="Prevents 500 error when tracker is nil during custom field assignment.",
        applied_branch="fix/tick-9001-safe-tracker-lookup",
    )

    mock_response = MagicMock()
    mock_response.parsed = expected_patch

    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_client.models.generate_content = AsyncMock(return_value=mock_response)

    agent = PatchEngineerAgent(client=mock_client, model="gemini-3.5-flash")

    snippets = {
        "app/controllers/issues_controller.rb": "def update\n  @issue.tracker.custom_fields\nend",
    }

    result = await agent.generate_patch(
        issue=issue,
        triage=triage,
        graph_context=graph,
        codebase_snippet_map=snippets,
    )

    # Assert model was called with structured schema configuration
    mock_client.models.generate_content.assert_awaited_once()
    call_kwargs = mock_client.models.generate_content.call_args[1]

    assert call_kwargs["model"] == "gemini-3.5-flash"
    assert "TICK-9001" in call_kwargs["contents"][0]
    assert "IssuesController#update" in call_kwargs["contents"][0]
    assert "app/controllers/issues_controller.rb" in call_kwargs["contents"][0]
    assert call_kwargs["config"].response_schema == CodePatch

    # Assert returned object
    assert isinstance(result, CodePatch)
    assert result.ticket_id == "TICK-9001"
    assert result.target_file == "app/controllers/issues_controller.rb"
    assert "safe navigate" in result.commit_message


# ==========================================
# Test 2: Timeout & Circuit Breaker Enforcement
# ==========================================

@pytest.mark.asyncio
async def test_generate_patch_timeout_circuit_breaker(
    sample_contexts: tuple[IssueContext, VisualTriageResult, GraphContext],
) -> None:
    """Test 2: Assert excessive model delay trips circuit breaker and raises AgentTimeoutError."""
    issue, triage, graph = sample_contexts

    async def slow_generation(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(30)
        return MagicMock()

    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_client.models.generate_content = slow_generation

    agent = PatchEngineerAgent(client=mock_client, model="gemini-3.5-flash")

    # generate_patch has circuit_breaker(timeout_seconds=25)
    # We can test timeout with a temporary small timeout or verifying AgentTimeoutError
    from utils.telemetry import circuit_breaker

    # Test circuit breaker timeout directly on an instance with small timeout
    @circuit_breaker(timeout_seconds=1)
    async def quick_timed_patch() -> None:
        await agent.generate_patch(issue, triage, graph)

    with pytest.raises(AgentTimeoutError) as exc_info:
        await quick_timed_patch()

    assert "timed out" in str(exc_info.value).lower()
