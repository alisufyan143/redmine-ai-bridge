"""Unit and integration tests for Phase 8: AIBridgeOrchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.graph_walker import GraphWalkerAgent
from agents.patch_engineer import PatchEngineerAgent
from agents.security_gate import SecurityGateAgent
from agents.visual_triage import VisualTriageAgent
from interceptor.redmine_client import RedmineAPIError, RedmineClient
from pipeline.orchestrator import AIBridgeOrchestrator
from sandbox.ruby_validator import RubySandboxValidator
from state import CodePatch, GraphContext, ValidationReport, VisualTriageResult


@pytest.fixture
def mock_orchestrator_components() -> tuple[
    MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, AIBridgeOrchestrator
]:
    """Fixture providing an AIBridgeOrchestrator with all mock agents."""
    mock_redmine = MagicMock(spec=RedmineClient)
    mock_triage_agent = MagicMock(spec=VisualTriageAgent)
    mock_graph_agent = MagicMock(spec=GraphWalkerAgent)
    mock_patch_agent = MagicMock(spec=PatchEngineerAgent)
    mock_sandbox = MagicMock(spec=RubySandboxValidator)
    mock_security = MagicMock(spec=SecurityGateAgent)

    orchestrator = AIBridgeOrchestrator(
        redmine_client=mock_redmine,
        visual_triage_agent=mock_triage_agent,
        graph_walker_agent=mock_graph_agent,
        patch_engineer_agent=mock_patch_agent,
        sandbox_validator=mock_sandbox,
        security_gate_agent=mock_security,
    )
    return (
        mock_redmine,
        mock_triage_agent,
        mock_graph_agent,
        mock_patch_agent,
        mock_sandbox,
        mock_security,
        orchestrator,
    )


# ==========================================
# Test 1: Full End-to-End Pipeline Success
# ==========================================

@pytest.mark.asyncio
async def test_orchestrator_end_to_end_success(
    mock_orchestrator_components: tuple[
        MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, AIBridgeOrchestrator
    ],
) -> None:
    """Test 1: Assert all pipeline agents are called once in sequence and comment is posted."""
    (
        mock_redmine,
        mock_triage_agent,
        mock_graph_agent,
        mock_patch_agent,
        mock_sandbox,
        mock_security,
        orchestrator,
    ) = mock_orchestrator_components

    now = datetime.now(timezone.utc)

    # 1. Mock Redmine fetch and attachment download
    mock_redmine.fetch_issue = AsyncMock(
        return_value={
            "issue": {
                "id": 1234,
                "subject": "Undefined method `name' for nil:NilClass",
                "description": "Crash when viewing project members without role assignment.",
                "author": {"name": "QA Engineer"},
                "priority": {"name": "High"},
                "attachments": [
                    {
                        "id": 88,
                        "content_type": "image/png",
                        "content_url": "https://redmine.example.com/attachments/download/88/shot.png",
                    }
                ],
            }
        }
    )
    mock_redmine.download_attachment = AsyncMock(return_value=b"\x89PNGfakeimagebytes")
    mock_redmine.post_issue_comment = AsyncMock(return_value=True)

    # 2. Mock Visual Triage
    mock_triage_res = VisualTriageResult(
        ticket_id="TICK-1234",
        processed_at=now,
        screenshot_urls=["https://redmine.example.com/attachments/download/88/shot.png"],
        ui_elements_detected=["MemberTable", "ErrorBanner"],
        anomaly_detected=True,
        triage_summary="NilClass crash on member.name lookup.",
        visual_confidence_score=0.99,
    )
    mock_triage_agent.analyze_issue = AsyncMock(return_value=mock_triage_res)

    # 3. Mock Graph Walker
    mock_graph_ctx = GraphContext(
        ticket_id="TICK-1234",
        extracted_at=now,
        query="Member name NilClass crash",
        relevant_nodes=["ProjectMembersController", "Member", "Role"],
        entity_relationships={"ProjectMembersController": ["Member"], "Member": ["Role"]},
        retrieved_code_snippets=[],
        subgraph_depth=1,
    )
    mock_graph_agent.build_context = AsyncMock(return_value=mock_graph_ctx)

    # 4. Mock Patch Engineer
    mock_code_patch = CodePatch(
        ticket_id="TICK-1234",
        generated_at=now,
        target_file="app/models/member.rb",
        diff="--- a/app/models/member.rb\n+++ b/app/models/member.rb\n@@ -20,1 +20,1 @@\n-  user.name\n+  user&.name || 'Unknown'",
        commit_message="fix(members): safe navigate user name lookup",
        rationale="Prevents 500 error when member user reference is nil.",
        applied_branch="fix/tick-1234-safe-member-name",
    )
    mock_patch_agent.generate_patch = AsyncMock(return_value=mock_code_patch)

    # 5. Mock Sandbox Validator
    mock_sandbox.validate_syntax = AsyncMock(return_value=(True, "Syntax OK"))

    # 6. Mock Security Gate
    mock_val_report = ValidationReport(
        ticket_id="TICK-1234",
        validated_at=now,
        is_valid=True,
        syntax_check_passed=True,
        unit_tests_passed=True,
        coverage_score=0.95,
        error_logs=[],
        summary="Audit passed cleanly. Safe navigation resolves the crash.",
    )
    mock_security.audit_patch = AsyncMock(return_value=mock_val_report)

    # Execute full pipeline
    success = await orchestrator.process_ticket(1234)

    assert success is True

    # Verify each agent was called exactly once in correct hand-off order
    mock_redmine.fetch_issue.assert_awaited_once_with(1234)
    mock_redmine.download_attachment.assert_awaited_once_with("https://redmine.example.com/attachments/download/88/shot.png")
    mock_triage_agent.analyze_issue.assert_awaited_once()
    mock_graph_agent.build_context.assert_awaited_once_with(triage=mock_triage_res)
    mock_patch_agent.generate_patch.assert_awaited_once()
    mock_sandbox.validate_syntax.assert_awaited_once_with(mock_code_patch.diff)
    mock_security.audit_patch.assert_awaited_once_with(
        patch=mock_code_patch,
        syntax_valid=True,
        syntax_message="Syntax OK",
    )

    # Verify final markdown comment was posted
    mock_redmine.post_issue_comment.assert_awaited_once()
    post_args = mock_redmine.post_issue_comment.call_args[1]
    assert post_args["issue_id"] == 1234
    posted_text = post_args["notes"]
    assert "System Failure Context Bridge" in posted_text
    assert "app/models/member.rb" in posted_text
    assert "safe navigate user name lookup" in posted_text
    assert "✅ PASSED" in posted_text


# ==========================================
# Test 2: Pipeline Abort on Fetch Failure
# ==========================================

@pytest.mark.asyncio
async def test_orchestrator_abort_on_fetch_failure(
    mock_orchestrator_components: tuple[
        MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, AIBridgeOrchestrator
    ],
) -> None:
    """Test 2: Assert pipeline halts gracefully and invokes no LLM agents when issue fetch fails."""
    (
        mock_redmine,
        mock_triage_agent,
        mock_graph_agent,
        mock_patch_agent,
        mock_sandbox,
        mock_security,
        orchestrator,
    ) = mock_orchestrator_components

    mock_redmine.fetch_issue = AsyncMock(side_effect=RedmineAPIError("HTTP 404 Issue Not Found", status_code=404))

    success = await orchestrator.process_ticket(9999)

    assert success is False
    mock_redmine.fetch_issue.assert_awaited_once_with(9999)

    # Assert no downstream agents were called
    mock_triage_agent.analyze_issue.assert_not_called()
    mock_graph_agent.build_context.assert_not_called()
    mock_patch_agent.generate_patch.assert_not_called()
    mock_sandbox.validate_syntax.assert_not_called()
    mock_security.audit_patch.assert_not_called()
    mock_redmine.post_issue_comment.assert_not_called()
