"""End-to-end multi-agent orchestration pipeline for System Failure Context Bridge."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from agents.graph_walker import GraphWalkerAgent
from agents.patch_engineer import PatchEngineerAgent
from agents.security_gate import SecurityGateAgent
from agents.visual_triage import VisualTriageAgent
from interceptor.redmine_client import RedmineClient
from sandbox.ruby_validator import RubySandboxValidator
from state import IssueContext
from utils.telemetry import get_json_logger

logger: logging.Logger = get_json_logger("pipeline.orchestrator")

# 1x1 transparent PNG fallback for tickets without attached screenshots
FALLBACK_PNG_BYTES: bytes = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class AIBridgeOrchestrator:
    """Orchestrates end-to-end bug interception, visual triage, Graph RAG, patch generation, and security auditing."""

    def __init__(
        self,
        redmine_client: RedmineClient,
        visual_triage_agent: VisualTriageAgent | None = None,
        graph_walker_agent: GraphWalkerAgent | None = None,
        patch_engineer_agent: PatchEngineerAgent | None = None,
        sandbox_validator: RubySandboxValidator | None = None,
        security_gate_agent: SecurityGateAgent | None = None,
        *,
        visual_triage: VisualTriageAgent | None = None,
        graph_walker: GraphWalkerAgent | None = None,
        patch_engineer: PatchEngineerAgent | None = None,
        ruby_validator: RubySandboxValidator | None = None,
        security_gate: SecurityGateAgent | None = None,
    ) -> None:
        self.redmine_client: RedmineClient = redmine_client
        self.visual_triage_agent: VisualTriageAgent = visual_triage_agent or visual_triage  # type: ignore[assignment]
        self.graph_walker_agent: GraphWalkerAgent = graph_walker_agent or graph_walker  # type: ignore[assignment]
        self.patch_engineer_agent: PatchEngineerAgent = patch_engineer_agent or patch_engineer  # type: ignore[assignment]
        self.sandbox_validator: RubySandboxValidator = sandbox_validator or ruby_validator  # type: ignore[assignment]
        self.security_gate_agent: SecurityGateAgent = security_gate_agent or security_gate  # type: ignore[assignment]


    async def process_ticket(self, issue_id: int) -> bool:
        """Executes the full automated diagnostic and remediation pipeline for a Redmine issue."""
        logger.info(
            f"Starting end-to-end pipeline execution for Redmine issue #{issue_id}",
            extra={"extra_payload": {"issue_id": issue_id}},
        )

        # 1. Fetch issue payload and primary image attachment from Redmine
        try:
            issue_payload = await self.redmine_client.fetch_issue(issue_id)
        except Exception as exc:
            logger.error(
                f"Pipeline aborted: failed to fetch Redmine issue #{issue_id}: {exc}",
                extra={"extra_payload": {"issue_id": issue_id, "error": str(exc)}},
            )
            return False

        issue_raw = issue_payload.get("issue", issue_payload)
        ticket_id = f"TICK-{issue_id}"
        issue_title = str(issue_raw.get("subject", "Untitled Bug Report"))
        issue_description = str(issue_raw.get("description", "No description provided."))
        author_info = issue_raw.get("author", {})
        reporter = str(author_info.get("name", "Redmine User") if isinstance(author_info, dict) else author_info)
        priority_info = issue_raw.get("priority", {})
        priority = str(priority_info.get("name", "normal") if isinstance(priority_info, dict) else priority_info)

        issue_context = IssueContext(
            ticket_id=ticket_id,
            title=issue_title,
            description=issue_description,
            reporter=reporter,
            created_at=datetime.now(timezone.utc),
            priority=priority,
        )

        # Find primary image attachment if present
        image_bytes: bytes = FALLBACK_PNG_BYTES
        mime_type: str = "image/png"
        attachments = issue_raw.get("attachments", [])

        for att in attachments:
            if isinstance(att, dict):
                content_type = str(att.get("content_type", "")).lower()
                content_url = att.get("content_url") or att.get("url")
                if content_type in ("image/png", "image/jpeg", "image/jpg") and content_url:
                    try:
                        downloaded = await self.redmine_client.download_attachment(str(content_url))
                        if downloaded:
                            image_bytes = downloaded
                            mime_type = "image/jpeg" if "jpeg" in content_type or "jpg" in content_type else "image/png"
                            break
                    except Exception as att_exc:
                        logger.warning(
                            f"Failed to download attachment {content_url}; using fallback image: {att_exc}",
                            extra={"extra_payload": {"error": str(att_exc)}},
                        )

        # 2. Multimodal Visual Triage
        logger.info(f"Step 2: Performing visual triage for #{issue_id}")
        triage_result = await self.visual_triage_agent.analyze_issue(
            issue_title=issue_context.title,
            issue_description=issue_context.description,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        # 3. Graph RAG Context Walk
        logger.info(f"Step 3: Building Graph RAG context for #{issue_id}")
        graph_context = await self.graph_walker_agent.build_context(triage=triage_result)

        # 4. Patch Engineering (Code Generation)
        logger.info(f"Step 4: Generating code patch for #{issue_id}")
        code_patch = await self.patch_engineer_agent.generate_patch(
            issue=issue_context,
            triage=triage_result,
            graph_context=graph_context,
        )

        # 5. Dockerized Sandbox Syntax Validation
        logger.info(f"Step 5: Validating patch syntax in Docker sandbox for #{issue_id}")
        syntax_valid, syntax_message = await self.sandbox_validator.validate_syntax(code_patch.diff)

        # 6. Security Gate Audit
        logger.info(f"Step 6: Auditing patch security for #{issue_id}")
        validation_report = await self.security_gate_agent.audit_patch(
            patch=code_patch,
            syntax_valid=syntax_valid,
            syntax_message=syntax_message,
        )

        # 7. Construct Markdown Report
        ui_elements_str = ", ".join(triage_result.ui_elements_detected) if triage_result.ui_elements_detected else "None"
        relevant_nodes_str = ", ".join(graph_context.relevant_nodes) if graph_context.relevant_nodes else "None"
        security_status = "✅ PASSED" if validation_report.is_valid else "❌ FAILED"
        syntax_status = "✅ Valid" if syntax_valid else f"❌ {syntax_message}"
        findings_str = "\n".join(f"- {f}" for f in validation_report.error_logs) if validation_report.error_logs else "- No security vulnerabilities identified."

        markdown_report = (
            f"## 🤖 System Failure Context Bridge — Automated Analysis & Fix\n\n"
            f"### 🚨 Triage Diagnostics\n"
            f"- **Anomaly Detected**: `{triage_result.anomaly_detected}`\n"
            f"- **Confidence Score**: `{triage_result.visual_confidence_score:.2f}`\n"
            f"- **Summary**: {triage_result.triage_summary}\n"
            f"- **UI Elements Detected**: {ui_elements_str}\n\n"
            f"### 🗺️ Subgraph Context Used\n"
            f"- **Relevant Code Nodes**: {relevant_nodes_str}\n"
            f"- **Active Edge Connections**: `{len(graph_context.entity_relationships)}`\n\n"
            f"### 🛠️ Proposed Ruby Patch\n"
            f"- **Target File**: `{code_patch.target_file}`\n"
            f"- **Branch**: `{code_patch.applied_branch}`\n"
            f"- **Commit Message**: `{code_patch.commit_message}`\n"
            f"- **Rationale**: {code_patch.rationale}\n\n"
            f"```diff\n"
            f"{code_patch.diff}\n"
            f"```\n\n"
            f"### 🛡️ Security Audit & Syntax Results\n"
            f"- **Overall Security Status**: {security_status}\n"
            f"- **Sandbox Syntax Check**: {syntax_status}\n"
            f"- **Audit Summary**: {validation_report.summary}\n"
            f"- **Findings**:\n{findings_str}\n"
        )

        # 8. Post Comment back to Redmine Issue
        logger.info(f"Step 8: Posting final diagnosis and patch comment to Redmine issue #{issue_id}")
        await self.redmine_client.post_issue_comment(
            issue_id=issue_id,
            notes=markdown_report,
            private=False,
        )

        logger.info(f"Pipeline successfully completed for issue #{issue_id}")
        return True
