"""Patch Engineer Agent module.

Uses Gemini reasoning to synthesize IssueContext, VisualTriageResult,
and GraphContext into a production-grade CodePatch unified diff.
"""

from __future__ import annotations

import logging
from typing import Any
from google import genai
from google.genai import types

from state import CodePatch, GraphContext, IssueContext, VisualTriageResult
from utils.telemetry import circuit_breaker, get_json_logger

logger: logging.Logger = get_json_logger("agents.patch_engineer")


class PatchEngineerAgent:
    """Synthesizes failure context, visual triage, and graph knowledge to generate bug fix patches."""

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        model: str = "gemini-3.5-flash",
    ) -> None:
        self.model: str = model
        if client is not None:
            self.client: Any = client
        else:
            self.client = genai.Client(api_key=api_key).aio

    @circuit_breaker(timeout_seconds=25)
    async def generate_patch(
        self,
        issue: IssueContext,
        triage: VisualTriageResult,
        graph_context: GraphContext,
        codebase_snippet_map: dict[str, str] | None = None,
    ) -> CodePatch:
        """Generates a structured CodePatch diff based on full pipeline diagnostics."""
        logger.info(
            f"Generating code patch for ticket '{issue.ticket_id}': {issue.title}",
            extra={"extra_payload": {"ticket_id": issue.ticket_id, "model": self.model}},
        )

        snippets_formatted = ""
        if codebase_snippet_map:
            for filepath, snippet in codebase_snippet_map.items():
                snippets_formatted += f"\n--- File: {filepath} ---\n{snippet}\n"
        else:
            snippets_formatted = "No external file snippets provided; refer to graph nodes."

        relationships_formatted = ""
        for node, neighbors in graph_context.entity_relationships.items():
            relationships_formatted += f"  - {node} -> {', '.join(neighbors)}\n"
        if not relationships_formatted:
            relationships_formatted = "  - None\n"

        prompt = (
            "You are a Senior Ruby on Rails Core Maintainer and Principal Software Engineer. "
            "Your objective is to diagnose the root cause of a reported system failure and produce "
            "a production-ready, minimal unified diff (git patch) resolving the issue.\n\n"
            "=== BUG TICKET CONTEXT ===\n"
            f"Ticket ID: {issue.ticket_id}\n"
            f"Title: {issue.title}\n"
            f"Description: {issue.description}\n"
            f"Reporter: {issue.reporter}\n"
            f"Priority: {issue.priority}\n\n"
            "=== VISUAL TRIAGE DIAGNOSTICS ===\n"
            f"Anomaly Detected: {triage.anomaly_detected}\n"
            f"Confidence Score: {triage.visual_confidence_score}\n"
            f"Triage Summary: {triage.triage_summary}\n"
            f"UI Elements Detected: {', '.join(triage.ui_elements_detected)}\n\n"
            "=== CODE GRAPH CONTEXT (GRAPH RAG) ===\n"
            f"Relevant Nodes: {', '.join(graph_context.relevant_nodes)}\n"
            f"Entity Relationships:\n{relationships_formatted}\n"
            "=== CODEBASE SNIPPETS ===\n"
            f"{snippets_formatted}\n\n"
            "=== INSTRUCTIONS ===\n"
            "1. Identify the exact failure point and formulate a minimal, robust code fix.\n"
            "2. Generate a standard unified diff ('diff') with '--- a/...' and '+++ b/...' headers.\n"
            "3. Do NOT include placeholders, ellipses, or '# TODO' comments in the patch diff.\n"
            "4. Provide a clear commit message adhering to Conventional Commits (e.g. 'fix(auth): ...').\n"
            "5. Provide a technical rationale explaining why this patch resolves the error.\n"
            f"6. Specify the suggested branch name in the format 'fix/{issue.ticket_id.lower()}-<topic>'.\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CodePatch,
        )

        try:
            response = await self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=config,
            )
        except Exception as exc:
            logger.error(
                f"Patch generation API request failed: {exc}",
                extra={"extra_payload": {"ticket_id": issue.ticket_id, "error": str(exc)}},
            )
            raise

        if hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, CodePatch):
                return response.parsed
            if isinstance(response.parsed, dict):
                return CodePatch.model_validate(response.parsed)

        if hasattr(response, "text") and response.text:
            return CodePatch.model_validate_json(response.text)

        raise ValueError("Gemini response did not return valid parsed CodePatch schema.")
