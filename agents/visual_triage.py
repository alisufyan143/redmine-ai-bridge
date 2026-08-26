"""Visual Triage Agent using Multimodal Gemini to analyze UI screenshots and bug reports."""

from __future__ import annotations

import logging
from typing import Any
from google import genai
from google.genai import types

from state import VisualTriageResult
from utils.telemetry import circuit_breaker, get_json_logger

logger: logging.Logger = get_json_logger("agents.visual_triage")


class VisualTriageAgent:
    """Agent that performs visual diagnostics on bug screenshots and issue text using Multimodal Gemini."""

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

    @circuit_breaker(timeout_seconds=20)
    async def analyze_issue(
        self,
        issue_title: str,
        issue_description: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> VisualTriageResult:
        """Analyzes a bug screenshot and text, returning a strictly structured VisualTriageResult."""
        logger.info(
            f"Initiating visual triage diagnostic analysis for: '{issue_title}'",
            extra={"extra_payload": {"issue_title": issue_title, "mime_type": mime_type}},
        )

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        prompt = (
            "You are an expert diagnostic engineer performing visual triage on an automated system failure report. "
            "Analyze the provided UI screenshot and the associated issue title and description. "
            "Identify visible UI elements, determine if any visual anomaly (e.g. 500 error banner, unhandled crash dialog, "
            "broken UI layout, unresponsive controls) is present, calculate a diagnostic confidence score between 0.0 and 1.0, "
            "and provide a concise technical triage summary.\n\n"
            f"Issue Title: {issue_title}\n"
            f"Issue Description: {issue_description}\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VisualTriageResult,
        )

        try:
            response = await self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part],
                config=config,
            )
        except Exception as exc:
            logger.error(
                f"Visual triage API request failed: {exc}",
                extra={"extra_payload": {"error": str(exc), "issue_title": issue_title}},
            )
            raise

        if hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, VisualTriageResult):
                return response.parsed
            if isinstance(response.parsed, dict):
                return VisualTriageResult.model_validate(response.parsed)

        if hasattr(response, "text") and response.text:
            return VisualTriageResult.model_validate_json(response.text)

        raise ValueError("Gemini response did not return valid parsed output or response text.")
