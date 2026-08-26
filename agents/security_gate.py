"""Security Gate Agent module.

Performs static and semantic security audits on generated Ruby patches,
checking for SQL injection, command execution, path traversal, and unsafe parameter handling.
"""

from __future__ import annotations

import logging
from typing import Any
from google import genai
from google.genai import types

from state import CodePatch, ValidationReport
from utils.telemetry import circuit_breaker, get_json_logger

logger: logging.Logger = get_json_logger("agents.security_gate")


class SecurityGateAgent:
    """Agent that audits code patches for security vulnerabilities, OWASP flaws, and syntax integrity."""

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
    async def audit_patch(
        self,
        patch: CodePatch,
        syntax_valid: bool,
        syntax_message: str,
    ) -> ValidationReport:
        """Audits a code patch for vulnerabilities, syntax health, and overall patch quality."""
        logger.info(
            f"Auditing code patch for ticket '{patch.ticket_id}' on file '{patch.target_file}'",
            extra={
                "extra_payload": {
                    "ticket_id": patch.ticket_id,
                    "target_file": patch.target_file,
                    "syntax_valid": syntax_valid,
                }
            },
        )

        prompt = (
            "You are a Principal Application Security Engineer specializing in Ruby on Rails security audits. "
            "Examine the proposed code patch, its syntax validation status, commit rationale, and target file.\n\n"
            "=== PATCH CONTEXT ===\n"
            f"Ticket ID: {patch.ticket_id}\n"
            f"Target File: {patch.target_file}\n"
            f"Applied Branch: {patch.applied_branch}\n"
            f"Commit Message: {patch.commit_message}\n"
            f"Rationale: {patch.rationale}\n\n"
            "=== UNIFIED DIFF ===\n"
            f"{patch.diff}\n\n"
            "=== SANDBOX SYNTAX VALIDATION STATUS ===\n"
            f"Syntax Valid: {syntax_valid}\n"
            f"Sandbox Message: {syntax_message}\n\n"
            "=== SECURITY AUDIT CHECKLIST ===\n"
            "1. SQL Injection: Raw string interpolation in ActiveRecord queries (e.g. where(\"id = #{params[:id]}\")).\n"
            "2. Command Injection: Unsafe execution methods (e.g. system, exec, `%x`, backticks, Open3, IO.popen).\n"
            "3. Path Traversal: Arbitrary file read/write using unvalidated user parameters (e.g. File.read(params[:path])).\n"
            "4. Mass Assignment & Unescaped Parameters: Unsafe permit / assign parameters.\n"
            "5. Syntax Integrity: If sandbox syntax check failed, 'is_valid' and 'syntax_check_passed' MUST be False.\n\n"
            "Provide an exhaustive validation report. If any vulnerability or syntax error exists, set 'is_valid=False' "
            "and list the concrete findings in 'error_logs'. If clean, set 'is_valid=True' and explain in 'summary'."
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ValidationReport,
        )

        try:
            response = await self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=config,
            )
        except Exception as exc:
            logger.error(
                f"Security audit API request failed: {exc}",
                extra={"extra_payload": {"ticket_id": patch.ticket_id, "error": str(exc)}},
            )
            raise

        if hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, ValidationReport):
                return response.parsed
            if isinstance(response.parsed, dict):
                return ValidationReport.model_validate(response.parsed)

        if hasattr(response, "text") and response.text:
            return ValidationReport.model_validate_json(response.text)

        raise ValueError("Gemini response did not return valid parsed ValidationReport schema.")
