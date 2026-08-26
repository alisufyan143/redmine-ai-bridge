"""Dockerized Ruby sandbox syntax validation module."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from utils.telemetry import circuit_breaker, get_json_logger

logger: logging.Logger = get_json_logger("sandbox.ruby_validator")


class RubySandboxValidator:
    """Validates Ruby syntax inside an isolated, resource-constrained Docker container."""

    def __init__(
        self,
        docker_image: str = "ruby:3.2-alpine",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.docker_image: str = docker_image
        self.timeout_seconds: float = timeout_seconds

    @circuit_breaker(timeout_seconds=15)
    async def validate_syntax(self, code_snippet: str) -> tuple[bool, str]:
        """Runs 'ruby -c' inside an isolated Docker sandbox using stdin streaming."""
        cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--memory",
            "128m",
            "--cpus",
            "0.5",
            self.docker_image,
            "ruby",
            "-c",
        ]

        input_bytes = code_snippet.encode("utf-8", errors="replace")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=input_bytes),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                logger.error("Sandbox syntax validation timed out after 10 seconds.")
                return False, f"Syntax validation timed out after {self.timeout_seconds} seconds."

            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                return True, "Syntax OK"

            error_msg = stderr_text or stdout_text or f"Process exited with non-zero status ({proc.returncode})"
            logger.warning(
                f"Ruby syntax validation failed: {error_msg}",
                extra={"extra_payload": {"error": error_msg, "exit_code": proc.returncode}},
            )
            return False, error_msg

        except Exception as exc:
            logger.error(
                f"Failed to spawn Docker sandbox: {exc}",
                extra={"extra_payload": {"error": str(exc)}},
            )
            return False, f"Sandbox execution error: {exc}"
