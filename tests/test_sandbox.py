"""Unit tests for Dockerized Ruby Sandbox Validator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from sandbox.ruby_validator import RubySandboxValidator


@pytest.fixture
def validator() -> RubySandboxValidator:
    """Fixture providing a RubySandboxValidator instance with short timeout."""
    return RubySandboxValidator(docker_image="ruby:3.2-alpine", timeout_seconds=1.0)


# ==========================================
# Test 1: Valid Syntax Handling (Exit 0)
# ==========================================

@pytest.mark.asyncio
async def test_validate_syntax_success(validator: RubySandboxValidator) -> None:
    """Test 1: Assert valid syntax returns (True, 'Syntax OK') when process returns code 0."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"Syntax OK\n", b""))

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc

        is_valid, message = await validator.validate_syntax("class ValidRuby; def ok; true; end; end")

        assert is_valid is True
        assert message == "Syntax OK"
        mock_exec.assert_awaited_once()
        cmd_args = mock_exec.call_args[0]
        assert "docker" in cmd_args
        assert "ruby:3.2-alpine" in cmd_args
        assert "ruby" in cmd_args
        assert "-c" in cmd_args


# ==========================================
# Test 2: Invalid Syntax Detection (Exit Non-Zero)
# ==========================================

@pytest.mark.asyncio
async def test_validate_syntax_failure(validator: RubySandboxValidator) -> None:
    """Test 2: Assert invalid syntax returns (False, error_msg) when process returns non-zero code."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"-:2: syntax error, unexpected end-of-input, expecting 'end'\n")
    )

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc

        is_valid, message = await validator.validate_syntax("class BrokenRuby\n  def unclosed")

        assert is_valid is False
        assert "syntax error" in message


# ==========================================
# Test 3: Sandbox Timeout Handling
# ==========================================

@pytest.mark.asyncio
async def test_validate_syntax_timeout(validator: RubySandboxValidator) -> None:
    """Test 3: Assert process timeout is caught, process is killed, and returns (False, timeout_msg)."""
    mock_proc = MagicMock()
    mock_proc.kill = MagicMock()

    async def mock_hanging_communicate(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
        await asyncio.sleep(5.0)
        return (b"", b"")

    mock_proc.communicate = mock_hanging_communicate

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc

        is_valid, message = await validator.validate_syntax("loop do; sleep 1; end")

        assert is_valid is False
        assert "timed out" in message.lower()
        mock_proc.kill.assert_called_once()
