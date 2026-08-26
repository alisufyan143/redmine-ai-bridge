"""Telemetry module for System Failure Context Bridge.

Provides structured JSON logging and circuit breaker protection
for asynchronous agent workflows.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import wraps
import json
import logging
import sys
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class AgentTimeoutError(Exception):
    """Raised when an asynchronous agent task exceeds circuit breaker timeout."""

    def __init__(
        self,
        message: str,
        function_name: str | None = None,
        timeout_seconds: float | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.function_name: str | None = function_name
        self.timeout_seconds: float | None = timeout_seconds
        self.args: tuple[Any, ...] = args or ()
        self.kwargs: dict[str, Any] = kwargs or {}

    def __str__(self) -> str:
        return (
            f"{self.message} (function={self.function_name}, "
            f"timeout={self.timeout_seconds}s)"
        )


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        extra_payload = getattr(record, "extra_payload", None)
        if isinstance(extra_payload, dict):
            for key, val in extra_payload.items():
                log_entry[key] = val

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_json_logger(name: str = "redmine_ai_bridge") -> logging.Logger:
    """Configures and returns a logger instance formatted with JSON output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_json_logger("circuit_breaker")


def circuit_breaker(
    timeout_seconds: int = 15,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Asynchronous decorator enforcing a timeout using asyncio.wait_for.

    Logs a CRITICAL JSON event and raises AgentTimeoutError upon timeout.
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        func_name = getattr(func, "__qualname__", getattr(func, "__name__", "unknown_async_func"))

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=float(timeout_seconds),
                )
            except asyncio.TimeoutError as exc:
                safe_args = [repr(a) for a in args]
                safe_kwargs = {k: repr(v) for k, v in kwargs.items()}

                event_payload: dict[str, Any] = {
                    "event": "CIRCUIT_BREAKER_TIMEOUT",
                    "function_name": func_name,
                    "timeout_seconds": float(timeout_seconds),
                    "args": safe_args,
                    "kwargs": safe_kwargs,
                    "error_type": "AgentTimeoutError",
                }

                logger.critical(
                    f"Circuit breaker timeout triggered for function '{func_name}' after {timeout_seconds}s",
                    extra={"extra_payload": event_payload},
                )

                raise AgentTimeoutError(
                    message=f"Function '{func_name}' timed out after {timeout_seconds} seconds.",
                    function_name=func_name,
                    timeout_seconds=float(timeout_seconds),
                    args=args,
                    kwargs=kwargs,
                ) from exc

        return wrapper

    return decorator
