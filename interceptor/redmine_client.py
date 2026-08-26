"""Asynchronous Redmine REST API client with exponential backoff and attachment security validation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
import aiohttp

from utils.telemetry import circuit_breaker, get_json_logger

logger: logging.Logger = get_json_logger("interceptor.redmine_client")

MAX_ATTACHMENT_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"image/png", "image/jpeg", "image/jpg"})


class RedmineAPIError(Exception):
    """Raised when Redmine REST API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.status_code: int | None = status_code
        self.response_body: str | None = response_body

    def __str__(self) -> str:
        return f"{self.message} (status_code={self.status_code})"


class InvalidAttachmentError(Exception):
    """Raised when an attachment has an unapproved MIME type."""


class AttachmentTooLargeError(Exception):
    """Raised when an attachment exceeds the 5MB size limit."""


class RedmineClient:
    """Asynchronous client for interacting with Redmine REST API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        session: aiohttp.ClientSession | None = None,
        max_retries: int = 3,
        base_backoff_seconds: float = 0.5,
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.api_key: str = api_key
        self._session: aiohttp.ClientSession | None = session
        self._owns_session: bool = session is None
        self.max_retries: int = max_retries
        self.base_backoff_seconds: float = base_backoff_seconds

    async def _get_session(self) -> aiohttp.ClientSession:
        """Retrieves or creates an active aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Closes the underlying aiohttp session if owned by this client."""
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> RedmineClient:
        await self._get_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @circuit_breaker(timeout_seconds=15)
    async def fetch_issue(self, issue_id: int) -> dict[str, Any]:
        """Fetches issue details by ID from Redmine REST API with exponential backoff on HTTP 429."""
        session = await self._get_session()
        url = f"{self.base_url}/issues/{issue_id}.json?include=attachments,relations,journals"
        headers = {
            "X-Redmine-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        last_status: int | None = None
        last_body: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(url, headers=headers) as resp:
                    last_status = resp.status
                    if resp.status == 200:
                        data: dict[str, Any] = await resp.json()
                        return data

                    last_body = await resp.text()

                    if resp.status == 429:
                        if attempt < self.max_retries:
                            backoff_duration = self.base_backoff_seconds * (2 ** attempt)
                            logger.warning(
                                f"HTTP 429 received from Redmine; retrying in {backoff_duration:.2f}s (attempt {attempt + 1}/{self.max_retries})",
                                extra={
                                    "extra_payload": {
                                        "event": "REDMINE_RATE_LIMITED",
                                        "issue_id": issue_id,
                                        "attempt": attempt + 1,
                                        "backoff_seconds": backoff_duration,
                                    }
                                },
                            )
                            await asyncio.sleep(backoff_duration)
                            continue
                        raise RedmineAPIError(
                            f"Redmine rate limit exceeded after {self.max_retries} retries.",
                            status_code=429,
                            response_body=last_body,
                        )

                    raise RedmineAPIError(
                        f"Redmine API returned HTTP {resp.status} for issue {issue_id}.",
                        status_code=resp.status,
                        response_body=last_body,
                    )
            except aiohttp.ClientError as exc:
                if attempt < self.max_retries:
                    backoff_duration = self.base_backoff_seconds * (2 ** attempt)
                    await asyncio.sleep(backoff_duration)
                    continue
                raise RedmineAPIError(
                    f"Redmine network request failed for issue {issue_id}: {exc}",
                    status_code=last_status,
                ) from exc

        raise RedmineAPIError(
            f"Failed to fetch issue {issue_id} after {self.max_retries} retries.",
            status_code=last_status,
            response_body=last_body,
        )

    @circuit_breaker(timeout_seconds=15)
    async def download_attachment(self, attachment_url: str) -> bytes:
        """Downloads an attachment into memory with strict MIME type and 5MB size limits."""
        session = await self._get_session()
        full_url = (
            attachment_url
            if attachment_url.startswith("http://") or attachment_url.startswith("https://")
            else f"{self.base_url}/{attachment_url.lstrip('/')}"
        )
        headers = {"X-Redmine-API-Key": self.api_key}

        async with session.get(full_url, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RedmineAPIError(
                    f"Failed to download attachment from '{full_url}' (HTTP {resp.status}).",
                    status_code=resp.status,
                    response_body=body,
                )

            # Validate Content-Type header
            content_type_header = resp.headers.get("Content-Type", "")
            mime_type = content_type_header.split(";")[0].strip().lower()
            if mime_type not in ALLOWED_MIME_TYPES:
                raise InvalidAttachmentError(
                    f"Forbidden MIME type '{mime_type}'. Only image/png and image/jpeg are permitted."
                )

            # Check advertised Content-Length if present
            content_length_header = resp.headers.get("Content-Length")
            if content_length_header is not None:
                try:
                    content_length = int(content_length_header)
                    if content_length > MAX_ATTACHMENT_SIZE_BYTES:
                        raise AttachmentTooLargeError(
                            f"Attachment size ({content_length} bytes) exceeds the maximum allowed limit of {MAX_ATTACHMENT_SIZE_BYTES} bytes."
                        )
                except ValueError:
                    pass

            # Stream chunks to prevent memory bloat
            buffer = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                buffer.extend(chunk)
                if len(buffer) > MAX_ATTACHMENT_SIZE_BYTES:
                    raise AttachmentTooLargeError(
                        f"Attachment exceeded 5MB size limit during download ({len(buffer)} bytes received so far)."
                    )

            return bytes(buffer)

    @circuit_breaker(timeout_seconds=15)
    async def post_issue_comment(self, issue_id: int, notes: str, private: bool = False) -> bool:
        """Posts a comment (journal note) to a Redmine issue via PUT /issues/{issue_id}.json."""
        session = await self._get_session()
        url = f"{self.base_url}/issues/{issue_id}.json"
        headers = {
            "X-Redmine-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "issue": {
                "notes": notes,
                "private_notes": private,
            }
        }

        async with session.put(url, headers=headers, json=payload) as resp:
            if resp.status in (200, 204):
                logger.info(
                    f"Successfully posted comment to Redmine issue {issue_id}",
                    extra={"extra_payload": {"issue_id": issue_id, "private": private}},
                )
                return True

            body = await resp.text()
            raise RedmineAPIError(
                f"Failed to post comment to Redmine issue {issue_id} (HTTP {resp.status}).",
                status_code=resp.status,
                response_body=body,
            )

