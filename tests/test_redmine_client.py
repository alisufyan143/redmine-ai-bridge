"""Unit tests for Redmine API Interceptor Client."""

from __future__ import annotations

import functools
from typing import Any, Generator
from unittest.mock import Mock

from aiohttp.client_reqrep import ClientResponse
from aiohttp.streams import StreamReader
from aioresponses import aioresponses
import pytest

from interceptor.redmine_client import (
    AttachmentTooLargeError,
    InvalidAttachmentError,
    RedmineAPIError,
    RedmineClient,
)

# Compatibility shims for aiohttp 3.11+ / aioresponses
_orig_response_init = ClientResponse.__init__


@functools.wraps(_orig_response_init)
def _patched_response_init(self: ClientResponse, method: str, url: Any, **kwargs: Any) -> None:
    if "stream_writer" not in kwargs or kwargs["stream_writer"] is None:
        sw = Mock()
        sw.output_size = 0
        kwargs["stream_writer"] = sw
    _orig_response_init(self, method, url, **kwargs)
    if hasattr(self, "_protocol") and self._protocol is not None:
        self._protocol._parser = Mock()


ClientResponse.__init__ = _patched_response_init

_orig_sr_init = StreamReader.__init__


@functools.wraps(_orig_sr_init)
def _patched_sr_init(self: StreamReader, protocol: Any, limit: int = 2**16, loop: Any = None) -> None:
    if protocol is not None and (not hasattr(protocol, "_parser") or getattr(protocol, "_parser", None) is None):
        protocol._parser = Mock()
    _orig_sr_init(self, protocol, limit, loop=loop)


StreamReader.__init__ = _patched_sr_init


@pytest.fixture
def redmine_client() -> Generator[RedmineClient, None, None]:
    """Fixture providing a RedmineClient instance."""
    client = RedmineClient(
        base_url="https://redmine.example.com",
        api_key="test_api_key",
        max_retries=2,
        base_backoff_seconds=0.01,
    )
    yield client


# ==========================================
# Test 1: Successful Issue Fetch & Image Download
# ==========================================

@pytest.mark.asyncio
async def test_fetch_issue_success(redmine_client: RedmineClient) -> None:
    """Assert successful retrieval of Redmine issue payload."""
    issue_url = "https://redmine.example.com/issues/1001.json?include=attachments,relations,journals"
    mock_payload = {
        "issue": {
            "id": 1001,
            "subject": "Crash on login",
            "description": "500 Internal Server Error when authenticating",
            "attachments": [
                {
                    "id": 50,
                    "filename": "screenshot.png",
                    "content_url": "https://redmine.example.com/attachments/download/50/screenshot.png",
                    "content_type": "image/png",
                }
            ],
        }
    }

    with aioresponses() as m:
        m.get(issue_url, status=200, payload=mock_payload)

        try:
            result = await redmine_client.fetch_issue(1001)
            assert result["issue"]["id"] == 1001
            assert result["issue"]["subject"] == "Crash on login"
            assert len(result["issue"]["attachments"]) == 1
        finally:
            await redmine_client.close()


@pytest.mark.asyncio
async def test_download_attachment_png_and_jpeg_success(redmine_client: RedmineClient) -> None:
    """Assert valid PNG and JPEG attachments under 5MB download successfully."""
    png_url = "https://redmine.example.com/attachments/download/50/screenshot.png"
    jpeg_url = "https://redmine.example.com/attachments/download/51/photo.jpg"

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake_png_data"
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"fake_jpeg_data"

    with aioresponses() as m:
        m.get(png_url, status=200, body=png_bytes, headers={"Content-Type": "image/png"})
        m.get(jpeg_url, status=200, body=jpeg_bytes, headers={"Content-Type": "image/jpeg"})

        try:
            downloaded_png = await redmine_client.download_attachment(png_url)
            assert downloaded_png == png_bytes

            downloaded_jpeg = await redmine_client.download_attachment(jpeg_url)
            assert downloaded_jpeg == jpeg_bytes
        finally:
            await redmine_client.close()


# ==========================================
# Test 2: Exponential Backoff on HTTP 429
# ==========================================

@pytest.mark.asyncio
async def test_fetch_issue_exponential_backoff_recovery(redmine_client: RedmineClient) -> None:
    """Assert client retries on 429 and succeeds when Redmine recovers on subsequent attempt."""
    issue_url = "https://redmine.example.com/issues/2002.json?include=attachments,relations,journals"
    mock_payload = {"issue": {"id": 2002, "subject": "Recovered Issue"}}

    with aioresponses() as m:
        m.get(issue_url, status=429, body="Rate limit exceeded")
        m.get(issue_url, status=200, payload=mock_payload)

        try:
            result = await redmine_client.fetch_issue(2002)
            assert result["issue"]["id"] == 2002
            assert result["issue"]["subject"] == "Recovered Issue"
        finally:
            await redmine_client.close()


@pytest.mark.asyncio
async def test_fetch_issue_exhausts_retries_on_persistent_429(redmine_client: RedmineClient) -> None:
    """Assert client raises RedmineAPIError when 429 persists through all retries."""
    issue_url = "https://redmine.example.com/issues/3003.json?include=attachments,relations,journals"

    with aioresponses() as m:
        m.get(issue_url, status=429, body="Rate limit exceeded")
        m.get(issue_url, status=429, body="Rate limit exceeded")
        m.get(issue_url, status=429, body="Rate limit exceeded")

        try:
            with pytest.raises(RedmineAPIError) as exc_info:
                await redmine_client.fetch_issue(3003)

            assert exc_info.value.status_code == 429
            assert "rate limit exceeded" in str(exc_info.value).lower()
        finally:
            await redmine_client.close()


# ==========================================
# Test 3: Security Validation (MIME & Size Limits)
# ==========================================

@pytest.mark.asyncio
async def test_download_attachment_rejects_invalid_mime_types(redmine_client: RedmineClient) -> None:
    """Assert non-image MIME types (PDF, executables, HTML) are rejected."""
    pdf_url = "https://redmine.example.com/attachments/download/90/doc.pdf"
    exe_url = "https://redmine.example.com/attachments/download/91/malicious.exe"

    with aioresponses() as m:
        m.get(pdf_url, status=200, body=b"%PDF-1.4", headers={"Content-Type": "application/pdf"})
        m.get(exe_url, status=200, body=b"MZ\x90\x00", headers={"Content-Type": "application/x-msdownload"})

        try:
            with pytest.raises(InvalidAttachmentError) as exc_pdf:
                await redmine_client.download_attachment(pdf_url)
            assert "application/pdf" in str(exc_pdf.value)

            with pytest.raises(InvalidAttachmentError) as exc_exe:
                await redmine_client.download_attachment(exe_url)
            assert "application/x-msdownload" in str(exc_exe.value)
        finally:
            await redmine_client.close()


@pytest.mark.asyncio
async def test_download_attachment_rejects_oversized_content_length(redmine_client: RedmineClient) -> None:
    """Assert attachments with Content-Length > 5MB are rejected before downloading body."""
    large_url = "https://redmine.example.com/attachments/download/99/large_image.png"
    size_6mb = 6 * 1024 * 1024

    with aioresponses() as m:
        m.get(
            large_url,
            status=200,
            body=b"header_only",
            headers={"Content-Type": "image/png", "Content-Length": str(size_6mb)},
        )

        try:
            with pytest.raises(AttachmentTooLargeError) as exc_info:
                await redmine_client.download_attachment(large_url)

            assert "5MB" in str(exc_info.value) or "exceeds" in str(exc_info.value)
        finally:
            await redmine_client.close()


@pytest.mark.asyncio
async def test_download_attachment_rejects_oversized_stream(redmine_client: RedmineClient) -> None:
    """Assert streaming body exceeding 5MB triggers AttachmentTooLargeError dynamically."""
    stream_url = "https://redmine.example.com/attachments/download/100/stream.png"
    oversized_body = b"A" * (5 * 1024 * 1024 + 1024)  # 5MB + 1KB

    with aioresponses() as m:
        m.get(
            stream_url,
            status=200,
            body=oversized_body,
            headers={"Content-Type": "image/png"},
        )

        try:
            with pytest.raises(AttachmentTooLargeError):
                await redmine_client.download_attachment(stream_url)
        finally:
            await redmine_client.close()
