"""Shared pytest fixtures for the CSE Stock Market Data Pipeline test suite.

Fixtures defined here are available to all tests automatically.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ─── Fixtures directory ───────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ─── PDF bytes ────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """A minimal valid PDF (1-page, no content) for testing download validation."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"trailer\n<< /Root 1 0 R /Size 4 >>\n"
        b"startxref\n0\n%%EOF"
    )


@pytest.fixture
def pdf_attachment_b64(minimal_pdf_bytes: bytes) -> str:
    """Base64-encoded PDF bytes as returned by the Gmail API."""
    return base64.urlsafe_b64encode(minimal_pdf_bytes).decode()


# ─── Gmail message payload helpers ───────────────────────────────────────────

def _make_header(name: str, value: str) -> dict:
    return {"name": name, "value": value}


def make_gmail_message(
    message_id: str = "msg_001",
    subject: str = "Comp Sheet - Feb 2024",
    date: str = "Mon, 19 Feb 2024 08:30:00 +0530",
    attachment_id: str | None = "att_001",
    filename: str = "comps.pdf",
    onedrive_url: str | None = None,
) -> dict:
    """Build a minimal Gmail API message payload dict for testing."""
    headers = [
        _make_header("Subject", subject),
        _make_header("Date", date),
        _make_header("From", "research@lolcsecurities.com"),
    ]

    if attachment_id:
        # Direct PDF attachment
        payload: dict[str, Any] = {
            "headers": headers,
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": base64.urlsafe_b64encode(b"Please find attached.").decode()},
                    "parts": [],
                },
                {
                    "mimeType": "application/pdf",
                    "filename": filename,
                    "body": {"attachmentId": attachment_id, "size": 12345},
                    "parts": [],
                },
            ],
        }
    elif onedrive_url:
        # OneDrive link in body
        body_text = f"Please download the comp sheet from:\n{onedrive_url}\n\nRegards"
        payload = {
            "headers": headers,
            "mimeType": "text/plain",
            "body": {
                "data": base64.urlsafe_b64encode(body_text.encode()).decode()
            },
            "parts": [],
        }
    else:
        # No attachment at all
        payload = {
            "headers": headers,
            "mimeType": "text/plain",
            "body": {"data": base64.urlsafe_b64encode(b"No attachment.").decode()},
            "parts": [],
        }

    return {"id": message_id, "threadId": "thread_001", "payload": payload}


@pytest.fixture
def gmail_message_with_attachment() -> dict:
    """Gmail message with a direct PDF attachment."""
    return make_gmail_message(attachment_id="att_001", onedrive_url=None)


@pytest.fixture
def gmail_message_with_onedrive() -> dict:
    """Gmail message with an OneDrive sharing link in the body."""
    return make_gmail_message(
        attachment_id=None,
        onedrive_url="https://onedrive.live.com/share?resid=ABC123",
    )


@pytest.fixture
def gmail_message_no_attachment() -> dict:
    """Gmail message with neither a PDF attachment nor OneDrive link."""
    return make_gmail_message(attachment_id=None, onedrive_url=None)


# ─── Mock Gmail service ───────────────────────────────────────────────────────

@pytest.fixture
def mock_gmail_service(
    gmail_message_with_attachment: dict,
    pdf_attachment_b64: str,
) -> MagicMock:
    """A MagicMock that mimics the Gmail API service object."""
    service = MagicMock()

    # messages().list() → one page with one message, no next page
    service.users().messages().list().execute.return_value = {
        "messages": [{"id": "msg_001", "threadId": "thread_001"}],
        "resultSizeEstimate": 1,
    }

    # messages().get() → the full message with attachment
    service.users().messages().get().execute.return_value = (
        gmail_message_with_attachment
    )

    # attachments().get() → base64-encoded PDF bytes
    service.users().messages().attachments().get().execute.return_value = {
        "size": 100,
        "data": pdf_attachment_b64,
    }

    return service


# ─── Temp data directory ──────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for PDF download tests."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def tmp_checkpoint_file(tmp_path: Path) -> Path:
    """Temporary checkpoint file path."""
    cp_dir = tmp_path / "data" / "checkpoints"
    cp_dir.mkdir(parents=True)
    return cp_dir / "backlog_checkpoint.json"
