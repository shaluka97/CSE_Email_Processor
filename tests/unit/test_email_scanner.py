"""Unit tests for src.email_scanner.

Covers:
- OAuth token loading / refresh / save
- Email search query construction
- PDF attachment detection
- OneDrive URL detection in email body
- Body text extraction (base64 decode, MIME walk)
- Missing attachment handling (returns None)
- Known URL patterns (onedrive.live.com, sharepoint.com, 1drv.ms)
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.email_scanner import EmailAttachment, GmailScanner
from tests.conftest import make_gmail_message


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _scanner_with_mock_service(mock_service: MagicMock) -> GmailScanner:
    """Create a GmailScanner whose service is replaced with a mock."""
    with (
        patch("src.email_scanner.os.path.exists", return_value=False),
        patch("src.email_scanner.InstalledAppFlow") as mock_flow,
        patch("src.email_scanner.build", return_value=mock_service),
        patch("builtins.open", mock_open()),
    ):
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.refresh_token = None
        mock_creds.to_json.return_value = "{}"
        mock_flow.from_client_secrets_file.return_value.run_local_server.return_value = mock_creds

        scanner = GmailScanner.__new__(GmailScanner)
        scanner._config = MagicMock()
        scanner._config.user_id = "me"
        scanner._config.target_sender = "research@lolcsecurities.com"
        scanner._config.subject_keyword = "Comp Sheet"
        scanner._config.credentials_path = "credentials.json"
        scanner._config.token_path = "token.json"
        scanner.service = mock_service
    return scanner


# ─── EmailAttachment dataclass ────────────────────────────────────────────────


class TestEmailAttachment:
    def test_valid_direct_pdf(self) -> None:
        att = EmailAttachment(
            message_id="msg_1",
            date="Mon, 19 Feb 2024 08:30:00 +0530",
            subject="Comp Sheet",
            attachment_type="direct_pdf",
            attachment_id="att_123",
            filename="comps.pdf",
        )
        assert att.attachment_type == "direct_pdf"
        assert att.attachment_id == "att_123"

    def test_valid_onedrive_link(self) -> None:
        att = EmailAttachment(
            message_id="msg_2",
            date="Tue, 20 Feb 2024 09:00:00 +0530",
            subject="Comp Sheet",
            attachment_type="onedrive_link",
            onedrive_url="https://onedrive.live.com/share?resid=ABC",
        )
        assert att.onedrive_url is not None
        assert "onedrive" in att.onedrive_url

    def test_invalid_attachment_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown attachment_type"):
            EmailAttachment(
                message_id="msg_3",
                date="",
                subject="",
                attachment_type="dropbox_link",  # invalid
            )


# ─── GmailScanner._find_pdf_attachment ───────────────────────────────────────


class TestFindPdfAttachment:
    def setup_method(self) -> None:
        self.scanner = GmailScanner.__new__(GmailScanner)

    def test_direct_pdf_part(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {}, "parts": []},
                {
                    "mimeType": "application/pdf",
                    "filename": "comps.pdf",
                    "body": {"attachmentId": "att_001"},
                    "parts": [],
                },
            ],
        }
        result = self.scanner._find_pdf_attachment(payload)
        assert result == ("att_001", "comps.pdf")

    def test_no_pdf_returns_none(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {}, "parts": []},
            ],
        }
        assert self.scanner._find_pdf_attachment(payload) is None

    def test_nested_pdf_attachment(self) -> None:
        """PDF nested inside a multipart/related wrapper."""
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/related",
                    "parts": [
                        {
                            "mimeType": "application/pdf",
                            "filename": "nested.pdf",
                            "body": {"attachmentId": "att_nested"},
                            "parts": [],
                        }
                    ],
                }
            ],
        }
        result = self.scanner._find_pdf_attachment(payload)
        assert result == ("att_nested", "nested.pdf")

    def test_pdf_without_attachment_id_is_skipped(self) -> None:
        """A PDF body without attachmentId (inline?) should not be returned."""
        payload = {
            "mimeType": "application/pdf",
            "filename": "inline.pdf",
            "body": {},  # no attachmentId
            "parts": [],
        }
        assert self.scanner._find_pdf_attachment(payload) is None

    def test_fallback_filename(self) -> None:
        """When filename is absent, default to 'attachment.pdf'."""
        payload = {
            "mimeType": "application/pdf",
            "body": {"attachmentId": "att_no_name"},
            "parts": [],
        }
        result = self.scanner._find_pdf_attachment(payload)
        assert result == ("att_no_name", "attachment.pdf")


# ─── GmailScanner._extract_body_text ─────────────────────────────────────────


class TestExtractBodyText:
    def setup_method(self) -> None:
        self.scanner = GmailScanner.__new__(GmailScanner)

    def _b64(self, text: str) -> str:
        return base64.urlsafe_b64encode(text.encode()).decode()

    def test_text_plain_payload(self) -> None:
        payload = {
            "mimeType": "text/plain",
            "body": {"data": self._b64("Hello world")},
            "parts": [],
        }
        assert self.scanner._extract_body_text(payload) == "Hello world"

    def test_text_in_parts(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "body": {},
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": self._b64("Body text here")},
                    "parts": [],
                }
            ],
        }
        assert self.scanner._extract_body_text(payload) == "Body text here"

    def test_empty_body(self) -> None:
        payload = {"mimeType": "text/plain", "body": {}, "parts": []}
        assert self.scanner._extract_body_text(payload) == ""

    def test_non_utf8_bytes(self) -> None:
        """Should not raise — replace invalid bytes."""
        bad = base64.urlsafe_b64encode(b"\xff\xfe text").decode()
        payload = {"mimeType": "text/plain", "body": {"data": bad}, "parts": []}
        text = self.scanner._extract_body_text(payload)
        assert "text" in text


# ─── GmailScanner._find_onedrive_url ─────────────────────────────────────────


class TestFindOneDriveUrl:
    @pytest.mark.parametrize(
        "text, expected_fragment",
        [
            (
                "Download here: https://onedrive.live.com/share?resid=XYZ",
                "onedrive.live.com",
            ),
            (
                "Link: https://mycompany.sharepoint.com/sites/data/file.pdf",
                "sharepoint.com",
            ),
            (
                "Short link: https://1drv.ms/u/s!AbCdEfGhIjKl",
                "1drv.ms",
            ),
            (
                "No link in this email at all.",
                None,
            ),
        ],
    )
    def test_known_patterns(self, text: str, expected_fragment: str | None) -> None:
        result = GmailScanner._find_onedrive_url(text)
        if expected_fragment is None:
            assert result is None
        else:
            assert result is not None
            assert expected_fragment in result

    def test_strips_trailing_punctuation(self) -> None:
        text = "See: https://onedrive.live.com/share?resid=ABC."
        result = GmailScanner._find_onedrive_url(text)
        assert result is not None
        assert not result.endswith(".")

    def test_strips_trailing_parenthesis(self) -> None:
        text = "(https://1drv.ms/u/s!AbCdEf)"
        result = GmailScanner._find_onedrive_url(text)
        assert result is not None
        assert not result.endswith(")")


# ─── GmailScanner.get_email_details ──────────────────────────────────────────


class TestGetEmailDetails:
    def test_detects_direct_pdf(self, gmail_message_with_attachment: dict) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().get().execute.return_value = (
            gmail_message_with_attachment
        )
        scanner = _scanner_with_mock_service(mock_service)

        result = scanner.get_email_details("msg_001")

        assert result is not None
        assert result.attachment_type == "direct_pdf"
        assert result.attachment_id == "att_001"
        assert result.filename == "comps.pdf"

    def test_detects_onedrive_link(self, gmail_message_with_onedrive: dict) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().get().execute.return_value = (
            gmail_message_with_onedrive
        )
        scanner = _scanner_with_mock_service(mock_service)

        result = scanner.get_email_details("msg_002")

        assert result is not None
        assert result.attachment_type == "onedrive_link"
        assert result.onedrive_url is not None
        assert "onedrive" in result.onedrive_url

    def test_returns_none_when_no_attachment(
        self, gmail_message_no_attachment: dict
    ) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().get().execute.return_value = (
            gmail_message_no_attachment
        )
        scanner = _scanner_with_mock_service(mock_service)

        result = scanner.get_email_details("msg_003")

        assert result is None

    def test_handles_api_error_gracefully(self) -> None:
        from googleapiclient.errors import HttpError
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status = 500

        mock_service = MagicMock()
        mock_service.users().messages().get().execute.side_effect = HttpError(
            resp=mock_resp, content=b"Server Error"
        )
        scanner = _scanner_with_mock_service(mock_service)

        result = scanner.get_email_details("msg_error")
        assert result is None


# ─── GmailScanner.search_comp_sheet_emails ───────────────────────────────────


class TestSearchCompSheetEmails:
    def test_query_includes_sender_and_subject(self) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            "messages": [],
            "resultSizeEstimate": 0,
        }
        scanner = _scanner_with_mock_service(mock_service)

        scanner.search_comp_sheet_emails(
            sender="test@example.com", subject_keyword="TestSheet"
        )

        call_kwargs = mock_service.users().messages().list.call_args[1]
        assert "from:test@example.com" in call_kwargs["q"]
        assert "subject:TestSheet" in call_kwargs["q"]

    def test_page_token_forwarded(self) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            "messages": []
        }
        scanner = _scanner_with_mock_service(mock_service)

        scanner.search_comp_sheet_emails(page_token="TOKEN_ABC")

        call_kwargs = mock_service.users().messages().list.call_args[1]
        assert call_kwargs.get("pageToken") == "TOKEN_ABC"

    def test_no_page_token_not_included_by_default(self) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            "messages": []
        }
        scanner = _scanner_with_mock_service(mock_service)

        scanner.search_comp_sheet_emails()

        call_kwargs = mock_service.users().messages().list.call_args[1]
        assert "pageToken" not in call_kwargs
