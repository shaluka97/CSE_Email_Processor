"""Integration tests for Gmail API authentication.

These tests REQUIRE:
- A valid credentials.json in the project root
- An active internet connection
- The first run will open a browser for OAuth consent

Mark: @pytest.mark.integration — excluded from CI unit test runs.
Run manually: pytest tests/integration/ -m integration -v
"""
from __future__ import annotations

import os
import pytest

from src.email_scanner import GmailScanner
from src.config import settings


@pytest.mark.integration
class TestGmailAuthentication:
    """Verify OAuth flow and basic Gmail API connectivity."""

    @pytest.fixture(autouse=True)
    def require_credentials(self) -> None:
        """Skip if credentials.json is not available."""
        if not os.path.exists(settings.gmail.credentials_path):
            pytest.skip(
                f"credentials.json not found at {settings.gmail.credentials_path!r}. "
                "Download from Google Cloud Console to run integration tests."
            )

    def test_authenticate_and_list_recent_emails(self) -> None:
        """After OAuth, the scanner should be able to list recent emails without error."""
        scanner = GmailScanner()

        assert scanner.service is not None, "Gmail service should be initialised"

        # List a small batch of recent emails (any emails, not just comp sheets)
        response = (
            scanner.service.users()
            .messages()
            .list(userId=settings.gmail.user_id, maxResults=5)
            .execute()
        )

        # The response should be a dict (even if inbox is empty)
        assert isinstance(response, dict)
        # resultSizeEstimate is always present
        assert "resultSizeEstimate" in response

    def test_search_comp_sheet_returns_dict(self) -> None:
        """search_comp_sheet_emails should return a well-formed response."""
        scanner = GmailScanner()

        response = scanner.search_comp_sheet_emails(
            sender=settings.gmail.target_sender,
            subject_keyword=settings.gmail.subject_keyword,
        )

        assert isinstance(response, dict)
        # messages key is present (may be empty list if no comp sheets yet)
        assert "resultSizeEstimate" in response

    def test_token_file_created_after_auth(self) -> None:
        """After authentication, token.json should exist on disk."""
        _scanner = GmailScanner()
        assert os.path.exists(settings.gmail.token_path), (
            f"token.json not found at {settings.gmail.token_path!r} after auth."
        )

    def test_token_refresh_does_not_crash(self) -> None:
        """Instantiating scanner a second time should silently refresh the token."""
        _scanner1 = GmailScanner()
        _scanner2 = GmailScanner()
        assert _scanner2.service is not None
