"""Gmail email scanner for CSE Comp Sheet emails.

Responsibilities
----------------
- OAuth2 authentication with automatic token refresh
- Search Gmail for emails from the CSE broker matching 'Comp Sheet'
- Detect whether each email carries a direct PDF attachment or an OneDrive link
- Return structured EmailAttachment dataclasses for downstream processing

Week 1 of the CSE Stock Market Data Pipeline.
"""
from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import GmailConfig, settings

logger = logging.getLogger(__name__)

# Read-only Gmail scope — we never modify emails
GMAIL_SCOPES: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]

# OneDrive / SharePoint URL patterns to search for in email bodies
_ONEDRIVE_PATTERNS: list[str] = [
    r"https://[^\s]*onedrive\.live\.com[^\s]*",
    r"https://[^\s]*sharepoint\.com[^\s]*",
    r"https://1drv\.ms/[^\s]*",
]


@dataclass
class EmailAttachment:
    """Holds everything the downloader needs to fetch one comp sheet PDF."""

    message_id: str
    date: str           # Raw RFC 2822 date string from email header
    subject: str
    attachment_type: str  # "direct_pdf" | "onedrive_link"

    # Set for direct_pdf attachments
    attachment_id: Optional[str] = None
    filename: Optional[str] = None

    # Set for onedrive_link attachments
    onedrive_url: Optional[str] = None
    # All OneDrive URLs found in the email (for filtering by filename)
    all_onedrive_urls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.attachment_type not in ("direct_pdf", "onedrive_link"):
            raise ValueError(
                f"Unknown attachment_type: {self.attachment_type!r}. "
                "Expected 'direct_pdf' or 'onedrive_link'."
            )


class GmailScanner:
    """Authenticated Gmail API client scoped to Comp Sheet email scanning.

    Usage
    -----
    >>> scanner = GmailScanner(config=settings.gmail)
    >>> for page in scanner.iter_comp_sheet_pages():
    ...     for msg in page:
    ...         attachment = scanner.get_email_details(msg["id"])
    """

    def __init__(self, config: Optional[GmailConfig] = None) -> None:
        self._config = config or settings.gmail
        self.service = self._build_service()

    # ─── Authentication ───────────────────────────────────────────────────────

    def _build_service(self):  # type: ignore[no-untyped-def]
        """Authenticate and return the Gmail API service object."""
        creds = self._load_credentials()

        if not creds or not creds.valid:
            creds = self._refresh_or_reauthorise(creds)

        return build("gmail", "v1", credentials=creds)

    def _load_credentials(self) -> Optional[Credentials]:
        """Load saved token from disk if it exists."""
        token_path = self._config.token_path
        if os.path.exists(token_path):
            return Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)
        return None

    def _refresh_or_reauthorise(
        self, creds: Optional[Credentials]
    ) -> Credentials:
        """Refresh an expired token or run the full OAuth flow."""
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired Gmail OAuth token.")
            creds.refresh(Request())
        else:
            logger.info("No valid token found — launching OAuth flow.")
            flow = InstalledAppFlow.from_client_secrets_file(
                self._config.credentials_path, GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)

        self._save_token(creds)
        return creds

    def _save_token(self, creds: Credentials) -> None:
        """Persist token to disk so future runs skip the browser flow."""
        token_path = self._config.token_path
        with open(token_path, "w") as fh:
            fh.write(creds.to_json())
        logger.debug("OAuth token saved to %s", token_path)

    # ─── Email Search ─────────────────────────────────────────────────────────

    def iter_comp_sheet_pages(
        self,
        sender: Optional[str] = None,
        subject_keyword: Optional[str] = None,
        page_token: Optional[str] = None,
    ) -> Iterator[list[dict]]:
        """Yield pages of message stubs (id + threadId) matching Comp Sheet emails.

        Parameters
        ----------
        sender:
            Gmail sender to filter on. Defaults to ``config.target_sender``.
        subject_keyword:
            Subject keyword filter. Defaults to ``config.subject_keyword``.
        page_token:
            Resume from this page (for backlog processor checkpoint/resume).

        Yields
        ------
        list[dict]
            Each dict has ``{"id": ..., "threadId": ...}``.
        """
        sender = sender or self._config.target_sender
        subject_keyword = subject_keyword or self._config.subject_keyword
        query = f"from:{sender} subject:{subject_keyword}"

        while True:
            params: dict = {
                "userId": self._config.user_id,
                "q": query,
                "maxResults": 50,
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                response = (
                    self.service.users().messages().list(**params).execute()
                )
            except HttpError as exc:
                logger.error("Gmail API error during list: %s", exc)
                raise

            messages = response.get("messages", [])
            if not messages:
                logger.debug("No messages on this page.")
                return

            yield messages

            page_token = response.get("nextPageToken")
            if not page_token:
                return

    def search_comp_sheet_emails(
        self,
        sender: Optional[str] = None,
        subject_keyword: Optional[str] = None,
        page_token: Optional[str] = None,
    ) -> dict:
        """Return a single page of results — convenience method for backlog processor.

        Returns the raw Gmail API response dict:
        ``{"messages": [...], "nextPageToken": "...", "resultSizeEstimate": N}``
        """
        sender = sender or self._config.target_sender
        subject_keyword = subject_keyword or self._config.subject_keyword
        query = f"from:{sender} subject:{subject_keyword}"

        params: dict = {
            "userId": self._config.user_id,
            "q": query,
            "maxResults": 50,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            return self.service.users().messages().list(**params).execute()
        except HttpError as exc:
            logger.error("Gmail API error during search: %s", exc)
            raise

    # ─── Email Detail Parsing ─────────────────────────────────────────────────

    def get_email_details(self, message_id: str) -> Optional[EmailAttachment]:
        """Fetch full message and return a structured EmailAttachment.

        Returns None if no PDF attachment or OneDrive link is found.
        """
        try:
            message = (
                self.service.users()
                .messages()
                .get(userId=self._config.user_id, id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:
            logger.error("Failed to fetch message %s: %s", message_id, exc)
            return None

        headers = {
            h["name"]: h["value"]
            for h in message["payload"].get("headers", [])
        }
        date = headers.get("Date", "")
        subject = headers.get("Subject", "")

        # Priority 1: direct PDF attachment
        attachment_info = self._find_pdf_attachment(message["payload"])
        if attachment_info:
            attachment_id, filename = attachment_info
            logger.info(
                "Message %s has direct PDF attachment: %s", message_id, filename
            )
            return EmailAttachment(
                message_id=message_id,
                date=date,
                subject=subject,
                attachment_type="direct_pdf",
                attachment_id=attachment_id,
                filename=filename,
            )

        # Priority 2: OneDrive link in body
        body_text = self._extract_body_text(message["payload"])
        all_onedrive_urls = self._find_all_onedrive_urls(body_text)
        if all_onedrive_urls:
            logger.info(
                "Message %s has %d OneDrive link(s): %s",
                message_id,
                len(all_onedrive_urls),
                all_onedrive_urls[0],
            )
            return EmailAttachment(
                message_id=message_id,
                date=date,
                subject=subject,
                attachment_type="onedrive_link",
                onedrive_url=all_onedrive_urls[0],
                all_onedrive_urls=all_onedrive_urls,
            )

        logger.warning(
            "No PDF attachment or OneDrive link found in message %s ('%s')",
            message_id,
            subject,
        )
        return None

    # ─── Internal Parsing Helpers ─────────────────────────────────────────────

    def _find_pdf_attachment(
        self, payload: dict
    ) -> Optional[tuple[str, str]]:
        """Recursively search payload parts for a PDF attachment.

        Prefers PDFs with "Comps" in the filename over other PDFs
        (e.g., "Market Insights").

        Returns
        -------
        (attachment_id, filename) or None
        """
        # Collect all PDF attachments first
        all_pdfs = self._collect_all_pdf_attachments(payload)

        if not all_pdfs:
            return None

        # Prefer PDFs with "Comps" in the filename (case-insensitive)
        for attachment_id, filename in all_pdfs:
            if re.search(r"comps", filename, re.IGNORECASE):
                return attachment_id, filename

        # Fallback to first PDF if no "Comps" match
        return all_pdfs[0]

    def _collect_all_pdf_attachments(
        self, payload: dict
    ) -> list[tuple[str, str]]:
        """Recursively collect all PDF attachments from a message payload.

        Returns
        -------
        List of (attachment_id, filename) tuples.
        """
        results: list[tuple[str, str]] = []
        mime_type = payload.get("mimeType", "")

        if mime_type == "application/pdf":
            body = payload.get("body", {})
            attachment_id = body.get("attachmentId")
            filename = payload.get("filename", "attachment.pdf")
            if attachment_id:
                results.append((attachment_id, filename))

        # Some mailers wrap attachments in multipart/mixed
        for part in payload.get("parts", []):
            results.extend(self._collect_all_pdf_attachments(part))

        return results

    def _extract_body_text(self, payload: dict) -> str:
        """Decode and return the plain-text body of an email.

        Walks the MIME tree to find the first text/plain part.
        """
        mime_type = payload.get("mimeType", "")

        if mime_type == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                decoded = base64.urlsafe_b64decode(data.encode())
                return decoded.decode("utf-8", errors="replace")

        for part in payload.get("parts", []):
            text = self._extract_body_text(part)
            if text:
                return text

        return ""

    @staticmethod
    def _find_onedrive_url(text: str) -> Optional[str]:
        """Extract a OneDrive / SharePoint sharing URL from plain text.

        Returns the first match, stripping trailing punctuation.
        For multiple URLs, use _find_all_onedrive_urls().
        """
        urls = GmailScanner._find_all_onedrive_urls(text)
        return urls[0] if urls else None

    @staticmethod
    def _find_all_onedrive_urls(text: str) -> list[str]:
        """Extract ALL OneDrive / SharePoint sharing URLs from plain text.

        Returns a list of URLs, each stripped of trailing punctuation.
        """
        urls = []
        for pattern in _ONEDRIVE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                url = match.group(0).rstrip(".,;)>\"'")
                # Remove invisible characters
                url = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00a0]", "", url)
                url = url.strip()
                if url and url not in urls:
                    urls.append(url)
        return urls
