"""PDF Downloader — handles both Gmail direct attachments and OneDrive sharing links.

Responsibilities
----------------
- Download raw PDF bytes from a Gmail attachment by attachment_id
- Parse and convert OneDrive sharing URLs to direct download URLs
- Save files to data/raw/{YYYY-MM-DD}_comps.pdf
- Skip duplicate files (idempotent re-runs)
- Gracefully handle HTTP errors: 403 (expired link), 404 (deleted), timeouts

Week 1 of the CSE Stock Market Data Pipeline.
"""
from __future__ import annotations

import base64
import logging
import re
from email.utils import parsedate_to_datetime
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from googleapiclient.errors import HttpError

from src.config import HttpConfig, StorageConfig, settings

logger = logging.getLogger(__name__)

# Magic bytes that identify a valid PDF file
_PDF_MAGIC = b"%PDF"


class PDFDownloader:
    """Downloads comp-sheet PDFs from Gmail attachments or OneDrive links.

    Usage
    -----
    >>> dl = PDFDownloader()
    >>> path = dl.download_attachment(
    ...     gmail_service=scanner.service,
    ...     user_id="me",
    ...     message_id="...",
    ...     attachment_id="...",
    ...     date_str="Mon, 19 Feb 2024 08:30:00 +0530",
    ... )
    """

    def __init__(
        self,
        storage_config: Optional[StorageConfig] = None,
        http_config: Optional[HttpConfig] = None,
    ) -> None:
        self._storage = storage_config or settings.storage
        self._http = http_config or settings.http
        self._data_dir = Path(self._storage.data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ─── File Naming / Deduplication ─────────────────────────────────────────

    def build_filepath(self, date_str: str) -> Path:
        """Convert an RFC 2822 email date string to the output PDF path.

        Example: "Mon, 19 Feb 2024 08:30:00 +0530" → data/raw/2024-02-19_comps.pdf
        """
        try:
            dt = parsedate_to_datetime(date_str)
            date_part = dt.strftime("%Y-%m-%d")
        except Exception:
            logger.warning(
                "Could not parse date %r — using today's date as fallback.", date_str
            )
            date_part = datetime.now().strftime("%Y-%m-%d")
        return self._data_dir / f"{date_part}_comps.pdf"

    def is_duplicate(self, filepath: Path) -> bool:
        """Return True if the file already exists on disk."""
        return filepath.exists() and filepath.stat().st_size > 0

    # ─── Gmail Attachment Download ────────────────────────────────────────────

    def download_attachment(
        self,
        gmail_service,  # type: ignore[no-untyped-def]
        user_id: str,
        message_id: str,
        attachment_id: str,
        date_str: str,
    ) -> Optional[Path]:
        """Download a PDF directly from a Gmail message attachment.

        Parameters
        ----------
        gmail_service:
            Authenticated Gmail API service object.
        user_id:
            Gmail user ID (``"me"`` for authenticated user).
        message_id:
            ID of the Gmail message containing the attachment.
        attachment_id:
            ID of the attachment (from email payload).
        date_str:
            RFC 2822 date from the email header, used to name the output file.

        Returns
        -------
        Path to saved file, or None on failure.
        """
        filepath = self.build_filepath(date_str)

        if self.is_duplicate(filepath):
            logger.info("Skipping duplicate (attachment): %s", filepath)
            return filepath

        try:
            attachment = (
                gmail_service.users()
                .messages()
                .attachments()
                .get(userId=user_id, messageId=message_id, id=attachment_id)
                .execute()
            )
        except HttpError as exc:
            logger.error(
                "Gmail API error fetching attachment %s from message %s: %s",
                attachment_id,
                message_id,
                exc,
            )
            return None

        raw_data = attachment.get("data", "")
        if not raw_data:
            logger.error("Attachment %s returned empty data.", attachment_id)
            return None

        pdf_bytes = base64.urlsafe_b64decode(raw_data.encode())

        if not self._is_valid_pdf(pdf_bytes):
            logger.error(
                "Downloaded bytes from attachment %s do not look like a PDF.",
                attachment_id,
            )
            return None

        filepath.write_bytes(pdf_bytes)
        logger.info(
            "Saved Gmail attachment → %s (%d bytes)", filepath, len(pdf_bytes)
        )
        return filepath

    # ─── OneDrive Download ────────────────────────────────────────────────────

    def download_from_onedrive(
        self,
        sharing_url: str,
        date_str: str,
    ) -> Optional[Path]:
        """Download a PDF from a OneDrive sharing URL.

        Handles:
        - onedrive.live.com personal sharing links
        - sharepoint.com business sharing links
        - 1drv.ms short URLs (followed to resolution)
        - Expired / deleted links (403, 404) — logs and returns None
        - Non-PDF responses — detected by magic bytes

        Parameters
        ----------
        sharing_url:
            The raw sharing URL from the email body.
        date_str:
            RFC 2822 date string for output file naming.

        Returns
        -------
        Path to saved file, or None on failure.
        """
        # Clean URL: strip trailing punctuation and invisible Unicode characters
        original_url = sharing_url
        sharing_url = self._sanitize_url(sharing_url)
        if sharing_url != original_url:
            logger.debug("Sanitized URL: %r → %r", original_url, sharing_url)

        filepath = self.build_filepath(date_str)

        if self.is_duplicate(filepath):
            logger.info("Skipping duplicate (OneDrive): %s", filepath)
            return filepath

        direct_url = self.convert_to_direct_download_url(sharing_url)
        if not direct_url:
            logger.error(
                "Could not resolve a direct download URL from: %s", sharing_url
            )
            return None

        logger.info("Converted OneDrive URL: %s → %s", sharing_url, direct_url)

        try:
            response = requests.get(
                direct_url,
                timeout=self._http.onedrive_timeout,
                allow_redirects=True,
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 403:
                logger.error(
                    "OneDrive link expired or access denied (403). "
                    "Original URL: %s",
                    sharing_url,
                )
            elif status == 404:
                logger.error(
                    "OneDrive file not found (404). Original URL: %s",
                    sharing_url,
                )
            else:
                logger.error(
                    "HTTP %s error downloading OneDrive file. URL: %s",
                    status,
                    sharing_url,
                )
            return None
        except requests.exceptions.Timeout:
            logger.error(
                "Timeout (%ds) downloading OneDrive file: %s",
                self._http.onedrive_timeout,
                sharing_url,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("Request failed for %s: %s", sharing_url, exc)
            return None

        pdf_bytes = response.content

        if not self._is_valid_pdf(pdf_bytes):
            content_type = response.headers.get("Content-Type", "unknown")
            logger.warning(
                "Anonymous download returned non-PDF (Content-Type: %s). "
                "Trying Microsoft Graph API with authentication...",
                content_type,
            )
            # Try authenticated download via Microsoft Graph API
            pdf_bytes = self._download_via_graph_api(sharing_url)
            if pdf_bytes is None:
                logger.error(
                    "Both anonymous and authenticated downloads failed for: %s",
                    sharing_url,
                )
                return None

        filepath.write_bytes(pdf_bytes)
        logger.info(
            "Saved OneDrive download → %s (%d bytes)", filepath, len(pdf_bytes)
        )
        return filepath

    def _download_via_graph_api(self, sharing_url: str) -> Optional[bytes]:
        """Download a SharePoint/OneDrive file using authenticated methods.

        Tries in order:
        1. Microsoft Graph API (if configured)
        2. Browser automation (Playwright)
        """
        # Try 1: Microsoft Graph API (if configured)
        pdf_bytes = self._try_graph_api(sharing_url)
        if pdf_bytes:
            return pdf_bytes

        # Try 2: Browser automation
        pdf_bytes = self._try_browser_download(sharing_url)
        if pdf_bytes:
            return pdf_bytes

        return None

    def _try_graph_api(self, sharing_url: str) -> Optional[bytes]:
        """Try downloading via Microsoft Graph API."""
        try:
            from src.microsoft_auth import get_microsoft_authenticator
        except ImportError:
            logger.debug("Microsoft auth module not available")
            return None

        auth = get_microsoft_authenticator()

        if not auth.is_configured():
            logger.debug("Microsoft OAuth not configured, skipping Graph API")
            return None

        pdf_bytes = auth.download_sharepoint_file(
            sharing_url=sharing_url,
            timeout=self._http.onedrive_timeout,
        )

        if pdf_bytes and self._is_valid_pdf(pdf_bytes):
            logger.info("Successfully downloaded via Microsoft Graph API")
            return pdf_bytes

        return None

    def _try_browser_download(
        self,
        sharing_url: str,
        require_comps: bool = False,
    ) -> Optional[bytes]:
        """Try downloading via browser automation (Playwright).

        Parameters
        ----------
        sharing_url:
            The SharePoint/OneDrive URL to download.
        require_comps:
            If True, only accept files with "Comps" in the filename.

        Returns
        -------
        PDF bytes, or None on failure.
        """
        try:
            from src.browser_downloader import download_sharepoint_file
        except ImportError as exc:
            logger.error(
                "Browser downloader not available. "
                "Run: pip install playwright && playwright install chromium. "
                "Error: %s",
                exc,
            )
            return None

        logger.info("Attempting browser-based download (may require login)...")
        result = download_sharepoint_file(sharing_url, headless=False)

        if result is None:
            return None

        pdf_bytes, filename = result

        if not self._is_valid_pdf(pdf_bytes):
            logger.error(
                "Browser download succeeded but content is not a PDF (first bytes: %r)",
                pdf_bytes[:8],
            )
            return None

        # Check filename filter
        if require_comps and not re.search(r"comps", filename, re.IGNORECASE):
            logger.info(
                "Downloaded file '%s' is not a Comps file, skipping...", filename
            )
            return None

        logger.info("Successfully downloaded via browser: %s", filename)
        return pdf_bytes

    def download_from_onedrive_urls(
        self,
        urls: list[str],
        date_str: str,
    ) -> Optional[Path]:
        """Download the Comps PDF from a list of OneDrive URLs.

        Tries each URL until it finds one with "Comps" in the filename.

        Parameters
        ----------
        urls:
            List of OneDrive/SharePoint URLs to try.
        date_str:
            RFC 2822 date string for output file naming.

        Returns
        -------
        Path to saved file, or None on failure.
        """
        filepath = self.build_filepath(date_str)

        if self.is_duplicate(filepath):
            logger.info("Skipping duplicate (OneDrive): %s", filepath)
            return filepath

        for i, url in enumerate(urls):
            logger.info("Trying OneDrive URL %d/%d...", i + 1, len(urls))
            url = self._sanitize_url(url)

            # Try browser download with filename check
            pdf_bytes = self._try_browser_download(url, require_comps=True)

            if pdf_bytes:
                filepath.write_bytes(pdf_bytes)
                logger.info(
                    "Saved OneDrive download → %s (%d bytes)", filepath, len(pdf_bytes)
                )
                return filepath

        logger.error("No Comps PDF found in any of the %d OneDrive URLs", len(urls))
        return None

    # ─── URL Conversion ───────────────────────────────────────────────────────

    def convert_to_direct_download_url(
        self, sharing_url: str
    ) -> Optional[str]:
        """Convert a OneDrive sharing URL to a direct-download URL.

        Strategy
        --------
        1. Expand 1drv.ms short URLs by following the redirect (HEAD request).
        2. Append ``download=1`` to onedrive.live.com personal links.
        3. Append ``download=1`` to sharepoint.com business links.

        Returns None if the URL cannot be resolved.
        """
        # Step 1: Expand short URL
        if "1drv.ms" in sharing_url:
            resolved = self._resolve_short_url(sharing_url)
            if not resolved:
                return None
            sharing_url = resolved

        # Step 2: Append download parameter
        if "onedrive.live.com" in sharing_url:
            return self._append_download_param(sharing_url)

        if "sharepoint.com" in sharing_url:
            # Return multiple URLs to try in order
            # The caller will try each one until one works
            return self._get_sharepoint_download_urls(sharing_url)

        # Fallback: return as-is and let the caller handle it
        logger.debug(
            "Unrecognised OneDrive URL pattern, using as-is: %s", sharing_url
        )
        return sharing_url

    def _get_sharepoint_download_urls(self, url: str) -> str:
        """Get the best download URL for SharePoint links.

        Tries multiple URL formats in order of likelihood to work.
        """
        # Try 1: download.aspx format (most reliable for personal OneDrive)
        converted = self._convert_sharepoint_personal_url(url)
        if converted:
            return converted

        # Try 2: Append download=1 to original URL
        return self._append_download_param(url)

    def _convert_sharepoint_personal_url(self, url: str) -> Optional[str]:
        """Convert SharePoint personal OneDrive sharing URL to direct download URL.

        Input format:
            https://<tenant>-my.sharepoint.com/:b:/g/personal/<user>/<id>
        Output format:
            https://<tenant>-my.sharepoint.com/personal/<user>/_layouts/15/download.aspx?share=<id>

        Returns None if the URL doesn't match the expected pattern.
        """
        # Match SharePoint personal OneDrive URLs
        # Pattern: https://<tenant>-my.sharepoint.com/:X:/g/personal/<user>/<share_id>
        # where X is a type indicator (b=binary, x=xlsx, etc.)
        pattern = r"^(https://[^/]+-my\.sharepoint\.com)/:[a-z]:/g/personal/([^/]+)/([^/?>\s]+)"
        match = re.match(pattern, url, re.IGNORECASE)

        if not match:
            return None

        base_url = match.group(1)  # https://<tenant>-my.sharepoint.com
        user = match.group(2)       # dimagis_lolcsecurities_com
        share_id = match.group(3)   # IQBbrfA5PO0xQZjUy6DVyf_vAeF9BKcUSFdFFF-ukiMiV44

        download_url = f"{base_url}/personal/{user}/_layouts/15/download.aspx?share={share_id}"
        logger.debug("Converted SharePoint URL: %s → %s", url, download_url)
        return download_url

    def _resolve_short_url(self, short_url: str) -> Optional[str]:
        """Follow a 1drv.ms short URL to its final destination."""
        try:
            response = requests.head(
                short_url,
                allow_redirects=True,
                timeout=self._http.onedrive_timeout,
            )
            resolved = response.url
            logger.debug("Resolved %s → %s", short_url, resolved)
            return resolved
        except requests.RequestException as exc:
            logger.error("Failed to resolve short URL %s: %s", short_url, exc)
            return None

    @staticmethod
    def _append_download_param(url: str) -> str:
        """Append ``download=1`` to a OneDrive / SharePoint URL."""
        if "download=1" in url:
            return url
        separator = "&" if "?" in url else "?"
        return url + separator + "download=1"

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Clean a URL by stripping trailing punctuation and invisible characters.

        Removes:
        - Zero-width spaces and other invisible Unicode characters
        - Trailing punctuation: . , ; ) > " ' <
        - HTML entities that may have leaked in
        """
        # FIRST: Remove ALL zero-width spaces and invisible Unicode from entire string
        url = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00a0]", "", url)
        # Remove any whitespace
        url = url.strip()
        # THEN: Strip trailing punctuation (now that invisible chars are gone)
        url = url.rstrip(".,;)>\"'<\n\r\t")
        # Final strip
        url = url.strip()
        return url

    # ─── Validation ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_pdf(data: bytes) -> bool:
        """Check PDF magic bytes (%PDF) to confirm the download is actually a PDF."""
        return data[:4] == _PDF_MAGIC
