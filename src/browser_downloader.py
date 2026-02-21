"""Browser-based downloader for SharePoint/OneDrive files requiring authentication.

Uses Playwright to automate a real browser, leveraging existing Microsoft login sessions
or prompting for login when needed.

This approach works when:
- You don't have permission to register Azure AD apps
- The SharePoint links require organizational authentication
- You're already logged into Microsoft in your browser
"""
from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _sanitize_url(url: str) -> str:
    """Clean a URL by stripping trailing punctuation and invisible characters."""
    # Remove ALL zero-width spaces and invisible Unicode from entire string
    url = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00a0]", "", url)
    # Remove any whitespace
    url = url.strip()
    # Strip trailing punctuation
    url = url.rstrip(".,;)>\"'<\n\r\t")
    # Final strip
    url = url.strip()
    return url

# Browser context storage path (persists login sessions)
_DEFAULT_USER_DATA_DIR = Path.home() / ".cse-pipeline" / "browser-data"


class BrowserDownloader:
    """Downloads files from SharePoint/OneDrive using browser automation.

    Uses a persistent browser context to maintain login sessions across runs.
    The first time you run this, you'll need to log in manually. After that,
    the session is cached.

    Usage
    -----
    >>> downloader = BrowserDownloader()
    >>> pdf_bytes = downloader.download_file("https://sharepoint.com/...")
    """

    def __init__(
        self,
        user_data_dir: Optional[Path] = None,
        headless: bool = False,
        timeout: int = 60000,
    ) -> None:
        """Initialize browser downloader.

        Parameters
        ----------
        user_data_dir:
            Directory to store browser profile/cookies. Defaults to ~/.cse-pipeline/browser-data
        headless:
            Run browser in headless mode. Set to False to see the browser (useful for login).
        timeout:
            Default timeout in milliseconds for browser operations.
        """
        self._user_data_dir = user_data_dir or _DEFAULT_USER_DATA_DIR
        self._user_data_dir.mkdir(parents=True, exist_ok=True)
        self._headless = headless
        self._timeout = timeout
        self._browser = None
        self._context = None
        self._playwright = None

    def _ensure_browser(self) -> None:
        """Launch browser if not already running."""
        if self._context is not None:
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        logger.info("Launching browser for SharePoint download...")
        self._playwright = sync_playwright().start()

        # Use persistent context to maintain login sessions
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self._user_data_dir),
            headless=self._headless,
            accept_downloads=True,
            # Increase viewport for better rendering
            viewport={"width": 1280, "height": 800},
        )
        logger.debug("Browser context created with user data dir: %s", self._user_data_dir)

    def close(self) -> None:
        """Close the browser and cleanup resources."""
        if self._context:
            self._context.close()
            self._context = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        logger.debug("Browser closed")

    def download_file(
        self,
        url: str,
        wait_for_login: bool = True,
        login_timeout: int = 120000,
    ) -> Optional[tuple[bytes, str]]:
        """Download a file from a SharePoint/OneDrive URL.

        Parameters
        ----------
        url:
            The SharePoint/OneDrive sharing URL.
        wait_for_login:
            If True, wait for user to complete login if prompted.
        login_timeout:
            Maximum time to wait for login (ms).

        Returns
        -------
        Tuple of (file_bytes, filename), or None on failure.
        """
        # Sanitize URL to remove invisible characters and trailing punctuation
        original_url = url
        url = _sanitize_url(url)
        if url != original_url:
            logger.info("Sanitized URL for browser: %s", url)

        self._ensure_browser()

        page = self._context.new_page()
        page.set_default_timeout(self._timeout)

        try:
            logger.info("Navigating to SharePoint URL...")
            page.goto(url, wait_until="domcontentloaded")

            # Check if we need to login
            if self._is_login_page(page):
                if wait_for_login:
                    logger.info(
                        "Login required. Please complete login in the browser window. "
                        "Waiting up to %d seconds...",
                        login_timeout // 1000,
                    )
                    # Wait for navigation away from login page
                    self._wait_for_login(page, login_timeout)
                else:
                    logger.error("Login required but wait_for_login=False")
                    return None

            # Wait for page to fully load
            page.wait_for_load_state("networkidle", timeout=30000)

            # Check for error pages
            if self._is_error_page(page):
                logger.error("SharePoint returned an error page. The link may be expired or invalid.")
                return None

            # Try to get the filename from the page before downloading
            page_filename = self._get_filename_from_page(page)
            if page_filename:
                logger.info("Detected filename from page: %s", page_filename)

            # Try to find and click download button
            result = self._trigger_download(page, url)

            # If we got a result but filename is unknown, use the page filename
            if result and result[1] == "unknown.pdf" and page_filename:
                result = (result[0], page_filename)

            return result

        except Exception as exc:
            logger.error("Browser download failed: %s", exc)
            return None
        finally:
            page.close()

    def _is_login_page(self, page) -> bool:
        """Check if the current page is a Microsoft login page."""
        url = page.url.lower()
        login_indicators = [
            "login.microsoftonline.com",
            "login.live.com",
            "login.windows.net",
            "/oauth2/",
            "/common/oauth2/",
        ]
        return any(indicator in url for indicator in login_indicators)

    def _is_error_page(self, page) -> bool:
        """Check if the page is showing an error."""
        try:
            # Check page title and content for common error indicators
            title = page.title().lower()
            error_titles = ["error", "something went wrong", "access denied", "not found"]
            if any(err in title for err in error_titles):
                return True

            # Check for common SharePoint error messages in page content
            error_selectors = [
                "text=Something went wrong",
                "text=This item might not exist",
                "text=Access Denied",
                "text=You need permission",
                "text=This link isn't available",
            ]
            for selector in error_selectors:
                try:
                    if page.locator(selector).count() > 0:
                        return True
                except Exception:
                    continue

            return False
        except Exception:
            return False

    def _get_filename_from_page(self, page) -> Optional[str]:
        """Try to extract the filename from a SharePoint page.

        Looks for the filename in various places:
        - Page title
        - Header elements
        - Specific SharePoint filename selectors
        """
        try:
            # Method 1: Try SharePoint-specific selectors
            filename_selectors = [
                '[data-automationid="FileNameCell"]',
                '[data-automationid="name"]',
                '.FileNameCell',
                'h1[class*="fileName"]',
                'span[class*="fileName"]',
                '[class*="FileTitle"]',
            ]

            for selector in filename_selectors:
                try:
                    element = page.locator(selector).first
                    if element.is_visible(timeout=1000):
                        filename = element.inner_text().strip()
                        if filename and filename.lower().endswith('.pdf'):
                            return filename
                except Exception:
                    continue

            # Method 2: Check page title (often "filename.pdf - SharePoint")
            title = page.title()
            if title:
                # Extract filename from title like "Comps - 19 Feb 2026.pdf - OneDrive"
                if ".pdf" in title.lower():
                    # Find the part before " - OneDrive" or " - SharePoint"
                    for sep in [" - OneDrive", " - SharePoint", " - Microsoft"]:
                        if sep in title:
                            filename = title.split(sep)[0].strip()
                            if filename.lower().endswith('.pdf'):
                                return filename
                    # If no separator, check if title itself is a filename
                    if title.lower().endswith('.pdf'):
                        return title.strip()

            # Method 3: Look for any visible text that looks like a PDF filename
            try:
                body_text = page.locator('body').inner_text()
                # Look for patterns like "Comps - 19 February 2026.pdf"
                import re
                pdf_pattern = r'([A-Za-z0-9\s\-_]+\.pdf)'
                matches = re.findall(pdf_pattern, body_text, re.IGNORECASE)
                for match in matches:
                    if len(match) > 5:  # Reasonable filename length
                        return match.strip()
            except Exception:
                pass

            return None

        except Exception as exc:
            logger.debug("Could not extract filename from page: %s", exc)
            return None

    def _wait_for_login(self, page, timeout: int) -> None:
        """Wait for user to complete login."""
        start_time = time.time()
        timeout_sec = timeout / 1000

        while self._is_login_page(page):
            if time.time() - start_time > timeout_sec:
                raise TimeoutError("Login timeout exceeded")
            time.sleep(1)

        logger.info("Login completed successfully")
        # Give the page time to load after login
        page.wait_for_load_state("networkidle", timeout=30000)

    def _trigger_download(self, page, original_url: str) -> Optional[tuple[bytes, str]]:
        """Trigger file download and capture the bytes.

        Returns
        -------
        Tuple of (file_bytes, filename) or None on failure.
        """
        # Method 1: Try direct download parameter
        if "download=1" not in page.url:
            download_url = original_url + ("&" if "?" in original_url else "?") + "download=1"
            try:
                with page.expect_download(timeout=30000) as download_info:
                    page.goto(download_url)
                download = download_info.value
                return self._save_download(download)
            except Exception:
                logger.debug("Direct download parameter didn't work, trying button click...")

        # Method 2: Try to find download button on the page
        download_selectors = [
            'button[data-automationid="downloadCommand"]',
            'button[name="Download"]',
            '[aria-label="Download"]',
            'button:has-text("Download")',
            '[data-icon-name="Download"]',
        ]

        for selector in download_selectors:
            try:
                button = page.locator(selector).first
                if button.is_visible(timeout=2000):
                    logger.debug("Found download button with selector: %s", selector)
                    with page.expect_download(timeout=30000) as download_info:
                        button.click()
                    download = download_info.value
                    return self._save_download(download)
            except Exception:
                continue

        # Method 3: Try the three-dot menu → Download
        try:
            # Click the "more actions" button
            more_button = page.locator('[data-automationid="moreActionsButton"]').first
            if more_button.is_visible(timeout=2000):
                more_button.click()
                page.wait_for_timeout(500)

                # Click download in the menu
                download_option = page.locator('button:has-text("Download")').first
                if download_option.is_visible(timeout=2000):
                    with page.expect_download(timeout=30000) as download_info:
                        download_option.click()
                    download = download_info.value
                    return self._save_download(download)
        except Exception:
            pass

        # Method 4: Check if page content is already the PDF
        content_type = page.evaluate("() => document.contentType")
        if content_type == "application/pdf":
            logger.debug("Page is already serving PDF content")
            # Get the response body directly
            response = page.request.get(page.url)
            return response.body(), "unknown.pdf"

        logger.error("Could not find a way to download the file")
        return None

    def _save_download(self, download) -> tuple[bytes, str]:
        """Save a Playwright download to bytes.

        Returns
        -------
        Tuple of (file_bytes, suggested_filename)
        """
        # Save to a temporary file, then read bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name

        download.save_as(tmp_path)
        pdf_bytes = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink()  # Clean up temp file

        filename = download.suggested_filename or "unknown.pdf"
        logger.info("Downloaded %d bytes via browser (filename: %s)", len(pdf_bytes), filename)
        return pdf_bytes, filename


# Module-level singleton
_browser_downloader: Optional[BrowserDownloader] = None


def get_browser_downloader(headless: bool = False) -> BrowserDownloader:
    """Get or create the browser downloader singleton."""
    global _browser_downloader
    if _browser_downloader is None:
        _browser_downloader = BrowserDownloader(headless=headless)
    return _browser_downloader


def download_sharepoint_file(url: str, headless: bool = False) -> Optional[tuple[bytes, str]]:
    """Convenience function to download a SharePoint file.

    Parameters
    ----------
    url:
        SharePoint/OneDrive sharing URL.
    headless:
        Run browser in headless mode (set False to see browser for login).

    Returns
    -------
    Tuple of (file_bytes, filename), or None on failure.
    """
    downloader = get_browser_downloader(headless=headless)
    return downloader.download_file(url)
