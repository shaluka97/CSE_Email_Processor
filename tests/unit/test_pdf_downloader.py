"""Unit tests for src.pdf_downloader.

Covers:
- File naming from RFC 2822 dates
- Duplicate detection (skip if file exists)
- Gmail attachment download (success, API error, non-PDF bytes)
- OneDrive download (success, 403, 404, timeout, non-PDF response)
- URL conversion (onedrive.live.com, sharepoint.com, 1drv.ms short URL)
- PDF magic byte validation
"""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.pdf_downloader import PDFDownloader
from src.config import StorageConfig, HttpConfig


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _downloader(tmp_data_dir: Path) -> PDFDownloader:
    storage = StorageConfig(data_dir=str(tmp_data_dir))
    http = HttpConfig(onedrive_timeout=10)
    return PDFDownloader(storage_config=storage, http_config=http)


# ─── build_filepath ───────────────────────────────────────────────────────────


class TestBuildFilepath:
    def test_rfc2822_date(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)
        path = dl.build_filepath("Mon, 19 Feb 2024 08:30:00 +0530")
        assert path.name == "2024-02-19_comps.pdf"
        assert path.parent == tmp_data_dir

    def test_invalid_date_uses_today(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)
        path = dl.build_filepath("not-a-date")
        # Should not raise; filename will be today's date
        assert path.name.endswith("_comps.pdf")

    def test_different_dates_give_different_files(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)
        p1 = dl.build_filepath("Mon, 19 Feb 2024 08:30:00 +0530")
        p2 = dl.build_filepath("Tue, 20 Feb 2024 08:30:00 +0530")
        assert p1 != p2


# ─── is_duplicate ─────────────────────────────────────────────────────────────


class TestIsDuplicate:
    def test_existing_file_is_duplicate(
        self, tmp_data_dir: Path, minimal_pdf_bytes: bytes
    ) -> None:
        dl = _downloader(tmp_data_dir)
        filepath = tmp_data_dir / "2024-02-19_comps.pdf"
        filepath.write_bytes(minimal_pdf_bytes)
        assert dl.is_duplicate(filepath) is True

    def test_missing_file_is_not_duplicate(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)
        filepath = tmp_data_dir / "nonexistent.pdf"
        assert dl.is_duplicate(filepath) is False

    def test_empty_file_is_not_duplicate(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)
        filepath = tmp_data_dir / "empty.pdf"
        filepath.write_bytes(b"")
        assert dl.is_duplicate(filepath) is False


# ─── download_attachment ──────────────────────────────────────────────────────


class TestDownloadAttachment:
    DATE = "Mon, 19 Feb 2024 08:30:00 +0530"

    def test_successful_download(
        self,
        tmp_data_dir: Path,
        minimal_pdf_bytes: bytes,
        pdf_attachment_b64: str,
    ) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().attachments().get().execute.return_value = {
            "size": len(minimal_pdf_bytes),
            "data": pdf_attachment_b64,
        }
        dl = _downloader(tmp_data_dir)

        result = dl.download_attachment(
            gmail_service=mock_service,
            user_id="me",
            message_id="msg_001",
            attachment_id="att_001",
            date_str=self.DATE,
        )

        assert result is not None
        assert result.exists()
        assert result.read_bytes() == minimal_pdf_bytes

    def test_skips_duplicate(
        self,
        tmp_data_dir: Path,
        minimal_pdf_bytes: bytes,
        pdf_attachment_b64: str,
    ) -> None:
        dl = _downloader(tmp_data_dir)
        # Pre-create the file
        expected_path = dl.build_filepath(self.DATE)
        expected_path.write_bytes(minimal_pdf_bytes)

        mock_service = MagicMock()
        result = dl.download_attachment(
            gmail_service=mock_service,
            user_id="me",
            message_id="msg_001",
            attachment_id="att_001",
            date_str=self.DATE,
        )

        # Should return path but NOT call the API
        assert result == expected_path
        mock_service.users().messages().attachments().get.assert_not_called()

    def test_returns_none_on_api_error(self, tmp_data_dir: Path) -> None:
        from googleapiclient.errors import HttpError

        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_service = MagicMock()
        mock_service.users().messages().attachments().get().execute.side_effect = (
            HttpError(resp=mock_resp, content=b"Server Error")
        )
        dl = _downloader(tmp_data_dir)

        result = dl.download_attachment(
            gmail_service=mock_service,
            user_id="me",
            message_id="msg_001",
            attachment_id="att_001",
            date_str=self.DATE,
        )
        assert result is None

    def test_returns_none_for_non_pdf_bytes(self, tmp_data_dir: Path) -> None:
        not_a_pdf = base64.urlsafe_b64encode(b"This is HTML, not PDF").decode()
        mock_service = MagicMock()
        mock_service.users().messages().attachments().get().execute.return_value = {
            "size": 20,
            "data": not_a_pdf,
        }
        dl = _downloader(tmp_data_dir)

        result = dl.download_attachment(
            gmail_service=mock_service,
            user_id="me",
            message_id="msg_001",
            attachment_id="att_001",
            date_str=self.DATE,
        )
        assert result is None

    def test_returns_none_for_empty_data(self, tmp_data_dir: Path) -> None:
        mock_service = MagicMock()
        mock_service.users().messages().attachments().get().execute.return_value = {
            "size": 0,
            "data": "",
        }
        dl = _downloader(tmp_data_dir)

        result = dl.download_attachment(
            gmail_service=mock_service,
            user_id="me",
            message_id="msg_001",
            attachment_id="att_001",
            date_str=self.DATE,
        )
        assert result is None


# ─── download_from_onedrive ───────────────────────────────────────────────────


class TestDownloadFromOneDrive:
    DATE = "Tue, 20 Feb 2024 09:00:00 +0530"
    URL = "https://onedrive.live.com/share?resid=ABC&download=1"

    def test_successful_download(
        self, tmp_data_dir: Path, minimal_pdf_bytes: bytes
    ) -> None:
        dl = _downloader(tmp_data_dir)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = minimal_pdf_bytes
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        with patch("src.pdf_downloader.requests.get", return_value=mock_response):
            result = dl.download_from_onedrive(self.URL, self.DATE)

        assert result is not None
        assert result.exists()
        assert result.read_bytes() == minimal_pdf_bytes

    def test_skips_duplicate(
        self, tmp_data_dir: Path, minimal_pdf_bytes: bytes
    ) -> None:
        dl = _downloader(tmp_data_dir)
        expected_path = dl.build_filepath(self.DATE)
        expected_path.write_bytes(minimal_pdf_bytes)

        with patch("src.pdf_downloader.requests.get") as mock_get:
            result = dl.download_from_onedrive(self.URL, self.DATE)

        assert result == expected_path
        mock_get.assert_not_called()

    def test_returns_none_on_403_expired_link(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)

        mock_response = MagicMock()
        mock_response.status_code = 403
        http_error = requests.exceptions.HTTPError(response=mock_response)

        with patch("src.pdf_downloader.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = http_error
            result = dl.download_from_onedrive(self.URL, self.DATE)

        assert result is None

    def test_returns_none_on_404_not_found(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)

        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)

        with patch("src.pdf_downloader.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = http_error
            result = dl.download_from_onedrive(self.URL, self.DATE)

        assert result is None

    def test_returns_none_on_timeout(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)

        with patch(
            "src.pdf_downloader.requests.get",
            side_effect=requests.exceptions.Timeout,
        ):
            result = dl.download_from_onedrive(self.URL, self.DATE)

        assert result is None

    def test_returns_none_for_non_pdf_response(self, tmp_data_dir: Path) -> None:
        dl = _downloader(tmp_data_dir)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Login required</html>"
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("src.pdf_downloader.requests.get", return_value=mock_response):
            result = dl.download_from_onedrive(self.URL, self.DATE)

        assert result is None


# ─── convert_to_direct_download_url ──────────────────────────────────────────


class TestConvertToDirectDownloadUrl:
    def setup_method(self) -> None:
        storage = StorageConfig(data_dir="/tmp/test_dl")
        self.dl = PDFDownloader(storage_config=storage)

    def test_onedrive_personal_appends_download(self) -> None:
        url = "https://onedrive.live.com/share?resid=ABC"
        result = self.dl.convert_to_direct_download_url(url)
        assert result is not None
        assert "download=1" in result

    def test_onedrive_personal_already_has_download(self) -> None:
        url = "https://onedrive.live.com/share?resid=ABC&download=1"
        result = self.dl.convert_to_direct_download_url(url)
        # Should not double-add download=1
        assert result is not None
        assert result.count("download=1") == 1

    def test_sharepoint_appends_download(self) -> None:
        url = "https://myorg.sharepoint.com/sites/data/:b:/r/file.pdf"
        result = self.dl.convert_to_direct_download_url(url)
        assert result is not None
        assert "download=1" in result

    def test_short_url_resolved(self) -> None:
        short_url = "https://1drv.ms/u/s!AbCdEfGhIjKl"
        resolved = "https://onedrive.live.com/share?resid=XYZ"

        mock_resp = MagicMock()
        mock_resp.url = resolved

        with patch("src.pdf_downloader.requests.head", return_value=mock_resp):
            result = self.dl.convert_to_direct_download_url(short_url)

        assert result is not None
        assert "onedrive.live.com" in result
        assert "download=1" in result

    def test_short_url_resolution_failure_returns_none(self) -> None:
        short_url = "https://1drv.ms/u/s!Bad"

        with patch(
            "src.pdf_downloader.requests.head",
            side_effect=requests.exceptions.ConnectionError,
        ):
            result = self.dl.convert_to_direct_download_url(short_url)

        assert result is None


# ─── _is_valid_pdf ────────────────────────────────────────────────────────────


class TestIsValidPdf:
    def test_valid_pdf_magic_bytes(self, minimal_pdf_bytes: bytes) -> None:
        assert PDFDownloader._is_valid_pdf(minimal_pdf_bytes) is True

    def test_html_is_not_pdf(self) -> None:
        assert PDFDownloader._is_valid_pdf(b"<html>content</html>") is False

    def test_empty_bytes_is_not_pdf(self) -> None:
        assert PDFDownloader._is_valid_pdf(b"") is False

    def test_truncated_pdf_header(self) -> None:
        assert PDFDownloader._is_valid_pdf(b"%PDF") is True

    def test_wrong_magic(self) -> None:
        assert PDFDownloader._is_valid_pdf(b"JFIF") is False
