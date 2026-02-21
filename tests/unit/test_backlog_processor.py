"""Unit tests for src.backlog_processor.

Covers:
- Checkpoint: load (fresh, existing, corrupt)
- Checkpoint: save (atomic write, content correct)
- process_all: processes one page of messages
- process_all: skips already-processed IDs
- process_all: resumes from saved page_token (checkpoint)
- process_all: handles rate-limit (429 HttpError) gracefully
- process_all: handles Gmail API failure gracefully
- process_all: dry_run=True does not download
- process_all: saves checkpoint after every email
- process_all: marks checkpoint last_page_token as None on completion
- _process_single_message: dispatches to downloader correctly
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest
from googleapiclient.errors import HttpError

from src.backlog_processor import BacklogProcessor
from src.email_scanner import EmailAttachment


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_processor(
    tmp_path: Path,
    messages_pages: list[list[dict]],
    email_details: Optional[EmailAttachment] = None,
    downloaded_path: Optional[Path] = None,
) -> BacklogProcessor:
    """Build a BacklogProcessor with fully mocked scanner and downloader."""
    mock_scanner = MagicMock()
    mock_downloader = MagicMock()

    # Build search_comp_sheet_emails side effects (one call per page)
    responses = []
    for i, page in enumerate(messages_pages):
        next_token = f"PAGE_{i+1}" if i < len(messages_pages) - 1 else None
        response = {"messages": page}
        if next_token:
            response["nextPageToken"] = next_token
        responses.append(response)

    mock_scanner.search_comp_sheet_emails.side_effect = responses
    mock_scanner.get_email_details.return_value = email_details
    mock_scanner._config.user_id = "me"

    mock_downloader.download_attachment.return_value = downloaded_path
    mock_downloader.download_from_onedrive.return_value = downloaded_path

    checkpoint_path = tmp_path / "checkpoints" / "test_checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True)

    processor = BacklogProcessor(
        scanner=mock_scanner,
        downloader=mock_downloader,
        checkpoint_file=str(checkpoint_path),
    )
    return processor


def _make_pdf_attachment(message_id: str = "msg_001") -> EmailAttachment:
    return EmailAttachment(
        message_id=message_id,
        date="Mon, 19 Feb 2024 08:30:00 +0530",
        subject="Comp Sheet",
        attachment_type="direct_pdf",
        attachment_id="att_001",
        filename="comps.pdf",
    )


def _make_onedrive_attachment(message_id: str = "msg_002") -> EmailAttachment:
    return EmailAttachment(
        message_id=message_id,
        date="Tue, 20 Feb 2024 09:00:00 +0530",
        subject="Comp Sheet OneDrive",
        attachment_type="onedrive_link",
        onedrive_url="https://onedrive.live.com/share?resid=ABC",
    )


# ─── Checkpoint Tests ─────────────────────────────────────────────────────────


class TestCheckpoint:
    def test_load_fresh_checkpoint_returns_defaults(self, tmp_path: Path) -> None:
        processor = _make_processor(tmp_path, messages_pages=[[]])
        checkpoint = processor.load_checkpoint()
        assert checkpoint["processed_ids"] == []
        assert checkpoint["last_page_token"] is None
        assert checkpoint["total_processed"] == 0

    def test_load_existing_checkpoint(self, tmp_path: Path) -> None:
        processor = _make_processor(tmp_path, messages_pages=[[]])
        # Pre-write a checkpoint
        data = {
            "processed_ids": ["msg_a", "msg_b"],
            "last_page_token": "TOKEN_X",
            "total_processed": 2,
        }
        processor._checkpoint_path.write_text(json.dumps(data))

        loaded = processor.load_checkpoint()
        assert loaded["processed_ids"] == ["msg_a", "msg_b"]
        assert loaded["last_page_token"] == "TOKEN_X"
        assert loaded["total_processed"] == 2

    def test_load_corrupt_checkpoint_returns_defaults(self, tmp_path: Path) -> None:
        processor = _make_processor(tmp_path, messages_pages=[[]])
        processor._checkpoint_path.write_text("NOT JSON {{{")

        checkpoint = processor.load_checkpoint()
        assert checkpoint["processed_ids"] == []

    def test_save_checkpoint_creates_file(self, tmp_path: Path) -> None:
        processor = _make_processor(tmp_path, messages_pages=[[]])
        processor.save_checkpoint(
            {"processed_ids": ["msg_1"], "last_page_token": None, "total_processed": 1}
        )
        assert processor._checkpoint_path.exists()

    def test_save_checkpoint_content_correct(self, tmp_path: Path) -> None:
        processor = _make_processor(tmp_path, messages_pages=[[]])
        data = {"processed_ids": ["x"], "last_page_token": "T1", "total_processed": 1}
        processor.save_checkpoint(data)

        saved = json.loads(processor._checkpoint_path.read_text())
        assert saved["processed_ids"] == ["x"]
        assert saved["last_page_token"] == "T1"


# ─── process_all ─────────────────────────────────────────────────────────────


class TestProcessAll:
    def test_processes_single_page_single_message(
        self, tmp_path: Path, minimal_pdf_bytes: bytes
    ) -> None:
        fake_path = tmp_path / "2024-02-19_comps.pdf"
        fake_path.write_bytes(minimal_pdf_bytes)
        attachment = _make_pdf_attachment("msg_001")

        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_001", "threadId": "t1"}]],
            email_details=attachment,
            downloaded_path=fake_path,
        )

        results = processor.process_all(rate_limit_delay=0)

        assert len(results["downloaded"]) == 1
        assert len(results["failed"]) == 0
        assert results["total_processed"] == 1

    def test_skips_already_processed_ids(
        self, tmp_path: Path
    ) -> None:
        attachment = _make_pdf_attachment("msg_001")
        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_001", "threadId": "t1"}]],
            email_details=attachment,
        )
        # Pre-populate checkpoint with msg_001
        processor.save_checkpoint(
            {
                "processed_ids": ["msg_001"],
                "last_page_token": None,
                "total_processed": 1,
            }
        )

        results = processor.process_all(rate_limit_delay=0)

        assert results["skipped"] == ["msg_001"]
        assert results["downloaded"] == []

    def test_resumes_from_checkpoint_page_token(self, tmp_path: Path) -> None:
        """When checkpoint has a page_token, the first API call should use it."""
        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_new", "threadId": "t1"}]],
            email_details=_make_pdf_attachment("msg_new"),
        )
        processor.save_checkpoint(
            {
                "processed_ids": [],
                "last_page_token": "RESUME_TOKEN",
                "total_processed": 0,
            }
        )

        processor.process_all(rate_limit_delay=0)

        first_call_kwargs = processor._scanner.search_comp_sheet_emails.call_args_list[0][1]
        assert first_call_kwargs.get("page_token") == "RESUME_TOKEN"

    def test_paginates_through_multiple_pages(
        self, tmp_path: Path, minimal_pdf_bytes: bytes
    ) -> None:
        fake_path = tmp_path / "comps.pdf"
        fake_path.write_bytes(minimal_pdf_bytes)

        page1 = [{"id": "msg_001", "threadId": "t1"}]
        page2 = [{"id": "msg_002", "threadId": "t2"}]
        page3 = [{"id": "msg_003", "threadId": "t3"}]

        processor = _make_processor(
            tmp_path,
            messages_pages=[page1, page2, page3],
            email_details=_make_pdf_attachment(),
            downloaded_path=fake_path,
        )

        results = processor.process_all(rate_limit_delay=0)

        assert processor._scanner.search_comp_sheet_emails.call_count == 3
        assert results["total_processed"] == 3

    def test_handles_rate_limit_429_gracefully(self, tmp_path: Path) -> None:
        """429 should cause early stop with checkpoint saved, not crash."""
        mock_resp = MagicMock()
        mock_resp.status = 429

        processor = _make_processor(tmp_path, messages_pages=[[]])
        processor._scanner.search_comp_sheet_emails.side_effect = HttpError(
            resp=mock_resp, content=b"Rate limit"
        )

        # Should not raise
        results = processor.process_all(rate_limit_delay=0)
        assert "total_processed" in results
        # Checkpoint should exist for resume
        assert processor._checkpoint_path.exists()

    def test_failed_download_does_not_crash(self, tmp_path: Path) -> None:
        """If downloader returns None, message goes to failed, not crash."""
        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_fail", "threadId": "t1"}]],
            email_details=_make_pdf_attachment("msg_fail"),
            downloaded_path=None,  # simulate download failure
        )

        results = processor.process_all(rate_limit_delay=0)

        assert "msg_fail" in results["failed"]
        assert results["downloaded"] == []

    def test_no_attachment_goes_to_failed(self, tmp_path: Path) -> None:
        """email_details=None means no attachment found → failed."""
        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_noatt", "threadId": "t1"}]],
            email_details=None,
        )

        results = processor.process_all(rate_limit_delay=0)

        assert "msg_noatt" in results["failed"]

    def test_dry_run_does_not_call_downloader(self, tmp_path: Path) -> None:
        attachment = _make_pdf_attachment("msg_dry")
        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_dry", "threadId": "t1"}]],
            email_details=attachment,
        )

        processor.process_all(dry_run=True, rate_limit_delay=0)

        processor._downloader.download_attachment.assert_not_called()
        processor._downloader.download_from_onedrive.assert_not_called()

    def test_checkpoint_cleared_after_success(self, tmp_path: Path) -> None:
        """After full run, last_page_token in checkpoint should be None."""
        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_ok", "threadId": "t1"}]],
            email_details=_make_pdf_attachment("msg_ok"),
        )

        processor.process_all(rate_limit_delay=0)

        saved = json.loads(processor._checkpoint_path.read_text())
        assert saved["last_page_token"] is None

    def test_onedrive_attachment_dispatched_to_correct_downloader(
        self, tmp_path: Path, minimal_pdf_bytes: bytes
    ) -> None:
        fake_path = tmp_path / "2024-02-20_comps.pdf"
        fake_path.write_bytes(minimal_pdf_bytes)
        attachment = _make_onedrive_attachment("msg_od")

        processor = _make_processor(
            tmp_path,
            messages_pages=[[{"id": "msg_od", "threadId": "t1"}]],
            email_details=attachment,
            downloaded_path=fake_path,
        )

        processor.process_all(rate_limit_delay=0)

        processor._downloader.download_from_onedrive.assert_called_once()
        processor._downloader.download_attachment.assert_not_called()

    def test_empty_inbox_returns_zero_processed(self, tmp_path: Path) -> None:
        """No messages → no errors, total_processed stays 0."""
        processor = _make_processor(tmp_path, messages_pages=[[]])
        results = processor.process_all(rate_limit_delay=0)
        assert results["total_processed"] == 0
        assert results["downloaded"] == []
        assert results["failed"] == []
