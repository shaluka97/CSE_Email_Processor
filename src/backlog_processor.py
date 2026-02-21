"""Backlog Processor — paginate through all historical Comp Sheet emails.

Responsibilities
----------------
- Iterate through every matching email in Gmail (oldest → newest via reverse pagination)
- Skip emails already processed (tracked in a JSON checkpoint file)
- Save progress after every email so the run can be safely interrupted and resumed
- Handle Gmail API rate limits (429) by saving state and raising for retry
- Support dry-run mode (list what would be processed, without downloading)

Week 1 of the CSE Stock Market Data Pipeline.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from googleapiclient.errors import HttpError

from src.config import settings
from src.email_scanner import EmailAttachment, GmailScanner
from src.pdf_downloader import PDFDownloader

logger = logging.getLogger(__name__)


class BacklogProcessor:
    """Process all historical comp-sheet emails, resuming from the last checkpoint.

    Checkpoint file format
    ----------------------
    ::

        {
            "processed_ids": ["msg_id_1", "msg_id_2", ...],
            "last_page_token": null,          # null when complete
            "total_processed": 42,
            "last_run_at": "2024-02-19T08:30:00"
        }

    Usage
    -----
    >>> processor = BacklogProcessor()
    >>> results = processor.process_all()
    >>> print(results)
    {"downloaded": [...], "skipped": [...], "failed": [...], "total_processed": 42}
    """

    def __init__(
        self,
        scanner: Optional[GmailScanner] = None,
        downloader: Optional[PDFDownloader] = None,
        checkpoint_file: Optional[str] = None,
    ) -> None:
        self._scanner = scanner or GmailScanner()
        self._downloader = downloader or PDFDownloader()
        self._checkpoint_path = Path(
            checkpoint_file or settings.storage.checkpoint_file
        )
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Checkpoint I/O ──────────────────────────────────────────────────────

    def load_checkpoint(self) -> dict:
        """Load checkpoint from disk, or return a fresh empty checkpoint."""
        if self._checkpoint_path.exists():
            try:
                with open(self._checkpoint_path) as fh:
                    data = json.load(fh)
                logger.info(
                    "Resuming from checkpoint: %d emails already processed.",
                    data.get("total_processed", 0),
                )
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Checkpoint file corrupt or unreadable (%s) — starting fresh.",
                    exc,
                )
        return {
            "processed_ids": [],
            "last_page_token": None,
            "total_processed": 0,
            "last_run_at": None,
        }

    def save_checkpoint(self, checkpoint: dict) -> None:
        """Persist checkpoint to disk (atomic write)."""
        from datetime import datetime, timezone

        checkpoint["last_run_at"] = datetime.now(timezone.utc).isoformat()

        # Write to a temp file then rename — prevents corruption on interruption
        tmp_path = self._checkpoint_path.with_suffix(".tmp")
        with open(tmp_path, "w") as fh:
            json.dump(checkpoint, fh, indent=2)
        tmp_path.replace(self._checkpoint_path)
        logger.debug("Checkpoint saved (%d processed).", checkpoint["total_processed"])

    def clear_checkpoint(self) -> None:
        """Remove checkpoint after a successful full-backlog run."""
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
            logger.info("Checkpoint cleared — full backlog processed successfully.")

    # ─── Main Entry Point ────────────────────────────────────────────────────

    def process_all(
        self,
        sender: Optional[str] = None,
        subject_keyword: Optional[str] = None,
        dry_run: bool = False,
        rate_limit_delay: float = 0.5,
    ) -> dict:
        """Process every matching email in Gmail, resuming from the last checkpoint.

        Parameters
        ----------
        sender:
            Gmail sender filter. Defaults to ``settings.gmail.target_sender``.
        subject_keyword:
            Subject keyword filter. Defaults to ``settings.gmail.subject_keyword``.
        dry_run:
            If True, log what would be downloaded without actually saving files.
        rate_limit_delay:
            Seconds to sleep between individual email fetches to avoid 429s.

        Returns
        -------
        dict with keys: ``downloaded``, ``skipped``, ``failed``, ``total_processed``.
        """
        sender = sender or settings.gmail.target_sender
        subject_keyword = subject_keyword or settings.gmail.subject_keyword

        checkpoint = self.load_checkpoint()
        processed_ids: set[str] = set(checkpoint.get("processed_ids", []))
        page_token: Optional[str] = checkpoint.get("last_page_token")
        total_processed: int = checkpoint.get("total_processed", 0)

        results: dict = {
            "downloaded": [],
            "skipped": [],
            "failed": [],
            "total_processed": total_processed,
        }

        logger.info(
            "Starting backlog processor. Already processed: %d. dry_run=%s",
            len(processed_ids),
            dry_run,
        )

        try:
            while True:
                # ── Fetch a page of message stubs ─────────────────────────────
                try:
                    response = self._scanner.search_comp_sheet_emails(
                        sender=sender,
                        subject_keyword=subject_keyword,
                        page_token=page_token,
                    )
                except HttpError as exc:
                    if exc.resp.status == 429:
                        logger.warning(
                            "Gmail API rate limit hit. Saving checkpoint and stopping."
                        )
                        self._persist_and_update(
                            checkpoint, processed_ids, page_token, total_processed
                        )
                        results["total_processed"] = total_processed
                        return results
                    raise

                messages = response.get("messages", [])
                if not messages:
                    logger.info("No more messages — backlog complete.")
                    break

                # ── Process each message on this page ─────────────────────────
                for msg in messages:
                    msg_id = msg["id"]

                    if msg_id in processed_ids:
                        logger.debug("Already processed %s — skipping.", msg_id)
                        results["skipped"].append(msg_id)
                        continue

                    if dry_run:
                        logger.info("[DRY RUN] Would process message: %s", msg_id)
                        processed_ids.add(msg_id)
                        total_processed += 1
                        continue

                    # Small delay to be polite to the Gmail API
                    time.sleep(rate_limit_delay)

                    filepath = self._process_single_message(msg_id)
                    if filepath is not None:
                        results["downloaded"].append(str(filepath))
                        logger.info("✓ %s → %s", msg_id, filepath)
                    else:
                        results["failed"].append(msg_id)

                    processed_ids.add(msg_id)
                    total_processed += 1

                    # Save after every email in case of interruption
                    self._persist_and_update(
                        checkpoint, processed_ids, page_token, total_processed
                    )

                # ── Advance to next page ──────────────────────────────────────
                page_token = response.get("nextPageToken")
                if not page_token:
                    break

                # Update checkpoint with the new page token
                checkpoint["last_page_token"] = page_token
                self.save_checkpoint(checkpoint)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Progress saved to checkpoint.")
            self._persist_and_update(
                checkpoint, processed_ids, page_token, total_processed
            )

        # Successful completion — clear page token so next run starts fresh
        checkpoint["last_page_token"] = None
        checkpoint["total_processed"] = total_processed
        self.save_checkpoint(checkpoint)

        results["total_processed"] = total_processed
        logger.info(
            "Backlog complete. downloaded=%d skipped=%d failed=%d total=%d",
            len(results["downloaded"]),
            len(results["skipped"]),
            len(results["failed"]),
            total_processed,
        )
        return results

    # ─── Single Message Processing ────────────────────────────────────────────

    def _process_single_message(self, message_id: str) -> Optional[Path]:
        """Fetch details and download for one message.  Returns file path or None."""
        attachment = self._scanner.get_email_details(message_id)
        if not attachment:
            logger.warning("No downloadable attachment in message %s", message_id)
            return None
        return self._download(attachment)

    def _download(self, attachment: EmailAttachment) -> Optional[Path]:
        """Dispatch download based on attachment type."""
        if attachment.attachment_type == "direct_pdf":
            if not attachment.attachment_id:
                logger.error(
                    "direct_pdf attachment missing attachment_id for message %s",
                    attachment.message_id,
                )
                return None
            return self._downloader.download_attachment(
                gmail_service=self._scanner.service,
                user_id=self._scanner._config.user_id,
                message_id=attachment.message_id,
                attachment_id=attachment.attachment_id,
                date_str=attachment.date,
            )

        if attachment.attachment_type == "onedrive_link":
            if not attachment.onedrive_url and not attachment.all_onedrive_urls:
                logger.error(
                    "onedrive_link attachment missing URL for message %s",
                    attachment.message_id,
                )
                return None

            # If we have multiple URLs, try each one until we find Comps
            if len(attachment.all_onedrive_urls) > 1:
                logger.info(
                    "Found %d OneDrive URLs, will search for Comps file...",
                    len(attachment.all_onedrive_urls),
                )
                return self._downloader.download_from_onedrive_urls(
                    urls=attachment.all_onedrive_urls,
                    date_str=attachment.date,
                )

            # Single URL - use original method
            return self._downloader.download_from_onedrive(
                sharing_url=attachment.onedrive_url,
                date_str=attachment.date,
            )

        logger.error(
            "Unknown attachment_type %r for message %s",
            attachment.attachment_type,
            attachment.message_id,
        )
        return None

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _persist_and_update(
        self,
        checkpoint: dict,
        processed_ids: set[str],
        page_token: Optional[str],
        total_processed: int,
    ) -> None:
        """Update checkpoint dict in-place and save to disk."""
        checkpoint["processed_ids"] = list(processed_ids)
        checkpoint["last_page_token"] = page_token
        checkpoint["total_processed"] = total_processed
        self.save_checkpoint(checkpoint)


def main() -> None:
    """CLI entry point: ``cse-backlog`` (defined in pyproject.toml)."""
    import argparse

    settings.configure_logging()

    parser = argparse.ArgumentParser(
        description="Process all historical CSE Comp Sheet emails."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List emails that would be processed without downloading.",
    )
    parser.add_argument(
        "--sender",
        default=settings.gmail.target_sender,
        help="Override the sender email filter.",
    )
    parser.add_argument(
        "--subject",
        default=settings.gmail.subject_keyword,
        help="Override the subject keyword filter.",
    )
    args = parser.parse_args()

    processor = BacklogProcessor()
    results = processor.process_all(
        sender=args.sender,
        subject_keyword=args.subject,
        dry_run=args.dry_run,
    )

    print("\n=== Backlog Processing Results ===")
    print(f"Downloaded : {len(results['downloaded'])}")
    print(f"Skipped    : {len(results['skipped'])}")
    print(f"Failed     : {len(results['failed'])}")
    print(f"Total seen : {results['total_processed']}")
    if results["failed"]:
        print("\nFailed message IDs:")
        for fid in results["failed"]:
            print(f"  {fid}")


if __name__ == "__main__":
    main()
