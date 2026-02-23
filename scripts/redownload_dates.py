#!/usr/bin/env python3
"""Redownload comp sheets for specific dates.

Usage:
    python scripts/redownload_dates.py 2026-02-16 2026-02-17 2026-02-18
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.email_scanner import GmailScanner
from src.pdf_downloader import PDFDownloader


def find_email_for_date(scanner: GmailScanner, target_date: str) -> dict | None:
    """Search Gmail for comp sheet email matching a specific date."""
    # Format: YYYY-MM-DD -> search for emails around that date
    dt = datetime.strptime(target_date, "%Y-%m-%d")

    # Gmail search query for specific date
    date_query = dt.strftime("%Y/%m/%d")
    query = f"from:{settings.gmail.target_sender} subject:{settings.gmail.subject_keyword} after:{date_query} before:{date_query}"

    print(f"  Searching: {query}")

    # Search for emails
    response = scanner.search_comp_sheet_emails(
        sender=settings.gmail.target_sender,
        subject_keyword=settings.gmail.subject_keyword,
    )

    messages = response.get("messages", [])

    # Find email matching the target date
    for msg in messages:
        attachment = scanner.get_email_details(msg["id"])
        if attachment:
            # Parse the email date
            from email.utils import parsedate_to_datetime
            try:
                email_dt = parsedate_to_datetime(attachment.date)
                email_date = email_dt.strftime("%Y-%m-%d")

                if email_date == target_date:
                    return {
                        "message_id": msg["id"],
                        "attachment": attachment,
                    }
            except:
                continue

    return None


def redownload_date(scanner: GmailScanner, downloader: PDFDownloader, target_date: str) -> bool:
    """Redownload comp sheet for a specific date."""
    print(f"\nProcessing {target_date}...")

    # Delete existing file
    pdf_path = Path(f"data/raw/{target_date}_comps.pdf")
    if pdf_path.exists():
        print(f"  Deleting existing file: {pdf_path}")
        pdf_path.unlink()

    # Find the email
    result = find_email_for_date(scanner, target_date)

    if not result:
        print(f"  ERROR: No email found for {target_date}")
        return False

    attachment = result["attachment"]
    message_id = result["message_id"]

    print(f"  Found email: {message_id}")
    print(f"  Type: {attachment.attachment_type}")

    if attachment.all_onedrive_urls:
        print(f"  OneDrive URLs: {len(attachment.all_onedrive_urls)}")
        for i, url in enumerate(attachment.all_onedrive_urls):
            print(f"    [{i+1}] {url[:80]}...")

    # Download
    if attachment.attachment_type == "direct_pdf":
        filepath = downloader.download_attachment(
            gmail_service=scanner.service,
            user_id=settings.gmail.user_id,
            message_id=attachment.message_id,
            attachment_id=attachment.attachment_id,
            date_str=attachment.date,
        )
    elif attachment.attachment_type == "onedrive_link":
        if len(attachment.all_onedrive_urls) > 1:
            filepath = downloader.download_from_onedrive_urls(
                urls=attachment.all_onedrive_urls,
                date_str=attachment.date,
            )
        else:
            filepath = downloader.download_from_onedrive(
                sharing_url=attachment.onedrive_url,
                date_str=attachment.date,
            )
    else:
        print(f"  ERROR: Unknown attachment type")
        return False

    if filepath:
        # Verify it's a real comp sheet
        from src.pdf_extractor import PDFTextExtractor
        text_ext = PDFTextExtractor(filepath)
        text = text_ext.extract_text()

        is_summary = 'Market Insights' in text[:1000] and 'N0000' not in text

        if is_summary:
            print(f"  WARNING: Downloaded file is Market Insights, not Comp Sheet!")
            print(f"  The email may not contain a Comp Sheet for this date.")
            return False
        else:
            print(f"  SUCCESS: Downloaded {filepath}")
            print(f"  Text size: {len(text):,} chars")
            return True
    else:
        print(f"  ERROR: Download failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Redownload comp sheets for specific dates")
    parser.add_argument("dates", nargs="+", help="Dates to redownload (YYYY-MM-DD format)")
    parser.add_argument("--list-urls", action="store_true", help="Just list OneDrive URLs without downloading")
    args = parser.parse_args()

    settings.configure_logging()

    scanner = GmailScanner()
    downloader = PDFDownloader()

    results = {"success": [], "failed": []}

    for date in args.dates:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {date} (expected YYYY-MM-DD)")
            results["failed"].append(date)
            continue

        if redownload_date(scanner, downloader, date):
            results["success"].append(date)
        else:
            results["failed"].append(date)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Success: {len(results['success'])} - {results['success']}")
    print(f"Failed:  {len(results['failed'])} - {results['failed']}")


if __name__ == "__main__":
    main()