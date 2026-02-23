#!/usr/bin/env python3
"""Batch PDF extraction script using Together.ai API.

Usage:
    python scripts/batch_extract.py                    # Process all PDFs (newest first)
    python scripts/batch_extract.py data/raw/file.pdf # Process single PDF
    python scripts/batch_extract.py --limit 10        # Process first 10 PDFs
    python scripts/batch_extract.py --reprocess       # Reprocess already completed PDFs
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import settings
from src.pdf_extractor import PDFTextExtractor, LLMExtractor, StockDataParser
from src.data_validator import ConfidenceScorer


# ─── Checkpoint Management ────────────────────────────────────────────────────

CHECKPOINT_FILE = Path("data/checkpoints/extraction_checkpoint.json")


def load_checkpoint() -> dict:
    """Load extraction checkpoint tracking processed PDFs."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed": {}, "failed": {}}


def save_checkpoint(checkpoint: dict) -> None:
    """Save extraction checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def is_processed(checkpoint: dict, pdf_name: str) -> bool:
    """Check if a PDF has already been successfully processed."""
    return pdf_name in checkpoint.get("processed", {})


def mark_processed(checkpoint: dict, pdf_name: str, output_file: str, stock_count: int) -> None:
    """Mark a PDF as successfully processed."""
    checkpoint.setdefault("processed", {})[pdf_name] = {
        "output_file": output_file,
        "stock_count": stock_count,
        "processed_at": datetime.now().isoformat(),
    }
    # Remove from failed if it was there
    checkpoint.get("failed", {}).pop(pdf_name, None)
    save_checkpoint(checkpoint)


def mark_failed(checkpoint: dict, pdf_name: str, error: str) -> None:
    """Mark a PDF as failed."""
    checkpoint.setdefault("failed", {})[pdf_name] = {
        "error": error,
        "failed_at": datetime.now().isoformat(),
    }
    save_checkpoint(checkpoint)


def smart_split(text: str, max_size: int = 2000) -> list[str]:
    """Split text into chunks at line boundaries.

    Default chunk size of 2000 chars keeps ~14 stocks per chunk, well within
    Llama 8B output token limits and improving full extraction reliability:
    - 2000 chars: ~20 chunks per PDF, ~14 stocks/chunk (recommended for Llama 8B)
    - 3500 chars: ~12 chunks per PDF, ~25 stocks/chunk (may miss stocks)
    - 5000 chars: ~8 chunks per PDF (for Llama 70B)
    """
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_size:
            if current:
                chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)
    return chunks


def extract_pdf(
    pdf_path: Path,
    llm: LLMExtractor,
    output_dir: Path,
    use_together_only: bool = True,
    request_delay: float = 0.1,
    use_chunking: bool = False,
) -> dict:
    """Extract data from a single PDF and save to JSON.

    Args:
        pdf_path: Path to PDF file
        llm: LLM extractor instance
        output_dir: Output directory for JSON files
        use_together_only: If True, only use Together.ai (no fallbacks)
        request_delay: Delay between API requests (for rate limiting)
        use_chunking: If True, split text into chunks (for small context models)
                      If False, process entire PDF in single request (recommended for Together.ai)
    """
    start_time = time.time()

    # Extract text
    text_extractor = PDFTextExtractor(pdf_path)
    raw_text = text_extractor.extract_text()
    trading_date = text_extractor.extract_trading_date()

    # Sanitize text
    clean_text = llm._sanitize_text(raw_text)

    # Extract from each chunk
    parser = StockDataParser()
    all_stocks = {}
    methods_used = set()
    failed_chunks = 0

    if use_chunking:
        # Split into chunks for small context models (Ollama)
        chunks = smart_split(clean_text)
        print(f"  Processing {len(chunks)} chunks...", end=" ", flush=True)
    else:
        # Process entire PDF in single request (Together.ai with 128K context)
        chunks = [clean_text]
        print(f"  Processing full text ({len(clean_text):,} chars)...", end=" ", flush=True)

    for i, chunk in enumerate(chunks):
        # Rate limiting delay between requests
        if i > 0 and request_delay > 0:
            time.sleep(request_delay)

        # Try extraction with retry on failure
        max_retries = 2
        data = None
        method = "failed"

        for attempt in range(max_retries):
            if use_together_only and llm.together_api_key:
                # Use Together.ai directly
                data = llm.extract_with_together(chunk)
                method = "together" if data else "failed"
            else:
                # Use normal extraction with fallbacks
                data, method = llm.extract(chunk)

            if data and "stocks" in data:
                # Success - break out of retry loop
                break
            elif attempt < max_retries - 1:
                # Failed - wait and retry
                time.sleep(2.0)

        if method not in ("none", "failed"):
            methods_used.add(method)
        else:
            failed_chunks += 1
            # Save the raw LLM response for failed chunks so they can be reprocessed later
            if llm._last_failed_response:
                failed_dir = output_dir / "failed_chunks" / pdf_path.stem
                failed_dir.mkdir(parents=True, exist_ok=True)
                failed_file = failed_dir / f"chunk_{i + 1:02d}_response.txt"
                failed_file.write_text(llm._last_failed_response)
                print(f"\n  [chunk {i + 1}] Saved failed response → {failed_file}")

        if data and "stocks" in data:
            before = len(all_stocks)
            df, _ = parser.parse(data)
            for _, row in df.iterrows():
                code = row.get("code", "")
                if code and code not in all_stocks:
                    all_stocks[code] = row.to_dict()
            added = len(all_stocks) - before
            print(f"\n  [chunk {i + 1}/{len(chunks)}] {added} stocks extracted", end="", flush=True)

    elapsed = time.time() - start_time
    failed_msg = f" ({failed_chunks} failed)" if failed_chunks else ""
    print(f"{len(all_stocks)} stocks in {elapsed:.1f}s{failed_msg}")

    if not all_stocks:
        return {"error": "No stocks extracted", "elapsed": elapsed}

    # Build DataFrame and calculate confidence
    df_all = pd.DataFrame(list(all_stocks.values()))
    sectors = sorted(df_all["sector"].dropna().unique().tolist())

    scorer = ConfidenceScorer()
    row_scores = scorer.score_dataframe(df_all)
    confidence = scorer.aggregate_score(row_scores)

    # Clean NaN values for JSON
    def clean_value(v):
        if pd.isna(v):
            return None
        return v

    stocks_clean = [
        {k: clean_value(v) for k, v in stock.items()}
        for stock in all_stocks.values()
    ]

    # Prepare output
    output = {
        "trading_date": str(trading_date) if trading_date else None,
        "source_file": pdf_path.name,
        "extraction_method": ", ".join(sorted(methods_used)) if methods_used else "none",
        "confidence_score": round(confidence, 4),
        "sectors": sectors,
        "stock_count": len(stocks_clean),
        "extraction_time_seconds": round(elapsed, 2),
        "stocks": stocks_clean,
    }

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use trading date for filename if available, otherwise use PDF name
    if trading_date:
        output_file = output_dir / f"{trading_date}_stocks.json"
    else:
        output_file = output_dir / f"{pdf_path.stem}_stocks.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    return {
        "output_file": str(output_file),
        "stock_count": len(stocks_clean),
        "confidence": confidence,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch PDF extraction")
    parser.add_argument("pdf_path", nargs="?", help="Single PDF to process")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to process")
    parser.add_argument("--input-dir", default="data/raw", help="Input directory")
    parser.add_argument("--output-dir", default="data/output", help="Output directory")
    parser.add_argument(
        "--together-only",
        action="store_true",
        default=True,
        help="Only use Together.ai API (no fallback to local Ollama)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.15,
        help="Delay between API requests in seconds (for rate limiting)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess already completed PDFs",
    )
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="Process oldest PDFs first (default is newest first)",
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Disable chunked processing and send full PDF in single request. "
             "Not recommended - may fail due to output token limits.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Check for API keys
    if settings.llm.together_api_key:
        print(f"Using Together.ai API ({settings.llm.together_model})")
    elif settings.llm.groq_api_key:
        print(f"Using Groq API ({settings.llm.groq_model})")
    elif settings.llm.anthropic_api_key:
        print("Using Claude API")
    else:
        print("Using local Ollama (slower)")

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_count = len(checkpoint.get("processed", {}))
    failed_count = len(checkpoint.get("failed", {}))
    print(f"Checkpoint: {processed_count} processed, {failed_count} failed")
    print()

    # Initialize extractor
    llm = LLMExtractor()

    # Get PDFs to process
    if args.pdf_path:
        pdfs = [Path(args.pdf_path)]
    else:
        # Sort by filename (which contains date) - newest first by default
        pdfs = sorted(input_dir.glob("*_comps.pdf"), reverse=not args.oldest_first)

        # Filter out already processed PDFs unless --reprocess is set
        if not args.reprocess:
            original_count = len(pdfs)
            pdfs = [p for p in pdfs if not is_processed(checkpoint, p.name)]
            skipped = original_count - len(pdfs)
            if skipped > 0:
                print(f"Skipping {skipped} already processed PDFs")

        if args.limit:
            pdfs = pdfs[:args.limit]

    if not pdfs:
        print(f"No PDFs to process in {input_dir}")
        return

    print(f"Processing {len(pdfs)} PDF(s) ({'oldest' if args.oldest_first else 'newest'} first)...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    total_start = time.time()
    results = []
    successful_count = 0
    failed_count = 0

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")

        try:
            result = extract_pdf(
                pdf_path,
                llm,
                output_dir,
                use_together_only=args.together_only,
                request_delay=args.request_delay,
                use_chunking=not args.no_chunking,
            )
            results.append(result)

            if "error" not in result:
                print(f"  → Saved: {result['output_file']}")
                print(f"  → {result['stock_count']} stocks, {result['confidence']:.1%} confidence")
                mark_processed(
                    checkpoint,
                    pdf_path.name,
                    result["output_file"],
                    result["stock_count"],
                )
                successful_count += 1
            else:
                mark_failed(checkpoint, pdf_path.name, result["error"])
                failed_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"error": str(e)})
            mark_failed(checkpoint, pdf_path.name, str(e))
            failed_count += 1

    # Summary
    total_time = time.time() - total_start
    successful = [r for r in results if "error" not in r]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {len(pdfs)} PDFs")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(pdfs) - len(successful)}")
    if len(pdfs) > 0:
        print(f"Total time: {total_time:.1f}s ({total_time/len(pdfs):.1f}s per PDF)")

    if successful:
        total_stocks = sum(r["stock_count"] for r in successful)
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        print(f"Total stocks: {total_stocks}")
        print(f"Average confidence: {avg_confidence:.1%}")

    # Show overall checkpoint status
    checkpoint = load_checkpoint()
    print(f"\nCheckpoint status: {len(checkpoint.get('processed', {}))} total processed")


if __name__ == "__main__":
    main()
