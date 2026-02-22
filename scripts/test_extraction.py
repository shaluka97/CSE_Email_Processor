#!/usr/bin/env python3
"""Test script for PDF extraction with local LLM (Ollama).

Usage:
    python scripts/test_extraction.py [pdf_path]

If no PDF path is provided, uses the most recent comp sheet.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.pdf_extractor import (
    CompSheetExtractor,
    PDFTextExtractor,
    LLMExtractor,
    StockDataParser,
)
from src.data_validator import validate_extraction


def find_latest_pdf() -> Path:
    """Find the most recent comp sheet PDF."""
    data_dir = Path("data/raw")
    pdfs = sorted(data_dir.glob("*_comps.pdf"), reverse=True)
    if not pdfs:
        raise FileNotFoundError("No comp sheet PDFs found in data/raw/")
    return pdfs[0]


def test_text_extraction(pdf_path: Path) -> str:
    """Test raw text extraction."""
    print(f"\n{'='*60}")
    print("STEP 1: PDF Text Extraction")
    print('='*60)

    extractor = PDFTextExtractor(pdf_path)

    # Get metadata
    date = extractor.extract_trading_date()
    pages = extractor.get_page_count()

    print(f"PDF: {pdf_path.name}")
    print(f"Trading Date: {date}")
    print(f"Page Count: {pages}")

    # Extract text
    text = extractor.extract_text()
    print(f"Text Length: {len(text):,} characters")
    print(f"\nFirst 500 characters:\n{'-'*40}")
    print(text[:500])

    return text


def test_ollama_connection() -> bool:
    """Test if Ollama is running."""
    print(f"\n{'='*60}")
    print("STEP 2: Ollama Connection Test")
    print('='*60)

    import requests

    try:
        response = requests.get(
            f"{settings.llm.ollama_base_url}/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"Ollama is running at {settings.llm.ollama_base_url}")
            print(f"Available models: {[m['name'] for m in models]}")
            print(f"Configured model: {settings.llm.ollama_model}")
            return True
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Ollama is not running!")
        print(f"Start it with: ollama serve")
        print(f"Then pull a model: ollama pull {settings.llm.ollama_model}")
        return False

    return False


def test_llm_extraction(text: str) -> dict | None:
    """Test LLM extraction."""
    print(f"\n{'='*60}")
    print("STEP 3: LLM Extraction")
    print('='*60)

    # Use only first ~4000 chars for reliable extraction with qwen3:8b
    # (Full extraction would process all pages in chunks)
    sample_text = text[:4000]
    print(f"Sending {len(sample_text):,} characters to LLM...")

    extractor = LLMExtractor()
    data, method = extractor.extract(sample_text)

    if data:
        print(f"Extraction successful using: {method}")
        stocks = data.get("stocks", [])
        print(f"Stocks extracted: {len(stocks)}")

        if stocks:
            print(f"\nFirst 3 stocks:")
            for stock in stocks[:3]:
                print(f"  - {stock.get('code')}: {stock.get('company_name')}")
                print(f"    Price: {stock.get('closing_price')}, PE: {stock.get('pe')}")

        return data
    else:
        print("Extraction failed!")
        return None


def test_parsing_and_validation(data: dict) -> None:
    """Test parsing and validation."""
    print(f"\n{'='*60}")
    print("STEP 4: Parsing & Validation")
    print('='*60)

    parser = StockDataParser()
    df, sectors = parser.parse(data)

    print(f"Parsed {len(df)} stocks across {len(sectors)} sectors")
    print(f"Sectors: {sectors}")

    # Validate
    report = validate_extraction(df)
    print(f"\nValidation Report:")
    print(f"  Valid Rows: {report.valid_rows}/{report.total_rows}")
    print(f"  Average Confidence: {report.average_confidence:.2%}")
    print(f"  Is Acceptable: {report.is_acceptable}")

    if report.common_issues:
        print(f"  Issues: {report.common_issues}")

    # Show DataFrame
    print(f"\nDataFrame Preview:")
    print(df[['code', 'company_name', 'closing_price', 'eps', 'pe']].head(10).to_string())


def test_full_extraction(pdf_path: Path) -> None:
    """Test full extraction pipeline."""
    print(f"\n{'='*60}")
    print("FULL EXTRACTION TEST")
    print('='*60)

    extractor = CompSheetExtractor()
    result = extractor.extract(pdf_path)

    print(f"\nExtraction Result:")
    print(f"  Trading Date: {result.trading_date}")
    print(f"  Stocks: {len(result.stocks)}")
    print(f"  Sectors: {len(result.sectors)}")
    print(f"  Method: {result.extraction_method}")
    print(f"  Confidence: {result.confidence_score:.2%}")
    print(f"  Is Valid: {result.is_valid}")

    if result.errors:
        print(f"  Errors: {result.errors}")


def main():
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = find_latest_pdf()

    print(f"Testing PDF extraction with: {pdf_path}")

    # Step 1: Text extraction
    text = test_text_extraction(pdf_path)

    # Step 2: Check Ollama
    if not test_ollama_connection():
        print("\nSkipping LLM extraction (Ollama not running)")
        print("To test with LLM, start Ollama and run this script again.")
        return

    # Step 3: LLM extraction
    data = test_llm_extraction(text)

    if data:
        # Step 4: Parsing & validation
        test_parsing_and_validation(data)

    # Optional: Full extraction test
    # test_full_extraction(pdf_path)


if __name__ == "__main__":
    main()
