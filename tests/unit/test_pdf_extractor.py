"""Unit tests for PDF extraction module (Week 2).

Tests cover:
- PDF text extraction with pdfplumber
- Date extraction from filename and content
- LLM prompt engineering
- Data parsing and normalization
- Parenthetical negative handling
- Missing value handling
- Sector detection
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pdf_extractor import (
    CompSheetExtractor,
    ExtractionResult,
    LLMExtractor,
    PDFTextExtractor,
    StockDataParser,
    StockRecord,
)


# ─── PDFTextExtractor Tests ──────────────────────────────────────────────────


class TestPDFTextExtractor:
    """Tests for PDF text extraction."""

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            PDFTextExtractor(tmp_path / "nonexistent.pdf")

    def test_extract_trading_date_from_filename(self, tmp_path: Path) -> None:
        """Test date extraction from filename pattern."""
        # Create a minimal PDF file
        pdf_path = tmp_path / "2026-02-20_comps.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

        with patch.object(PDFTextExtractor, "extract_text", return_value=""):
            extractor = PDFTextExtractor.__new__(PDFTextExtractor)
            extractor.pdf_path = pdf_path
            result = extractor.extract_trading_date()

        assert result == date(2026, 2, 20)

    def test_extract_trading_date_various_formats(self) -> None:
        """Test date parsing from various filename formats."""
        test_cases = [
            ("2025-12-31_comps.pdf", date(2025, 12, 31)),
            ("2024-01-15_comps.pdf", date(2024, 1, 15)),
            ("data_2026-06-01_comps.pdf", date(2026, 6, 1)),
        ]

        for filename, expected in test_cases:
            extractor = PDFTextExtractor.__new__(PDFTextExtractor)
            extractor.pdf_path = Path(filename)

            with patch.object(extractor, "extract_text", return_value=""):
                result = extractor.extract_trading_date()

            assert result == expected, f"Failed for {filename}"


# ─── StockDataParser Tests ───────────────────────────────────────────────────


class TestStockDataParser:
    """Tests for stock data parsing and normalization."""

    @pytest.fixture
    def parser(self) -> StockDataParser:
        return StockDataParser()

    def test_parse_number_normal(self, parser: StockDataParser) -> None:
        """Test parsing normal numeric values."""
        assert parser._parse_number(123.45) == 123.45
        assert parser._parse_number(100) == 100.0
        assert parser._parse_number("456.78") == 456.78

    def test_parse_number_with_commas(self, parser: StockDataParser) -> None:
        """Test parsing numbers with thousand separators."""
        assert parser._parse_number("1,234.56") == 1234.56
        assert parser._parse_number("1,234,567") == 1234567.0

    def test_parse_number_parenthetical_negative(
        self, parser: StockDataParser
    ) -> None:
        """Test parsing negative numbers in parentheses."""
        assert parser._parse_number("(247.00)") == -247.00
        assert parser._parse_number("(1,234.56)") == -1234.56

    def test_parse_number_percentage(self, parser: StockDataParser) -> None:
        """Test parsing percentage values."""
        assert parser._parse_number("5.5%") == 5.5
        assert parser._parse_number("-1.23%") == -1.23

    def test_parse_number_missing_values(self, parser: StockDataParser) -> None:
        """Test parsing missing value indicators."""
        assert parser._parse_number(None) is None
        assert parser._parse_number("-") is None
        assert parser._parse_number("") is None
        assert parser._parse_number("N/A") is None
        assert parser._parse_number("n/a") is None

    def test_normalize_sector_exact_match(self, parser: StockDataParser) -> None:
        """Test sector normalization with exact match."""
        assert parser._normalize_sector("Banks") == "Banks"
        assert parser._normalize_sector("Capital Goods") == "Capital Goods"

    def test_normalize_sector_case_insensitive(
        self, parser: StockDataParser
    ) -> None:
        """Test case-insensitive sector matching."""
        assert parser._normalize_sector("banks") == "Banks"
        assert parser._normalize_sector("BANKS") == "Banks"
        assert parser._normalize_sector("CAPITAL GOODS") == "Capital Goods"

    def test_normalize_sector_partial_match(self, parser: StockDataParser) -> None:
        """Test partial sector matching."""
        # Test when input contains part of a known sector
        assert parser._normalize_sector("Telecommunication") == "Telecommunication Services"
        assert parser._normalize_sector("Real Estate") == "Real Estate Management & Development"

    def test_normalize_sector_unknown(self, parser: StockDataParser) -> None:
        """Test handling of unknown sectors."""
        assert parser._normalize_sector("") == "Unknown"
        assert parser._normalize_sector(None) == "Unknown"

    def test_parse_empty_data(self, parser: StockDataParser) -> None:
        """Test parsing empty stock data."""
        df, sectors = parser.parse({"stocks": []})
        assert len(df) == 0
        assert len(sectors) == 0

    def test_parse_single_stock(self, parser: StockDataParser) -> None:
        """Test parsing a single stock record."""
        data = {
            "stocks": [
                {
                    "code": "COMB.N0000",
                    "company_name": "COMMERCIAL BANK OF CEYLON PLC",
                    "sector": "Banks",
                    "closing_price": 230.50,
                    "price_change_pct": -1.07,
                    "turnover": 51010412,
                    "earnings_4qt": 71353.77,
                    "eps": 43.59,
                    "pe": 5.29,
                    "navps": 193.80,
                    "pbv": 1.19,
                    "roe_pct": 22.49,
                    "dps": 9.20,
                    "dy_pct": 3.99,
                }
            ]
        }

        df, sectors = parser.parse(data)

        assert len(df) == 1
        assert df.iloc[0]["code"] == "COMB.N0000"
        assert df.iloc[0]["closing_price"] == 230.50
        assert "Banks" in sectors

    def test_parse_negative_eps_stock(self, parser: StockDataParser) -> None:
        """Test parsing stock with negative EPS (loss-making company)."""
        data = {
            "stocks": [
                {
                    "code": "HDFC.N0000",
                    "company_name": "HDFC BANK",
                    "sector": "Banks",
                    "closing_price": 55.00,
                    "eps": -3.82,
                    "pe": -14.41,
                    "earnings_4qt": "(247.00)",  # Parenthetical negative
                }
            ]
        }

        df, _ = parser.parse(data)

        assert df.iloc[0]["eps"] == -3.82
        assert df.iloc[0]["pe"] == -14.41
        assert df.iloc[0]["earnings_4qt"] == -247.00

    def test_parse_missing_dividend_data(self, parser: StockDataParser) -> None:
        """Test parsing stock with missing dividend data."""
        data = {
            "stocks": [
                {
                    "code": "TEST.N0000",
                    "company_name": "TEST COMPANY",
                    "sector": "Banks",
                    "closing_price": 100.00,
                    "dps": "-",
                    "dy_pct": None,
                }
            ]
        }

        df, _ = parser.parse(data)

        assert pd.isna(df.iloc[0]["dps"])
        assert pd.isna(df.iloc[0]["dy_pct"])

    def test_parse_zero_turnover(self, parser: StockDataParser) -> None:
        """Test parsing stock with zero or no turnover."""
        data = {
            "stocks": [
                {
                    "code": "TEST.N0000",
                    "company_name": "TEST COMPANY",
                    "sector": "Banks",
                    "closing_price": 100.00,
                    "turnover": 0,
                }
            ]
        }

        df, _ = parser.parse(data)
        assert df.iloc[0]["turnover"] == 0.0

    def test_enforce_schema_adds_missing_columns(
        self, parser: StockDataParser
    ) -> None:
        """Test that schema enforcement adds missing columns."""
        df = pd.DataFrame(
            [{"code": "TEST.N0000", "company_name": "TEST", "sector": "Banks"}]
        )

        result = parser._enforce_schema(df)

        assert "closing_price" in result.columns
        assert "eps" in result.columns
        assert "pe" in result.columns

    def test_multiple_sectors(self, parser: StockDataParser) -> None:
        """Test parsing stocks from multiple sectors."""
        data = {
            "stocks": [
                {"code": "BANK1.N0000", "company_name": "Bank 1", "sector": "Banks"},
                {"code": "CAP1.N0000", "company_name": "Cap 1", "sector": "Capital Goods"},
                {"code": "BANK2.N0000", "company_name": "Bank 2", "sector": "Banks"},
            ]
        }

        df, sectors = parser.parse(data)

        assert len(df) == 3
        assert len(sectors) == 2
        assert "Banks" in sectors
        assert "Capital Goods" in sectors


# ─── LLMExtractor Tests ──────────────────────────────────────────────────────


class TestLLMExtractor:
    """Tests for LLM extraction."""

    @pytest.fixture
    def extractor(self) -> LLMExtractor:
        return LLMExtractor(
            ollama_base_url="http://localhost:11434",
            ollama_model="test-model",
            anthropic_api_key="test-key",
        )

    def test_parse_json_response_clean(self, extractor: LLMExtractor) -> None:
        """Test parsing clean JSON response."""
        response = '{"stocks": [{"code": "TEST.N0000"}]}'
        result = extractor._parse_json_response(response)

        assert result is not None
        assert "stocks" in result

    def test_parse_json_response_with_code_block(
        self, extractor: LLMExtractor
    ) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        response = '```json\n{"stocks": [{"code": "TEST.N0000"}]}\n```'
        result = extractor._parse_json_response(response)

        assert result is not None
        assert "stocks" in result

    def test_parse_json_response_invalid(self, extractor: LLMExtractor) -> None:
        """Test parsing invalid JSON."""
        response = "This is not JSON"
        result = extractor._parse_json_response(response)

        assert result is None

    def test_validate_basic_structure_valid(self, extractor: LLMExtractor) -> None:
        """Test validation of valid structure."""
        data = {
            "stocks": [
                {"code": "TEST.N0000", "company_name": "TEST COMPANY"}
            ]
        }
        assert extractor._validate_basic_structure(data) is True

    def test_validate_basic_structure_missing_stocks(
        self, extractor: LLMExtractor
    ) -> None:
        """Test validation fails without stocks key."""
        data = {"other_key": []}
        assert extractor._validate_basic_structure(data) is False

    def test_validate_basic_structure_empty_stocks(
        self, extractor: LLMExtractor
    ) -> None:
        """Test validation fails with empty stocks list."""
        data = {"stocks": []}
        assert extractor._validate_basic_structure(data) is False

    def test_validate_basic_structure_missing_code(
        self, extractor: LLMExtractor
    ) -> None:
        """Test validation fails when stocks missing code."""
        data = {"stocks": [{"company_name": "TEST"}]}
        assert extractor._validate_basic_structure(data) is False

    @patch("requests.post")
    def test_extract_with_ollama_success(
        self, mock_post: MagicMock, extractor: LLMExtractor
    ) -> None:
        """Test successful Ollama extraction."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"stocks": [{"code": "TEST.N0000", "company_name": "TEST"}]}'
        }
        mock_post.return_value = mock_response

        result = extractor.extract_with_ollama("test text")

        assert result is not None
        assert "stocks" in result

    @patch("requests.post")
    def test_extract_with_ollama_connection_error(
        self, mock_post: MagicMock, extractor: LLMExtractor
    ) -> None:
        """Test Ollama extraction with connection error."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError()

        result = extractor.extract_with_ollama("test text")

        assert result is None


# ─── ExtractionResult Tests ──────────────────────────────────────────────────


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_is_valid_with_valid_data(self) -> None:
        """Test is_valid returns True for valid extraction."""
        result = ExtractionResult(
            trading_date=date(2026, 2, 20),
            stocks=pd.DataFrame([{"code": "TEST.N0000"}]),
            sectors=["Banks"],
            raw_text="test",
            extraction_method="ollama",
            confidence_score=0.85,
            row_scores=[0.85],
        )

        assert result.is_valid is True

    def test_is_valid_with_low_confidence(self) -> None:
        """Test is_valid returns False for low confidence."""
        result = ExtractionResult(
            trading_date=date(2026, 2, 20),
            stocks=pd.DataFrame([{"code": "TEST.N0000"}]),
            sectors=["Banks"],
            raw_text="test",
            extraction_method="ollama",
            confidence_score=0.5,  # Below threshold
            row_scores=[0.5],
        )

        assert result.is_valid is False

    def test_is_valid_with_empty_stocks(self) -> None:
        """Test is_valid returns False with no stocks."""
        result = ExtractionResult(
            trading_date=date(2026, 2, 20),
            stocks=pd.DataFrame(),
            sectors=[],
            raw_text="test",
            extraction_method="ollama",
            confidence_score=0.85,
            row_scores=[],
        )

        assert result.is_valid is False

    def test_is_valid_with_no_date(self) -> None:
        """Test is_valid returns False without trading date."""
        result = ExtractionResult(
            trading_date=None,
            stocks=pd.DataFrame([{"code": "TEST.N0000"}]),
            sectors=["Banks"],
            raw_text="test",
            extraction_method="ollama",
            confidence_score=0.85,
            row_scores=[0.85],
        )

        assert result.is_valid is False


# ─── Integration-style Tests ─────────────────────────────────────────────────


class TestStockDataParserIntegration:
    """Integration tests for parsing real-world-like data."""

    @pytest.fixture
    def parser(self) -> StockDataParser:
        return StockDataParser()

    def test_parse_bank_sector_sample(self, parser: StockDataParser) -> None:
        """Test parsing sample bank sector data."""
        data = {
            "trading_date": "2026-02-20",
            "stocks": [
                {
                    "code": "ABL.N0000",
                    "company_name": "AMANA BANK PLC",
                    "sector": "Banks",
                    "closing_price": 29.70,
                    "price_change_pct": -1.00,
                    "turnover": 3136920,
                    "earnings_4qt": 2262.58,
                    "eps": 4.10,
                    "pe": 7.24,
                    "navps": 44.83,
                    "pbv": 0.66,
                    "roe_pct": 9.16,
                    "dps": 1.30,
                    "dy_pct": 4.38,
                },
                {
                    "code": "CBNK.N0000",
                    "company_name": "CARGILLS BANK PLC",
                    "sector": "Banks",
                    "closing_price": 9.60,
                    "price_change_pct": -1.03,
                    "turnover": 16382784,
                    "earnings_4qt": 805.44,
                    "eps": 0.85,
                    "pe": 11.27,
                    "navps": 12.65,
                    "pbv": 0.76,
                    "roe_pct": 6.73,
                    "dps": None,  # Missing dividend
                    "dy_pct": None,
                },
                {
                    "code": "HDFC.N0000",
                    "company_name": "HDFC BANK",
                    "sector": "Banks",
                    "closing_price": 55.00,
                    "price_change_pct": -0.72,
                    "turnover": 3262565,
                    "earnings_4qt": -247.00,  # Negative earnings
                    "eps": -3.82,
                    "pe": -14.41,
                    "navps": 120.72,
                    "pbv": 0.46,
                    "roe_pct": -3.16,
                    "dps": None,
                    "dy_pct": None,
                },
            ],
        }

        df, sectors = parser.parse(data)

        # Check row count
        assert len(df) == 3

        # Check sector detection
        assert sectors == ["Banks"]

        # Check normal stock
        abl = df[df["code"] == "ABL.N0000"].iloc[0]
        assert abl["closing_price"] == 29.70
        assert abl["eps"] == 4.10
        assert abl["dps"] == 1.30

        # Check stock with missing dividend
        cbnk = df[df["code"] == "CBNK.N0000"].iloc[0]
        assert pd.isna(cbnk["dps"])
        assert pd.isna(cbnk["dy_pct"])

        # Check stock with negative values
        hdfc = df[df["code"] == "HDFC.N0000"].iloc[0]
        assert hdfc["earnings_4qt"] == -247.00
        assert hdfc["eps"] == -3.82
        assert hdfc["pe"] == -14.41
