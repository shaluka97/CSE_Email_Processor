"""PDF Extractor — LLM-based structured data extraction from CSE comp sheets.

Week 2 of the CSE Stock Market Data Pipeline.

This module extracts stock data from LOLC Securities comp sheet PDFs using:
1. pdfplumber for raw text extraction
2. Local LLM (Ollama) for structured JSON extraction
3. Claude API as fallback when local LLM fails validation
4. Pandas DataFrame output with normalized fields
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
import requests

from src.config import settings

logger = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Result of PDF extraction including metadata and confidence."""

    trading_date: date | None
    stocks: pd.DataFrame
    sectors: list[str]
    raw_text: str
    extraction_method: str  # "ollama" or "claude"
    confidence_score: float
    row_scores: list[float]
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if extraction produced valid results."""
        return (
            self.trading_date is not None
            and len(self.stocks) > 0
            and self.confidence_score >= 0.7
        )


@dataclass
class StockRecord:
    """Single stock record from comp sheet."""

    code: str
    company_name: str
    sector: str
    closing_price: float | None
    price_change_pct: float | None
    turnover: float | None
    earnings_4qt: float | None
    eps: float | None
    pe: float | None
    navps: float | None
    pbv: float | None
    roe_pct: float | None
    dps: float | None
    dy_pct: float | None


# ─── PDF Text Extraction ─────────────────────────────────────────────────────

class PDFTextExtractor:
    """Extract raw text from comp sheet PDFs using pdfplumber."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    def extract_text(self) -> str:
        """Extract all text from PDF, preserving table structure."""
        all_text: list[str] = []

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                logger.info(f"Extracting text from {len(pdf.pages)} pages")

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Try table extraction first for better structure
                    tables = page.extract_tables()
                    if tables:
                        page_text = self._tables_to_text(tables)
                    else:
                        # Fall back to raw text
                        page_text = page.extract_text() or ""

                    all_text.append(f"--- Page {page_num} ---\n{page_text}")

        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise

        return "\n\n".join(all_text)

    def _tables_to_text(self, tables: list[list[list[str | None]]]) -> str:
        """Convert extracted tables to tab-separated text."""
        lines: list[str] = []

        for table in tables:
            for row in table:
                if row:
                    # Clean and join cells with tabs
                    cells = [self._clean_cell(cell) for cell in row]
                    lines.append("\t".join(cells))
            lines.append("")  # Empty line between tables

        return "\n".join(lines)

    def _clean_cell(self, cell: str | None) -> str:
        """Clean individual cell value."""
        if cell is None:
            return ""
        # Normalize whitespace
        text = " ".join(str(cell).split())
        # Fix broken numbers from PDF extraction
        text = self._fix_broken_numbers(text)
        return text

    def _fix_broken_numbers(self, text: str) -> str:
        """Fix numbers that were split by PDF extraction.

        Common issues:
        - '1 ,519,190' → '1,519,190' (space before comma)
        - '5 04.23' → '504.23' (space within number)
        - '6 .50' → '6.50' (space before decimal)
        - '1 6,382' → '16,382' (space before digit followed by comma)
        - '( 247.00)' → '(247.00)' (space after parenthesis)
        """
        # Remove space before comma in numbers: "1 ,234" → "1,234"
        text = re.sub(r"(\d)\s+,(\d)", r"\1,\2", text)

        # Remove space before decimal point: "6 .50" → "6.50"
        text = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", text)

        # Remove space after decimal point: "6. 50" → "6.50"
        text = re.sub(r"(\d\.)\s+(\d)", r"\1\2", text)

        # Join split numbers where second part has comma: "1 6,382" → "16,382"
        text = re.sub(r"(\d)\s+(\d,\d)", r"\1\2", text)

        # Join split numbers: "5 04" → "504" (digit space digit(s))
        text = re.sub(r"(\d)\s+(\d{2,}(?:[\.,]\d+)?)", r"\1\2", text)

        # Join single digit splits: "1 5.00" → "15.00"
        text = re.sub(r"(\d)\s+(\d\.\d)", r"\1\2", text)

        # Fix parenthetical negatives: "( 247.00)" → "(247.00)"
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)

        return text

    def extract_trading_date(self) -> date | None:
        """Extract trading date from PDF filename or content."""
        # Try filename pattern: YYYY-MM-DD_comps.pdf
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", self.pdf_path.stem)
        if match:
            try:
                return date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            except ValueError:
                pass

        # Try extracting from content
        text = self.extract_text()[:2000]  # First 2000 chars
        # Pattern: "20/02/2026" or "20 February 2026"
        date_patterns = [
            r"(\d{2})/(\d{2})/(\d{4})",
            r"(\d{2})-(\d{2})-(\d{4})",
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if len(match.groups()) == 3 and match.group(2).isalpha():
                        # Month name format
                        month_map = {
                            "January": 1, "February": 2, "March": 3,
                            "April": 4, "May": 5, "June": 6,
                            "July": 7, "August": 8, "September": 9,
                            "October": 10, "November": 11, "December": 12,
                        }
                        return date(
                            int(match.group(3)),
                            month_map[match.group(2)],
                            int(match.group(1)),
                        )
                    else:
                        # Numeric format DD/MM/YYYY
                        return date(
                            int(match.group(3)),
                            int(match.group(2)),
                            int(match.group(1)),
                        )
                except (ValueError, KeyError):
                    continue

        logger.warning(f"Could not extract date from {self.pdf_path}")
        return None

    def get_page_count(self) -> int:
        """Return number of pages in PDF."""
        with pdfplumber.open(self.pdf_path) as pdf:
            return len(pdf.pages)


# ─── LLM Extraction ──────────────────────────────────────────────────────────

class LLMExtractor:
    """Extract structured stock data using LLM (Ollama primary, Claude fallback)."""

    # JSON schema for stock data
    STOCK_SCHEMA = {
        "type": "object",
        "properties": {
            "trading_date": {"type": "string", "format": "date"},
            "stocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "company_name": {"type": "string"},
                        "sector": {"type": "string"},
                        "closing_price": {"type": ["number", "null"]},
                        "price_change_pct": {"type": ["number", "null"]},
                        "turnover": {"type": ["number", "null"]},
                        "earnings_4qt": {"type": ["number", "null"]},
                        "eps": {"type": ["number", "null"]},
                        "pe": {"type": ["number", "null"]},
                        "navps": {"type": ["number", "null"]},
                        "pbv": {"type": ["number", "null"]},
                        "roe_pct": {"type": ["number", "null"]},
                        "dps": {"type": ["number", "null"]},
                        "dy_pct": {"type": ["number", "null"]},
                    },
                    "required": ["code", "company_name", "sector"],
                },
            },
        },
        "required": ["stocks"],
    }

    def __init__(
        self,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        anthropic_api_key: str | None = None,
    ) -> None:
        self.ollama_base_url = ollama_base_url or settings.llm.ollama_base_url
        self.ollama_model = ollama_model or settings.llm.ollama_model
        self.anthropic_api_key = anthropic_api_key or settings.llm.anthropic_api_key
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load extraction prompt from file or use default."""
        prompt_path = Path("prompts/extraction_prompt.txt")
        if prompt_path.exists():
            return prompt_path.read_text()
        return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Default extraction prompt with schema and examples."""
        return '''Extract stock data from this CSE comp sheet as a JSON array.

RULES:
- (247.00) means -247.00 (negative)
- "-" means null
- Sector headers (Banks, Capital Goods, etc.) appear before their stocks

OUTPUT FORMAT - Return ONLY valid JSON:
{"stocks": [{"code": "XXX.N0000", "company_name": "Name", "sector": "Sector", "closing_price": 0.0, "eps": 0.0, "pe": 0.0, "navps": 0.0, "pbv": 0.0, "dps": null, "dy_pct": null}]}

INPUT:
{text}

JSON:'''

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for LLM input by removing problematic control characters."""
        # Replace tabs with spaces
        text = text.replace("\t", "  ")
        # Remove other control characters except newlines
        text = "".join(
            char if char == "\n" or (ord(char) >= 32 and ord(char) < 127) or ord(char) > 127
            else " "
            for char in text
        )
        # Normalize multiple spaces
        text = re.sub(r"  +", "  ", text)
        return text

    def extract_with_ollama(self, text: str) -> dict[str, Any] | None:
        """Extract data using local Ollama LLM."""
        # Sanitize text to remove control characters
        text = self._sanitize_text(text)
        # Use replace instead of format to avoid issues with JSON braces
        prompt = self._prompt_template.replace("{text}", text)

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0,
                        "num_ctx": 8192,
                        "num_predict": 4096,
                    },
                },
                timeout=300,  # 5 minutes for large PDFs
            )
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            # Parse JSON from response
            result = self._parse_json_response(raw_response)

            # Handle columnar format if needed
            if result and self._is_columnar_format(result):
                result = self._convert_columnar_to_rows(result)

            return result

        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not available - connection refused")
            return None
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama extraction failed: {e}")
            return None

    def extract_with_claude(self, text: str) -> dict[str, Any] | None:
        """Extract data using Claude API as fallback."""
        if not self.anthropic_api_key:
            logger.warning("Claude API key not configured")
            return None

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            # Sanitize text to remove control characters
            text = self._sanitize_text(text)
            # Use replace instead of format to avoid issues with JSON braces
            prompt = self._prompt_template.replace("{text}", text)

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            raw_response = message.content[0].text
            return self._parse_json_response(raw_response)

        except ImportError:
            logger.error("anthropic package not installed")
            return None
        except Exception as e:
            logger.error(f"Claude extraction failed: {e}")
            return None

    def _parse_json_response(self, response: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            return None

    def extract(self, text: str) -> tuple[dict[str, Any] | None, str]:
        """
        Extract stock data using LLM with fallback.

        Returns tuple of (data, method) where method is "ollama" or "claude".
        """
        # Try Ollama first
        logger.info("Attempting extraction with Ollama...")
        result = self.extract_with_ollama(text)
        if result and self._validate_basic_structure(result):
            # Normalize field names to standard format
            result = self._normalize_llm_output(result)
            logger.info("Ollama extraction successful")
            return result, "ollama"

        # Fall back to Claude
        logger.info("Falling back to Claude API...")
        result = self.extract_with_claude(text)
        if result and self._validate_basic_structure(result):
            # Normalize field names to standard format
            result = self._normalize_llm_output(result)
            logger.info("Claude extraction successful")
            return result, "claude"

        logger.error("Both Ollama and Claude extraction failed")
        return None, "none"

    def _validate_basic_structure(self, data: dict[str, Any]) -> bool:
        """Validate basic JSON structure."""
        if not isinstance(data, dict):
            return False

        # Check for columnar format (each key is a list)
        # Convert to row-based format if needed
        if self._is_columnar_format(data):
            data = self._convert_columnar_to_rows(data)

        # Find the stocks array - accept multiple possible key names
        stocks_keys = ["stocks", "market_data", "data", "records", "companies"]
        stocks = None
        for key in stocks_keys:
            if key in data and isinstance(data[key], list):
                stocks = data[key]
                # Normalize to "stocks" key
                if key != "stocks":
                    data["stocks"] = stocks
                    del data[key]
                break

        if stocks is None:
            # Maybe the data itself is a list
            if isinstance(data.get("stocks"), list):
                stocks = data["stocks"]
            else:
                return False

        if len(stocks) == 0:
            return False

        # Check first few stocks have required fields
        # Accept both snake_case and camelCase formats
        for stock in stocks[:3]:
            if not isinstance(stock, dict):
                return False
            # Check for code field (accept various names)
            has_code = any(
                k in stock
                for k in ("code", "Code", "ticker", "symbol", "stock_code")
            )
            if not has_code:
                return False
            # Check for company name (accept multiple formats)
            has_name = any(
                k in stock
                for k in ("company_name", "companyName", "name", "company", "Company")
            )
            if not has_name:
                return False

        return True

    def _is_columnar_format(self, data: dict[str, Any]) -> bool:
        """Check if data is in columnar format (each key is a list)."""
        # Check if we have code/company_name as lists
        code_keys = ["code", "Code", "ticker", "symbol"]
        for key in code_keys:
            if key in data and isinstance(data[key], list):
                return True
        return False

    def _convert_columnar_to_rows(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert columnar format to row-based format."""
        # Find the length from any list column
        length = 0
        for value in data.values():
            if isinstance(value, list):
                length = max(length, len(value))
                break

        if length == 0:
            return {"stocks": []}

        # Build rows
        stocks = []
        for i in range(length):
            row = {}
            for key, value in data.items():
                if isinstance(value, list) and i < len(value):
                    row[key] = value[i]
                elif isinstance(value, list):
                    row[key] = None
                else:
                    row[key] = value
            stocks.append(row)

        return {"stocks": stocks}

    def _normalize_llm_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize LLM output to standard snake_case format."""
        if "stocks" not in data:
            return data

        # Field mapping from various formats to standard
        field_map = {
            # Code variations
            "Code": "code",
            "ticker": "code",
            "symbol": "code",
            "stock_code": "code",
            # Company name variations
            "companyName": "company_name",
            "company": "company_name",
            "name": "company_name",
            "Company": "company_name",
            # Price variations
            "closingPrice": "closing_price",
            "marketPrice": "closing_price",
            "market_price": "closing_price",
            "price": "closing_price",
            # Price change variations
            "priceChange": "price_change_pct",
            "priceChangePct": "price_change_pct",
            "price_change": "price_change_pct",
            # Earnings variations
            "earnings4qt": "earnings_4qt",
            "latest4QTearnings": "earnings_4qt",
            "earnings": "earnings_4qt",
            # EPS variations
            "latest4QTEPS": "eps",
            "EPS": "eps",
            # NAVPS variations
            "navps": "navps",
            "NAVPS": "navps",
            # PBV variations
            "pbv": "pbv",
            "PBV": "pbv",
            # ROE variations
            "roE": "roe_pct",
            "roe": "roe_pct",
            "ROE": "roe_pct",
            # DPS variations
            "ttmDps": "dps",
            "DPS": "dps",
            "ttm_dps": "dps",
            # DY variations
            "dy": "dy_pct",
            "DY": "dy_pct",
            "dyPct": "dy_pct",
        }

        normalized_stocks = []
        for stock in data["stocks"]:
            normalized = {}
            for key, value in stock.items():
                # Use mapped key if available, otherwise convert camelCase to snake_case
                if key in field_map:
                    new_key = field_map[key]
                else:
                    # Convert camelCase to snake_case
                    new_key = "".join(
                        f"_{c.lower()}" if c.isupper() else c for c in key
                    ).lstrip("_")
                normalized[new_key] = value
            normalized_stocks.append(normalized)

        data["stocks"] = normalized_stocks
        return data


# ─── Data Parsing and Normalization ──────────────────────────────────────────

class StockDataParser:
    """Parse and normalize LLM-extracted stock data."""

    # Known CSE sectors
    KNOWN_SECTORS = [
        "Automobiles & Components",
        "Banks",
        "Capital Goods",
        "Commercial & Professional Services",
        "Consumer Durables & Apparel",
        "Consumer Services",
        "Diversified Financials",
        "Energy",
        "Food & Staples Retailing",
        "Food, Beverage & Tobacco",
        "Healthcare Equipment & Services",
        "Household & Personal Products",
        "Insurance",
        "Materials",
        "Real Estate Management & Development",
        "Retailing",
        "Software & Services",
        "Telecommunication Services",
        "Transportation",
        "Utilities",
    ]

    def __init__(self) -> None:
        pass

    def parse(self, data: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        """
        Parse LLM output into normalized DataFrame.

        Returns tuple of (DataFrame, list of detected sectors).
        """
        stocks = data.get("stocks", [])
        if not stocks:
            return pd.DataFrame(), []

        records: list[dict[str, Any]] = []
        sectors_found: set[str] = set()

        for stock in stocks:
            record = self._normalize_stock(stock)
            records.append(record)
            if record.get("sector"):
                sectors_found.add(record["sector"])

        df = pd.DataFrame(records)

        # Ensure correct column order and types
        df = self._enforce_schema(df)

        return df, sorted(sectors_found)

    def _normalize_stock(self, stock: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single stock record."""
        return {
            "code": self._clean_string(stock.get("code", "")),
            "company_name": self._clean_string(stock.get("company_name", "")),
            "sector": self._normalize_sector(stock.get("sector", "")),
            "closing_price": self._parse_number(stock.get("closing_price")),
            "price_change_pct": self._parse_number(stock.get("price_change_pct")),
            "turnover": self._parse_number(stock.get("turnover")),
            "earnings_4qt": self._parse_number(stock.get("earnings_4qt")),
            "eps": self._parse_number(stock.get("eps")),
            "pe": self._parse_number(stock.get("pe")),
            "navps": self._parse_number(stock.get("navps")),
            "pbv": self._parse_number(stock.get("pbv")),
            "roe_pct": self._parse_number(stock.get("roe_pct")),
            "dps": self._parse_number(stock.get("dps")),
            "dy_pct": self._parse_number(stock.get("dy_pct")),
        }

    def _clean_string(self, value: Any) -> str:
        """Clean string value."""
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_sector(self, sector: str | None) -> str:
        """Normalize sector name to known sectors."""
        if not sector:
            return "Unknown"

        sector = sector.strip()

        # Direct match
        if sector in self.KNOWN_SECTORS:
            return sector

        # Case-insensitive match
        sector_lower = sector.lower()
        for known in self.KNOWN_SECTORS:
            if known.lower() == sector_lower:
                return known

        # Partial match
        for known in self.KNOWN_SECTORS:
            if sector_lower in known.lower() or known.lower() in sector_lower:
                return known

        return sector  # Return as-is if no match

    def _parse_number(self, value: Any) -> float | None:
        """
        Parse numeric value handling various formats.

        - None, "-", "" → None
        - (247.00) → -247.00 (parenthetical negatives)
        - "1,234.56" → 1234.56 (comma thousands)
        - "5.5%" → 5.5 (strip percentage sign)
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            value = value.strip()

            # Check for missing value indicators
            if value in ("", "-", "N/A", "n/a", "null", "None"):
                return None

            # Handle parenthetical negatives: (247.00) → -247.00
            if value.startswith("(") and value.endswith(")"):
                value = "-" + value[1:-1]

            # Remove percentage sign
            value = value.rstrip("%")

            # Remove thousand separators (comma)
            value = value.replace(",", "")

            try:
                return float(value)
            except ValueError:
                logger.debug(f"Could not parse number: {value}")
                return None

        return None

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has correct columns and types."""
        expected_columns = [
            "code",
            "company_name",
            "sector",
            "closing_price",
            "price_change_pct",
            "turnover",
            "earnings_4qt",
            "eps",
            "pe",
            "navps",
            "pbv",
            "roe_pct",
            "dps",
            "dy_pct",
        ]

        # Add missing columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        # Reorder columns
        df = df[expected_columns]

        # Set numeric types (allowing None)
        numeric_cols = expected_columns[3:]  # All after sector
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# ─── Main Extractor Interface ────────────────────────────────────────────────

class CompSheetExtractor:
    """Main interface for extracting data from comp sheet PDFs."""

    def __init__(
        self,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        anthropic_api_key: str | None = None,
    ) -> None:
        self.llm_extractor = LLMExtractor(
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            anthropic_api_key=anthropic_api_key,
        )
        self.parser = StockDataParser()

    def extract(self, pdf_path: str | Path) -> ExtractionResult:
        """
        Extract stock data from a comp sheet PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractionResult with stocks DataFrame and metadata
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Extracting data from {pdf_path.name}")

        errors: list[str] = []

        # Step 1: Extract raw text
        try:
            text_extractor = PDFTextExtractor(pdf_path)
            raw_text = text_extractor.extract_text()
            trading_date = text_extractor.extract_trading_date()
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ExtractionResult(
                trading_date=None,
                stocks=pd.DataFrame(),
                sectors=[],
                raw_text="",
                extraction_method="none",
                confidence_score=0.0,
                row_scores=[],
                errors=[str(e)],
            )

        # Step 2: LLM extraction with fallback
        data, method = self.llm_extractor.extract(raw_text)
        if data is None:
            errors.append("LLM extraction failed for both Ollama and Claude")
            return ExtractionResult(
                trading_date=trading_date,
                stocks=pd.DataFrame(),
                sectors=[],
                raw_text=raw_text,
                extraction_method="none",
                confidence_score=0.0,
                row_scores=[],
                errors=errors,
            )

        # Step 3: Parse and normalize
        try:
            df, sectors = self.parser.parse(data)
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            errors.append(f"Parsing failed: {e}")
            return ExtractionResult(
                trading_date=trading_date,
                stocks=pd.DataFrame(),
                sectors=[],
                raw_text=raw_text,
                extraction_method=method,
                confidence_score=0.0,
                row_scores=[],
                errors=errors,
            )

        # Step 4: Confidence scoring (imported from data_validator)
        # For now, use placeholder - will be replaced by data_validator
        from src.data_validator import ConfidenceScorer

        scorer = ConfidenceScorer()
        row_scores = scorer.score_dataframe(df)
        confidence_score = scorer.aggregate_score(row_scores)

        logger.info(
            f"Extraction complete: {len(df)} stocks, "
            f"{len(sectors)} sectors, confidence={confidence_score:.2%}"
        )

        return ExtractionResult(
            trading_date=trading_date,
            stocks=df,
            sectors=sectors,
            raw_text=raw_text,
            extraction_method=method,
            confidence_score=confidence_score,
            row_scores=row_scores,
            errors=errors,
        )


# ─── Convenience Functions ───────────────────────────────────────────────────

def extract_comp_sheet(pdf_path: str | Path) -> ExtractionResult:
    """
    Extract data from a comp sheet PDF.

    Convenience function using default settings.
    """
    extractor = CompSheetExtractor()
    return extractor.extract(pdf_path)


def extract_text_only(pdf_path: str | Path) -> str:
    """Extract raw text from PDF without LLM processing."""
    extractor = PDFTextExtractor(pdf_path)
    return extractor.extract_text()
