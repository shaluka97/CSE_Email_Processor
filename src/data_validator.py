"""Data Validator — metamorphic confidence scoring for extracted comp sheet data.

Week 2 of the CSE Stock Market Data Pipeline.

This module validates LLM-extracted stock data using:
1. JSON schema validation
2. Cross-field metamorphic checks (PE × EPS ≈ Price, PBV × NAVPS ≈ Price)
3. Per-row confidence scores
4. Handling of edge cases (negative EPS, missing values, etc.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of validation for a single stock record."""

    code: str
    is_valid: bool
    confidence_score: float
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Summary report of data quality for an extraction."""

    total_rows: int
    valid_rows: int
    average_confidence: float
    min_confidence: float
    max_confidence: float
    common_issues: dict[str, int]
    row_results: list[ValidationResult]

    @property
    def is_acceptable(self) -> bool:
        """Check if overall data quality is acceptable."""
        return (
            self.average_confidence >= 0.7
            and self.valid_rows / self.total_rows >= 0.8
            if self.total_rows > 0
            else False
        )


# ─── Confidence Scoring ──────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Calculate confidence scores for extracted stock data.

    Uses metamorphic checks to verify data consistency:
    - PE × EPS ≈ Price (within tolerance)
    - PBV × NAVPS ≈ Price (within tolerance)
    - DY = DPS / Price × 100 (within tolerance)
    """

    # Tolerance for metamorphic checks (percentage difference allowed)
    DEFAULT_TOLERANCE = 0.05  # 5%

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE) -> None:
        self.tolerance = tolerance

    def score_row(self, row: pd.Series) -> float:
        """
        Calculate confidence score for a single row.

        Returns score between 0.0 and 1.0.
        """
        checks_weight = {
            "has_code": 0.15,
            "has_company_name": 0.10,
            "has_price": 0.15,
            "pe_eps_check": 0.20,
            "pbv_navps_check": 0.20,
            "dy_check": 0.10,
            "reasonable_pe": 0.05,
            "reasonable_pbv": 0.05,
        }

        score = 0.0

        # Basic field presence checks
        if pd.notna(row.get("code")) and row.get("code"):
            score += checks_weight["has_code"]

        if pd.notna(row.get("company_name")) and row.get("company_name"):
            score += checks_weight["has_company_name"]

        price = row.get("closing_price")
        if pd.notna(price) and price is not None and price > 0:
            score += checks_weight["has_price"]

        # Metamorphic check: PE × EPS ≈ Price
        pe_score = self._check_pe_eps_price(row)
        score += checks_weight["pe_eps_check"] * pe_score

        # Metamorphic check: PBV × NAVPS ≈ Price
        pbv_score = self._check_pbv_navps_price(row)
        score += checks_weight["pbv_navps_check"] * pbv_score

        # Metamorphic check: DY = DPS / Price × 100
        dy_score = self._check_dy_calculation(row)
        score += checks_weight["dy_check"] * dy_score

        # Reasonableness checks
        if self._is_reasonable_pe(row):
            score += checks_weight["reasonable_pe"]

        if self._is_reasonable_pbv(row):
            score += checks_weight["reasonable_pbv"]

        return min(1.0, max(0.0, score))

    def _check_pe_eps_price(self, row: pd.Series) -> float:
        """
        Check if PE × EPS ≈ Price.

        For stocks with negative EPS, PE should be negative or N/A.
        Returns 1.0 if check passes, 0.0-1.0 based on accuracy, or 0.5 for N/A.
        """
        price = row.get("closing_price")
        pe = row.get("pe")
        eps = row.get("eps")

        # Skip if essential values missing
        if pd.isna(price) or price is None or price <= 0:
            return 0.5  # Can't verify, neutral score
        if pd.isna(pe) or pd.isna(eps):
            return 0.5  # Can't verify, neutral score

        # Handle negative EPS cases
        if eps < 0:
            # PE should be negative for stocks with negative earnings
            if pe < 0:
                return 1.0  # Correct handling
            else:
                return 0.3  # Inconsistent

        if eps == 0:
            return 0.5  # Can't calculate, neutral

        # Calculate expected price
        expected_price = pe * eps

        # Check if within tolerance
        if expected_price > 0:
            difference = abs(price - expected_price) / price
            if difference <= self.tolerance:
                return 1.0
            elif difference <= self.tolerance * 2:
                return 0.7
            elif difference <= self.tolerance * 4:
                return 0.4
            else:
                return 0.2

        return 0.5

    def _check_pbv_navps_price(self, row: pd.Series) -> float:
        """
        Check if PBV × NAVPS ≈ Price.

        Returns 1.0 if check passes, 0.0-1.0 based on accuracy.
        """
        price = row.get("closing_price")
        pbv = row.get("pbv")
        navps = row.get("navps")

        # Skip if essential values missing
        if pd.isna(price) or price is None or price <= 0:
            return 0.5
        if pd.isna(pbv) or pd.isna(navps):
            return 0.5

        # Handle negative NAVPS (company with negative equity)
        if navps < 0:
            if pbv < 0:
                return 1.0  # Correct handling
            else:
                return 0.3  # Inconsistent

        if navps == 0:
            return 0.5  # Can't calculate

        # Calculate expected price
        expected_price = pbv * navps

        # Check if within tolerance
        if expected_price > 0:
            difference = abs(price - expected_price) / price
            if difference <= self.tolerance:
                return 1.0
            elif difference <= self.tolerance * 2:
                return 0.7
            elif difference <= self.tolerance * 4:
                return 0.4
            else:
                return 0.2

        return 0.5

    def _check_dy_calculation(self, row: pd.Series) -> float:
        """
        Check if DY = DPS / Price × 100.

        Returns 1.0 if check passes, 0.0-1.0 based on accuracy.
        """
        price = row.get("closing_price")
        dps = row.get("dps")
        dy = row.get("dy_pct")

        # If no dividend data, it's valid for both to be null
        if pd.isna(dps) and pd.isna(dy):
            return 1.0  # Both missing is consistent

        if pd.isna(dps) or dps == 0:
            # DY should also be 0 or null
            if pd.isna(dy) or dy == 0:
                return 1.0
            else:
                return 0.3  # Inconsistent

        # Skip if price missing
        if pd.isna(price) or price is None or price <= 0:
            return 0.5

        if pd.isna(dy):
            return 0.5  # Can't verify

        # Calculate expected DY
        expected_dy = (dps / price) * 100

        # Check if within tolerance
        difference = abs(dy - expected_dy)
        if difference <= 0.5:  # Within 0.5 percentage points
            return 1.0
        elif difference <= 1.0:
            return 0.7
        elif difference <= 2.0:
            return 0.4
        else:
            return 0.2

    def _is_reasonable_pe(self, row: pd.Series) -> bool:
        """Check if PE ratio is within reasonable bounds."""
        pe = row.get("pe")
        if pd.isna(pe):
            return True  # Missing is acceptable

        # Reasonable PE bounds (-500 to 500 for edge cases)
        return -500 <= pe <= 500

    def _is_reasonable_pbv(self, row: pd.Series) -> bool:
        """Check if PBV ratio is within reasonable bounds."""
        pbv = row.get("pbv")
        if pd.isna(pbv):
            return True  # Missing is acceptable

        # Reasonable PBV bounds (-50 to 100 for edge cases)
        return -50 <= pbv <= 100

    def score_dataframe(self, df: pd.DataFrame) -> list[float]:
        """
        Calculate confidence scores for all rows in DataFrame.

        Returns list of scores (one per row).
        """
        if df.empty:
            return []

        scores = []
        for _, row in df.iterrows():
            score = self.score_row(row)
            scores.append(score)

        return scores

    def aggregate_score(self, scores: list[float]) -> float:
        """
        Calculate aggregate confidence score from row scores.

        Uses weighted average with lower scores weighted more heavily
        (to penalize inconsistent data).
        """
        if not scores:
            return 0.0

        # Simple average for now, can be made more sophisticated
        avg = sum(scores) / len(scores)

        # Penalize if many low scores
        low_score_count = sum(1 for s in scores if s < 0.5)
        low_score_penalty = (low_score_count / len(scores)) * 0.2

        return max(0.0, avg - low_score_penalty)


# ─── Data Validator ──────────────────────────────────────────────────────────

class DataValidator:
    """
    Validate extracted stock data for completeness and consistency.
    """

    # Required fields that should always be present
    REQUIRED_FIELDS = ["code", "company_name", "sector"]

    # Numeric fields that should be valid numbers when present
    NUMERIC_FIELDS = [
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

    def __init__(self, scorer: ConfidenceScorer | None = None) -> None:
        self.scorer = scorer or ConfidenceScorer()

    def validate_row(self, row: pd.Series) -> ValidationResult:
        """Validate a single row of stock data."""
        code = row.get("code", "UNKNOWN")
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        warnings: list[str] = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            value = row.get(field)
            if pd.notna(value) and value:
                checks_passed.append(f"has_{field}")
            else:
                checks_failed.append(f"missing_{field}")

        # Check stock code format
        if pd.notna(row.get("code")):
            code_str = str(row.get("code"))
            if self._is_valid_stock_code(code_str):
                checks_passed.append("valid_code_format")
            else:
                warnings.append(f"unusual_code_format: {code_str}")

        # Check price is positive
        price = row.get("closing_price")
        if pd.notna(price):
            if price > 0:
                checks_passed.append("positive_price")
            else:
                checks_failed.append("non_positive_price")

        # Check for unreasonable values
        if self._has_unreasonable_values(row):
            warnings.append("unreasonable_values")

        # Calculate confidence score
        confidence = self.scorer.score_row(row)

        is_valid = (
            len(checks_failed) == 0
            and confidence >= 0.5
        )

        return ValidationResult(
            code=str(code),
            is_valid=is_valid,
            confidence_score=confidence,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
        )

    def _is_valid_stock_code(self, code: str) -> bool:
        """Check if stock code follows CSE format."""
        # CSE codes typically: XXXX.N0000 or XXXX.X0000
        if not code:
            return False

        # Allow various CSE formats
        parts = code.split(".")
        if len(parts) != 2:
            return False

        symbol, suffix = parts
        if len(symbol) < 2 or len(symbol) > 10:
            return False
        if suffix not in ("N0000", "X0000", "P0000"):
            return False

        return True

    def _has_unreasonable_values(self, row: pd.Series) -> bool:
        """Check for obviously unreasonable values."""
        # Check for extreme PE ratios
        pe = row.get("pe")
        if pd.notna(pe) and (pe > 1000 or pe < -1000):
            return True

        # Check for extreme PBV
        pbv = row.get("pbv")
        if pd.notna(pbv) and (pbv > 200 or pbv < -100):
            return True

        # Check for extreme ROE
        roe = row.get("roe_pct")
        if pd.notna(roe) and (roe > 500 or roe < -500):
            return True

        return False

    def validate_dataframe(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate all rows in a DataFrame.

        Returns a comprehensive quality report.
        """
        if df.empty:
            return DataQualityReport(
                total_rows=0,
                valid_rows=0,
                average_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                common_issues={},
                row_results=[],
            )

        row_results: list[ValidationResult] = []
        issue_counts: dict[str, int] = {}

        for _, row in df.iterrows():
            result = self.validate_row(row)
            row_results.append(result)

            # Count issues
            for issue in result.checks_failed:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            for warning in result.warnings:
                issue_counts[warning] = issue_counts.get(warning, 0) + 1

        # Calculate statistics
        scores = [r.confidence_score for r in row_results]
        valid_count = sum(1 for r in row_results if r.is_valid)

        return DataQualityReport(
            total_rows=len(df),
            valid_rows=valid_count,
            average_confidence=sum(scores) / len(scores) if scores else 0.0,
            min_confidence=min(scores) if scores else 0.0,
            max_confidence=max(scores) if scores else 0.0,
            common_issues=issue_counts,
            row_results=row_results,
        )


# ─── Schema Validation ───────────────────────────────────────────────────────

class SchemaValidator:
    """Validate data against expected schema."""

    EXPECTED_COLUMNS = [
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

    COLUMN_TYPES = {
        "code": str,
        "company_name": str,
        "sector": str,
        "closing_price": float,
        "price_change_pct": float,
        "turnover": float,
        "earnings_4qt": float,
        "eps": float,
        "pe": float,
        "navps": float,
        "pbv": float,
        "roe_pct": float,
        "dps": float,
        "dy_pct": float,
    }

    def validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate DataFrame against expected schema.

        Returns (is_valid, list of errors).
        """
        errors: list[str] = []

        # Check for missing columns
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # Check for extra columns
        extra_cols = set(df.columns) - set(self.EXPECTED_COLUMNS)
        if extra_cols:
            # Extra columns are warnings, not errors
            logger.warning(f"Extra columns in data: {extra_cols}")

        # Check column types (flexible - pandas handles type coercion)
        for col in self.EXPECTED_COLUMNS:
            if col in df.columns:
                expected_type = self.COLUMN_TYPES[col]
                if expected_type == float:
                    # Numeric columns should be numeric dtype
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        # Try to coerce
                        try:
                            pd.to_numeric(df[col], errors="raise")
                        except (ValueError, TypeError):
                            errors.append(
                                f"Column '{col}' contains non-numeric values"
                            )

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_json_structure(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate JSON structure from LLM output.

        Returns (is_valid, list of errors).
        """
        errors: list[str] = []

        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return False, errors

        if "stocks" not in data:
            errors.append("Missing 'stocks' key")
            return False, errors

        stocks = data["stocks"]
        if not isinstance(stocks, list):
            errors.append("'stocks' must be a list")
            return False, errors

        if len(stocks) == 0:
            errors.append("'stocks' list is empty")
            return False, errors

        # Validate first few stocks
        for i, stock in enumerate(stocks[:5]):
            if not isinstance(stock, dict):
                errors.append(f"Stock at index {i} is not a dictionary")
                continue

            if "code" not in stock:
                errors.append(f"Stock at index {i} missing 'code'")

            if "company_name" not in stock:
                errors.append(f"Stock at index {i} missing 'company_name'")

        is_valid = len(errors) == 0
        return is_valid, errors


# ─── Convenience Functions ───────────────────────────────────────────────────

def validate_extraction(df: pd.DataFrame) -> DataQualityReport:
    """
    Validate extracted stock data.

    Convenience function using default settings.
    """
    validator = DataValidator()
    return validator.validate_dataframe(df)


def calculate_confidence(df: pd.DataFrame) -> tuple[float, list[float]]:
    """
    Calculate confidence scores for extracted data.

    Returns (aggregate_score, list of row scores).
    """
    scorer = ConfidenceScorer()
    row_scores = scorer.score_dataframe(df)
    aggregate = scorer.aggregate_score(row_scores)
    return aggregate, row_scores
