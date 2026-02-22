"""Unit tests for Data Validator module (Week 2).

Tests cover:
- Confidence scoring with metamorphic checks
- PE × EPS ≈ Price validation
- PBV × NAVPS ≈ Price validation
- DY = DPS / Price × 100 validation
- Negative EPS handling
- Missing value handling
- Schema validation
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_validator import (
    ConfidenceScorer,
    DataQualityReport,
    DataValidator,
    SchemaValidator,
    ValidationResult,
    calculate_confidence,
    validate_extraction,
)


# ─── ConfidenceScorer Tests ──────────────────────────────────────────────────


class TestConfidenceScorer:
    """Tests for confidence scoring with metamorphic checks."""

    @pytest.fixture
    def scorer(self) -> ConfidenceScorer:
        return ConfidenceScorer(tolerance=0.05)

    def test_score_complete_valid_row(self, scorer: ConfidenceScorer) -> None:
        """Test scoring a complete, valid stock row."""
        # PE × EPS = 5.29 × 43.59 = 230.59 ≈ 230.50 (valid)
        # PBV × NAVPS = 1.19 × 193.80 = 230.62 ≈ 230.50 (valid)
        # DY = 9.20 / 230.50 × 100 = 3.99% (valid)
        row = pd.Series({
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
        })

        score = scorer.score_row(row)

        # Should score very high (>0.9) for well-formed data
        assert score >= 0.90

    def test_score_row_with_negative_eps(self, scorer: ConfidenceScorer) -> None:
        """Test scoring a row with negative EPS (loss-making company)."""
        row = pd.Series({
            "code": "HDFC.N0000",
            "company_name": "HDFC BANK",
            "sector": "Banks",
            "closing_price": 55.00,
            "price_change_pct": -0.72,
            "turnover": 3262565,
            "earnings_4qt": -247.00,
            "eps": -3.82,
            "pe": -14.41,  # Negative PE for negative earnings
            "navps": 120.72,
            "pbv": 0.46,
            "roe_pct": -3.16,
            "dps": None,
            "dy_pct": None,
        })

        score = scorer.score_row(row)

        # Should still score reasonably for consistent negative data
        assert score >= 0.70

    def test_score_row_with_missing_values(self, scorer: ConfidenceScorer) -> None:
        """Test scoring a row with missing dividend data."""
        row = pd.Series({
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
            "dps": None,
            "dy_pct": None,
        })

        score = scorer.score_row(row)

        # Should still score well with consistent missing dividend data
        assert score >= 0.70

    def test_check_pe_eps_price_valid(self, scorer: ConfidenceScorer) -> None:
        """Test PE × EPS ≈ Price check with valid data."""
        # PE × EPS = 7.24 × 4.10 = 29.68 ≈ 29.70
        row = pd.Series({
            "closing_price": 29.70,
            "eps": 4.10,
            "pe": 7.24,
        })

        score = scorer._check_pe_eps_price(row)
        assert score >= 0.9

    def test_check_pe_eps_price_invalid(self, scorer: ConfidenceScorer) -> None:
        """Test PE × EPS ≈ Price check with inconsistent data."""
        # PE × EPS = 10 × 10 = 100 ≠ 50 (50% off)
        row = pd.Series({
            "closing_price": 50.0,
            "eps": 10.0,
            "pe": 10.0,
        })

        score = scorer._check_pe_eps_price(row)
        assert score < 0.5

    def test_check_pe_eps_price_missing(self, scorer: ConfidenceScorer) -> None:
        """Test PE × EPS check with missing values."""
        row = pd.Series({
            "closing_price": 100.0,
            "eps": None,
            "pe": None,
        })

        score = scorer._check_pe_eps_price(row)
        assert score == 0.5  # Neutral score for missing data

    def test_check_pbv_navps_price_valid(self, scorer: ConfidenceScorer) -> None:
        """Test PBV × NAVPS ≈ Price check with valid data."""
        # PBV × NAVPS = 0.66 × 44.83 = 29.59 ≈ 29.70
        row = pd.Series({
            "closing_price": 29.70,
            "navps": 44.83,
            "pbv": 0.66,
        })

        score = scorer._check_pbv_navps_price(row)
        assert score >= 0.7

    def test_check_pbv_navps_price_negative_navps(
        self, scorer: ConfidenceScorer
    ) -> None:
        """Test PBV × NAVPS check with negative NAVPS (negative equity)."""
        row = pd.Series({
            "closing_price": 7.30,
            "navps": -47.70,
            "pbv": -0.15,  # Negative PBV for negative NAVPS
        })

        score = scorer._check_pbv_navps_price(row)
        assert score == 1.0  # Should pass for consistent negative equity

    def test_check_dy_calculation_valid(self, scorer: ConfidenceScorer) -> None:
        """Test DY = DPS / Price × 100 check with valid data."""
        # DY = 1.30 / 29.70 × 100 = 4.38%
        row = pd.Series({
            "closing_price": 29.70,
            "dps": 1.30,
            "dy_pct": 4.38,
        })

        score = scorer._check_dy_calculation(row)
        assert score >= 0.9

    def test_check_dy_calculation_no_dividend(
        self, scorer: ConfidenceScorer
    ) -> None:
        """Test DY check when no dividend is paid."""
        row = pd.Series({
            "closing_price": 55.00,
            "dps": None,
            "dy_pct": None,
        })

        score = scorer._check_dy_calculation(row)
        assert score == 1.0  # Both null is consistent

    def test_check_dy_calculation_zero_dividend(
        self, scorer: ConfidenceScorer
    ) -> None:
        """Test DY check when dividend is zero."""
        row = pd.Series({
            "closing_price": 55.00,
            "dps": 0,
            "dy_pct": 0,
        })

        score = scorer._check_dy_calculation(row)
        assert score == 1.0  # Zero dividend with zero DY is consistent

    def test_reasonable_pe_bounds(self, scorer: ConfidenceScorer) -> None:
        """Test PE reasonableness check."""
        assert scorer._is_reasonable_pe(pd.Series({"pe": 10})) == True
        assert scorer._is_reasonable_pe(pd.Series({"pe": -50})) == True
        assert scorer._is_reasonable_pe(pd.Series({"pe": 1000})) == False
        # None values return True (missing is acceptable)
        row_with_none = pd.Series({"pe": float("nan")})
        assert scorer._is_reasonable_pe(row_with_none) == True

    def test_reasonable_pbv_bounds(self, scorer: ConfidenceScorer) -> None:
        """Test PBV reasonableness check."""
        assert scorer._is_reasonable_pbv(pd.Series({"pbv": 2.5})) == True
        assert scorer._is_reasonable_pbv(pd.Series({"pbv": -10})) == True
        assert scorer._is_reasonable_pbv(pd.Series({"pbv": 200})) == False
        # None values return True (missing is acceptable)
        row_with_none = pd.Series({"pbv": float("nan")})
        assert scorer._is_reasonable_pbv(row_with_none) == True

    def test_score_dataframe(self, scorer: ConfidenceScorer) -> None:
        """Test scoring a full DataFrame."""
        df = pd.DataFrame([
            {
                "code": "TEST1.N0000",
                "company_name": "TEST 1",
                "closing_price": 100.0,
                "eps": 10.0,
                "pe": 10.0,
                "navps": 50.0,
                "pbv": 2.0,
                "dps": 5.0,
                "dy_pct": 5.0,
            },
            {
                "code": "TEST2.N0000",
                "company_name": "TEST 2",
                "closing_price": 50.0,
                "eps": 5.0,
                "pe": 10.0,
                "navps": 25.0,
                "pbv": 2.0,
                "dps": None,
                "dy_pct": None,
            },
        ])

        scores = scorer.score_dataframe(df)

        assert len(scores) == 2
        assert all(0 <= s <= 1 for s in scores)

    def test_aggregate_score(self, scorer: ConfidenceScorer) -> None:
        """Test aggregating row scores."""
        scores = [0.9, 0.85, 0.8, 0.75]
        aggregate = scorer.aggregate_score(scores)

        assert 0.7 <= aggregate <= 0.9

    def test_aggregate_score_with_low_scores(
        self, scorer: ConfidenceScorer
    ) -> None:
        """Test aggregation penalizes many low scores."""
        high_scores = [0.9, 0.9, 0.9]
        mixed_scores = [0.9, 0.3, 0.3]

        high_agg = scorer.aggregate_score(high_scores)
        mixed_agg = scorer.aggregate_score(mixed_scores)

        assert high_agg > mixed_agg

    def test_aggregate_score_empty(self, scorer: ConfidenceScorer) -> None:
        """Test aggregation with empty list."""
        assert scorer.aggregate_score([]) == 0.0


# ─── DataValidator Tests ─────────────────────────────────────────────────────


class TestDataValidator:
    """Tests for data validation."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        return DataValidator()

    def test_validate_valid_row(self, validator: DataValidator) -> None:
        """Test validation of a valid row."""
        row = pd.Series({
            "code": "COMB.N0000",
            "company_name": "COMMERCIAL BANK OF CEYLON PLC",
            "sector": "Banks",
            "closing_price": 230.50,
            "eps": 43.59,
            "pe": 5.29,
            "navps": 193.80,
            "pbv": 1.19,
        })

        result = validator.validate_row(row)

        assert result.is_valid is True
        assert "has_code" in result.checks_passed
        assert "has_company_name" in result.checks_passed
        assert "positive_price" in result.checks_passed

    def test_validate_row_missing_code(self, validator: DataValidator) -> None:
        """Test validation fails for missing code."""
        row = pd.Series({
            "code": None,
            "company_name": "TEST COMPANY",
            "sector": "Banks",
            "closing_price": 100.0,
        })

        result = validator.validate_row(row)

        assert "missing_code" in result.checks_failed

    def test_validate_row_negative_price(self, validator: DataValidator) -> None:
        """Test validation flags negative price."""
        row = pd.Series({
            "code": "TEST.N0000",
            "company_name": "TEST COMPANY",
            "sector": "Banks",
            "closing_price": -10.0,
        })

        result = validator.validate_row(row)

        assert "non_positive_price" in result.checks_failed

    def test_is_valid_stock_code(self, validator: DataValidator) -> None:
        """Test stock code format validation."""
        assert validator._is_valid_stock_code("COMB.N0000") is True
        assert validator._is_valid_stock_code("COMB.X0000") is True
        assert validator._is_valid_stock_code("AAF.P0000") is True
        assert validator._is_valid_stock_code("INVALID") is False
        assert validator._is_valid_stock_code("TOO_LONG_CODE.N0000") is False
        assert validator._is_valid_stock_code("") is False

    def test_has_unreasonable_values(self, validator: DataValidator) -> None:
        """Test detection of unreasonable values."""
        normal_row = pd.Series({"pe": 10, "pbv": 2, "roe_pct": 15})
        assert validator._has_unreasonable_values(normal_row) is False

        extreme_pe = pd.Series({"pe": 5000, "pbv": 2, "roe_pct": 15})
        assert validator._has_unreasonable_values(extreme_pe) is True

        extreme_pbv = pd.Series({"pe": 10, "pbv": 500, "roe_pct": 15})
        assert validator._has_unreasonable_values(extreme_pbv) is True

    def test_validate_dataframe(self, validator: DataValidator) -> None:
        """Test validation of full DataFrame."""
        df = pd.DataFrame([
            {
                "code": "GOOD.N0000",
                "company_name": "GOOD COMPANY",
                "sector": "Banks",
                "closing_price": 100.0,
                "eps": 10.0,
                "pe": 10.0,
            },
            {
                "code": None,  # Invalid
                "company_name": "BAD COMPANY",
                "sector": "Banks",
                "closing_price": 50.0,
            },
        ])

        report = validator.validate_dataframe(df)

        assert report.total_rows == 2
        assert report.valid_rows == 1  # Only one is valid
        assert "missing_code" in report.common_issues

    def test_validate_empty_dataframe(self, validator: DataValidator) -> None:
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        report = validator.validate_dataframe(df)

        assert report.total_rows == 0
        assert report.valid_rows == 0
        assert report.average_confidence == 0.0


# ─── SchemaValidator Tests ───────────────────────────────────────────────────


class TestSchemaValidator:
    """Tests for schema validation."""

    @pytest.fixture
    def validator(self) -> SchemaValidator:
        return SchemaValidator()

    def test_validate_schema_valid(self, validator: SchemaValidator) -> None:
        """Test schema validation with valid DataFrame."""
        df = pd.DataFrame([{
            "code": "TEST.N0000",
            "company_name": "TEST",
            "sector": "Banks",
            "closing_price": 100.0,
            "price_change_pct": -1.0,
            "turnover": 1000000,
            "earnings_4qt": 500.0,
            "eps": 5.0,
            "pe": 20.0,
            "navps": 50.0,
            "pbv": 2.0,
            "roe_pct": 10.0,
            "dps": 2.5,
            "dy_pct": 2.5,
        }])

        is_valid, errors = validator.validate_schema(df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_schema_missing_columns(
        self, validator: SchemaValidator
    ) -> None:
        """Test schema validation with missing columns."""
        df = pd.DataFrame([{
            "code": "TEST.N0000",
            "company_name": "TEST",
            # Missing many columns
        }])

        is_valid, errors = validator.validate_schema(df)

        assert is_valid is False
        assert any("Missing columns" in e for e in errors)

    def test_validate_json_structure_valid(
        self, validator: SchemaValidator
    ) -> None:
        """Test JSON structure validation with valid data."""
        data = {
            "stocks": [
                {"code": "TEST.N0000", "company_name": "TEST"}
            ]
        }

        is_valid, errors = validator.validate_json_structure(data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_json_structure_missing_stocks(
        self, validator: SchemaValidator
    ) -> None:
        """Test JSON validation without stocks key."""
        data = {"other": []}

        is_valid, errors = validator.validate_json_structure(data)

        assert is_valid is False
        assert any("Missing 'stocks'" in e for e in errors)

    def test_validate_json_structure_empty_stocks(
        self, validator: SchemaValidator
    ) -> None:
        """Test JSON validation with empty stocks list."""
        data = {"stocks": []}

        is_valid, errors = validator.validate_json_structure(data)

        assert is_valid is False
        assert any("empty" in e.lower() for e in errors)


# ─── DataQualityReport Tests ─────────────────────────────────────────────────


class TestDataQualityReport:
    """Tests for DataQualityReport."""

    def test_is_acceptable_high_quality(self) -> None:
        """Test is_acceptable for high quality data."""
        report = DataQualityReport(
            total_rows=10,
            valid_rows=9,
            average_confidence=0.85,
            min_confidence=0.7,
            max_confidence=0.95,
            common_issues={},
            row_results=[],
        )

        assert report.is_acceptable is True

    def test_is_acceptable_low_confidence(self) -> None:
        """Test is_acceptable fails for low confidence."""
        report = DataQualityReport(
            total_rows=10,
            valid_rows=9,
            average_confidence=0.5,  # Too low
            min_confidence=0.3,
            max_confidence=0.7,
            common_issues={},
            row_results=[],
        )

        assert report.is_acceptable is False

    def test_is_acceptable_few_valid_rows(self) -> None:
        """Test is_acceptable fails with few valid rows."""
        report = DataQualityReport(
            total_rows=10,
            valid_rows=5,  # Only 50% valid
            average_confidence=0.85,
            min_confidence=0.7,
            max_confidence=0.95,
            common_issues={},
            row_results=[],
        )

        assert report.is_acceptable is False

    def test_is_acceptable_empty_data(self) -> None:
        """Test is_acceptable for empty data."""
        report = DataQualityReport(
            total_rows=0,
            valid_rows=0,
            average_confidence=0.0,
            min_confidence=0.0,
            max_confidence=0.0,
            common_issues={},
            row_results=[],
        )

        assert report.is_acceptable is False


# ─── Convenience Functions Tests ─────────────────────────────────────────────


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_validate_extraction(self) -> None:
        """Test validate_extraction convenience function."""
        df = pd.DataFrame([{
            "code": "TEST.N0000",
            "company_name": "TEST",
            "sector": "Banks",
            "closing_price": 100.0,
            "eps": 10.0,
            "pe": 10.0,
            "navps": 50.0,
            "pbv": 2.0,
        }])

        report = validate_extraction(df)

        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 1

    def test_calculate_confidence(self) -> None:
        """Test calculate_confidence convenience function."""
        df = pd.DataFrame([{
            "code": "TEST.N0000",
            "company_name": "TEST",
            "closing_price": 100.0,
            "eps": 10.0,
            "pe": 10.0,
            "navps": 50.0,
            "pbv": 2.0,
            "dps": 5.0,
            "dy_pct": 5.0,
        }])

        aggregate, row_scores = calculate_confidence(df)

        assert 0 <= aggregate <= 1
        assert len(row_scores) == 1
        assert 0 <= row_scores[0] <= 1
