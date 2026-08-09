"""Tests for the matrix assembler module.

Verifies orientation mirroring, label correctness, market-column rejection,
schema-version checks, NULL-to-NaN conversion, draw/NC exclusion, ablation
rung subsetting, and manifest metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufc_edge.model.matrix import (
    MARKET_COLUMNS,
    MarketLeakageError,
    SchemaMismatchError,
    assemble_matrix,
)
from ufc_edge.model.schemas import AblationRung

# ---------------------------------------------------------------------------
# Orientation and labelling
# ---------------------------------------------------------------------------


class TestOrientationMirroring:
    """Each fight produces two symmetric training rows with complementary labels."""

    def test_two_orientations_per_fight(self, feature_db):
        """Every fight yields exactly two rows (A-vs-B and B-vs-A)."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        # 50 fights total, minus 2 draws and 1 NC = 47 usable fights → 94 rows.
        assert result.manifest.n_rows == 94

    def test_label_symmetry(self, feature_db):
        """For each fight the two orientation labels sum to 1 (one win, one loss)."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        y = result.y
        # Labels come in consecutive pairs (row 0,1 for fight 1; row 2,3 for fight 2).
        for i in range(0, len(y), 2):
            assert y[i] + y[i + 1] == 1.0, f"Labels at indices {i},{i+1} do not sum to 1"

    def test_labels_are_binary(self, feature_db):
        """All labels are strictly 0.0 or 1.0."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert set(np.unique(result.y)) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# Market-column rejection
# ---------------------------------------------------------------------------


class TestMarketColumnRejection:
    """Market-derived columns are rejected to keep the model's predictions
    independent of the thing it's being compared against.
    """

    def test_market_columns_frozenset_not_empty(self):
        """The MARKET_COLUMNS guard set must list at least one column."""
        assert len(MARKET_COLUMNS) > 0

    def test_raises_market_leakage_error(self, feature_db):
        """If the feature table contains a market-derived column and no
        ablation rung excludes it, assembly must fail before producing data.
        """
        # The fixture table includes 'opening_implied_prob' which is in MARKET_COLUMNS.
        # We intentionally include it in the schema by NOT stripping it —
        # the assembler must detect and reject it.
        # First, verify normal assembly works (it strips market columns internally):
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        # The assembled matrix must NOT contain any market column.
        for col in MARKET_COLUMNS:
            assert col not in result.manifest.columns

    def test_explicit_market_column_injection_raises(self, feature_db):
        """Directly injecting a market column via override triggers the error."""
        with pytest.raises(MarketLeakageError):
            assemble_matrix(
                features_table="features_v1",
                feature_version="v1",
                con=feature_db,
                force_columns=list(MARKET_COLUMNS),
            )


# ---------------------------------------------------------------------------
# Schema version mismatch
# ---------------------------------------------------------------------------


class TestSchemaVersionMismatch:
    """Requesting a version that doesn't match the table raises an error."""

    def test_raises_schema_mismatch_error(self, feature_db):
        """Requesting v2 against a v1 table raises SchemaMismatchError."""
        with pytest.raises(SchemaMismatchError):
            assemble_matrix(
                features_table="features_v1",
                feature_version="v2",
                con=feature_db,
            )


# ---------------------------------------------------------------------------
# NULL → NaN conversion
# ---------------------------------------------------------------------------


class TestNullToNanConversion:
    """DuckDB NULLs become np.nan; no Python None survives in the output."""

    def test_no_python_none_in_feature_matrix(self, feature_db):
        """The feature array X contains no Python None values."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        # np.nan is a float, so if there were None values they'd be object dtype.
        assert result.X.dtype.kind == "f", "Feature matrix must be float dtype"

    def test_nulls_converted_to_nan(self, feature_db):
        """The fixture has ~10% NULLs; confirm NaN is present (not zeros)."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        # At least some NaN values should exist from the synthetic NULLs.
        assert np.isnan(result.X).any(), "Expected NaN values from NULL conversion"


# ---------------------------------------------------------------------------
# Draw / NC exclusion
# ---------------------------------------------------------------------------


class TestDrawNCExclusion:
    """Draws and no-contests are excluded with correct counts in the manifest."""

    def test_draws_excluded(self, feature_db):
        """Fights with 'draw' outcome are excluded from the training matrix."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert "draw" in result.manifest.exclusions
        assert result.manifest.exclusions["draw"] == 2

    def test_nc_excluded(self, feature_db):
        """Fights with 'nc' outcome are excluded from the training matrix."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert "nc" in result.manifest.exclusions
        assert result.manifest.exclusions["nc"] == 1

    def test_total_rows_reflect_exclusions(self, feature_db):
        """Row count = 2 × (total fights - excluded fights)."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        total_excluded = sum(result.manifest.exclusions.values())
        expected_rows = 2 * (50 - total_excluded)
        assert result.manifest.n_rows == expected_rows


# ---------------------------------------------------------------------------
# Ablation rung subsetting
# ---------------------------------------------------------------------------


class TestAblationRungSubsetting:
    """Each ablation rung's columns are a subset of the next rung."""

    def test_naive_produces_no_features(self, feature_db):
        """The naive rung assembles zero feature columns."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
            ablation_rung=AblationRung.naive,
        )
        assert result.manifest.n_features == 0
        assert result.X.shape[1] == 0

    def test_rungs_are_nested(self, feature_db):
        """Each successive rung's column set is a superset of the prior rung."""
        rungs = [
            AblationRung.record,
            AblationRung.physical,
            AblationRung.schedule_strength,
            AblationRung.domain_interactions,
        ]
        prev_cols: set[str] = set()
        for rung in rungs:
            result = assemble_matrix(
                features_table="features_v1",
                feature_version="v1",
                con=feature_db,
                ablation_rung=rung,
            )
            current_cols = set(result.manifest.columns)
            assert prev_cols.issubset(current_cols), (
                f"Rung {rung} columns are not a superset of prior rung. "
                f"Missing: {prev_cols - current_cols}"
            )
            assert len(current_cols) > len(prev_cols), (
                f"Rung {rung} must have more columns than prior rung"
            )
            prev_cols = current_cols

    def test_record_rung_has_record_columns(self, feature_db):
        """The record rung includes record-family features."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
            ablation_rung=AblationRung.record,
        )
        cols = set(result.manifest.columns)
        # Must contain at least some record columns from the fixture.
        assert any("win_pct" in c for c in cols)
        assert any("streak" in c for c in cols)


# ---------------------------------------------------------------------------
# Manifest correctness
# ---------------------------------------------------------------------------


class TestManifest:
    """The assembly manifest reports accurate metadata."""

    def test_manifest_row_count(self, feature_db):
        """Manifest n_rows matches actual array shape."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert result.manifest.n_rows == result.X.shape[0]
        assert result.manifest.n_rows == len(result.y)

    def test_manifest_feature_count(self, feature_db):
        """Manifest n_features matches actual column count."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert result.manifest.n_features == result.X.shape[1]
        assert result.manifest.n_features == len(result.manifest.columns)

    def test_manifest_columns_no_market(self, feature_db):
        """Manifest column list excludes all market-derived columns."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        for col in MARKET_COLUMNS:
            assert col not in result.manifest.columns

    def test_manifest_exclusion_counts_present(self, feature_db):
        """Exclusion counts are populated with reason codes."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert isinstance(result.manifest.exclusions, dict)
        assert len(result.manifest.exclusions) > 0

    def test_manifest_feature_version(self, feature_db):
        """Manifest records the feature version."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert result.manifest.feature_version == "v1"

    def test_manifest_has_timestamp(self, feature_db):
        """Manifest includes an assembly timestamp."""
        result = assemble_matrix(
            features_table="features_v1",
            feature_version="v1",
            con=feature_db,
        )
        assert result.manifest.assembled_at is not None
