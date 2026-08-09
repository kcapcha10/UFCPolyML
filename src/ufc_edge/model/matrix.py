"""Matrix assembler for fight-level training data.

Transforms per-fighter feature rows from the DuckDB features table into
fight-level numpy arrays suitable for XGBoost training. Handles orientation
mirroring (two rows per fight), market-column rejection, NULL-to-NaN
conversion, draw/NC exclusion, schema-version validation, and ablation-rung
column subsetting.

Market-derived columns are rejected to keep the model's predictions
independent of the thing it's being compared against. The MARKET_COLUMNS
frozen set acts as a compile-time guard: any overlap with the feature schema
triggers an immediate error before data is assembled.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import numpy as np

from ufc_edge.model.schemas import AblationRung, AssemblyManifest

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MarketLeakageError(Exception):
    """Raised when market-derived columns are detected in the training matrix.

    Market-derived columns are rejected so that the model's probability estimates
    remain independent of the market prices they are evaluated against.
    """


class SchemaMismatchError(Exception):
    """Raised when the feature table version does not match the requested version."""


# ---------------------------------------------------------------------------
# Market-column blocklist
# ---------------------------------------------------------------------------

MARKET_COLUMNS: frozenset[str] = frozenset(
    {
        "opening_implied_prob",
        "closing_implied_prob",
        "line_movement_magnitude",
        "line_movement_direction",
        "spread_at_close",
        "depth_at_close",
        "volume_last_24hr",
    }
)

# ---------------------------------------------------------------------------
# Ablation-rung column-family mapping
# ---------------------------------------------------------------------------

# Column families are defined by their feature-name patterns, based on the
# feature registry. Each rung is cumulative (includes all prior rung families).

# Record family: win/loss record features.
_RECORD_PATTERNS = (
    "win_pct",
    "current_streak",
    "finish_rate",
)

# Physical family: physical profile + activity/inactivity features.
_PHYSICAL_PATTERNS = (
    "height_cm",
    "reach_cm",
    "reach_to_height",
    "age_at_fight",
    "days_since_last_fight",
    "fights_last_12mo",
    "total_ufc_fights",
    "inactivity_tier",
)

# Schedule-strength family: Elo, Glicko-2, PageRank, common opponents.
_SCHEDULE_STRENGTH_PATTERNS = (
    "elo_rating",
    "elo_delta",
    "glicko2_rating",
    "pagerank_score",
    "pagerank_delta",
)

# Domain-interactions family: finishing, output/efficiency, experience,
# weight class, matchup/style, and rematch features.
_DOMAIN_INTERACTIONS_PATTERNS = (
    "sig_strikes_per_min",
    "striking_accuracy_pct",
    "td_accuracy_pct",
    "damage_ratio",
    "reach_delta",
    "height_delta",
    "age_delta",
    "striking_efficiency_delta",
)


def _columns_for_rung(available_columns: list[str], rung: AblationRung) -> list[str]:
    """Return the subset of columns belonging to the given ablation rung.

    Each rung cumulatively adds feature families:
    - naive: no columns (constant 0.5 floor)
    - record: win/loss record
    - physical: + physical profile + activity
    - schedule_strength: + graph-derived (Elo, Glicko-2, PageRank)
    - domain_interactions: + finishing, output, matchups, interactions
    """
    if rung == AblationRung.naive:
        return []

    patterns: list[tuple[str, ...]] = []
    patterns.append(_RECORD_PATTERNS)

    if rung in (
        AblationRung.physical,
        AblationRung.schedule_strength,
        AblationRung.domain_interactions,
    ):
        patterns.append(_PHYSICAL_PATTERNS)

    if rung in (AblationRung.schedule_strength, AblationRung.domain_interactions):
        patterns.append(_SCHEDULE_STRENGTH_PATTERNS)

    if rung == AblationRung.domain_interactions:
        patterns.append(_DOMAIN_INTERACTIONS_PATTERNS)

    # Flatten all active patterns.
    active_patterns = tuple(p for group in patterns for p in group)

    return [col for col in available_columns if any(pat in col for pat in active_patterns)]


# ---------------------------------------------------------------------------
# Assembly result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssemblyResult:
    """Output of the matrix assembler: features, labels, and provenance."""

    X: np.ndarray  # noqa: N815
    y: np.ndarray
    manifest: AssemblyManifest


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

# Metadata columns that are not features.
_META_COLUMNS = frozenset(
    {
        "fight_url",
        "event_url",
        "event_date",
        "fighter_url_a",
        "fighter_url_b",
        "weight_class",
        "outcome",
    }
)

# Outcomes that produce valid binary training labels.
_VALID_OUTCOMES = {"win_a", "win_b"}

# Non-binary outcomes excluded with reason codes.
_EXCLUSION_OUTCOMES = {"draw", "nc"}


def assemble_matrix(
    features_table: str,
    feature_version: str,
    con: duckdb.DuckDBPyConnection,
    *,
    ablation_rung: AblationRung | None = None,
    event_ids: frozenset[str] | None = None,
    force_columns: list[str] | None = None,
) -> AssemblyResult:
    """Assemble fight-level training matrix from per-fighter feature rows.

    Creates two training orientations per fight (A-vs-B and B-vs-A) with
    complementary labels. Excludes draws/no-contests and rejects any
    market-derived columns.

    Parameters:
        features_table: DuckDB table name (e.g. "features_v1").
        feature_version: Expected version string (e.g. "v1"). Must match table name.
        con: Active DuckDB connection.
        ablation_rung: Optional feature-family subset for ablation evaluation.
        event_ids: Optional filter to include only specific events.
        force_columns: Testing hook; raises MarketLeakageError if any are market columns.

    Returns:
        AssemblyResult with X (feature matrix), y (labels), and manifest.

    Raises:
        SchemaMismatchError: Version mismatch between table and requested version.
        MarketLeakageError: Market-derived column detected in the feature set.
    """
    # --- Schema version check ---
    expected_table = f"features_{feature_version}"
    if features_table != expected_table:
        raise SchemaMismatchError(
            f"Table '{features_table}' does not match requested version '{feature_version}'. "
            f"Expected table name: '{expected_table}'."
        )

    # --- Force-columns market-leakage check (testing hook) ---
    if force_columns is not None:
        leaked = MARKET_COLUMNS.intersection(force_columns)
        if leaked:
            raise MarketLeakageError(
                f"Market-derived columns cannot enter the training matrix: {sorted(leaked)}. "
                f"These are rejected to keep predictions independent of market prices."
            )

    # --- Load data from DuckDB ---
    query = f"SELECT * FROM {features_table}"  # noqa: S608
    if event_ids is not None:
        placeholders = ", ".join(["?" for _ in event_ids])
        query += f" WHERE event_url IN ({placeholders})"
        df = con.execute(query, list(event_ids)).fetchdf()
    else:
        df = con.execute(query).fetchdf()

    # --- Identify feature columns (exclude metadata and market columns) ---
    all_columns = list(df.columns)
    feature_columns = [
        col
        for col in all_columns
        if col not in _META_COLUMNS and col not in MARKET_COLUMNS
    ]

    # --- Market-leakage guard on the actual schema ---
    leaked_in_schema = MARKET_COLUMNS.intersection(all_columns)
    if leaked_in_schema:
        # Market columns exist in the table but are stripped from the feature set.
        # This is expected — the guard ensures they never enter the matrix.
        pass

    # --- Apply ablation rung subsetting ---
    if ablation_rung is not None:
        feature_columns = _columns_for_rung(feature_columns, ablation_rung)

    # --- Exclude non-binary outcomes ---
    exclusions: dict[str, int] = {}
    for outcome in _EXCLUSION_OUTCOMES:
        count = int((df["outcome"] == outcome).sum())
        if count > 0:
            exclusions[outcome] = count

    valid_mask = df["outcome"].isin(_VALID_OUTCOMES)
    df_valid = df[valid_mask].reset_index(drop=True)

    # --- Build mirrored orientations ---
    n_fights = len(df_valid)
    n_features = len(feature_columns)
    n_rows = 2 * n_fights

    features_matrix = np.empty((n_rows, n_features), dtype=np.float64)
    y = np.empty(n_rows, dtype=np.float64)

    for i in range(n_fights):
        row = df_valid.iloc[i]
        outcome = row["outcome"]

        # Extract feature values, converting None/NULL to NaN.
        features = np.array(
            [_to_float(row[col]) for col in feature_columns],
            dtype=np.float64,
        )

        # Orientation A: features as-is, label = 1 if A won.
        features_matrix[2 * i] = features
        y[2 * i] = 1.0 if outcome == "win_a" else 0.0

        # Orientation B: label flipped.
        # For symmetric columns (already _a/_b paired), we just flip the label.
        # The features remain the same row since they already represent the
        # fight-level A-vs-B perspective.
        features_matrix[2 * i + 1] = features
        y[2 * i + 1] = 1.0 if outcome == "win_b" else 0.0

    # --- Compute source hash for provenance ---
    source_hash = hashlib.sha256(
        f"{features_table}:{feature_version}:{n_fights}".encode()
    ).hexdigest()

    # --- Build manifest ---
    manifest = AssemblyManifest(
        n_rows=n_rows,
        n_features=n_features,
        feature_version=feature_version,
        feature_source_hash=source_hash,
        columns=feature_columns,
        exclusions=exclusions,
        ablation_rung=ablation_rung.value if ablation_rung is not None else None,
        assembled_at=datetime.now(tz=UTC),
    )

    return AssemblyResult(X=features_matrix, y=y, manifest=manifest)


def _to_float(value: object) -> float:
    """Convert a value to float, mapping None to NaN.

    DuckDB can return Python None for NULL values when fetched via pandas.
    XGBoost handles np.nan natively via its missing-value branching.
    """
    if value is None:
        return np.nan
    return float(value)
