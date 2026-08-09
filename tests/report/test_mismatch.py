"""Tests for mismatch computation, bucket assignment, and magnitude gating.

Validates the core arithmetic (model minus market), bucket boundary logic,
and the gate decision: a mismatch only counts as a signal if it's bigger
than the model's typical error in that probability range, otherwise it's
likely just noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ufc_edge.report.mismatch import apply_gate, assign_bucket, compute_mismatch
from ufc_edge.report.schemas import GateVerdict

# ── Fixtures ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BucketEntry:
    """Minimal bucket entry matching the shape consumed by the gate logic."""

    bucket_id: str
    lower: float
    upper: float
    n: int
    calibration_error: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class BucketArtifact:
    """Container holding a list of bucket entries, as produced by the model eval."""

    buckets: list[BucketEntry]


@pytest.fixture
def bucket_artifact() -> BucketArtifact:
    """Four-bucket artifact with known calibration errors for deterministic tests."""
    return BucketArtifact(
        buckets=[
            BucketEntry(
                bucket_id="0.1-0.3",
                lower=0.1,
                upper=0.3,
                n=50,
                calibration_error=0.04,
                ci_lower=0.02,
                ci_upper=0.06,
            ),
            BucketEntry(
                bucket_id="0.3-0.5",
                lower=0.3,
                upper=0.5,
                n=80,
                calibration_error=0.03,
                ci_lower=0.01,
                ci_upper=0.05,
            ),
            BucketEntry(
                bucket_id="0.5-0.7",
                lower=0.5,
                upper=0.7,
                n=70,
                calibration_error=0.05,
                ci_lower=0.03,
                ci_upper=0.07,
            ),
            BucketEntry(
                bucket_id="0.7-0.9",
                lower=0.7,
                upper=0.9,
                n=40,
                calibration_error=0.06,
                ci_lower=0.04,
                ci_upper=0.08,
            ),
        ]
    )


@pytest.fixture
def default_buckets(bucket_artifact: BucketArtifact) -> list[BucketEntry]:
    """The raw bucket list from the artifact, for assign_bucket tests."""
    return bucket_artifact.buckets


# ── compute_mismatch ─────────────────────────────────────────────────────────


class TestComputeMismatch:
    """Pure arithmetic: model probability minus market midpoint."""

    def test_positive_mismatch(self) -> None:
        """Model higher than market produces a positive mismatch."""
        result = compute_mismatch(p_model=0.7, p_market_mid=0.5)
        assert result == pytest.approx(0.2)

    def test_negative_mismatch(self) -> None:
        """Model lower than market produces a negative mismatch."""
        result = compute_mismatch(p_model=0.3, p_market_mid=0.6)
        assert result == pytest.approx(-0.3)

    def test_zero_mismatch(self) -> None:
        """Identical probabilities produce zero mismatch."""
        result = compute_mismatch(p_model=0.55, p_market_mid=0.55)
        assert result == pytest.approx(0.0)

    def test_small_difference(self) -> None:
        """Fractional differences are preserved without rounding."""
        result = compute_mismatch(p_model=0.512, p_market_mid=0.499)
        assert result == pytest.approx(0.013)


# ── assign_bucket ─────────────────────────────────────────────────────────────


class TestAssignBucket:
    """Bucket assignment with half-open intervals and boundary handling."""

    def test_interior_first_bucket(self, default_buckets: list[BucketEntry]) -> None:
        """Probability inside the first bucket maps correctly."""
        entry = assign_bucket(0.2, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.1-0.3"

    def test_interior_middle_bucket(self, default_buckets: list[BucketEntry]) -> None:
        """Probability inside a middle bucket maps correctly."""
        entry = assign_bucket(0.45, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.3-0.5"

    def test_interior_last_bucket(self, default_buckets: list[BucketEntry]) -> None:
        """Probability inside the final bucket maps correctly."""
        entry = assign_bucket(0.85, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.7-0.9"

    def test_boundary_assigned_to_lower_bucket(
        self, default_buckets: list[BucketEntry]
    ) -> None:
        """Probability on a boundary between two buckets goes to the lower one."""
        entry = assign_bucket(0.3, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.1-0.3"

    def test_boundary_0_5_to_lower(self, default_buckets: list[BucketEntry]) -> None:
        """Boundary 0.5 is assigned to the 0.3-0.5 bucket."""
        entry = assign_bucket(0.5, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.3-0.5"

    def test_boundary_0_7_to_lower(self, default_buckets: list[BucketEntry]) -> None:
        """Boundary 0.7 is assigned to the 0.5-0.7 bucket."""
        entry = assign_bucket(0.7, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.5-0.7"

    def test_upper_inclusive_last_bucket(
        self, default_buckets: list[BucketEntry]
    ) -> None:
        """The final bucket's upper bound (0.9) is inclusive."""
        entry = assign_bucket(0.9, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.7-0.9"

    def test_below_all_buckets_uses_nearest(
        self, default_buckets: list[BucketEntry]
    ) -> None:
        """Probability below 0.1 falls to the nearest (first) bucket."""
        entry = assign_bucket(0.05, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.1-0.3"

    def test_above_all_buckets_uses_nearest(
        self, default_buckets: list[BucketEntry]
    ) -> None:
        """Probability above 0.9 falls to the nearest (last) bucket."""
        entry = assign_bucket(0.95, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.7-0.9"

    def test_exactly_lower_bound_first_bucket(
        self, default_buckets: list[BucketEntry]
    ) -> None:
        """Probability exactly at 0.1 (lower bound of first bucket) maps to it."""
        entry = assign_bucket(0.1, default_buckets)
        assert entry is not None
        assert entry.bucket_id == "0.1-0.3"

    def test_empty_bucket_list_returns_none(self) -> None:
        """Empty bucket list yields None."""
        entry = assign_bucket(0.5, [])
        assert entry is None


# ── apply_gate ────────────────────────────────────────────────────────────────


class TestApplyGate:
    """Gate logic: flag mismatches larger than k × calibration error for the bucket."""

    def test_flagged_positive_mismatch(
        self, bucket_artifact: BucketArtifact
    ) -> None:
        """Large positive mismatch exceeds threshold and gets flagged."""
        # Bucket 0.3-0.5: error=0.03, k=2.0, threshold=0.06
        # Mismatch 0.10 > 0.06 → FLAGGED
        result = apply_gate(
            mismatch=0.10, p_model=0.4, bucket_artifact=bucket_artifact, k=2.0
        )
        assert result.verdict == GateVerdict.FLAGGED
        assert result.bucket_id == "0.3-0.5"
        assert result.bucket_n == 80
        assert result.bucket_calibration_error == pytest.approx(0.03)

    def test_flagged_negative_mismatch(
        self, bucket_artifact: BucketArtifact
    ) -> None:
        """Large negative mismatch (absolute value) exceeds threshold."""
        # Bucket 0.5-0.7: error=0.05, k=2.0, threshold=0.10
        # |−0.15| = 0.15 > 0.10 → FLAGGED
        result = apply_gate(
            mismatch=-0.15, p_model=0.6, bucket_artifact=bucket_artifact, k=2.0
        )
        assert result.verdict == GateVerdict.FLAGGED
        assert result.bucket_id == "0.5-0.7"

    def test_within_noise(self, bucket_artifact: BucketArtifact) -> None:
        """Mismatch below the threshold is within expected model noise."""
        # Bucket 0.3-0.5: error=0.03, k=2.0, threshold=0.06
        # |0.04| < 0.06 → WITHIN_NOISE
        result = apply_gate(
            mismatch=0.04, p_model=0.4, bucket_artifact=bucket_artifact, k=2.0
        )
        assert result.verdict == GateVerdict.WITHIN_NOISE
        assert result.bucket_id == "0.3-0.5"

    def test_exactly_at_threshold_is_within_noise(
        self, bucket_artifact: BucketArtifact
    ) -> None:
        """Mismatch exactly equal to threshold does not flag (strict inequality)."""
        # Bucket 0.3-0.5: error=0.03, k=2.0, threshold=0.06
        # |0.06| == 0.06, not strictly greater → WITHIN_NOISE
        result = apply_gate(
            mismatch=0.06, p_model=0.4, bucket_artifact=bucket_artifact, k=2.0
        )
        assert result.verdict == GateVerdict.WITHIN_NOISE

    def test_no_bucket_data_empty_list(self) -> None:
        """Empty bucket artifact yields NO_BUCKET_DATA."""
        empty = BucketArtifact(buckets=[])
        result = apply_gate(mismatch=0.10, p_model=0.5, bucket_artifact=empty, k=2.0)
        assert result.verdict == GateVerdict.NO_BUCKET_DATA

    def test_gate_respects_k_parameter(
        self, bucket_artifact: BucketArtifact
    ) -> None:
        """Lowering k makes the same mismatch trip the flag."""
        # Bucket 0.3-0.5: error=0.03, k=1.0, threshold=0.03
        # |0.04| > 0.03 → FLAGGED with k=1.0
        result = apply_gate(
            mismatch=0.04, p_model=0.4, bucket_artifact=bucket_artifact, k=1.0
        )
        assert result.verdict == GateVerdict.FLAGGED

    def test_gate_uses_nearest_bucket_for_extreme_probability(
        self, bucket_artifact: BucketArtifact
    ) -> None:
        """Probability outside [0.1, 0.9] uses the nearest bucket for gating."""
        # p_model=0.05 → nearest bucket is 0.1-0.3: error=0.04, k=2.0, threshold=0.08
        # |0.12| > 0.08 → FLAGGED
        result = apply_gate(
            mismatch=0.12, p_model=0.05, bucket_artifact=bucket_artifact, k=2.0
        )
        assert result.verdict == GateVerdict.FLAGGED
        assert result.bucket_id == "0.1-0.3"

    def test_gate_transparency_columns(
        self, bucket_artifact: BucketArtifact
    ) -> None:
        """GateResult carries full bucket transparency (n, error, CI bounds)."""
        result = apply_gate(
            mismatch=0.10, p_model=0.4, bucket_artifact=bucket_artifact, k=2.0
        )
        assert result.bucket_n == 80
        assert result.bucket_calibration_error == pytest.approx(0.03)
        assert result.ci_lower == pytest.approx(0.01)
        assert result.ci_upper == pytest.approx(0.05)

    def test_boundary_probability_uses_lower_bucket_for_gate(
        self, bucket_artifact: BucketArtifact
    ) -> None:
        """When p_model sits on a boundary, the gate uses the lower bucket."""
        # p_model=0.3, sits on boundary → lower bucket is 0.1-0.3
        # error=0.04, k=2.0, threshold=0.08
        # |0.05| < 0.08 → WITHIN_NOISE
        result = apply_gate(
            mismatch=0.05, p_model=0.3, bucket_artifact=bucket_artifact, k=2.0
        )
        assert result.verdict == GateVerdict.WITHIN_NOISE
        assert result.bucket_id == "0.1-0.3"
