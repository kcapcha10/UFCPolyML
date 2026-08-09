"""Symmetric inference for fight-level win-probability predictions.

Guarantees that a matchup prediction does not depend on which fighter is listed
first. Given fighters A and B, the predicted probability for A is exactly
1 minus the predicted probability for B, regardless of argument order.

The protocol:
  1. Establish canonical order (lexicographically smaller fighter URL first).
  2. Compute raw model predictions for both orientations.
  3. Average the two orientations: p_raw = 0.5 * (p_ab + (1 - p_ba)).
  4. Calibrate once on the canonical raw probability.
  5. For the reversed query, return 1 - p_calibrated.

Both the raw-score predictor and calibrator are injected as callables so this
module has no hard dependency on specific training or calibration implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingDataError(Exception):
    """Raised when a fighter's feature vector is unavailable for prediction.

    Symmetric inference requires feature data for both fighters in the matchup.
    If either side's features are None or otherwise missing, prediction cannot
    proceed.
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmetricPrediction:
    """Calibrated win probability with exact symmetry guarantee.

    A matchup prediction must not depend on which fighter is listed first.
    The canonical fighter is the one whose URL sorts lexicographically smaller.
    p_calibrated gives the probability that the canonical fighter wins.
    """

    fight_url: str
    canonical_fighter_url: str
    canonical_opponent_url: str
    p_calibrated: float
    raw_p_ab: float
    raw_p_ba: float
    calibrator_method: str


# ---------------------------------------------------------------------------
# Type aliases for injected dependencies
# ---------------------------------------------------------------------------

# A raw-score predictor: takes a feature vector and returns P(row-A wins).
RawPredictor = Callable[[np.ndarray], float]

# A calibrator: takes a raw probability and returns a calibrated probability.
# Also exposes a `method` attribute naming the calibration method used.
CalibratorCallable = Callable[[float], float]


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------


def predict_symmetric(
    raw_predict: RawPredictor,
    calibrate: CalibratorCallable,
    calibrator_method: str,
    fighter_a_features: np.ndarray | None,
    fighter_b_features: np.ndarray | None,
    fighter_a_url: str,
    fighter_b_url: str,
    fight_url: str,
) -> SymmetricPrediction:
    """Produce a symmetric calibrated prediction for a two-fighter matchup.

    Averages raw predictions from both orientations and calibrates once on
    the canonical ordering. The result is invariant to argument order: calling
    with (A, B) yields exactly 1 - p compared to calling with (B, A).

    Parameters:
        raw_predict: Callable that takes a 1-D feature array and returns a raw
            probability (e.g. XGBoost's output for binary:logistic).
        calibrate: Callable that maps a raw probability to a calibrated one.
        calibrator_method: Name of the calibration method (e.g. "platt", "isotonic").
        fighter_a_features: Feature vector for fighter A, or None if unavailable.
        fighter_b_features: Feature vector for fighter B, or None if unavailable.
        fighter_a_url: URL identifier for fighter A.
        fighter_b_url: URL identifier for fighter B.
        fight_url: URL identifier for this fight.

    Returns:
        SymmetricPrediction with calibrated probability for the canonical fighter.

    Raises:
        MissingDataError: If either fighter's features are None.
    """
    # --- Validate that both fighters have feature data ---
    if fighter_a_features is None:
        raise MissingDataError(
            f"Missing feature data for fighter: {fighter_a_url}"
        )
    if fighter_b_features is None:
        raise MissingDataError(
            f"Missing feature data for fighter: {fighter_b_url}"
        )

    # --- Determine canonical ordering (lexicographically smaller URL first) ---
    if fighter_a_url <= fighter_b_url:
        canonical_url = fighter_a_url
        opponent_url = fighter_b_url
        canonical_features = fighter_a_features
        opponent_features = fighter_b_features
    else:
        canonical_url = fighter_b_url
        opponent_url = fighter_a_url
        canonical_features = fighter_b_features
        opponent_features = fighter_a_features

    # --- Raw predictions for both orientations ---
    # p_ab: P(canonical wins) when canonical is in the "A" slot.
    # p_ba: P(opponent wins) when opponent is in the "A" slot.
    p_ab = raw_predict(canonical_features)
    p_ba = raw_predict(opponent_features)

    # --- Symmetric averaging: eliminates orientation bias ---
    # If the model were perfectly symmetric, p_ab == 1 - p_ba.
    # Averaging the two orientations removes any residual asymmetry.
    p_raw = 0.5 * (p_ab + (1.0 - p_ba))

    # --- Single calibration pass on the canonical raw probability ---
    p_calibrated = calibrate(p_raw)

    return SymmetricPrediction(
        fight_url=fight_url,
        canonical_fighter_url=canonical_url,
        canonical_opponent_url=opponent_url,
        p_calibrated=p_calibrated,
        raw_p_ab=p_ab,
        raw_p_ba=p_ba,
        calibrator_method=calibrator_method,
    )
