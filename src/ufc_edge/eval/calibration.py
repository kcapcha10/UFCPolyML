"""Calibrator fitting and per-fold selection for probability recalibration.

Provides three calibration methods and a selection protocol that picks the
method whose transformed probabilities best match observed outcomes on a
held-out evaluation slice. The winner is selected by expected calibration
error, with Brier score as tiebreak.

Methods:
- Platt: logistic regression on the logit of the raw score (2 parameters).
- Isotonic: non-parametric monotone mapping via isotonic regression.
- Beta: 3-parameter generalization fitting sigmoid(a·ln(s) - b·ln(1-s) + c)
  via negative log-likelihood minimization.

Guards:
- CalibrationSizingError if the calibration set is too small for reliable fitting.
- CalibrationLeakageError if holdout-window events are detected in calibration inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ufc_edge.eval.metrics import brier_score, expected_calibration_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HOLDOUT_START = "2026-01-01"
_DEFAULT_HOLDOUT_END = "2026-08-31"
_DEFAULT_MIN_CALIBRATION_SIZE = 250
_CLIP_EPS = 1e-7

# ---------------------------------------------------------------------------
# Public protocols and data types
# ---------------------------------------------------------------------------


class CalibratorProtocol(Protocol):
    """Interface that all fitted calibrators satisfy."""

    method: str

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw model probabilities to calibrated probabilities."""
        ...


@dataclass(frozen=True)
class PlattCalibrator:
    """Logistic regression on logit(raw_prob) — 2-parameter Platt scaling."""

    method: str = "platt"
    _model: LogisticRegression = None  # type: ignore[assignment]

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply the fitted logistic mapping to raw probabilities."""
        clipped = np.clip(raw_probs, _CLIP_EPS, 1.0 - _CLIP_EPS)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self._model.predict_proba(logits)[:, 1]


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Non-parametric monotone calibration via isotonic regression."""

    method: str = "isotonic"
    _model: IsotonicRegression = None  # type: ignore[assignment]

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply the fitted isotonic mapping to raw probabilities."""
        return self._model.predict(raw_probs)


@dataclass(frozen=True)
class BetaCalibrator:
    """3-parameter beta calibration: sigmoid(a·ln(s) - b·ln(1-s) + c).

    Generalizes Platt scaling by allowing the log-odds transformation to
    weight positive and negative evidence asymmetrically.
    """

    method: str = "beta"
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply the fitted beta calibration map."""
        clipped = np.clip(raw_probs, _CLIP_EPS, 1.0 - _CLIP_EPS)
        z = self.a * np.log(clipped) - self.b * np.log(1.0 - clipped) + self.c
        return _sigmoid(z)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CalibrationSizingError(Exception):
    """Raised when the calibration set is too small for reliable fitting.

    Proceeding with an undersized calibration set would produce an unreliable
    probability mapping — the calibrator might overfit to noise in a handful
    of samples rather than learning a true probability correction.
    """


class CalibrationLeakageError(Exception):
    """Raised when holdout-window events are detected in calibration inputs.

    The holdout period is reserved for a single final evaluation after all model
    decisions are locked. Allowing holdout data into calibrator fitting would
    compromise the integrity of that evaluation.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_calibrators(
    raw_probs: np.ndarray,
    labels: np.ndarray,
    *,
    event_dates: list[date] | None = None,
    min_calibration_size: int = _DEFAULT_MIN_CALIBRATION_SIZE,
    holdout_start: str = _DEFAULT_HOLDOUT_START,
    holdout_end: str = _DEFAULT_HOLDOUT_END,
) -> dict[str, CalibratorProtocol]:
    """Fit all three calibration methods on the given calibration data.

    Params:
        raw_probs: Raw model-output probabilities for the calibration slice.
        labels: Binary outcomes (0 or 1) corresponding to each prediction.
        event_dates: Optional per-sample event dates for holdout leakage detection.
            When provided, any date within the holdout window triggers an error.
        min_calibration_size: Minimum number of samples required. If the input is
            smaller, CalibrationSizingError is raised before fitting.
        holdout_start: Start of holdout window (inclusive), ISO format.
        holdout_end: End of holdout window (inclusive), ISO format.

    Returns:
        Dictionary mapping method names to fitted calibrator objects.

    Raises:
        CalibrationSizingError: If len(raw_probs) < min_calibration_size.
        CalibrationLeakageError: If event_dates overlap the holdout window.
    """
    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    _guard_sizing(len(raw_probs), min_calibration_size)
    if event_dates is not None:
        _guard_holdout_leakage(event_dates, holdout_start, holdout_end)

    platt = _fit_platt(raw_probs, labels)
    isotonic = _fit_isotonic(raw_probs, labels)
    beta = _fit_beta(raw_probs, labels)

    return {"platt": platt, "isotonic": isotonic, "beta": beta}


def select_calibrator(
    calibrators: dict[str, CalibratorProtocol],
    eval_raw_probs: np.ndarray,
    eval_labels: np.ndarray,
) -> tuple[str, CalibratorProtocol]:
    """Select the best calibrator by evaluating each on a held-out slice.

    The method that best matches predicted probabilities to observed outcomes
    on held-out data wins. Selection criterion is expected calibration error;
    ties are broken by Brier score (lower is better for both).

    Params:
        calibrators: Mapping of method name to fitted calibrator.
        eval_raw_probs: Raw probabilities for the evaluation slice (disjoint
            from calibration fitting data).
        eval_labels: Binary outcomes for the evaluation slice.

    Returns:
        Tuple of (winning_method_name, winning_calibrator).
    """
    eval_raw_probs = np.asarray(eval_raw_probs, dtype=np.float64)
    eval_labels = np.asarray(eval_labels, dtype=np.float64)

    best_name: str | None = None
    best_calibrator: CalibratorProtocol | None = None
    best_ece = float("inf")
    best_brier = float("inf")

    for name, calibrator in calibrators.items():
        calibrated = calibrator.transform(eval_raw_probs)
        ece = expected_calibration_error(calibrated, eval_labels)
        brier = brier_score(calibrated, eval_labels)

        # Lower ECE wins; ties broken by lower Brier
        if (ece < best_ece) or (ece == best_ece and brier < best_brier):
            best_name = name
            best_calibrator = calibrator
            best_ece = ece
            best_brier = brier

    # At least one calibrator must exist
    assert best_name is not None  # noqa: S101
    assert best_calibrator is not None  # noqa: S101
    return best_name, best_calibrator


# ---------------------------------------------------------------------------
# Internal: fitting implementations
# ---------------------------------------------------------------------------


def _fit_platt(raw_probs: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    """Fit Platt scaling via logistic regression on logit-transformed scores."""
    clipped = np.clip(raw_probs, _CLIP_EPS, 1.0 - _CLIP_EPS)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    lr = LogisticRegression(solver="lbfgs", C=np.inf, max_iter=1000)
    lr.fit(logits, labels)

    return PlattCalibrator(_model=lr)


def _fit_isotonic(raw_probs: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    """Fit isotonic regression for monotone non-parametric calibration."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, labels)

    return IsotonicCalibrator(_model=iso)


def _fit_beta(raw_probs: np.ndarray, labels: np.ndarray) -> BetaCalibrator:
    """Fit 3-parameter beta calibration via negative log-likelihood minimization.

    The model is: calibrated = sigmoid(a·ln(s) - b·ln(1-s) + c)
    where s is the raw probability. Parameters (a, b, c) are found by minimizing
    the negative log-likelihood of the binary outcomes under the calibrated probs.
    """
    clipped = np.clip(raw_probs, _CLIP_EPS, 1.0 - _CLIP_EPS)
    log_s = np.log(clipped)
    log_1ms = np.log(1.0 - clipped)

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b, c = params
        z = a * log_s - b * log_1ms + c
        cal_probs = _sigmoid(z)
        # Clip to avoid log(0)
        cal_probs = np.clip(cal_probs, _CLIP_EPS, 1.0 - _CLIP_EPS)
        ll = labels * np.log(cal_probs) + (1.0 - labels) * np.log(1.0 - cal_probs)
        return -float(np.mean(ll))

    result = minimize(
        neg_log_likelihood,
        x0=np.array([1.0, 1.0, 0.0]),
        method="L-BFGS-B",
    )

    a, b, c = result.x
    return BetaCalibrator(a=float(a), b=float(b), c=float(c))


# ---------------------------------------------------------------------------
# Internal: guards
# ---------------------------------------------------------------------------


def _guard_sizing(n_samples: int, minimum: int) -> None:
    """Reject calibration attempts on insufficient data."""
    if n_samples < minimum:
        msg = (
            f"Calibration set has {n_samples} samples, below the minimum of "
            f"{minimum}. Fitting a calibrator on too few samples produces an "
            f"unreliable probability mapping."
        )
        raise CalibrationSizingError(msg)


def _guard_holdout_leakage(
    event_dates: list[date],
    holdout_start: str,
    holdout_end: str,
) -> None:
    """Reject if any event dates fall within the holdout window."""
    start = date.fromisoformat(holdout_start)
    end = date.fromisoformat(holdout_end)

    for d in event_dates:
        if start <= d <= end:
            msg = (
                f"Event date {d} falls within the holdout window "
                f"[{holdout_start}, {holdout_end}]. Holdout data must not "
                f"enter calibrator fitting."
            )
            raise CalibrationLeakageError(msg)


# ---------------------------------------------------------------------------
# Internal: utilities
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )
