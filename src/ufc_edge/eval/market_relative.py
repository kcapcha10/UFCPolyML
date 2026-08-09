"""Market-relative Brier skill evaluation.

Scores the market's own implied probabilities as if the market were a predictor,
so the model's skill can be compared against a real benchmark instead of an
arbitrary floor (like constant 0.5). The headline metric is:

    Brier_skill = 1 − (Brier_model / Brier_market)

Positive values mean the model outperforms the market; zero means parity;
negative means the market is more accurate.

IMPORTANT: This module reads market data ONLY for scoring/evaluation purposes
after predictions are already made. Market probabilities are never used as model
inputs or training labels — they serve exclusively as a comparison benchmark in
post-prediction evaluation. The model itself is strictly odds-free.

The function accepts pre-joined data (a list of per-fight rows containing model
prediction, market probability, and outcome) rather than querying a specific
table directly. This keeps the evaluation logic testable and decoupled from the
data source; the caller wires the real market_fight_links table at integration
time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketFightRow:
    """A single fight with model prediction, optional market probability, and outcome.

    Params: model_prob (model's predicted win prob), market_prob (market implied
    prob or None if unmatched), outcome (1=win, 0=loss).
    """

    model_prob: float
    market_prob: float | None
    outcome: int


@dataclass(frozen=True)
class BrierSkillResult:
    """Output of the market-relative evaluation.

    Contains absolute Brier scores for model and market (both computed on the
    identical subset of fights with matched market data), the relative skill
    score, and exclusion accounting.
    """

    brier_model: float | None
    brier_market: float | None
    brier_skill: float | None
    n_scored: int
    n_excluded: int


def compute_brier_skill(rows: list[MarketFightRow]) -> BrierSkillResult:
    """Compute Brier skill of the model relative to market-implied probabilities.

    Excludes fights without a matched market probability and reports the
    exclusion count. Both model and market are scored on the identical fight
    subset (same denominator) to ensure a fair comparison.

    Returns BrierSkillResult with None values if no fights have market matches.
    """
    matched: list[MarketFightRow] = [r for r in rows if r.market_prob is not None]
    n_excluded = len(rows) - len(matched)
    n_scored = len(matched)

    if n_scored == 0:
        return BrierSkillResult(
            brier_model=None,
            brier_market=None,
            brier_skill=None,
            n_scored=0,
            n_excluded=n_excluded,
        )

    model_probs = np.array([r.model_prob for r in matched], dtype=np.float64)
    market_probs = np.array([r.market_prob for r in matched], dtype=np.float64)
    outcomes = np.array([r.outcome for r in matched], dtype=np.float64)

    brier_model = float(np.mean((model_probs - outcomes) ** 2))
    brier_market = float(np.mean((market_probs - outcomes) ** 2))

    brier_skill = 1.0 - (brier_model / brier_market)

    return BrierSkillResult(
        brier_model=brier_model,
        brier_market=brier_market,
        brier_skill=brier_skill,
        n_scored=n_scored,
        n_excluded=n_excluded,
    )
