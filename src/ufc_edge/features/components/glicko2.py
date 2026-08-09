"""Glicko-2 rating system state component.

Tracks rating (μ), rating deviation (RD), and volatility (σ) per fighter.
Unlike plain Elo which only estimates a point rating, Glicko-2 also models
how confident we are in that rating (RD) — a fighter who hasn't competed
recently has a higher RD, signaling less certainty about their true skill.

The algorithm follows Glickman's "Example of the Glicko-2 System" paper:
 1. Convert ratings to internal Glicko-2 scale (divide by 173.7178).
 2. Compute g(φ) and E(μ, μ_j, φ_j) for opponents.
 3. Update volatility σ via iterative Illinois algorithm.
 4. Update RD and rating using the new volatility.
 5. Convert back to the original scale.

Between fights, RD grows to reflect increasing uncertainty during inactivity,
capped at the initial RD. Injury stoppages and no-contests are excluded from
updates since they carry no information about relative skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from ufc_edge.features.contracts import FightOutcomeView, FrozenState

# Glicko-2 internal scale factor: 400 / ln(10)
_SCALE = 400.0 / math.log(10)  # ≈ 173.7178

# Convergence tolerance for the volatility iteration (Illinois algorithm)
_EPSILON = 1e-6

# Methods treated as uninformative — no rating update applied
_INJURY_KEYWORDS = frozenset({
    "could not continue",
    "injury",
    "doctor's stoppage",
    "nc",
    "no contest",
    "overturned",
})


def _is_injury_or_nc(method: str) -> bool:
    """Return True if the fight method indicates a non-informative outcome."""
    lower = method.lower()
    return any(keyword in lower for keyword in _INJURY_KEYWORDS)


# ---------------------------------------------------------------------------
# Internal record for mutable state
# ---------------------------------------------------------------------------


@dataclass
class _Glicko2Record:
    """Mutable internal record for a single fighter's Glicko-2 state."""

    mu: float
    rd: float
    sigma: float
    last_fight_date: date | None
    fight_count: int


# ---------------------------------------------------------------------------
# Frozen record exposed via freeze()
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Glicko2Record:
    """Immutable snapshot of one fighter's Glicko-2 state."""

    mu: float
    rd: float
    sigma: float
    last_fight_date: date | None
    fight_count: int


# ---------------------------------------------------------------------------
# Frozen state container
# ---------------------------------------------------------------------------


class Glicko2FrozenState(FrozenState):
    """Read-only snapshot of all fighters' Glicko-2 ratings.

    Returns default initial values for unknown fighters (those not yet in the
    system), allowing emitters to query any fighter without key errors.
    """

    __slots__ = ("_records", "_initial_mu", "_initial_rd", "_initial_sigma")

    def __init__(
        self,
        records: dict[str, Glicko2Record],
        initial_mu: float,
        initial_rd: float,
        initial_sigma: float,
    ) -> None:
        # Bypass FrozenState.__setattr__ for initialization
        object.__setattr__(self, "_records", records)
        object.__setattr__(self, "_initial_mu", initial_mu)
        object.__setattr__(self, "_initial_rd", initial_rd)
        object.__setattr__(self, "_initial_sigma", initial_sigma)

    def get_record(self, fighter_url: str) -> Glicko2Record:
        """Get a fighter's Glicko-2 record, returning defaults if unknown."""
        if fighter_url in self._records:
            return self._records[fighter_url]
        return Glicko2Record(
            mu=self._initial_mu,
            rd=self._initial_rd,
            sigma=self._initial_sigma,
            last_fight_date=None,
            fight_count=0,
        )


# ---------------------------------------------------------------------------
# Glicko-2 math helpers
# ---------------------------------------------------------------------------


def _g(phi: float) -> float:
    """Glicko-2 g(φ) function: reduces impact of opponent with high RD."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _expected(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score E(μ, μ_j, φ_j) — probability of winning."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(
    sigma: float, phi: float, delta: float, v: float, tau: float
) -> float:
    """Compute new volatility using the Illinois algorithm (Step 5 of paper).

    Finds σ' such that the system's estimated variance is consistent with the
    observed outcome surprise. Uses a bracketed root-finding approach.
    """
    a = math.log(sigma**2)
    phi_sq = phi**2
    delta_sq = delta**2

    def f(x: float) -> float:
        ex = math.exp(x)
        denom = phi_sq + v + ex
        term1 = ex * (delta_sq - phi_sq - v - ex) / (2.0 * denom**2)
        term2 = (x - a) / (tau**2)
        return term1 - term2

    # Set initial bounds for bracketing
    fa = f(a)
    if delta_sq > phi_sq + v:
        b = math.log(delta_sq - phi_sq - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        b = a - k * tau

    fb = f(b)

    # Illinois algorithm iteration
    while abs(b - a) > _EPSILON:
        c = a + (a - b) * fa / (fb - fa)
        fc = f(c)
        if fc * fb <= 0:
            a = b
            fa = fb
        else:
            fa = fa / 2.0
        b = c
        fb = fc

    return math.exp(b / 2.0)


# ---------------------------------------------------------------------------
# Glicko2Tracker — the StateComponent implementation
# ---------------------------------------------------------------------------


class Glicko2Tracker:
    """Glicko-2 rating tracker for UFC fighters.

    Maintains per-fighter rating (μ), rating deviation (RD), and volatility (σ).
    RD captures our confidence in the rating — it shrinks with more data and
    grows during inactivity. Volatility captures how consistently a fighter
    performs relative to their rating.

    Params:
        initial_mu: Starting rating for new fighters (default 1500).
        initial_rd: Starting rating deviation (default 350 — high uncertainty).
        tau: System constant controlling volatility change speed.
             Standard default ~0.5; smaller = more stable volatility.
        rating_period_days: Days per rating period for RD growth calculation.
    """

    def __init__(
        self,
        initial_mu: float = 1500.0,
        initial_rd: float = 350.0,
        tau: float = 0.5,  # Glickman's paper default; configurable via graph.yaml
        rating_period_days: int = 180,  # placeholder until owner specifies
    ) -> None:
        self._initial_mu = initial_mu
        self._initial_rd = initial_rd
        self._initial_sigma = 0.06  # Standard initial volatility from the paper
        self._tau = tau
        self._rating_period_days = rating_period_days
        self._records: dict[str, _Glicko2Record] = {}

    def _get_or_create(self, fighter_url: str) -> _Glicko2Record:
        """Retrieve existing record or create a new one with initial values."""
        if fighter_url not in self._records:
            self._records[fighter_url] = _Glicko2Record(
                mu=self._initial_mu,
                rd=self._initial_rd,
                sigma=self._initial_sigma,
                last_fight_date=None,
                fight_count=0,
            )
        return self._records[fighter_url]

    def _apply_rd_growth(self, record: _Glicko2Record, current_date: date) -> None:
        """Grow RD proportional to elapsed rating periods since last fight.

        When a fighter is inactive, our confidence in their rating decreases.
        RD grows based on elapsed time, capped at the initial RD.
        """
        if record.last_fight_date is None:
            return
        days_elapsed = (current_date - record.last_fight_date).days
        if days_elapsed <= 0:
            return
        periods_elapsed = days_elapsed / self._rating_period_days
        # Convert to Glicko-2 internal scale for growth computation
        phi = record.rd / _SCALE
        sigma = record.sigma
        # New phi after inactivity: phi' = sqrt(phi^2 + periods * sigma^2)
        phi_new = math.sqrt(phi**2 + periods_elapsed * sigma**2)
        # Convert back and cap at initial RD
        new_rd = min(phi_new * _SCALE, self._initial_rd)
        record.rd = new_rd

    def update(self, fight: FightOutcomeView) -> None:
        """Apply one fight outcome to internal state.

        Skips the rating update for injury stoppages, no-contests, and draws
        (winner_url is None). For valid outcomes, applies the full Glicko-2
        algorithm to both fighters simultaneously.
        """
        # Skip uninformative outcomes
        if _is_injury_or_nc(fight.method):
            return
        if fight.winner_url is None:
            # Draws and no-contests carry no win/loss information
            return

        fighter_a_url = fight.fighter_a_url
        fighter_b_url = fight.fighter_b_url
        rec_a = self._get_or_create(fighter_a_url)
        rec_b = self._get_or_create(fighter_b_url)

        # Apply RD growth for inactivity before computing the update
        self._apply_rd_growth(rec_a, fight.event_date)
        self._apply_rd_growth(rec_b, fight.event_date)

        # Determine scores: 1 for win, 0 for loss
        score_a = 1.0 if fight.winner_url == fighter_a_url else 0.0
        score_b = 1.0 - score_a

        # Update both fighters
        self._update_single(rec_a, rec_b, score_a, fight.event_date)
        self._update_single(rec_b, rec_a, score_b, fight.event_date)

    def _update_single(
        self,
        player: _Glicko2Record,
        opponent: _Glicko2Record,
        score: float,
        event_date: date,
    ) -> None:
        """Apply Glicko-2 update for a single fighter given one opponent result.

        Steps from Glickman's paper:
         1. Convert to Glicko-2 scale.
         2. Compute g(φ_j) and E (expected score).
         3. Compute estimated variance v.
         4. Compute estimated improvement Δ.
         5. Compute new volatility σ'.
         6. Update pre-rating-period φ* and then new φ, μ.
         7. Convert back to original scale.
        """
        # Step 1: Convert to internal Glicko-2 scale
        mu = (player.mu - self._initial_mu) / _SCALE
        phi = player.rd / _SCALE
        sigma = player.sigma
        mu_j = (opponent.mu - self._initial_mu) / _SCALE
        phi_j = opponent.rd / _SCALE

        # Step 2: Compute helper quantities
        g_phi_j = _g(phi_j)
        e_val = _expected(mu, mu_j, phi_j)

        # Step 3: Estimated variance v
        v = 1.0 / (g_phi_j**2 * e_val * (1.0 - e_val))

        # Step 4: Estimated improvement delta
        delta = v * g_phi_j * (score - e_val)

        # Step 5: New volatility
        new_sigma = _new_volatility(sigma, phi, delta, v, self._tau)

        # Step 6: Update phi (pre-rating period) then new phi and mu
        phi_star = math.sqrt(phi**2 + new_sigma**2)
        new_phi = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        new_mu = mu + new_phi**2 * g_phi_j * (score - e_val)

        # Step 7: Convert back to original scale
        player.mu = new_mu * _SCALE + self._initial_mu
        player.rd = new_phi * _SCALE
        player.sigma = new_sigma
        player.last_fight_date = event_date
        player.fight_count += 1

    def freeze(self) -> Glicko2FrozenState:
        """Return a deeply-frozen snapshot of all fighters' current ratings.

        The snapshot is independent of internal state — later updates do not
        affect previously frozen objects.
        """
        frozen_records = {
            url: Glicko2Record(
                mu=rec.mu,
                rd=rec.rd,
                sigma=rec.sigma,
                last_fight_date=rec.last_fight_date,
                fight_count=rec.fight_count,
            )
            for url, rec in self._records.items()
        }
        return Glicko2FrozenState(
            records=frozen_records,
            initial_mu=self._initial_mu,
            initial_rd=self._initial_rd,
            initial_sigma=self._initial_sigma,
        )
