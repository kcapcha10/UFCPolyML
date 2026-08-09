"""Record emitter — win/loss record features from CareerAccumulator state.

Reads frozen CareerFighterState snapshots and emits overall win percentages,
windowed recent-fight percentages, streak, finish/decision breakdowns,
UFC-specific record, debut detection with opponent context, and contender
series event identification.
"""

from __future__ import annotations

from ufc_edge.features.components.career import CareerFighterState, CareerSnapshot
from ufc_edge.features.contracts import EmitContext


def _is_contender_series(event_url: str) -> bool:
    """Detect Dana White's Contender Series from event URL pattern."""
    return "contender-series" in event_url.lower()


class RecordEmitter:
    """Stateless emitter for win/loss record features.

    Reads the 'career' component from EmitContext and derives percentages,
    streaks, and debut-related features. Returns None for any metric requiring
    history that does not yet exist.
    """

    name: str = "record"

    def emit(self, context: EmitContext) -> dict[str, float | None]:
        """Emit record features for the focal fighter."""
        career_snap: CareerSnapshot = context.components["career"]  # type: ignore[assignment]
        fighter_state = career_snap.get(context.fighter_url)
        opponent_state = career_snap.get(context.opponent_url)

        is_debut = fighter_state is None or fighter_state.total_fights == 0

        return {
            "win_pct_all": self._win_pct_all(fighter_state),
            "win_pct_last3": self._win_pct_last_n(fighter_state, 3),
            "win_pct_last5": self._win_pct_last_n(fighter_state, 5),
            "current_streak": self._current_streak(fighter_state),
            "win_pct_by_finish": self._win_pct_by_finish(fighter_state),
            "win_pct_by_decision": self._win_pct_by_decision(fighter_state),
            "loss_pct_by_finish": self._loss_pct_by_finish(fighter_state),
            "loss_pct_by_decision": self._loss_pct_by_decision(fighter_state),
            "ufc_win_pct": self._ufc_win_pct(fighter_state),
            "ufc_record_fights_count": self._ufc_fights_count(fighter_state),
            "is_ufc_debut": 1.0 if is_debut else 0.0,
            "debut_opponent_ufc_experience": self._debut_opp_experience(
                is_debut, opponent_state
            ),
            "debut_opponent_ufc_win_pct": self._debut_opp_win_pct(
                is_debut, opponent_state
            ),
            "contender_series_win": (
                1.0 if _is_contender_series(context.event_url) else 0.0
            ),
        }

    def _win_pct_all(self, state: CareerFighterState | None) -> float | None:
        """Win percentage across all UFC fights (wins / total_fights)."""
        if state is None or state.total_fights == 0:
            return None
        return state.wins / state.total_fights

    def _win_pct_last_n(
        self, state: CareerFighterState | None, n: int
    ) -> float | None:
        """Win percentage over the last N fights. None if fewer than N fights.

        Uses overall win rate — frozen state does not carry per-fight outcome
        history. Becomes exact once CareerAccumulator adds a recent-outcomes deque.
        """
        if state is None or state.total_fights < n:
            return None
        return state.wins / state.total_fights

    def _current_streak(self, state: CareerFighterState | None) -> float:
        """Current streak: positive = win streak, negative = loss streak, 0 = no fights."""
        if state is None:
            return 0.0
        return float(state.current_streak)

    def _win_pct_by_finish(self, state: CareerFighterState | None) -> float | None:
        """Fraction of wins that came by finish (KO/TKO or submission)."""
        if state is None or state.wins == 0:
            return None
        finish_wins = state.ko_wins + state.submission_wins
        return finish_wins / state.wins

    def _win_pct_by_decision(self, state: CareerFighterState | None) -> float | None:
        """Fraction of wins that came by decision."""
        if state is None or state.wins == 0:
            return None
        return state.decision_wins / state.wins

    def _loss_pct_by_finish(self, state: CareerFighterState | None) -> float | None:
        """Fraction of losses that came by finish (KO/TKO or submission)."""
        if state is None or state.losses == 0:
            return None
        finish_losses = state.times_finished_by_ko + state.times_finished_by_sub
        return finish_losses / state.losses

    def _loss_pct_by_decision(self, state: CareerFighterState | None) -> float | None:
        """Fraction of losses that came by decision."""
        if state is None or state.losses == 0:
            return None
        finish_losses = state.times_finished_by_ko + state.times_finished_by_sub
        decision_losses = state.losses - finish_losses
        return decision_losses / state.losses

    def _ufc_win_pct(self, state: CareerFighterState | None) -> float | None:
        """Win percentage in UFC fights (same as overall since we only track UFC)."""
        if state is None or state.total_fights == 0:
            return None
        return state.wins / state.total_fights

    def _ufc_fights_count(self, state: CareerFighterState | None) -> float:
        """Total UFC fights count."""
        if state is None:
            return 0.0
        return float(state.total_fights)

    def _debut_opp_experience(
        self, is_debut: bool, opponent_state: CareerFighterState | None
    ) -> float | None:
        """Opponent's UFC fight count, only populated when focal fighter is debuting."""
        if not is_debut:
            return None
        if opponent_state is None or opponent_state.total_fights == 0:
            return None
        return float(opponent_state.total_fights)

    def _debut_opp_win_pct(
        self, is_debut: bool, opponent_state: CareerFighterState | None
    ) -> float | None:
        """Opponent's UFC win pct, only populated when focal fighter is debuting."""
        if not is_debut:
            return None
        if opponent_state is None or opponent_state.total_fights == 0:
            return None
        return opponent_state.wins / opponent_state.total_fights
