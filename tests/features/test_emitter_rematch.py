"""Tests for RematchEmitter.

Validates rematch detection, fights-since counting, result of first meeting,
method tracking, competitive-fight classification (decision-only), and
score-delta calculation from the first encounter between two fighters.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_edge.features.contracts import EmitContext, FighterProfile, FightOutcomeView
from ufc_edge.features.emitters.rematch import RematchAccumulator, RematchEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"
FIGHTER_C = "http://fighter/c"


def _outcome(
    *,
    fight_url: str = "http://fight/1",
    event_url: str = "http://event/1",
    event_date: date = date(2024, 6, 1),
    fighter_a_url: str = FIGHTER_A,
    fighter_b_url: str = FIGHTER_B,
    winner_url: str | None = FIGHTER_A,
    method: str = "Decision - Unanimous",
    ending_round: int = 3,
    ending_time: str = "5:00",
    weight_class: str = "Lightweight",
    bout_order: int | None = None,
) -> FightOutcomeView:
    return FightOutcomeView(
        fight_url=fight_url,
        event_url=event_url,
        event_date=event_date,
        fighter_a_url=fighter_a_url,
        fighter_b_url=fighter_b_url,
        winner_url=winner_url,
        method=method,
        ending_round=ending_round,
        ending_time=ending_time,
        weight_class=weight_class,
        bout_order=bout_order,
    )


def _build_context(
    accumulator: RematchAccumulator,
    *,
    fighter_url: str = FIGHTER_A,
    opponent_url: str = FIGHTER_B,
    event_date: date = date(2024, 6, 1),
) -> EmitContext:
    """Build a minimal EmitContext with a rematch snapshot."""
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=FighterProfile(fighter_url=fighter_url),
        opponent_url=opponent_url,
        opponent_profile=FighterProfile(fighter_url=opponent_url),
        event_date=event_date,
        event_url="http://event/test",
        weight_class="Lightweight",
        fight_url="http://fight/test",
        bout_order=None,
        components={"rematch": accumulator.freeze()},
    )


# ---------------------------------------------------------------------------
# Non-rematch tests — all outputs None/False
# ---------------------------------------------------------------------------


class TestNonRematch:
    """When fighters have not met before, all rematch features are None or False."""

    def test_no_prior_history_returns_false_and_nones(self):
        accumulator = RematchAccumulator()
        context = _build_context(accumulator)
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["is_rematch"] == 0.0
        assert result["fights_since_first_meeting"] is None
        assert result["result_of_first_meeting"] is None
        assert result["first_meeting_method"] is None
        assert result["first_meeting_competitive"] is None
        assert result["first_meeting_score_delta"] is None

    def test_fighter_has_history_but_not_with_opponent(self):
        accumulator = RematchAccumulator()
        # A fought C, not B
        accumulator.update(_outcome(
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_C,
            winner_url=FIGHTER_A,
        ))
        context = _build_context(accumulator, opponent_url=FIGHTER_B)
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["is_rematch"] == 0.0
        assert result["fights_since_first_meeting"] is None


# ---------------------------------------------------------------------------
# Rematch detection — correct fields populated
# ---------------------------------------------------------------------------


class TestRematchDetected:
    """When fighters have met before, is_rematch is 1.0 and history fields populate."""

    def test_immediate_rematch_one_fight_since(self):
        accumulator = RematchAccumulator()
        # First meeting: A beats B by KO in round 2
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2023, 1, 1),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_B,
            winner_url=FIGHTER_A,
            method="KO/TKO",
            ending_round=2,
        ))
        # A fights C (builds fight count)
        accumulator.update(_outcome(
            fight_url="http://fight/middle",
            event_date=date(2023, 6, 1),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_C,
            winner_url=FIGHTER_A,
        ))

        context = _build_context(
            accumulator,
            event_date=date(2024, 1, 1),
        )
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["is_rematch"] == 1.0
        # A had 1 fight between the first meeting and this rematch
        assert result["fights_since_first_meeting"] == 1.0
        # A won the first meeting
        assert result["result_of_first_meeting"] == 1.0
        assert result["first_meeting_method"] == "KO/TKO"
        # KO is not a decision — not competitive
        assert result["first_meeting_competitive"] == 0.0
        # Ended in round 2
        assert result["first_meeting_score_delta"] == 2.0

    def test_rematch_where_focal_fighter_lost_first(self):
        accumulator = RematchAccumulator()
        # B beats A by submission in round 1
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2022, 3, 15),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_B,
            winner_url=FIGHTER_B,
            method="Submission",
            ending_round=1,
        ))

        context = _build_context(
            accumulator,
            fighter_url=FIGHTER_A,
            opponent_url=FIGHTER_B,
            event_date=date(2023, 3, 15),
        )
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["is_rematch"] == 1.0
        # No fights between — A's fight count since first meeting is 0
        assert result["fights_since_first_meeting"] == 0.0
        # A lost the first meeting
        assert result["result_of_first_meeting"] == 0.0
        assert result["first_meeting_method"] == "Submission"
        assert result["first_meeting_competitive"] == 0.0
        assert result["first_meeting_score_delta"] == 1.0

    def test_rematch_draw_in_first_meeting(self):
        accumulator = RematchAccumulator()
        # Draw (no winner)
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2022, 5, 1),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_B,
            winner_url=None,
            method="Decision - Split",
            ending_round=3,
        ))

        context = _build_context(
            accumulator,
            event_date=date(2023, 5, 1),
        )
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["is_rematch"] == 1.0
        # Draw → 0.5
        assert result["result_of_first_meeting"] == 0.5
        assert result["first_meeting_method"] == "Decision - Split"
        # Split decision → competitive
        assert result["first_meeting_competitive"] == 1.0
        # Decision goes full rounds; score_delta = ending_round
        assert result["first_meeting_score_delta"] == 3.0

    def test_fights_since_counts_focal_fighter_fights_only(self):
        accumulator = RematchAccumulator()
        # First meeting
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2022, 1, 1),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_B,
            winner_url=FIGHTER_A,
            method="Decision - Unanimous",
            ending_round=3,
        ))
        # A fights C twice
        accumulator.update(_outcome(
            fight_url="http://fight/2",
            event_date=date(2022, 6, 1),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_C,
            winner_url=FIGHTER_A,
        ))
        accumulator.update(_outcome(
            fight_url="http://fight/3",
            event_date=date(2023, 1, 1),
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_C,
            winner_url=FIGHTER_C,
        ))
        # B fights C (should not count for A's fights_since)
        accumulator.update(_outcome(
            fight_url="http://fight/4",
            event_date=date(2023, 3, 1),
            fighter_a_url=FIGHTER_B,
            fighter_b_url=FIGHTER_C,
            winner_url=FIGHTER_B,
        ))

        context = _build_context(
            accumulator,
            event_date=date(2024, 1, 1),
        )
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["is_rematch"] == 1.0
        # A had 2 fights between first meeting with B and this rematch
        assert result["fights_since_first_meeting"] == 2.0


# ---------------------------------------------------------------------------
# Competitive classification — decision-only
# ---------------------------------------------------------------------------


class TestCompetitiveClassification:
    """first_meeting_competitive is 1.0 only for decision methods."""

    def test_unanimous_decision_is_competitive(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2023, 1, 1),
            method="Decision - Unanimous",
            ending_round=5,
            winner_url=FIGHTER_A,
        ))

        context = _build_context(accumulator, event_date=date(2024, 1, 1))
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["first_meeting_competitive"] == 1.0

    def test_split_decision_is_competitive(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2023, 1, 1),
            method="Decision - Split",
            ending_round=3,
            winner_url=FIGHTER_A,
        ))

        context = _build_context(accumulator, event_date=date(2024, 1, 1))
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["first_meeting_competitive"] == 1.0

    def test_majority_decision_is_competitive(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2023, 1, 1),
            method="Decision - Majority",
            ending_round=3,
            winner_url=FIGHTER_B,
        ))

        context = _build_context(accumulator, event_date=date(2024, 1, 1))
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["first_meeting_competitive"] == 1.0

    def test_ko_is_not_competitive(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2023, 1, 1),
            method="KO/TKO",
            ending_round=1,
            winner_url=FIGHTER_A,
        ))

        context = _build_context(accumulator, event_date=date(2024, 1, 1))
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["first_meeting_competitive"] == 0.0

    def test_submission_is_not_competitive(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2023, 1, 1),
            method="Submission",
            ending_round=2,
            winner_url=FIGHTER_B,
        ))

        context = _build_context(accumulator, event_date=date(2024, 1, 1))
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["first_meeting_competitive"] == 0.0

    def test_tko_doctors_stoppage_is_not_competitive(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2023, 1, 1),
            method="TKO - Doctor's Stoppage",
            ending_round=3,
            winner_url=FIGHTER_A,
        ))

        context = _build_context(accumulator, event_date=date(2024, 1, 1))
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["first_meeting_competitive"] == 0.0


# ---------------------------------------------------------------------------
# Symmetry — first meeting uses chronologically earliest encounter
# ---------------------------------------------------------------------------


class TestSymmetry:
    """Multiple meetings: always uses the first (earliest) meeting."""

    def test_triple_fight_uses_first_meeting(self):
        accumulator = RematchAccumulator()
        # First meeting: decision
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2020, 1, 1),
            winner_url=FIGHTER_B,
            method="Decision - Unanimous",
            ending_round=3,
        ))
        # Second meeting: KO
        accumulator.update(_outcome(
            fight_url="http://fight/second",
            event_date=date(2021, 6, 1),
            winner_url=FIGHTER_A,
            method="KO/TKO",
            ending_round=1,
        ))

        context = _build_context(
            accumulator,
            event_date=date(2023, 1, 1),
        )
        emitter = RematchEmitter()

        result = emitter.emit(context)

        assert result["is_rematch"] == 1.0
        # Should reference the FIRST meeting (decision), not the second (KO)
        assert result["first_meeting_method"] == "Decision - Unanimous"
        assert result["first_meeting_competitive"] == 1.0
        # A lost the first meeting (B won)
        assert result["result_of_first_meeting"] == 0.0

    def test_perspective_reversal_gives_opposite_result(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/first",
            event_date=date(2022, 1, 1),
            winner_url=FIGHTER_A,
            method="Decision - Split",
            ending_round=5,
        ))

        # From A's perspective: won first meeting
        ctx_a = _build_context(
            accumulator,
            fighter_url=FIGHTER_A,
            opponent_url=FIGHTER_B,
            event_date=date(2024, 1, 1),
        )
        # From B's perspective: lost first meeting
        ctx_b = _build_context(
            accumulator,
            fighter_url=FIGHTER_B,
            opponent_url=FIGHTER_A,
            event_date=date(2024, 1, 1),
        )
        emitter = RematchEmitter()

        result_a = emitter.emit(ctx_a)
        result_b = emitter.emit(ctx_b)

        assert result_a["result_of_first_meeting"] == 1.0
        assert result_b["result_of_first_meeting"] == 0.0
        # Method and competitive are the same regardless of perspective
        assert result_a["first_meeting_method"] == result_b["first_meeting_method"]
        assert result_a["first_meeting_competitive"] == result_b["first_meeting_competitive"]


# ---------------------------------------------------------------------------
# RematchAccumulator state correctness
# ---------------------------------------------------------------------------


class TestRematchAccumulator:
    """Unit tests for the accumulator's freeze/update cycle."""

    def test_freeze_returns_immutable_snapshot(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome())
        snapshot = accumulator.freeze()

        with pytest.raises(AttributeError):
            snapshot.some_field = "mutation"  # type: ignore[attr-defined]

    def test_freeze_independent_of_subsequent_updates(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fight_url="http://fight/1",
            event_date=date(2023, 1, 1),
        ))
        snapshot_before = accumulator.freeze()

        # Further updates should not affect the frozen snapshot
        accumulator.update(_outcome(
            fight_url="http://fight/2",
            event_date=date(2023, 6, 1),
        ))
        snapshot_after = accumulator.freeze()

        # The first snapshot should still reflect only one fight
        matchup = snapshot_before.get_matchup_history(FIGHTER_A, FIGHTER_B)
        assert len(matchup) == 1

        matchup_after = snapshot_after.get_matchup_history(FIGHTER_A, FIGHTER_B)
        assert len(matchup_after) == 2

    def test_bidirectional_tracking(self):
        accumulator = RematchAccumulator()
        accumulator.update(_outcome(
            fighter_a_url=FIGHTER_A,
            fighter_b_url=FIGHTER_B,
            winner_url=FIGHTER_A,
        ))
        snapshot = accumulator.freeze()

        # Both directions should find the same fight
        ab = snapshot.get_matchup_history(FIGHTER_A, FIGHTER_B)
        ba = snapshot.get_matchup_history(FIGHTER_B, FIGHTER_A)
        assert len(ab) == 1
        assert len(ba) == 1
        assert ab[0].fight_url == ba[0].fight_url
