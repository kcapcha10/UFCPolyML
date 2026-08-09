"""Tests for ExperienceEmitter.

Validates title fight experience counting, championship detection from weight_class,
days as champion computation, main event experience, five-round fight tracking from
time_format, five-round win percentage, and zero-experience edge cases.
"""

from __future__ import annotations

from datetime import date

from ufc_edge.features.components.career import CareerAccumulator, CareerSnapshot
from ufc_edge.features.contracts import EmitContext, FighterProfile, FightOutcomeView
from ufc_edge.features.emitters.experience import (
    ExperienceAccumulator,
    ExperienceEmitter,
    ExperienceSnapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTER_A = "http://fighter/a"
FIGHTER_B = "http://fighter/b"


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
    career_snapshot: CareerSnapshot,
    experience_snapshot: ExperienceSnapshot,
    *,
    fighter_url: str = FIGHTER_A,
    opponent_url: str = FIGHTER_B,
    event_date: date = date(2024, 6, 1),
    weight_class: str = "Lightweight",
    bout_order: int | None = None,
) -> EmitContext:
    """Build a minimal EmitContext with career and experience snapshots."""
    return EmitContext(
        fighter_url=fighter_url,
        fighter_profile=FighterProfile(fighter_url=fighter_url),
        opponent_url=opponent_url,
        opponent_profile=FighterProfile(fighter_url=opponent_url),
        event_date=event_date,
        event_url="http://event/test",
        weight_class=weight_class,
        fight_url="http://fight/test",
        bout_order=bout_order,
        components={"career": career_snapshot, "experience": experience_snapshot},
    )


# ---------------------------------------------------------------------------
# Championship detection from weight_class field
# ---------------------------------------------------------------------------


class TestChampionshipDetection:
    """Verify title fights are detected from weight_class containing 'Title'."""

    def test_title_bout_detected_from_weight_class(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(weight_class="Lightweight Title Bout"),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.title_fights == 1

    def test_interim_title_bout_detected(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(weight_class="Interim Heavyweight Title Bout"),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.title_fights == 1

    def test_non_title_bout_not_counted(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(weight_class="Lightweight"),
            time_format="3 Rnd (5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.title_fights == 0

    def test_winner_marked_as_champion(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(
                weight_class="Lightweight Title Bout",
                winner_url=FIGHTER_A,
                event_date=date(2024, 1, 1),
            ),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.has_been_champion is True

    def test_loser_of_title_fight_not_champion(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(
                weight_class="Lightweight Title Bout",
                winner_url=FIGHTER_A,
                event_date=date(2024, 1, 1),
            ),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_B)
        assert state is not None
        assert state.has_been_champion is False

    def test_draw_in_title_fight_no_champion(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(
                weight_class="Welterweight Title Bout",
                winner_url=None,
                method="Draw",
                event_date=date(2024, 3, 1),
            ),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state_a = snapshot.get(FIGHTER_A)
        state_b = snapshot.get(FIGHTER_B)
        assert state_a is not None and state_a.has_been_champion is False
        assert state_b is not None and state_b.has_been_champion is False


# ---------------------------------------------------------------------------
# Five-round detection from time_format
# ---------------------------------------------------------------------------


class TestFiveRoundDetection:
    """Verify five-round fights are detected from time_format field."""

    def test_five_round_format_detected(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(weight_class="Lightweight Title Bout"),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.five_round_fights == 1

    def test_three_round_format_not_counted(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(weight_class="Lightweight"),
            time_format="3 Rnd (5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.five_round_fights == 0

    def test_five_round_win_tracked(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(winner_url=FIGHTER_A, weight_class="Lightweight Title Bout"),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.five_round_wins == 1

    def test_five_round_loss_not_counted_as_win(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(winner_url=FIGHTER_A, weight_class="Lightweight Title Bout"),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_B)
        assert state is not None
        assert state.five_round_fights == 1
        assert state.five_round_wins == 0


# ---------------------------------------------------------------------------
# Zero experience edge cases
# ---------------------------------------------------------------------------


class TestZeroExperience:
    """Fighters with no prior fights should emit all-zero features."""

    def test_unknown_fighter_returns_zeros(self) -> None:
        career_acc = CareerAccumulator()
        exp_acc = ExperienceAccumulator()
        emitter = ExperienceEmitter()
        ctx = _build_context(
            career_acc.freeze(),
            exp_acc.freeze(),
            fighter_url=FIGHTER_A,
        )
        result = emitter.emit(ctx)
        assert result["title_fight_experience"] == 0.0
        assert result["has_been_champion"] == 0.0
        assert result["days_as_champion"] == 0.0
        assert result["main_event_experience"] == 0.0
        assert result["five_round_experience"] == 0.0
        assert result["five_round_win_pct"] is None


# ---------------------------------------------------------------------------
# Days as champion calculation
# ---------------------------------------------------------------------------


class TestDaysAsChampion:
    """Verify days_as_champion is computed from first title win to event date."""

    def test_days_as_champion_from_first_win(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(
                fight_url="http://fight/1",
                event_date=date(2024, 1, 1),
                weight_class="Lightweight Title Bout",
                winner_url=FIGHTER_A,
            ),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        emitter = ExperienceEmitter()
        career_acc = CareerAccumulator()
        career_acc.update(
            _outcome(
                fight_url="http://fight/1",
                event_date=date(2024, 1, 1),
                weight_class="Lightweight Title Bout",
                winner_url=FIGHTER_A,
            )
        )
        ctx = _build_context(
            career_acc.freeze(),
            acc.freeze(),
            fighter_url=FIGHTER_A,
            event_date=date(2024, 7, 1),
        )
        result = emitter.emit(ctx)
        # 182 days between Jan 1 and Jul 1, 2024
        assert result["days_as_champion"] == 182.0

    def test_non_champion_has_zero_days(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(
                event_date=date(2024, 1, 1),
                weight_class="Lightweight",
                winner_url=FIGHTER_A,
            ),
            time_format="3 Rnd (5-5-5)",
        )
        emitter = ExperienceEmitter()
        career_acc = CareerAccumulator()
        career_acc.update(
            _outcome(
                event_date=date(2024, 1, 1),
                weight_class="Lightweight",
                winner_url=FIGHTER_A,
            )
        )
        ctx = _build_context(
            career_acc.freeze(),
            acc.freeze(),
            fighter_url=FIGHTER_A,
            event_date=date(2024, 7, 1),
        )
        result = emitter.emit(ctx)
        assert result["days_as_champion"] == 0.0


# ---------------------------------------------------------------------------
# Main event experience
# ---------------------------------------------------------------------------


class TestMainEventExperience:
    """Verify main event experience counts from bout_order."""

    def test_highest_bout_order_is_main_event(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(bout_order=12),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.main_event_fights == 1

    def test_non_main_event_not_counted(self) -> None:
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(bout_order=5),
            time_format="3 Rnd (5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.main_event_fights == 0

    def test_five_round_non_title_is_main_event(self) -> None:
        """Five-round non-title fights (main/co-main events) are main events."""
        acc = ExperienceAccumulator()
        acc.update_with_context(
            _outcome(bout_order=5, weight_class="Lightweight"),
            time_format="5 Rnd (5-5-5-5-5)",
        )
        snapshot = acc.freeze()
        state = snapshot.get(FIGHTER_A)
        assert state is not None
        assert state.main_event_fights == 1


# ---------------------------------------------------------------------------
# Full emitter output with accumulated history
# ---------------------------------------------------------------------------


class TestEmitterOutput:
    """Integration tests for the full ExperienceEmitter output."""

    def test_accumulated_title_fight_experience(self) -> None:
        career_acc = CareerAccumulator()
        exp_acc = ExperienceAccumulator()

        # Two title fights
        for i in range(1, 3):
            outcome = _outcome(
                fight_url=f"http://fight/{i}",
                event_url=f"http://event/{i}",
                event_date=date(2024, i, 1),
                weight_class="Lightweight Title Bout",
                winner_url=FIGHTER_A,
            )
            career_acc.update(outcome)
            exp_acc.update_with_context(outcome, time_format="5 Rnd (5-5-5-5-5)")

        emitter = ExperienceEmitter()
        ctx = _build_context(
            career_acc.freeze(),
            exp_acc.freeze(),
            fighter_url=FIGHTER_A,
            event_date=date(2024, 6, 1),
        )
        result = emitter.emit(ctx)
        assert result["title_fight_experience"] == 2.0
        assert result["has_been_champion"] == 1.0
        assert result["five_round_experience"] == 2.0

    def test_five_round_win_pct_calculation(self) -> None:
        career_acc = CareerAccumulator()
        exp_acc = ExperienceAccumulator()

        # Win one five-round fight
        outcome1 = _outcome(
            fight_url="http://fight/1",
            event_date=date(2024, 1, 1),
            weight_class="Lightweight Title Bout",
            winner_url=FIGHTER_A,
        )
        career_acc.update(outcome1)
        exp_acc.update_with_context(outcome1, time_format="5 Rnd (5-5-5-5-5)")

        # Lose another five-round fight
        outcome2 = _outcome(
            fight_url="http://fight/2",
            event_url="http://event/2",
            event_date=date(2024, 3, 1),
            weight_class="Welterweight Title Bout",
            winner_url=FIGHTER_B,
        )
        career_acc.update(outcome2)
        exp_acc.update_with_context(outcome2, time_format="5 Rnd (5-5-5-5-5)")

        emitter = ExperienceEmitter()
        ctx = _build_context(
            career_acc.freeze(),
            exp_acc.freeze(),
            fighter_url=FIGHTER_A,
            event_date=date(2024, 6, 1),
        )
        result = emitter.emit(ctx)
        assert result["five_round_experience"] == 2.0
        assert result["five_round_win_pct"] == 0.5

    def test_protocol_compliance(self) -> None:
        """Emitter has name attribute and emit returns dict."""
        emitter = ExperienceEmitter()
        assert emitter.name == "experience"

    def test_no_five_round_fights_returns_none_for_pct(self) -> None:
        career_acc = CareerAccumulator()
        exp_acc = ExperienceAccumulator()
        outcome = _outcome(
            event_date=date(2024, 1, 1),
            weight_class="Lightweight",
            winner_url=FIGHTER_A,
        )
        career_acc.update(outcome)
        exp_acc.update_with_context(outcome, time_format="3 Rnd (5-5-5)")

        emitter = ExperienceEmitter()
        ctx = _build_context(
            career_acc.freeze(),
            exp_acc.freeze(),
            fighter_url=FIGHTER_A,
            event_date=date(2024, 6, 1),
        )
        result = emitter.emit(ctx)
        assert result["five_round_win_pct"] is None
