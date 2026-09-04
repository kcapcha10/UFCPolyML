"""Due-diligence eval harness: measures how often the due-diligence step's
verdicts match what a human reviewer would have concluded, so the component
earns trust through a measured track record instead of being assumed reliable.

Runs the production due-diligence runner on a hand-labeled fixture set,
computes precision and recall treating CONFIRM/QUALIFY as "concern flagged"
vs. VETO as "no concern", and gates at precision >= 0.80 and recall >= 0.60.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from ufc_edge.report.due_diligence import LLMClient, SearchClient, run_due_diligence
from ufc_edge.report.schemas import DueDiligenceVerdictType

# ── Data structures ───────────────────────────────────────────────────────────

_PRECISION_THRESHOLD = 0.80
_RECALL_THRESHOLD = 0.60


@dataclass(frozen=True)
class LabeledFight:
    """One hand-labeled fixture entry for eval."""

    fight_url: str
    event_date: str
    fighter_a: str
    fighter_b: str
    ground_truth_concern: bool
    notes: str


@dataclass(frozen=True)
class EvalResult:
    """Outcome of running the eval harness on a labeled fixture set."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    total: int


# ── Public API ────────────────────────────────────────────────────────────────


def load_labels(fixture_path: Path) -> list[LabeledFight]:
    """Load labeled fights from a JSON fixture file.

    Each entry needs: fight_url, event_date, fighter_a, fighter_b,
    ground_truth_concern (bool), notes.
    """
    raw = json.loads(fixture_path.read_text())
    labels = []
    for entry in raw:
        if "_comment" in entry and "fight_url" not in entry:
            continue
        labels.append(
            LabeledFight(
                fight_url=entry["fight_url"],
                event_date=entry["event_date"],
                fighter_a=entry["fighter_a"],
                fighter_b=entry["fighter_b"],
                ground_truth_concern=entry["ground_truth_concern"],
                notes=entry.get("notes", ""),
            )
        )
    return labels


def _verdict_signals_concern(verdict: DueDiligenceVerdictType) -> bool:
    """Map a due-diligence verdict to a binary concern flag.

    QUALIFY and VETO both indicate the LLM found something noteworthy.
    CONFIRM means no concern detected.
    """
    return verdict in (DueDiligenceVerdictType.QUALIFY, DueDiligenceVerdictType.VETO)


def run_eval(
    labels: list[LabeledFight],
    llm_client: LLMClient,
    search_client: SearchClient,
    conn: duckdb.DuckDBPyConnection,
) -> EvalResult:
    """Run the production due-diligence pipeline on each labeled fight and
    compute precision/recall against ground-truth concern labels.

    Uses the same run_due_diligence code path as production — no separate
    reimplementation of prompt construction, parsing, or schema validation.
    The eval run uses a fixed report_run_id prefix so verdicts are isolated
    from real report runs.
    """
    tp = fp = tn = fn = 0

    for label in labels:
        checklist_context = {
            "fighter_a": label.fighter_a,
            "fighter_b": label.fighter_b,
            "event_date": label.event_date,
        }

        verdict_obj = run_due_diligence(
            fight_url=label.fight_url,
            report_run_id="eval-harness",
            checklist_context=checklist_context,
            llm_client=llm_client,
            search_client=search_client,
            conn=conn,
        )

        predicted_concern = _verdict_signals_concern(verdict_obj.verdict)
        actual_concern = label.ground_truth_concern

        if predicted_concern and actual_concern:
            tp += 1
        elif predicted_concern and not actual_concern:
            fp += 1
        elif not predicted_concern and actual_concern:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return EvalResult(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        total=len(labels),
    )


def gate_check(eval_result: EvalResult) -> bool:
    """Return True if the eval result meets minimum quality thresholds.

    Precision must be >= 0.80 and recall must be >= 0.60 for the
    due-diligence component to pass its quality gate.
    """
    return eval_result.precision >= _PRECISION_THRESHOLD and eval_result.recall >= _RECALL_THRESHOLD
