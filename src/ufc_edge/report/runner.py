"""End-to-end report-generation runner for the mismatch-report pipeline.

Ties the report primitives together into one append-only pipeline execution:
resolve fight/market links, load the calibrated model's reliability artifact from
MLflow, compute the model-vs-market mismatch, gate it, tag sparse history, persist
paper signals, then annotate FLAGGED signals with due diligence and schedule
post-signal snapshots. Market fields are used only for comparison here and are
never passed into model inference.

The calibrated inference function is injected as a port (mocked at the integration
boundary per the spec); the runner itself consumes the MLflow run directly only for
the reliability (bucket) artifact and provenance parameters. Wiring predict_symmetric
to a booster/calibrator and serve-time features belongs to the CLI layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime

import duckdb
import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

from ufc_edge.data.schemas import _FrozenModel
from ufc_edge.report.due_diligence import (
    DueDiligenceError,
    LLMClient,
    SearchClient,
    run_due_diligence,
)
from ufc_edge.report.matching import resolve_links
from ufc_edge.report.mismatch import apply_gate, compute_mismatch
from ufc_edge.report.schemas import (
    GateVerdict,
    MarketFightLink,
    MatchStatus,
    PaperSignal,
    ReportRun,
)
from ufc_edge.report.snapshots import schedule_snapshots
from ufc_edge.report.sparse_history import tag_sparse_history
from ufc_edge.report.storage import write_paper_signal, write_report_run

_LOGGER = logging.getLogger(__name__)

# A calibrated, symmetric matchup predictor: given the two fighter URLs and the
# fight URL, returns the model's win probability for the market's referenced
# outcome, or None when a prediction is unavailable. Market data is never passed.
MatchupPredictor = Callable[[str, str, str], float | None]

# Top-level MLflow artifacts a complete model run must expose before a report may
# be written. "model" is the booster directory; the other two are files.
_REQUIRED_ARTIFACTS: tuple[str, ...] = ("reliability.json", "model", "calibrator.pkl")

# Latest order-book snapshot at or before the report cutoff for one token.
# Static SQL with bound parameters — no external input reaches the string.
_LATEST_SNAPSHOT_SQL = """
    SELECT market_id, mid_price, best_bid, best_ask, best_bid_size,
           best_ask_size, captured_at, tick_id
    FROM order_book_snapshots
    WHERE token_id = ? AND captured_at <= ?
    ORDER BY captured_at DESC
    LIMIT 1
"""


class MissingArtifactError(RuntimeError):
    """Raised when the MLflow run lacks a required model/calibrator/bucket artifact.

    A report without the calibrated model and its reliability (bucket) artifact is
    meaningless, so the runner raises before writing any row.
    """


class UpcomingFight(_FrozenModel):
    """One upcoming bout to evaluate. Names/date/event-start feed signal rows.

    fighter_a_url/fighter_b_url/fight_url are the identifiers routed to matching
    and inference; the remaining fields are recorded on the paper signal.
    """

    fight_url: str
    fighter_a_url: str
    fighter_b_url: str
    fighter_a_name: str
    fighter_b_name: str
    weight_class: str | None = None
    event_date: date | None = None
    event_start_time: datetime | None = None


@dataclass(frozen=True)
class BucketEntry:
    """One calibration bucket adapted from the MLflow reliability artifact.

    Satisfies the structural BucketEntryLike protocol consumed by apply_gate.
    """

    bucket_id: str
    lower: float
    upper: float
    n: int
    calibration_error: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class BucketArtifact:
    """Concrete bucket artifact (list of BucketEntry) satisfying BucketArtifactLike."""

    buckets: list[BucketEntry]


@dataclass(frozen=True)
class RunnerConfig:
    """Resolved report configuration the runner needs (from Hydra, high-level)."""

    gate_k: float
    sparse_history_threshold: int
    due_diligence_enabled: bool
    due_diligence_prompt_version: str
    due_diligence_model_name: str
    due_diligence_model_version: str


@dataclass(frozen=True)
class _MarketState:
    """Comparison-time market view for a matched token; never reaches inference."""

    market_id: str | None
    p_market_mid: float | None
    best_bid: float | None
    best_ask: float | None
    best_bid_size: float | None
    best_ask_size: float | None
    snapshot_timestamp: datetime | None
    tick_id: str | None


@dataclass(frozen=True)
class _RunContext:
    """Run-scoped inputs and injected dependencies threaded through bout evaluation.

    Bundles the values constant for one report execution so per-bout helpers keep
    small signatures. This is a dependency/context bundle, not a persisted model.
    """

    report_run_id: str
    mlflow_run_id: str
    as_of: datetime
    feature_version: str
    data_revision: str
    config_hash: str
    created_at: datetime
    config: RunnerConfig
    bucket_artifact: BucketArtifact
    predict_matchup: MatchupPredictor
    new_id: Callable[[], str]


def _utcnow() -> datetime:
    """Return current UTC time as a naive datetime (DuckDB-compatible)."""
    return datetime.now(UTC).replace(tzinfo=None)


def _new_id() -> str:
    """Return a fresh UUID4 string for a run or signal identifier."""
    return str(uuid.uuid4())


def run_report(
    as_of: datetime,
    mlflow_run_id: str,
    fights: list[UpcomingFight],
    config: RunnerConfig,
    conn: duckdb.DuckDBPyConnection,
    *,
    predict_matchup: MatchupPredictor,
    llm_client: LLMClient,
    search_client: SearchClient,
    now: Callable[[], datetime] = _utcnow,
    new_id: Callable[[], str] = _new_id,
) -> ReportRun:
    """Execute one append-only report run over the given upcoming bouts.

    Behavior: verifies the MLflow run's artifacts, loads the bucket artifact,
    resolves links, evaluates each bout (matched bouts get model/market/gate
    columns; others get null probability fields), writes the run and signals,
    then annotates FLAGGED signals with due diligence and schedules snapshots.
    Params: as_of cutoff; mlflow_run_id for the calibrated model run; fights to
    score; config thresholds; conn DuckDB; injected predictor/LLM/search clients;
    now/new_id factories for deterministic tests. Returns the persisted ReportRun.
    Errors: MissingArtifactError before any write if a required artifact is absent.
    """
    # Preconditions first: never write a partial report from an incomplete run.
    _require_run_artifacts(mlflow_run_id)
    bucket_artifact = load_bucket_artifact(mlflow_run_id)
    feature_version, data_revision = _load_run_provenance(mlflow_run_id)

    # Resolve (and persist) fight/market links; idempotent for already-matched rows.
    links = resolve_links([_link_input(fight) for fight in fights], as_of, conn)
    link_by_fight = {link.fight_url: link for link in links}

    ctx = _RunContext(
        report_run_id=new_id(),
        mlflow_run_id=mlflow_run_id,
        as_of=as_of,
        feature_version=feature_version,
        data_revision=data_revision,
        config_hash=_config_hash(config),
        created_at=now(),
        config=config,
        bucket_artifact=bucket_artifact,
        predict_matchup=predict_matchup,
        new_id=new_id,
    )

    # Build every signal in memory first so the run's counts are known up front.
    signals = [_evaluate_bout(fight, link_by_fight[fight.fight_url], conn, ctx) for fight in fights]
    flagged = [signal for signal in signals if signal.gate_verdict == GateVerdict.FLAGGED]

    # Persist the run before its signals (FK), then the signals themselves.
    report_run = ReportRun(
        report_run_id=ctx.report_run_id,
        as_of_timestamp=as_of,
        mlflow_run_id=mlflow_run_id,
        data_revision=data_revision,
        feature_version=feature_version,
        config_hash=ctx.config_hash,
        bout_count=len(fights),
        flagged_count=len(flagged),
        created_at=ctx.created_at,
    )
    write_report_run(conn, report_run)
    for signal in signals:
        write_paper_signal(conn, signal)

    # Annotate-only: due diligence + snapshot scheduling never suppress a signal.
    fight_by_url = {fight.fight_url: fight for fight in fights}
    for signal in flagged:
        fight = fight_by_url[signal.fight_url]
        _annotate_flagged(signal, fight, ctx, llm_client, search_client, conn)
        schedule_snapshots(signal, fight.event_start_time, conn)

    _LOGGER.info(
        "Report run %s complete: %d bouts, %d flagged.",
        ctx.report_run_id,
        len(fights),
        len(flagged),
    )
    return report_run


def _link_input(fight: UpcomingFight) -> dict[str, str]:
    """Project an UpcomingFight to the url-triple dict resolve_links expects."""
    return {
        "fight_url": fight.fight_url,
        "fighter_a_url": fight.fighter_a_url,
        "fighter_b_url": fight.fighter_b_url,
    }


def _require_run_artifacts(mlflow_run_id: str) -> None:
    """Raise MissingArtifactError unless the run exposes all required artifacts.

    Checks presence only (no loading) so an incomplete run fails before writes.
    """
    present = {info.path for info in _mlflow_client().list_artifacts(mlflow_run_id)}
    missing = [name for name in _REQUIRED_ARTIFACTS if name not in present]
    if missing:
        raise MissingArtifactError(
            f"MLflow run {mlflow_run_id!r} is missing required artifacts: {missing}. "
            "A report requires a calibrated model, its calibrator, and the "
            "reliability (bucket) artifact; refusing to write a partial report."
        )


def load_bucket_artifact(mlflow_run_id: str) -> BucketArtifact:
    """Load the reliability artifact from MLflow and adapt it to a BucketArtifact.

    Reads runs:/<id>/reliability.json (ReliabilityBucket schema) and maps each
    entry to a BucketEntry, synthesizing bucket_id from the bounds and mapping
    n_fights to n. Assumes _require_run_artifacts has confirmed the file exists.
    """
    payload = mlflow.artifacts.load_dict(f"runs:/{mlflow_run_id}/reliability.json")
    buckets = [
        BucketEntry(
            bucket_id=_bucket_id(entry["lower"], entry["upper"]),
            lower=float(entry["lower"]),
            upper=float(entry["upper"]),
            n=int(entry["n_fights"]),
            calibration_error=float(entry["calibration_error"]),
            ci_lower=float(entry["ci_lower"]),
            ci_upper=float(entry["ci_upper"]),
        )
        for entry in payload.get("buckets", [])
    ]
    return BucketArtifact(buckets=buckets)


def _bucket_id(lower: float, upper: float) -> str:
    """Format a stable bucket id from its bounds, e.g. (0.1, 0.3) -> '0.1-0.3'."""
    return f"{lower:.1f}-{upper:.1f}"


def _load_run_provenance(mlflow_run_id: str) -> tuple[str, str]:
    """Return (feature_version, data_revision) from the run's params, else 'unknown'."""
    params = _mlflow_client().get_run(mlflow_run_id).data.params
    return params.get("feature_version", "unknown"), params.get("data_revision", "unknown")


def _mlflow_client() -> MlflowClient:
    """Return a client bound to the active MLflow tracking store."""
    return MlflowClient(tracking_uri=mlflow.get_tracking_uri())


def _config_hash(config: RunnerConfig) -> str:
    """Return a deterministic SHA-256 hex digest of the resolved runner config."""
    canonical = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evaluate_bout(
    fight: UpcomingFight,
    link: MarketFightLink,
    conn: duckdb.DuckDBPyConnection,
    ctx: _RunContext,
) -> PaperSignal:
    """Build one paper signal from a bout and its resolved link.

    Every bout gets sparse-history and match-status columns. Only MATCHED bouts
    receive model, market, and gate columns; all others keep null probability
    fields (no guessing on an unresolved market).
    """
    sparse = tag_sparse_history(
        fight.fighter_a_url,
        fight.fighter_b_url,
        fight.fight_url,
        conn,
        ctx.config.sparse_history_threshold,
    )
    fields: dict[str, object] = {
        "signal_id": ctx.new_id(),
        "report_run_id": ctx.report_run_id,
        "fight_url": fight.fight_url,
        "event_date": fight.event_date,
        "fighter_a_url": fight.fighter_a_url,
        "fighter_b_url": fight.fighter_b_url,
        "fighter_a_name": fight.fighter_a_name,
        "fighter_b_name": fight.fighter_b_name,
        "weight_class": fight.weight_class,
        "min_prior_ufc_fights": sparse.min_prior_ufc_fights,
        "sparse_history": sparse.sparse_history,
        "match_status": link.match_status,
        "mlflow_run_id": ctx.mlflow_run_id,
        "data_revision": ctx.data_revision,
        "feature_version": ctx.feature_version,
        "config_hash": ctx.config_hash,
        "created_at": ctx.created_at,
    }
    if link.match_status == MatchStatus.MATCHED:
        fields.update(_matched_fields(fight, link, conn, ctx))
    return PaperSignal(**fields)


def _matched_fields(
    fight: UpcomingFight,
    link: MarketFightLink,
    conn: duckdb.DuckDBPyConnection,
    ctx: _RunContext,
) -> dict[str, object]:
    """Compute model/market/gate columns for a matched bout.

    Fetches the latest market snapshot for comparison and the calibrated model
    probability from the injected predictor. Mismatch and gate are computed only
    when both a model probability and a market midpoint are available.
    """
    market = _latest_market_snapshot(conn, link.token_id, ctx.as_of)
    p_model = ctx.predict_matchup(fight.fighter_a_url, fight.fighter_b_url, fight.fight_url)

    matched: dict[str, object] = {"token_id": link.token_id, "p_model": p_model}
    if market is not None:
        matched.update(
            {
                "market_id": market.market_id,
                "p_market_mid": market.p_market_mid,
                "best_bid": market.best_bid,
                "best_ask": market.best_ask,
                "best_bid_size": market.best_bid_size,
                "best_ask_size": market.best_ask_size,
                "snapshot_timestamp": market.snapshot_timestamp,
                "tick_id": market.tick_id,
            }
        )

    if p_model is not None and market is not None and market.p_market_mid is not None:
        mismatch = compute_mismatch(p_model, market.p_market_mid)
        gate = apply_gate(mismatch, p_model, ctx.bucket_artifact, ctx.config.gate_k)
        matched.update(
            {
                "mismatch": mismatch,
                "gate_verdict": gate.verdict,
                "bucket_id": gate.bucket_id,
                "bucket_n": gate.bucket_n,
                "bucket_calibration_error": gate.bucket_calibration_error,
                "bucket_ci_lower": gate.ci_lower,
                "bucket_ci_upper": gate.ci_upper,
            }
        )
    return matched


def _latest_market_snapshot(
    conn: duckdb.DuckDBPyConnection,
    token_id: str,
    as_of: datetime,
) -> _MarketState | None:
    """Return the most recent order-book snapshot at or before as_of for a token.

    Falls back to a bid/ask midpoint when the stored mid_price is null. Returns
    None when no snapshot exists (leaving market columns null).
    """
    row = conn.execute(_LATEST_SNAPSHOT_SQL, [token_id, as_of]).fetchone()
    if row is None:
        return None

    market_id, mid_price, best_bid, best_ask, best_bid_size, best_ask_size, captured_at, tick_id = (
        row
    )
    if mid_price is None:
        mid_price = _mid_from_book(best_bid, best_ask)
    return _MarketState(
        market_id=market_id,
        p_market_mid=mid_price,
        best_bid=best_bid,
        best_ask=best_ask,
        best_bid_size=best_bid_size,
        best_ask_size=best_ask_size,
        snapshot_timestamp=captured_at,
        tick_id=tick_id,
    )


def _mid_from_book(best_bid: float | None, best_ask: float | None) -> float | None:
    """Return the bid/ask midpoint, or None if either side is missing."""
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def _annotate_flagged(
    signal: PaperSignal,
    fight: UpcomingFight,
    ctx: _RunContext,
    llm_client: LLMClient,
    search_client: SearchClient,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Run annotate-only due diligence for a flagged signal.

    The signal row is already persisted; this only adds an informational verdict.
    A due-diligence failure is logged (the runner records the failed attempt) and
    never blocks the report or suppresses the signal.
    """
    if not ctx.config.due_diligence_enabled:
        return

    checklist_context = {
        "fighter_a": fight.fighter_a_name,
        "fighter_b": fight.fighter_b_name,
        "event_date": fight.event_date.isoformat() if fight.event_date else "",
    }
    try:
        run_due_diligence(
            signal.fight_url,
            signal.report_run_id,
            checklist_context,
            llm_client,
            search_client,
            conn,
            prompt_version=ctx.config.due_diligence_prompt_version,
            model_name=ctx.config.due_diligence_model_name,
            model_version=ctx.config.due_diligence_model_version,
        )
    except DueDiligenceError:
        _LOGGER.warning(
            "Due-diligence failed for flagged signal %s; verdict left null, "
            "signal preserved (annotate-only).",
            signal.fight_url,
        )
