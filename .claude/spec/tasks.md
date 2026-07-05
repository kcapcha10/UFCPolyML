# tasks.md — ufc-edge build plan

> **Living document.** Tasks enter this list only when their design section has
> locked (see [Key Design Decisions.md](Key%20Design%20Decisions.md)). Tags:
> `[LAYER]` = DATA/FEATURES/REPS/PREDICT/STRATEGY/SIM/ONLINE/EVAL/OPS, plus kind:
> `[build]`, `[test]`, `[experiment]`, `[infra]`.

## Task list

### EVAL spine (design locked — D10)

- [ ] **T-E1** `[EVAL][build]` `WalkForwardSplitter` + `Fold` model in
      `src/ufc_edge/eval/` (expanding window, event-grouped, deterministic)
  - [ ] T-E1a `[EVAL][test]` P3 property test — hypothesis-generated event
        calendars; ordering, card-integrity, determinism
- [ ] **T-E2** `[EVAL][build]` Calibration stage: `fit_calibrator` with
      Platt/isotonic switch at `ISOTONIC_MIN_SAMPLES` (Hydra-config)
  - [ ] T-E2a `[EVAL][test]` P3 clause 1 applied to calibration slice
- [ ] **T-E3** `[EVAL][build]` Metrics module: Brier / log-loss / ECE /
      reliability, per-fold + pooled; event-level cluster bootstrap CIs
- [ ] **T-E5** `[EVAL][infra]` MLflow run wrapper logging DVC rev, feature
      version, resolved config, seeds, library versions
- [ ] **T-E4** `[EVAL][experiment]` Era-drift ablation (pooled Brier vs.
      training-window length) — *blocked on FEATURES + PREDICT existing*

### DATA — validation suite (design locked — D8/D11/D12)

- [x] **T-D1** `[DATA][build]` Validation package: invariants, quarantine,
      runner, report; `make validate` CI gate (2026-07-05)
- [x] **T-D1a** `[DATA][test]` Planted-violation tests: every invariant family,
      era scoping, quarantine refresh (12 tests, 2026-07-05)
- [x] **T-D2** `[DATA][experiment]` First real run (2026-07-05, post-backfill).
      Verdicts: `BREAKDOWN_PARTITION_MISMATCH` fired zero times on 58,660 rows —
      **kept**. Two invariants were miscalibrated, not the data: overtime
      formats ('1 Rnd + OT') now skipped by the ending-round check (29 false
      positives), weight upper bound raised to 360 kg (Yarbrough, 349.3 kg —
      real). After recalibration: **0 violations database-wide**.
- [ ] **T-D3** `[DATA][build]` Quarantine anti-join helper for the FEATURES
      layer — *blocked on FEATURES design (Decision #4)*
- [ ] **T-D4** `[DATA][experiment]` UFC 1 (1993-11-12, "The Beginning") is
      absent from the scrape — earliest event is UFC 2. Determine whether
      ufcstats omits it from the completed-events list or the spider's list
      parsing skips a row; feature-history-only impact (early accumulators).

### FEATURES — as-of replay engine (design COMPLETE — D13/D14/D15; buildable on user go)

- [ ] **T-F1** `[FEATURES][build]` Replay loop, two-phase tick, component
      protocol + registry (duplicate-name rejection), EmitContext
- [ ] **T-F2** `[FEATURES][test]` P1 deletion-oracle + determinism property
      tests — *same wave as T-F1, never later*
- [ ] **T-F3** `[FEATURES][build]` Materialized wide `features_v{N}` table +
      DVC versioning + quarantine anti-join (absorbs T-D3)
- [ ] **T-F4** `[FEATURES][build]` `FEATURE_VERSION` + lockfile hash guard
      pytest (D15)
- [ ] **T-F5** `[FEATURES][build]` Components per the design.md map (§1–§12;
      §5a deferred on card-position data gap; §9d excluded pending TODO(human))

### OPS (running / user-owned)

- [x] **T-O1** `[OPS]` tick_id column implemented + verified locally (2026-07-04)
- [ ] **T-O2** `[OPS]` `(user)` Fly volume extend + deploy (one restart covers both)
- [ ] **T-O3** `[OPS]` `(user)` GDrive OAuth → first `make backup` push
- [ ] **T-O4** `[OPS]` ufcstats backfill (running since 2026-07-05; resumable)

## Notes

- **Design order ≠ diagram order.** LLD decisions proceed in dependency/risk
  order per D9/D10: EVAL → DATA-validation → FEATURES → PREDICT (completing the
  XGBoost end-to-end slice) → STRATEGY v1 → SIM → REPS → ONLINE. The diagram
  order (REPS before PREDICT) would design the DL encoders before a baseline
  number exists — exactly what D9 rejected.
- Capture cron is P0; nothing in this list may interrupt it.
- T-E1/E3/E5 are buildable now against synthetic data; T-E2 needs only synthetic
  raw scores. None of the EVAL wave waits on the scrape finishing.

## Task Dependency Graph

```json
{
  "completed": ["T-O1", "T-O2", "T-O4", "T-D1", "T-D1a", "T-D2"],
  "waves": [
    { "wave": 0, "tasks": ["T-E1", "T-E3", "T-E5", "T-F1", "T-F4", "T-O3"],
      "note": "fully parallel; T-F1 and T-F2 land together (P1 in the same wave); T-O3 is user-only (GDrive OAuth)" },
    { "wave": 1, "tasks": ["T-E1a", "T-E2", "T-F2", "T-F3"],
      "blockedBy": ["T-E1", "T-F1"] },
    { "wave": 2, "tasks": ["T-E2a", "T-F5"], "blockedBy": ["T-E2", "T-F1", "T-F3"] },
    { "wave": "later", "tasks": ["T-E4", "T-D4"],
      "blockedBy": ["PREDICT design (Decision #5)", "T-F5"] }
  ]
}
```
