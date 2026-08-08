# Spec-writing orchestration brief

Three specs under `.kiro/specs/`: `feature-engine/`, `model-and-eval/`,
`mismatch-report/`. Each contains `requirements.md`, `design.md`, `tasks.md`.
Match the structure and DENSITY of `.kiro/specs/reference/spec-style-guide.md`
(extracted from the owner's exemplar). Decisions D28–D39 in
`/Users/kcapcha/UFCPolyML/.kiro/specs/DECISIONS.md` are BINDING — specs implement them,
never relitigate them.

## Shared context files (read all)

- `.kiro/specs/reference/spec-style-guide.md` — format + density contract
- `.kiro/specs/DECISIONS.md` — all decisions incl. D28–D39 (recent, at end of file)
- `.kiro/specs/DESIGN.md`, `.kiro/specs/REQUIREMENTS.md` (historical; superseded by .kiro/specs/, recoverable from git history)
- Code audit findings (see existing codebase under `src/ufc_edge/` for what code exists today)
- `.kiro/specs/reference/legacy-features-registry.md` — owner's feature definitions + v1 triage (feature-engine spec especially)

## Requirement ID namespaces (no collisions)

- feature-engine: `FE-1`, `FE-2`, … acceptance criteria `FE-1.1`, `FE-1.2`, …
- model-and-eval: `ME-…`
- mismatch-report: `MR-…`

Cross-spec references use the full ID (e.g., a mismatch-report requirement may
depend on `ME-4.2`).

## Cross-spec interface contract (each spec treats the others as black boxes)

1. **feature-engine OWNS:** the replay engine, state components, emitters,
   `features_v{N}` DuckDB table (wide, keyed by `fight_url` + `fighter_url`,
   NULL→NaN contract), `FEATURE_VERSION` + source-hash guard, graph configs
   (`configs/graph.yaml`: Elo, Glicko-2, PageRank), Kaggle validation protocol
   (D30), regenerated `.kiro/specs/FEATURES.md` (from legacy registry + D39 — verbatim
   definitions, nothing invented; owner reviews before merge).
2. **model-and-eval OWNS:** matrix assembler (rejects market-derived columns),
   XGBoost training (fixed rounds, ~12 candidates, 8-axis surface per D19/D25),
   symmetric inference (D19–D21), calibrator selection per fold (D33),
   stratified reliability artifact (D33), ablation ladder (D29: naive → record →
   +physical → +schedule-strength(Elo+Glicko-2+PageRank+common-opp) → +domain
   interactions), market-relative eval (D28), power analysis + paired
   permutation test + cumulative tracking (D35), history-depth-stratified
   metrics (D37), MLflow logging (D27). CONSUMES: `features_v{N}`,
   `market_fight_links` (for D28 historical market Brier).
3. **mismatch-report OWNS:** `market_fight_links` table + name matching + human
   confirmation flow (D38), mismatch computation, magnitude gate + bucket
   transparency columns (D34), `SPARSE_HISTORY` tagging (D37), paper-signal log
   with bid/ask/depth + post-signal snapshots at ~T+1h/4h/24h/fight (D36),
   LLM due-diligence service: schema, runner, eval harness (30–50 labeled
   fights, P≥0.80/R≥0.60 gate), verdict scoreboard, annotate-only (D32),
   `report_runs`/`paper_signals` persistence (existing DESIGN.md §7 semantics).
   CONSUMES: calibrated probabilities + bucket calibration-error table from
   model-and-eval; capture snapshots from existing polymarket module (schema
   extension for bid/ask/depth is an MR task).

## tasks.md concurrency contract (all three specs)

The owner builds with parallel AI subagents (orchestrated workflow). tasks.md
MUST be organized as an execution timeline:

- Group tasks into **Waves** (Wave 0, Wave 1, …). Everything inside a wave is
  safe to run concurrently by independent subagents (no shared files, no
  dependency edges).
- Every task carries: `Depends on:` (task IDs, possibly cross-spec, or "none"),
  `Files:` (paths it creates/modifies — used to prove non-overlap within a
  wave), `_Requirements: <IDs>_` traceability per the style guide.
- Tasks are sized for one subagent session: one module or one coherent test
  suite per task; every implementation task pairs with its test task (TDD:
  test task explicitly first or same task with test-first bullets).
- Each spec's tasks.md ends with a **Timeline summary** table: wave → tasks →
  what runs in parallel → cross-spec joins (e.g., "ME Wave 0 can start
  immediately; ME Wave 2 blocks on FE Wave 3 (features_v1 materialized)").
- Include explicit verification tasks (run `make test`, `make lint`; leakage
  tests; determinism tests) as first-class checkboxes, not afterthoughts.

## Known cross-spec dependency spine (encode it, don't rediscover it)

- FE Wave 0 (scaffolding, protocols, configs) → nothing blocks it.
- ME Waves 0–1 (assembler interface against a FIXTURE feature table, metrics,
  calibrators, power analysis — all testable without real features) → can run
  parallel to FE.
- MR Waves 0–1 (link table + matching + capture schema extension + LLM service
  scaffolding with stubbed model probabilities) → can run parallel to both.
- Real-data joins happen late: ME training on `features_v1` blocks on FE
  completion; MR report generation blocks on ME calibrated model + bucket
  table; MR historical market Brier feed blocks on MR link table + ME eval.
- 5a card-position features: FE contains a verification task "confirm scraper
  captures card position / bout ordering; if absent, add scraper field +
  fixture" BEFORE the 5a emitter task.

## Style guardrails

- EARS-style SHALL acceptance criteria with UPPERCASE WHEN/IF/WHERE, per style guide
- design.md names exact files, classes, functions, DuckDB DDL, mermaid diagrams,
  correctness properties tagged `Validates: Requirements <ID>`
- No invented features, no scope beyond D23+D28–D39
- Python 3.12, uv, DuckDB, Hydra, frozen Pydantic at boundaries, fixture-based
  tests only (per AGENTS.md)
