# UFCPolyML contributor guide

## Project scope

V1 is a batch applied-ML system:

`DATA → FEATURES → XGBoost + EVAL → mismatch report / paper-signal log`

The model is strictly odds-free. Market probability is a report comparator, never
a feature or training label. No deep learning, LLM enrichment of training data,
automated trading, Kelly sizing, RL/OPE, CLOB simulation, or online learning
belongs in v1. The LLM enters only as an annotate-only due-diligence layer at
signal time (D32).

## Non-negotiable correctness rules

- A feature for fight `X` may use only information available strictly before `X`.
- Training, calibration, and evaluation use event-grouped temporal splits; never
  random row splits.
- Calibration data must be disjoint from model-training data and precede test data.
- Features are emitted before a fight updates any accumulator.
- Validation failures are quarantined with reason codes; source rows are never
  silently dropped or auto-corrected.
- XGBoost consumes no historical or live market-derived feature.
- A served matchup probability must be symmetric: reversing fighter order returns
  exactly `1 - p`.

## Engineering conventions

- Python 3.12 and `uv`; use `uv run`, never global `pip` installs.
- DuckDB is the single datastore. Hydra holds configuration; no magic constants.
- Public functions are typed. Use frozen Pydantic models at external boundaries.
- Prefer small, named functions with one responsibility. Format with `ruff`.
- Run fixture-based tests only; scraper tests must never require the live site.
- Never commit secrets, DuckDB files, or raw data blobs.
- MLflow is the sole artifact store for model runs (D27).

## Skills rule

Every agent working in this repo MUST read and follow
`.kiro/skills/code-conventions/SKILL.md` before writing any code. It defines
naming, structure, commenting, test, and design conventions that are binding.

## Commit rules

- One commit per completed task, immediately after that task's verification passes
  (build + tests green).
- Commit messages follow Conventional Commits with the task ID in the subject:
  `feat(features): Implement EloState component [FE-1.1]`
- Never batch multiple tasks into one commit.
- Never commit to `main` directly; only the owner/orchestrator merges to `main`.

## Branching rules

Work happens on three long-lived spec branches:

| Branch | Spec | Owned paths |
|---|---|---|
| `feature-engine` | Feature Engine | `src/ufc_edge/features/`, `tests/features/`, `configs/graph.yaml` |
| `model-and-eval` | Model and Evaluation | `src/ufc_edge/model/`, `src/ufc_edge/eval/`, `tests/model/`, `tests/eval/`, `configs/model/`, `configs/eval/` |
| `mismatch-report` | Mismatch Report | `src/ufc_edge/report/`, `tests/report/`, `configs/report/` |

Each spec's tasks touch only files that spec owns (see ownership table in
`.kiro/specs/reference/spec-orchestration-brief.md`). Cross-spec integration tasks
and merges to `main` happen at wave gates defined in each spec's `tasks.md`
timeline.

## Documentation rules

- `.kiro/specs/DECISIONS.md` is the single decision record.
- `.kiro/specs/FEATURES.md` is the human-owned feature registry.
- `.kiro/specs/feature-engine/` — feature engine requirements, design, tasks.
- `.kiro/specs/model-and-eval/` — model training, calibration, evaluation spec.
- `.kiro/specs/mismatch-report/` — mismatch report and paper-signal log spec.
- `.kiro/specs/reference/` — orchestration brief, style guide, legacy registry.

Do not invent feature ideas; mark unclear definitions `TODO(human)`.

## Current status

| Area | Status |
|---|---|
| Data ingestion, storage, validation | Built |
| Polymarket market capture | Built; production cron deployed on Fly |
| Feature engine spec | Complete (requirements, design, tasks) |
| Model and evaluation spec | Complete (requirements, design, tasks) |
| Mismatch report spec | Complete (requirements, design, tasks) |
| Implementation | Not started (Wave 0 next) |

## Common commands

```bash
make setup
make validate
make test
make lint
make format
make scrape
make capture
make backup
```
