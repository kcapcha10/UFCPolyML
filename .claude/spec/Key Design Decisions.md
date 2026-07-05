# Key Design Decisions

The plain-language "why" record for the low-level design. One entry per **key**
decision — the calls that shape the system, not every engineering choice (those go
in [DECISIONS.md](../../docs/DECISIONS.md)). Numbering continues from D1–D7 in
[DECISIONS_architecture_redesign.md](../../docs/DECISIONS_architecture_redesign.md) so the
whole project shares one decision namespace.

Each entry answers: why do we want this, how do we accomplish it, what does it
replace, and how the day-to-day process changes (Before → After).

---

## D8 — Drop Kaggle; we validate our own data and (later) scrape our own pre-UFC history

**Decided:** 2026-07-04 · **Owner:** human

### Why do we want this
Kaggle played two roles, and it was quietly bad at both. As a *cross-check* for our
scraper it looked independent but isn't — the popular Kaggle UFC datasets are
themselves scrapes of ufcstats.com, so comparing against them mostly re-tests the
same source. As a *pre-UFC data source* it can't deliver at all: ufcstats-derived
sets only contain UFC fights. On top of that, their precomputed columns have known
leakage problems (career stats that include the current fight), and while our
pipeline would never have consumed those columns, depending on a dataset whose
authors made that mistake is a bad foundation for a project whose whole identity is
"we don't leak."

### How we accomplish it
- **Cross-check replacement:** a self-consistency validation suite in
  `src/ufc_edge/data/validation/` — internal invariants (per-round stats sum to
  fight totals, win/loss records reconcile across pages, event fight counts match)
  plus spot-checks against genuinely independent sources (e.g. Wikipedia event
  pages). Designed in LLD v1; runs in CI forever, catching parser regressions the
  Kaggle diff would have caught and more.
- **Pre-UFC data:** a self-scraped source (Sherdog or Tapology) as a **separate
  stretch milestone**, not part of LLD v1. Its hard problem is fighter-identity
  matching across sites (two UFC "Bruno Silva"s exist) — a silent-corruption risk
  that gets its own design pass.
- **Immediate registry consequences (human-decided):** `natural_weight_class` is
  redefined as *first observed UFC weight class*; the four `pre_ufc_*` features in
  FEATURES.md §3a are deferred until the pre-UFC source exists.

### What it replaces
The `kaggle_crosscheck` stage in `dvc.yaml` and the never-implemented
`src/ufc_edge/data/kaggle.py`, plus the assumption that Kaggle would someday supply
pre-UFC records.

### Change to process (Before → After)
- **Before:** data quality = "someday diff against Kaggle." Concretely: a parser bug
  that halves takedown counts would only surface if the Kaggle diff ran, was fresh,
  and the same bug wasn't in the Kaggle scraper too.
- **After:** data quality = invariants that run on every test pass. The same
  takedown bug fails `round sums == fight totals` immediately, with no external
  dependency. A new scrape is trusted only after the validation suite passes on it.
- **Use case:** before training the first model, we run the suite over the full
  DuckDB; any fighter whose record doesn't reconcile is quarantined from the
  training set instead of silently corrupting features.

---

## D9 — Build XGBoost-first end-to-end, with two contract locks that keep the deep-learning path open

**Decided:** 2026-07-04 · **Owner:** human · Resolves the open "D3 sequencing" item
in the architecture redesign.

### Why do we want this
The single most important number in this project is a **real walk-forward
calibration curve** — it tells us whether the edge thesis is alive. Every week spent
building multi-model plumbing before that number exists is a week of risk that the
plumbing serves a thesis that doesn't work. There's also an interface-quality
argument: an abstraction designed with one implementation is a guess; extracting it
once a second model actually exists shapes it with evidence. Research backs the
ordering: on medium-sized tabular data (~10k rows — we have ~7k fights), gradient
boosted trees still beat deep learning (Grinsztajn et al., NeurIPS 2022; Shwartz-Ziv
& Armon 2022), so XGBoost is the right *first* champion, not a throwaway.

**Deep learning is not cut — it is sequenced.** The REPS layer (sequence encoder +
fight-graph GNN) remains the project's DL centerpiece; this decision makes its
eventual entry *credible* by giving it a strong, calibrated baseline to beat under
walk-forward evaluation. That baseline-then-ablate discipline is how quant research
teams actually admit model complexity.

### How we accomplish it
One concrete pipeline: features → XGBoost → calibration → walk-forward eval. No
`Predictor` protocol, no model registry yet. But the LLD locks two **data
contracts** now, because they're the places XGBoost-isms would otherwise silently
ossify:

1. **Raw scores and calibration stay separate stages** (already invariant #3 of the
   architecture). The model never hands strategy anything but calibrated `P(win)`
   through the shared calibration step.
2. **The feature matrix documents its missingness semantics explicitly.** FEATURES.md
   leans on "return NaN, XGBoost learns the default branch" — that's a *contract on
   the data*, not a model detail. Writing it down means a future MLP adds an
   imputation adapter instead of renegotiating the feature layer.

### What it replaces
The alternative sequencing (multi-model `Predictor` interface, Hydra model registry,
and embedding plumbing built before any model trains). Deferred, not rejected — the
interface gets extracted when model #2 (the REPS-fed MLP) arrives.

### Change to process (Before → After)
- **Before (interface-first world):** first weeks go to protocol design, registry
  config, and embedding-shaped placeholders; the calibration curve arrives late and
  any interface misjudgment is discovered after it's load-bearing.
- **After:** first milestone ends with a walk-forward Brier/ECE number for
  calibrated XGBoost. When REPS lands, the ablation is pre-registered: the
  encoder/GNN embeddings must beat that number to enter the model. If they don't,
  that ablation table *is* the finding — and the honest one.
- **Use case (interview framing):** "my Transformer had to beat my calibrated
  XGBoost under walk-forward evaluation to earn its place — here's the ablation"
  is a stronger deep-learning story than "I used a Transformer."
- **Accepted cost (eyes open):** when model #2 arrives there will be a refactor
  week to extract the interface that Option B would have prepaid.

---

## D10 — Evaluation protocol: expanding-window walk-forward, split by event, calibrated inside the fold

**Decided:** 2026-07-05 · **Owner:** human · First LLD decision; designed before any
model because it is the yardstick every other component is measured by.

### Why do we want this
The scariest failure in this project is a **plausible backtest number that is
quietly wrong** — the pipeline runs, the reliability curve looks beautiful, and
it's fake. The two quiet ways that happens: (1) the calibrator sees data from the
period it's later judged on, and (2) fights from the same card get split across
train and test, leaking same-night context. Both fail silently: nothing errors,
the metrics just get better than reality. With only ~7k fights we also can't
afford to throw data away, which is why the window expands instead of rolling.

### How we accomplish it
- **Splits:** expanding-window walk-forward. The unit of splitting is the
  **event (fight card), never the individual fight** — all fights on one card land
  on the same side of every split. Train = all events strictly before test block
  *k*; refit each block.
- **Calibration inside the fold:** the calibrator (Platt below ~1,000 calibration
  samples, isotonic above — isotonic overfits small samples, Niculescu-Mizil &
  Caruana 2005) is fit only on a trailing validation slice of that fold's training
  window. Never on anything from the test block. Pytest-enforced as correctness
  property P3.
- **Rider — era-drift ablation:** one deliberate experiment (pooled Brier vs.
  training-window length) so that switching to a rolling window would be a
  data-driven call, not a vibe.
- **Uncertainty:** confidence intervals via cluster bootstrap at the event level
  (fight-level resampling would pretend same-card fights are independent).

### What it replaces
- Naive k-fold or random splits (banned repo-wide already — invariant #4).
- Rolling fixed-width windows (kept as a fallback the drift ablation can trigger).
- Purged combinatorial CV (López de Prado) — built for finance's overlapping
  labels; fight labels resolve on event night, so its machinery is dead weight here.

### Change to process (Before → After)
- **Before:** "evaluate the model" was an intention with no protocol; any script
  could quietly fit isotonic on the full history and report a gorgeous ECE.
- **After:** there is exactly one splitter and one calibration stage, both
  property-tested (P3): for every fold, the latest timestamp used in training or
  calibration precedes the earliest test-event timestamp, and no card straddles
  the boundary. Every eval run logs data snapshot, feature version, config, and
  seeds to MLflow — same inputs, same numbers.
- **Use case:** when REPS lands, its ablation runs under this exact protocol, so
  "the GNN beat XGBoost" can't be an artifact of a friendlier split.

---

## D11 — Two data universes: train on 2010+, compute features from everything

**Decided:** 2026-07-05 · **Owner:** human (cutoff year and the two-universe split
both human-locked)

### Why do we want this
Pre-modern UFC is a different sport (rules maturation, talent depth, event
cadence), so old fights make poor *training examples*. But stateful features —
career records, Elo, the fight graph — are accumulators: their value at fight X
depends on everything before X. Deleting old rows wouldn't just drop irrelevant
training data; it would truncate active careers mid-stream: a 2011 champion would
enter 2010s training data with a 2-fight record and a cold 1500 Elo, and XGBoost
would learn to distrust exactly the graph features the project is proudest of.
The failure is silent — nothing errors, the features are just fiction near the
boundary.

### How we accomplish it
- **Label universe:** only fights with `event_date >= label_start_date`
  (config `configs/data/default.yaml`, default **2010-01-01**) are trained on,
  calibrated on, and evaluated on.
- **Feature-history universe:** all scraped rows, all-time, feed feature
  computation and accumulator warm-up (Elo burn-in, graph construction, career
  stats). Nothing is deleted; the cutoff is a filter, not a purge.
- The D10 era-drift ablation sweeps `label_start_date` (e.g. 2008/2010/2012/2015),
  so the human's 2010 judgment call is empirically checkable later.

### What it replaces
The single-universe framing where one date governs both scraping/feature data and
training data ("we only want fights after 2010"), which contained the truncation
trap above.

### Change to process (Before → After)
- **Before:** "cut at 2010" would have meant scraping/keeping only modern rows —
  smaller DB, corrupted boundary-era features, cold-start Elo for everyone.
- **After:** the scraper backfills everything; FEATURES reads everything; only the
  training/eval joins apply the label filter. One config value moves the boundary.
- **Use case:** a 2012 fight between two 30-fight veterans gets features computed
  from both full careers, but a 2008 fight never appears as a training row.

---

## D12 — Validation failure semantics: quarantine with era-scoped alarms

**Decided:** 2026-07-05 · **Owner:** human · Completes D8 (the validation suite
that replaced Kaggle).

### Why do we want this
Hard-failing on any bad row means thirty-year-old data quirks permanently block
work — and the realistic human response, globally weakening invariants, guts the
suite. Silently dropping bad rows is worse: deletion that correlates with era or
fighter obscurity biases the training distribution exactly where the edge thesis
lives (obscure, short-notice fighters are where Polymarket misprices). So:
violating rows are **quarantined with reason codes**, never silently dropped and
never auto-"fixed", and the alarm that fails the build is **era-scoped**.

### How we accomplish it
- **Quarantine:** rows failing an *impossibility* invariant (e.g. landed >
  attempted) are recorded in a `validation_quarantine` table with reason codes
  and excluded from feature computation, regardless of era.
- **Impossible ≠ incomplete:** expected-missing data (e.g. no per-round stats
  before ~2001) is NOT a violation — conditional invariants skip what cannot be
  evaluated; the Missingness Policy handles absence.
- **Era-scoped alarm:** any violation in the label universe (`event_date >=
  label_start_date`, per D11) fails the suite outright (0% tolerance — modern
  violations mean parser bugs, and aggregate thresholds would hide a fresh
  regression for weeks). Pre-cutoff violations are report-only.
- **Anti-silent rider:** quarantine counts and reason codes surface in the
  validation report and in every MLflow eval run — quarantine without a visible
  report is silent data loss with extra steps.

### What it replaces
Option A (hard-fail everything — brittle against historical quirks) and Option C
(auto-fixing "known" patterns — silent mutation of source data, rejected
outright).

### Change to process (Before → After)
- **Before:** a parser regression zeroing control time would ingest quietly for
  weeks; a corrupt row would flow into every rolling average and Elo chain.
- **After:** `make validate` (and CI) runs the suite; a modern-era violation is a
  same-day loud failure; historical quirks are visible in the report but block
  nothing; every eval run carries the quarantine census.
- **Use case:** ufcstats shifts a column and new fights parse with
  `takedowns_landed > takedowns_attempted`; the next validation run fails with
  reason `LANDED_GT_ATTEMPTED` on post-2010 rows — caught before any feature or
  model consumes them.

---

## D13 — Feature engine: chronological state replay, emit-before-update

**Decided:** 2026-07-05 · **Owner:** human · **Status: recorded, NOT implemented**
— internals (state model, feature-function interface) are Decision #4, next
session.

### Why do we want this
Every feature must be point-in-time correct, and the two rejected architectures
each leave leakage to per-query or per-function discipline — one forgotten as-of
filter and a fighter's debut row carries their career-final win rate (the classic
bug that makes public UFC models report fake accuracy). Replay makes safety
*structural*: fights are processed in strict date order against per-fighter state,
and features for fight X are emitted **before** X's outcome updates that state —
the future cannot be read because it has not been replayed yet. It is also the
only architecture where iterative features (Elo, the fight graph, streaks) are
natural rather than bolted on: they simply *are* the state.

### How we accomplish it
- One replay loop over all fights (full feature-history universe, D11), ordered
  by event date; per-fighter state objects hold accumulators, Elo, graph refs.
- **The load-bearing rule: emit features from current state first, then apply
  the fight's outcome to the state.** A one-line inversion of that order makes
  Elo clairvoyant by exactly one fight — plausible-looking, invisible in
  aggregate — so correctness property **P1 (the deletion oracle: recompute fight
  X's features after deleting all rows dated ≥ X; must be bit-identical) ships
  in the same build wave as the engine, not after.**
- Output materialized to a DuckDB table keyed
  `(fight_url, fighter_url, feature_version)`, DVC-versioned per FEATURES.md.
- **One engine, one code path:** training reads the table; predicting an
  upcoming card is just a replay that ends today. No separate inference path,
  no train-serve skew.
- Quarantined rows (D12) are excluded from replay input via anti-join.

### What it replaces
(a) On-the-fly SQL window functions — cannot express Elo/graph at all;
(b) per-feature batch functions each re-walking history with their own as-of
filter — every function a fresh leakage risk and an O(fights × features) cost.

### Change to process (Before → After)
- **Before:** "compute features" meant ad-hoc queries, each needing its own
  temporal-discipline review; iterative features had no home.
- **After:** adding a feature = writing a function from (state, upcoming-fight
  context) → value, registered with a feature version. The replay loop and P1
  guard are shared infrastructure nobody re-implements.
- **Use case:** to predict next week's card, run the same replay through
  yesterday, emit feature rows for the announced matchups, and hand them to
  PREDICT — identical code to what built the training table.

---

## D14 — Replay state is composable components with a two-phase tick

**Decided:** 2026-07-05 · **Owner:** human · Refines D13 (Decision #4a).

### Why do we want this
Two silent failure modes drive this. First, a monolithic per-fighter state class
(~100 fields, one shared update path) means every future feature addition edits
code that every existing feature depends on — a wrong branch while adding camp
features can quietly stop streak updates for one subpopulation, and the
corruption is temporally consistent, so even the P1 deletion oracle passes.
Second, features read *across* families (§9c weights common opponents by Elo at
fight time), so if components update mid-fight in registration order, an emitter
can read post-fight state — one-fight clairvoyance that only appears in
cross-family features. Correctness must not depend on registration order.

### How we accomplish it
- One **state component per FEATURES.md family** (RecordState, EloState,
  ActivityState, …), each owning its accumulators, its update rule, and its
  tests. **Global components** (the fight graph) sit alongside per-fighter ones.
- **Two-phase tick, hard rule:** processing fight X, phase 1 emits *all*
  features against frozen pre-X state (any emitter may read any component);
  phase 2 applies X's outcome to *all* components. Ordering becomes irrelevant
  by construction — this generalizes D13's emit-before-update to components.
- P1's property test exercises cross-family features specifically, since that
  is where phase-mixing bugs hide.

### What it replaces
(a) A monolithic FighterState god-object — cheap now, a blast-radius liability
forever; (b) untyped nested dicts — rejected outright (types at boundaries is a
house rule).

### Change to process (Before → After)
- **Before:** adding a feature family means editing shared state code all other
  families depend on; component interactions are implicit.
- **After:** adding a family = one new component + its emitters + its tests;
  existing families are untouchable from its blast radius. Cross-family reads
  are safe by construction, not by code-review vigilance.
- **Use case:** §14 camp features land as `CampState` months from now without
  opening the Elo, record, or streak code paths at all.

---

## D15 — Emitter protocol, versioning with a hash guard, wide output table

**Decided:** 2026-07-05 · **Owner:** human · Completes Decision #4 (with D13/D14).

### Why do we want this
Three closing choices for the feature engine. The dangerous one is versioning:
if feature code changes without the cached table's version bumping, the table
silently mixes rows produced by different code — MLflow logs the same version
for both, and no experiment is reproducible anymore. You only notice when a
metric moves between two "identical" runs and you burn a day finding out why.

### How we accomplish it
- **Protocol:** components own state + an `update(fight)` rule (phase 2);
  emitters are registered functions `EmitContext -> dict[feature_name, value]`
  (phase 1), where `EmitContext` is the frozen pre-fight view (both fighters'
  components, the global graph, as-of fight info). A central registry rejects
  duplicate feature names at startup and defines the output schema.
  `None` → NULL → NaN is the missingness contract (D9).
- **Versioning: manual constant + hash guard.** A human bumps
  `FEATURE_VERSION`; a committed lockfile records the source hash of the
  features package per version; a pytest fails whenever code and lockfile
  diverge — forgetting to bump is *impossible but loud*, and versions stay
  human-readable (v4, not a hash).
- **Output: wide table** `features_v{N}`, one row per
  `(fight_url, fighter_url)`, one column per registered feature. Labels are
  NOT stored here — they come from joining `fights.winner_url` downstream.

### What it replaces
Auto-content-hash versioning (unreadable ids, churns on refactors) and
honor-system manual versioning (silent staleness). Long/narrow output format
(schema-stable, but requires a pivot for every human inspection; versioned
tables already solve schema evolution).

### Change to process (Before → After)
- **Before:** no defined path from a FEATURES.md row to code; caches trusted on
  faith.
- **After:** adding a feature = write an emitter, register names, run tests;
  changing any feature code without bumping `FEATURE_VERSION` fails CI in the
  same commit. Debugging a value = SELECT one row from the wide table.
