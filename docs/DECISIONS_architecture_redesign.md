# DECISION: Architecture redesign — four layers → six, learning-capable

**Date:** 2026-06-21
**Status:** Accepted (high-level design locked; low-level design owned by human, pending)
**Supersedes:** original four-layer design (data · features · model · strategy)

---

## Context

The original repo established a four-layer separation contract whose purpose was to
keep the edge honest: a calibrated predictor estimates `P(win)`, and a separate
strategy layer compares that to the market and sizes bets. Two layers were built
(data, Polymarket capture); features/model/strategy were empty scaffolds.

This redesign keeps the separation contract intact — it is the repo's single most
valuable asset and the structural guarantee against circularity and leakage — and
grows the model and strategy layers into something that actually learns. It adds two
new layers (representation, simulator) and one cross-cutting layer (online feedback).

**Framing decision:** keep the spine, evolve the organs. Not a from-scratch rewrite.

---

## Locked architecture

```
        ufcstats ─┐
        Kaggle ───┼──►  ┌─────────── DATA ──────────────┐
   Polymarket ────┘     │ scrapers + capture cron        │
   (Gamma+CLOB)         │ + OFFLINE ENRICHMENT (cached,  │
                        │   LLM-sourced, as-of:          │
                        │   short-notice/injury/camp →   │
                        │   structured columns)          │
                        │            ↓ DuckDB (one store)│
                        └────┬──────────────────────┬────┘
                             │                       │
                ┌────────────┴───────┐      ┌────────┴───────────┐
                │ FEATURES (as-of)   │      │ SIM (CLOB replay)  │
                │ tabular: stats +   │      │ rebuild book @ any │
                │ short-notice/injury│      │ decision time      │
                └──────────┬─────────┘      └────────┬───────────┘
                           │                         │
                ┌──────────┴─────────┐               │
                │ REPS (as-of)       │               │
                │ • seq encoder      │               │
                │   (GRU/Transformer)│               │
                │ • fight-graph GNN  │               │
                │ [2 leakage surfaces│               │
                │  pytest-guarded]   │               │
                └──────────┬─────────┘               │
                           │ tabular ⊕ embeddings    │
                ┌──────────┴─────────┐               │
                │ PREDICT            │               │
                │ gbt|mlp|ensemble   │               │
                │  (XGBoost=baseline)│               │
                │ → calibrate (Platt │               │
                │   →isotonic)→P(win)│               │
                └────┬───────────┬───┘               │
                     │           │ calibrated P(win) │
   resolution +      │           ▼                   ▼
   ground truth      │   ┌──────────────────────────────────────────┐
        │            │   │ STRATEGY (sequential decision)             │
        │            │   │ state=(P, price, depth, ttr, position)     │
        │            │   │ Kelly/shrinkage → bandit → offline RL      │
        │            │   │ Type A (hold) vs Type B (converge)         │
        │            │   │ market-derived feats (ex-§16) live here    │
        │            │   │ eval: OPE first; sim rollouts; RL later    │
        │            │   └───────────────┬────────────────────────────┘
        │            │                   │ orders @ sim prices
   ┌────┴────────────┴───────┐           ▼
   │ ONLINE (feedback)       │   ┌──────────────────────────────┐
   │ • neural: Adam warm-    │   │ EVAL / REPRO spine           │
   │   start + replay buffer ├───┤ walk-forward · calibration   │
   │ • gbt: retrain cadence  │   │ (ECE/Brier/reliability)      │
   │ • rolling recalibration │   │ edge attribution: model-edge │
   │ • drift → retrigger     │   │   vs execution-edge          │
   └─────────────────────────┘   │ MLflow · DVC · Hydra         │
                                  └──────────────────────────────┘
```

---

## Invariants (build must hold all five)

1. **Layer separation.** The predictor consumes only `P(win)`-relevant signal
   (hard stats, learned reps, short-notice/injury). Behavioral and market-derived
   signal — including all odds, historical or live — live strategy-side. This is
   what makes the dropped anti-circularity *rule* (old §16) redundant rather than
   missing: separation enforces it structurally.
2. **Temporal integrity, extended.** As-of `event_date < fight_date` propagates
   through features, BOTH reps encoders, the enrichment columns, and the simulator.
   Enrichment additionally gates on `source_published_at < fight_date` (see D7).
   Every new component is a new leakage surface with its own pytest guard.
3. **Calibration before strategy.** Strategy consumes calibrated `P(win)` only,
   never raw model scores. Platt → isotonic.
4. **Walk-forward only.** No random splits on a time series, anywhere.
5. **Simulator fidelity.** A policy may only observe order-book state that existed
   at decision time. No future-price lookahead into the action.

---

## Decisions

### D1 — Keep the four-layer spine; do not rewrite from scratch
The separation contract (predictive vs. behavioral/market) is the strongest signal
in the project and the structural defense against circularity and leakage. Redesign
extends it rather than replacing it.
**Rejected:** clean-slate rewrite — would discard the one genuinely sophisticated
invariant for no gain.

### D2 — Add a representation layer: sequence encoder + fight-graph GNN
Deep learning earns its place as *learned representations*, not a bolt-on head.
- **Sequence encoder** (GRU or small Transformer) over a fighter's career as an
  ordered sequence of fight-vectors — learns recency weighting instead of
  hand-picking last-3 vs. career vs. exponential decay.
- **Fight-graph GNN** over the who-fought-whom graph — learns strength-of-schedule
  ("PageRank but learned") instead of a crude SOS stat.
Both emit as-of fighter embeddings consumed by PREDICT alongside tabular features.
**Both are new leakage surfaces:** the sequence encoder may only read fights strictly
before T; the graph must be rebuilt with edges `event_date < fight_date`, or it leaks
future opponent quality backward. Both pytest-guarded like features.
**Caveat on record:** on ~7k fights these may not beat a tuned GBT. The architecture's
job is to make that *answerable* via ablation, not to assume DL wins.

### D3 — PREDICT is model-agnostic; XGBoost is the baseline and a live model class
One interface: `[tabular ⊕ embeddings] → raw score → calibrate → P(win)`, with
swappable implementations: XGBoost (baseline), MLP head, ensemble. XGBoost is the
control that reps/MLP must beat to justify their complexity, AND a candidate
production model / ensemble member if it wins.
**Open (implementation sequencing, human-owned):** ship XGBoost-only end-to-end first
to get a real walk-forward calibration curve, *then* add reps + ablate — vs. build the
multi-model interface up front. Not locked here.

### D4 — ONLINE feedback loop with model-class-asymmetric updates
On fight resolution: score the prediction (Brier/log-loss), warm-start fine-tune the
neural components with Adam at low LR on new examples + a replay buffer (guards against
catastrophic forgetting), refit calibration on a rolling window every event, and
full-retrain the GBT on a cadence (trees don't warm-start cleanly). A drift monitor on
rolling calibration triggers retrains.
**Rationale for the asymmetry:** model class drives update strategy — neural updates
online, trees retrain periodically. This is a deliberate design point, not an
inconsistency.

### D5 — STRATEGY as a sequential decision problem; eval is OPE-first
State = (calibrated `P`, market price, book depth, time-to-resolution, position);
action = (size, enter/hold/exit); reward = realized log-wealth. Progression:
Kelly-with-shrinkage (closed-form baseline) → contextual bandit (sizing) → offline RL
(drop-in later, only if sample size ever justifies it). Type A (hold) and Type B
(convergence) are two policies the simulator evaluates head-to-head.
**Evaluation order — LOCKED: OPE first, offline RL later.** OPE (IPS / weighted-IPS /
doubly-robust / FQE) gives uncertainty-quantified policy value from logged data, and
exists *before* there is a learned policy to evaluate — correct dependency direction.
The CLOB replay simulator is a second (model-based) evaluator; disagreement between sim
rollouts and OPE is itself a diagnostic. Manual breakdown trajectories
(prediction + trade type + entry/exit + outcome) serve as expert demonstrations: a
human baseline to beat and a behavior policy to seed offline RL from later.
**Rejected:** offline-RL-first. Data-hungry and unstable at hundreds of fights /
fewer trades; distributional shift and extrapolation error dominate; and it doesn't
remove the need for a Kelly baseline anyway. Deferred to a drop-in once justified.

### D6 — Old §16 market-derived features move to the strategy layer
`opening_implied_prob`, `closing_implied_prob`, `line_movement_magnitude`,
`line_movement_direction`, `spread_at_close`, `depth_at_close`, `volume_last_24hr`
are dropped from FEATURES.md as model features and **relocated to the strategy layer**,
where they drive convergence-trade logic and the slippage model. "Dropped from
FEATURES.md" means "documented strategy-side," NOT "deleted from the system."
**Rejected:** any odds-derived feature in the model (see D6b).

### D6b — Old §7 (Performance vs Expectation) dropped entirely
§7 proposed feeding *historical* closing odds into the model via residual features
(`performance_vs_expectation_last5`, etc.). Dropped in full.
**Rationale:** §7 passes the temporal-leakage test (it's about prior fights) but fails
the *economic circularity* test that the project's thesis depends on. Historical
closing odds are autocorrelated with the current line being traded; consuming them
partially trains the "independent" predictor to reconstruct market consensus, which
muddies the model-edge vs. execution-edge attribution — the rare, high-value analysis.
Dropping §7 keeps the model fully odds-free and yields a crisp answer to the
"isn't your edge circular?" interview question: the model structurally never sees odds.
**Rejected alternatives:** (a) keep §7 whole; (b) keep only the residual fragility
features and drop raw `avg_closing_odds_last5`. Both rejected in favor of a clean
no-odds-in-model boundary; the marginal fragility signal wasn't worth the attribution
ambiguity.

### D7 — Short-notice / injury / camp as an offline, LLM-sourced, as-of enrichment column
A short-notice / opponent-swap / camp-change / layoff / known-injury signal is a
genuine `P(win)` predictor (not circular — it is not the line) and belongs in the
MODEL layer. It is produced by an **offline, cached enrichment step** that emits
structured columns (`is_short_notice`, `days_notice`, `opponent_changed`,
`known_injury`, …). The extraction source is an **LLM (approved)**, run ahead of time,
NOT in the inference path.
**Two independent problems, two independent fixes — do not conflate:**
- *Reproducibility / robustness:* solved by moving the LLM offline (no flaky model in
  the prediction path). Right call for the role.
- *Leakage:* solved SEPARATELY by a timestamp gate. Moving the LLM offline buys ZERO
  leakage protection — an offline job can still read a post-fight article and write a
  flag onto a pre-fight row. The guard is on the *source*: every enrichment fact gates
  on `source_published_at < fight_date`. A post-fight reveal ("I was injured in camp")
  is excluded by construction because its source postdates the fight. Pytest-asserted:
  no enrichment fact feeding fight T may cite a source published on or after T.
**Semantics, labeled honestly:** even with perfect gating, this feature means
"publicly-known-pre-fight injury," not "was injured" — a biased subsample (big names
get more coverage). That is exactly the information a bettor had at the time, so it is
the correct target, and FEATURES.md states this conditioning explicitly.

---

## Tooling / repro spine (unchanged)
uv · Python 3.12 · httpx · Hydra (config, never literals) · DVC · MLflow · DuckDB
(single datastore) · detect-secrets pre-commit · documentation-first workflow.
EVAL spine: walk-forward backtesting, calibration metrics (ECE / Brier / reliability),
edge attribution (model-edge vs. execution-edge).

## Ownership note
Feature ideation remains human-owned. This entry records *machinery and structure*
(encoders, interfaces, feedback loop, guards), not new predictive features. FEATURES.md
remains a registry, not a brainstorm.

## Open items (human-owned, not blocking)
- D3 sequencing: XGBoost-only v1 vs. multi-model interface up front.
  — **RESOLVED 2026-07-04:** XGBoost-first with two contract locks; see
  [.claude/spec/Key Design Decisions.md](../.claude/spec/Key%20Design%20Decisions.md) D9.
- Low-level design of each layer (tradeoffs, schemas, interfaces) — Step 2, human-owned.
