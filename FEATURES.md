# FEATURES.md

> **Features are human-specified. This file is a registry, not a brainstorm.**
> Claude Code does not add, "improve," or speculatively expand this list. Where a
> definition is genuinely ambiguous it is flagged `TODO(human):` rather than guessed.

> **Architecture note (2026-06-21 redesign).** The system moved from four layers to
> six (see [DECISIONS_architecture_redesign.md](DECISIONS_architecture_redesign.md)).
> Consequences reflected in this registry: the **model is structurally odds-free** —
> Section 7 is **dropped entirely** (D6b) and Section 16 market features are
> **strategy-layer only** (D6). LLM-extracted signals are produced by an **offline,
> cached enrichment step** gated on `source_published_at < fight_date` (D7), never in
> the inference path. Learned representations (sequence encoder + fight-graph GNN) and
> the predictor/strategy internals are machinery, not features, and live in the
> redesign doc — not here.

## How to read this registry

Every feature carries four attributes (per the documentation contract):

- **Definition** — one line: what is computed.
- **Layer** — `model` (predictive, fed to XGBoost) or `strategy` (post-model
  behavioral/execution adjustment, **never** in the gradient path). See the layer
  separation contract in [CLAUDE.md](CLAUDE.md).
- **As-of cutoff** — the temporal-integrity rule for the feature. The global
  invariant: any feature used for fight *X* is computed strictly from data with
  `event_date < fight_date`. Per-feature notes call out tighter handling.
- **Status** — `confirmed` (in scope, decided), `deferred` (decided but gated on a
  precondition, e.g. an LLM eval threshold), or `abandoned` (rejected; reason in
  [DECISIONS.md](DECISIONS.md)).

Sections inherit a default Layer / As-of cutoff / Status stated in their header;
individual features override only where they differ.

## Philosophy

Features are not hand-weighted. Importance is learned by XGBoost through
gradient-boosted splits. This document defines what is computed, how, and why — not
how much it matters. All features are temporally enforced: any feature used for
fight *X* is computed strictly from data with `event_date < fight_date`. Leakage
tests in `tests/unit/test_leakage.py` will cover every feature function (the test
module and the feature functions do not exist yet — the model/feature layer is
scaffolded but unimplemented; see [CLAUDE.md](CLAUDE.md) scope map).

Features fall into three computational sources:

- **Quantitative** — computed deterministically from ufcstats + Kaggle data.
- **Graph-derived** — computed from the fight graph (Elo, PageRank, common opponents).
- **LLM-extracted** — preprocessed offline, cached to DuckDB, treated as static
  inputs. Never recomputed at inference time.

Matchup-level features are deltas: `fighter_A_value - fighter_B_value`. Positive
delta favors fighter A by convention throughout.

LLM features are included only if the eval suite shows **precision ≥ 0.80, recall ≥
0.60, non-null rate ≥ 0.30** across the training set. Features below threshold are
logged in [DECISIONS.md](DECISIONS.md) with scores attached.

> **LLM provider conflict — RESOLVED (2026-06-21, redesign D7).** The LLM is an
> **offline, cached enrichment source** that emits structured columns ahead of time
> and is **never in the inference path**; leakage is handled by a
> `source_published_at < fight_date` gate (not by being offline). This rejects the
> *grounded-search-at-inference* approach (old Gemini 2.5 Flash draft) in favor of the
> offline-cached approach. The *specific* offline model (e.g. the brief's local Llama
> 3.1 70B vs. another offline/batch model) is a low-level detail still owned by the
> human; the LLM sections below remain written model-neutral. See
> [DECISIONS.md](DECISIONS.md): "LLM provider conflict — RESOLVED".

---

## Section 1 — Fighter Physical Profile

**Layer:** model · **As-of cutoff:** static per fighter, evaluated as-of fight date
(uses `age_at_fight`) · **Status:** confirmed

Computed once per fighter from static data. Updated only if fighter changes weight class.

| Feature | Definition | Source |
|---|---|---|
| `height_cm` | Fighter height | ufcstats |
| `reach_cm` | Fighter reach | ufcstats |
| `reach_to_height_ratio` | Normalized reach-advantage proxy | derived |
| `stance` | Orthodox / Southpaw / Switch | ufcstats |
| `age_at_fight` | Age on fight date | derived |
| `weight_class` | Current weight class | ufcstats |
| `natural_weight_class` | Weight class where fighter started career | derived |
| `weight_class_delta` | Signed difference from natural class (cutting down = negative) | derived |

> `TODO(human):` `natural_weight_class` derivation is underspecified. ufcstats does
> not reliably carry pre-UFC career data; define the source (Kaggle? first observed
> UFC class?) and the rule for "started career."

---

## Section 2 — Activity & Inactivity

**Layer:** model · **As-of cutoff:** all counts/windows strictly prior to fight date
· **Status:** confirmed

| Feature | Definition |
|---|---|
| `days_since_last_fight` | Days between previous fight and current fight date |
| `fights_last_12mo` | Fight count in prior 12 months |
| `fights_last_3yr` | Fight count in prior 3 years |
| `fights_last_5yr` | Fight count in prior 5 years |
| `total_pro_fights` | Total professional fights as of fight date |
| `total_ufc_fights` | UFC fights only as of fight date |
| `last_fight_injury_stoppage` | Boolean: last fight ended by doctor stoppage or NC-injury |
| `age_x_inactivity` | `age_at_fight * days_since_last_fight` (interaction term) |
| `inactivity_tier` | Bucketed: 0=<6mo, 1=6–12mo, 2=1–2yr, 3=2yr+ |

**Note on inactivity:** Elo decay handles part of this signal by pulling ratings
toward the mean during layoffs. These features capture what Elo misses: the reason
for inactivity, the interaction with age, and the last fight's method of ending.

---

## Section 3 — Win/Loss Record (Corrected)

**Layer:** model · **As-of cutoff:** records computed from fights strictly prior to
fight date · **Status:** confirmed

Raw win/loss is a weak signal. These features correct for context.

| Feature | Definition |
|---|---|
| `win_pct_all` | Overall win percentage |
| `win_pct_last3` | Win % in last 3 fights |
| `win_pct_last5` | Win % in last 5 fights |
| `current_streak` | Signed: +3 = 3-fight win streak, −2 = 2-fight loss streak |
| `win_pct_by_finish` | % of wins that were finishes |
| `win_pct_by_decision` | % of wins that went to judges |
| `loss_pct_by_finish` | % of losses that were stoppages (durability proxy) |
| `loss_pct_by_decision` | % of losses that went to distance |
| `ufc_win_pct` | Win % in UFC fights only |
| `ufc_record_fights_count` | How many UFC fights used to compute `ufc_win_pct` |

### 3a — Debut Fighter Adjustment

**Layer:** model · **Status:** confirmed (except as noted below)

A UFC debut is not a neutral event. The direction of mispricing depends on the path
to the debut and the opponent being faced.

| Feature | Definition |
|---|---|
| `is_ufc_debut` | Boolean |
| `pre_ufc_record_wins` | Wins before UFC |
| `pre_ufc_record_losses` | Losses before UFC |
| `pre_ufc_opponent_avg_win_pct` | Average win % of opponents faced before UFC |
| `pre_ufc_finish_rate` | Finish rate in pre-UFC career |
| `debut_opponent_ufc_experience` | Number of UFC fights opponent has (hostile vs soft debut) |
| `debut_opponent_ufc_win_pct` | Win % of debut opponent in UFC |
| `contender_series_win` | Boolean: fighter won a DWCS bout to get contract |
| `regional_circuit_quality_tier` | LLM-extracted: elite / solid / thin / unknown — **deferred** (gated on LLM eval) |

**Context:** The market prices debuts on record and name recognition. A fighter who
went 12-0 against losing-record opponents in a thin regional circuit is not
equivalent to one who went 8-0 against established regional competition. The debut
opponent context matters as much as the fighter's own record.

> `TODO(human):` `pre_ufc_*` features require pre-UFC career data that ufcstats does
> not carry. Confirm the source (Kaggle dataset coverage?) or mark these deferred.

---

## Section 4 — Finishing Profile

**Layer:** model · **As-of cutoff:** computed from fights strictly prior to fight
date · **Status:** confirmed

| Feature | Definition |
|---|---|
| `finish_rate` | Finishes / total wins |
| `ko_rate` | KO or TKO wins / total wins |
| `submission_rate` | Submission wins / total wins |
| `early_finish_rate` | Round-1 finishes / total wins |
| `avg_fight_duration_sec` | Mean fight duration in seconds |
| `fight_duration_variance` | Variance in fight duration (high = bimodal finisher) |
| `has_ever_been_finished` | Boolean |
| `times_finished_by_ko` | Count |
| `times_finished_by_sub` | Count |
| `has_been_finished_r1` | Boolean: ever finished in round 1 |

### 4a — Never Been Finished (Conditional Signal)

**Layer:** model · **Status:** confirmed

"Never been finished" is only meaningful when the opponent has credible finishing
power. As a standalone feature it is weak; as an interaction it is informative.

| Feature | Definition |
|---|---|
| `never_been_finished` | Boolean: 0 career stoppages |
| `never_been_finished_x_opp_finish_rate` | Interaction: `never_finished * opponent finish rate` |

**Context:** A fighter who has never been stopped going into bouts against power
punchers is genuinely predictive. The same flag on a fighter who has only faced
decision-fighters tells you almost nothing. The product term gives the model the
interaction directly.

---

## Section 5 — Output & Efficiency Metrics

**Layer:** model · **As-of cutoff:** rolling averages over last *N* fights, strictly
as-of fight date · **Status:** confirmed

| Feature | Definition |
|---|---|
| `sig_strikes_per_min` | Significant strikes landed per minute |
| `sig_strikes_absorbed_per_min` | Significant strikes received per minute |
| `striking_accuracy_pct` | Sig strikes landed / total thrown |
| `striking_defense_pct` | % of incoming strikes avoided |
| `td_per_15min` | Takedowns landed per 15 minutes |
| `td_accuracy_pct` | Takedowns landed / attempted |
| `td_defense_pct` | % of opponent takedowns defended |
| `sub_attempts_per_15min` | Submission attempts per 15 minutes |
| `knockdown_rate` | Knockdowns landed per significant strike thrown |
| `damage_ratio` | `sig_strikes_landed / sig_strikes_absorbed` |
| `grappling_dominance` | `(TD landed + control time) / (TD absorbed + control time absorbed)` |
| `control_time_per_fight` | Average ground control time (seconds) |
| `distance_strike_pct` | % of strikes thrown at distance |
| `clinch_strike_pct` | % of strikes thrown in clinch |
| `ground_strike_pct` | % of strikes thrown on ground |
| `head_target_pct` | % of strikes aimed at head |
| `body_target_pct` | % of strikes aimed at body |
| `leg_target_pct` | % of strikes aimed at legs |

### 5a — Output Variance by Card Position

**Layer:** model · **Status:** confirmed · Requires 3+ UFC fights (else NaN; see
Missingness Policy)

Captures entertainment-seeking fighters whose output changes on big stages.

| Feature | Definition |
|---|---|
| `sig_strikes_main_card_avg` | Average sig strikes/min in main-card appearances |
| `sig_strikes_prelim_avg` | Average sig strikes/min in prelim appearances |
| `td_rate_main_card_avg` | Takedown rate in main-card appearances |
| `td_rate_prelim_avg` | Takedown rate in prelim appearances |
| `grappling_abandonment_delta` | `td_rate_prelim - td_rate_main` |
| `output_variance_by_position` | Variance in sig strikes across card positions |

---

## Section 6 — Experience & Championship Context

**Layer:** model · **As-of cutoff:** counts strictly prior to fight date ·
**Status:** confirmed

| Feature | Definition |
|---|---|
| `title_fight_experience` | Count of career title fights |
| `has_been_champion` | Boolean |
| `days_as_champion` | Total days holding a UFC title |
| `main_event_experience` | Count of main-event appearances |
| `five_round_experience` | Count of scheduled 5-round fights (stamina proxy) |
| `five_round_win_pct` | Win % in 5-round fights specifically |


## Section 8 — Weight Class & Physical Dominance

**Layer:** model · **As-of cutoff:** strictly prior to fight date · **Status:** confirmed

### 8a — Weight Class Migration

| Feature | Definition |
|---|---|
| `is_weight_class_change` | Boolean: fighting at different class than last fight |
| `direction_of_change` | Moving up / moving down / same |
| `fights_at_current_class` | UFC fights at current weight class as of fight date |
| `win_pct_at_current_class` | Win % specifically at current weight class |
| `prior_class_win_pct` | Win % at previous weight class |

### 8b — Weight Bully (Interaction, not a standalone flag)

| Feature | Definition |
|---|---|
| `is_large_for_class` | Boolean: reach + height both in top quartile for weight class |
| `grappling_utilization_rate` | TD attempts + control time, normalized by fight count |
| `weight_bully_score` | `is_large_for_class * grappling_utilization_rate` (**product term**) |

**Context:** Being large for the class only converts to an edge when paired with a
style that imposes it. A big fighter who does not grapple does not get this score.
The product term is the correct operationalization.

---

## Section 9 — Graph-Derived Features

**Layer:** model · **As-of cutoff:** graph recomputed as-of each fight date; **no
future fight results enter the graph** · **Status:** confirmed (9d gated — see below)

Functions to live in `src/features/graph_features.py` with leakage coverage in
`tests/unit/test_leakage.py` (not yet implemented).

> **Context (not a feature):** the recursive neighborhood-quality estimation used
> below (a node's quality depends on its neighbors' quality, recursively) is
> structurally identical to an academic ego-network approach to query-performance
> prediction. Recorded as framing per the human; it introduces no new feature.

### 9a — Elo Rating

| Feature | Definition |
|---|---|
| `elo_rating` | Current Elo as-of fight date (pre-fight) |
| `elo_trajectory_last5` | Linear-regression slope of last 5 Elo values |
| `elo_peak` | Highest Elo ever achieved as-of fight date |
| `elo_current_vs_peak` | `elo_rating / elo_peak` (decay-from-peak signal) |

Elo update uses a variable K factor (method bonus + recency) and decays toward the
mean (1500) during inactivity. Injury stoppages are Elo-neutral (K=0). DQ outcomes
use K × 0.1. See [DECISIONS.md](DECISIONS.md): Elo Configuration.

### 9b — PageRank

| Feature | Definition |
|---|---|
| `pagerank_score` | PageRank on directed win graph as-of fight date |

Edges run loser → winner. Edge weights encode finish-type bonus, recency decay
(exponential, λ tuned by Optuna), and early-finish bonus. Damping α = 0.85. All
hyperparameters in `configs/graph.yaml` (not yet created).

### 9c — Common Opponent Analysis

Three-year lookback. NaN when no common opponents exist (XGBoost learns the default
branch).

| Feature | Definition |
|---|---|
| `n_common_opponents` | Count of common opponents in prior 3 years |
| `common_opp_score_a` | Fighter A's mean performance score vs common opponents |
| `common_opp_score_b` | Fighter B's mean performance score vs common opponents |
| `common_opp_score_delta` | `common_opp_score_a − common_opp_score_b` |
| `common_opp_a_win_rate` | Fighter A's win rate vs common opponents |
| `common_opp_b_win_rate` | Fighter B's win rate vs common opponents |

Performance scores weight opponent quality by Elo and PageRank at the time of fight
and apply recency decay within the 3-year window. See `MATH.md` (not yet created):
Common Opponent Weighting Derivation.

### 9d — Opponent Trajectory Adjustment ⚠️ LEAKAGE TRAP

**Status:** confirmed · **As-of cutoff:** ⚠️ **this is the repo's primary leakage
trap.** These features use opponents' *post-fight* results, which are future events
relative to the fight being predicted. They are only admissible at an **inference
as-of date late enough that the opponents' subsequent fights have already occurred**,
and must **never** fold an opponent's post-fight outcome back into features for the
original fight (or any earlier fight). Every function here requires an explicit
dedicated leakage test before it may enter the model.

| Feature | Definition |
|---|---|
| `opp_avg_post_fight_win_pct` | Mean win % of all opponents in their next 3 fights after facing this fighter |
| `opp_trajectory_score` | Composite: did opponents improve, hold, or decline after this fight |
| `loss_to_future_contender` | Boolean: any loss came against a fighter who reached top-5 within 18mo |
| `win_over_declining_opp` | Boolean: any win came against a fighter who went 0-3 or was cut in next 3 fights |

**Context:** A fighter who is 0-2 in the UFC where both losses came against future
top contenders should not be penalized like a fighter who is 0-2 against journeymen.
This surfaces that distinction — at the cost of being the easiest place in the whole
system to leak the future.

> `TODO(human):` Specify the exact as-of evaluation date semantics for these
> features (e.g. "computed only for fights ≥ N days old, using opponent fights with
> `event_date <= inference_date`"). The model layer must encode this precisely before
> these features are enabled.

---

## Section 10 — Matchup-Level Features

**Layer:** model · **As-of cutoff:** deltas of features each computed strictly
as-of fight date · **Status:** confirmed · Convention: positive favors fighter A.

| Feature | Definition |
|---|---|
| `reach_delta` | `reach_A − reach_B` |
| `height_delta` | `height_A − height_B` |
| `age_delta` | `age_A − age_B` |
| `stance_matchup` | Encoded: ortho_v_ortho, ortho_v_south, south_v_south, switch_involved |
| `southpaw_matchup` | Boolean: one fighter is southpaw |
| `elo_delta` | `elo_A − elo_B` |
| `pagerank_delta` | `pagerank_A − pagerank_B` |
| `finish_rate_delta` | `finish_rate_A − finish_rate_B` |
| `striking_efficiency_delta` | `striking_accuracy_A − striking_accuracy_B` |
| `td_accuracy_delta` | `td_accuracy_A − td_accuracy_B` |
| `damage_ratio_delta` | `damage_ratio_A − damage_ratio_B` |
| `avg_fight_duration_delta` | `avg_duration_A − avg_duration_B` (pace mismatch) |
| `fight_duration_variance_delta` | `variance_A − variance_B` (chaos vs consistency) |
| `five_round_experience_delta` | `five_round_exp_A − five_round_exp_B` |
| `ufc_experience_delta` | `ufc_fights_A − ufc_fights_B` |
| `title_fight_exp_delta` | `title_fights_A − title_fights_B` |

### 10a — Grappling Sub-Type Matchup

| Feature | Definition |
|---|---|
| `wrestler_score_a` | `td_accuracy * td_per_15 * td_defense` (wrestling-output proxy) |
| `submission_score_a` | `sub_attempts_per_15 * submission_rate` (submission-threat proxy) |
| `wrestler_score_b` | Same for fighter B |
| `submission_score_b` | Same for fighter B |
| `wrestling_delta` | `wrestler_score_A − wrestler_score_B` |
| `submission_delta` | `submission_score_A − submission_score_B` |
| `grappling_type_mismatch` | High wrestler_score for one fighter, high submission_score for opponent |

**Context:** The market conflates all grapplers. A wrestler without submission skills
facing a submission specialist is in danger on the ground despite being the
"grappler" in the matchup. Splitting wrestling vs submission threat surfaces the
asymmetry for any two grapplers.

### 10b — Style Interaction

| Feature | Definition |
|---|---|
| `striker_vs_grappler` | Boolean: one primarily striker, other primarily grappler |
| `pressure_vs_counter` | Boolean: one high-pace volume, other counter-striker |
| `pace_mismatch_score` | `sig_strikes_per_min delta * fight_duration_variance delta` |
| `southpaw_orthodox_history` | Fighter A's historical win % specifically vs southpaws (or vice versa) |

**Context:** `southpaw_orthodox_history` is **fighter-specific stance performance
history**, not a generic stance premium. The signal is whether *this* fighter has
evidence of struggling against southpaws, derived from their own fight history.

---

## Section 11 — Rematch Features

**Layer:** model · **As-of cutoff:** first-meeting data is strictly prior;
`style_change_since_first_meeting` is LLM-extracted (deferred) · **Status:** confirmed
(except the LLM field)

| Feature | Definition |
|---|---|
| `is_rematch` | Boolean |
| `fights_since_first_meeting` | How many fights each fighter has had since fight 1 |
| `result_of_first_meeting` | Encoded: A won / B won / NC / Draw |
| `first_meeting_method` | KO / Sub / Decision / NC |
| `first_meeting_competitive` | Boolean: went to decision or was split/majority |
| `style_change_since_first_meeting` | LLM-extracted: did either fighter change style or camp — **deferred** (gated on LLM eval) |
| `first_meeting_score_delta` | Performance score of each fighter in first meeting |

---

## Section 12 — Home Advantage & Geographic Context

**Layer:** model · **As-of cutoff:** event location + training base known at fight
date · **Status:** confirmed

| Feature | Definition |
|---|---|
| `fighter_a_home_country_fight` | Boolean: fight in fighter A's home country |
| `fighter_b_home_country_fight` | Boolean: fight in fighter B's home country |
| `home_advantage_delta` | Encoded asymmetry: A at home vs neutral vs B at home |
| `fighter_a_travel_required` | Boolean: fight country differs from training base |
| `fighter_b_travel_required` | Same for fighter B |
| `event_country` | Country where event takes place |

**Context:** Home advantage exists but is partly matchmaking (e.g. 2013–2017
Brazilian cards systematically matchmade for local fighters), not just crowd energy.
Both effects are real and only partially priced by the market.

---

## Section 13 — Behavioral Features (Post-Model Adjustment Layer)

**Layer:** strategy (**excluded from XGBoost by design**) · **As-of cutoff:**
LLM-extracted offline, cached, treated as static · **Status:** deferred (gated on LLM
eval; behavioral coverage is sparse/qualitative)

These are **not** fed into XGBoost. They are applied as post-model adjustments to win
probability in the strategy layer, after the model outputs a base probability. See
[DECISIONS.md](DECISIONS.md): Behavioral Discount Architecture.

**Rationale (why excluded from the model):** behavioral profiles have sparse,
qualitative coverage insufficient to survive cross-validation as training features.
In XGBoost they would produce near-zero importance for most fighters or overfit to a
handful of well-documented cases. They belong in the decision layer. Cached to DuckDB
table `llm_behavioral_profiles`.

| Feature | Definition | Adjustment Direction |
|---|---|---|
| `entertainment_seeker` | Boolean: prioritizes crowd approval over optimal strategy | Negative on big cards |
| `entertainment_seeker_evidence` | Source string for audit | — |
| `gameplan_adherence_tier` | high / medium / low | Low = negative vs disciplined opponents |
| `impulsivity_signal` | reactive / neutral / disciplined | Reactive = negative vs counter-strikers |
| `pressure_response` | tightens / resets / reckless / unknown | Reckless = negative in championship rounds |
| `performs_to_skill_ceiling` | Boolean: consistently executes at actual level | False = add variance |
| `adjusts_between_rounds` | Boolean: corner adjustments visible in footage | False = negative in long fights |
| `profile_confidence` | 0–1 LLM self-assessment | Discount applied only if ≥ 0.65 |

---

## Section 14 — Camp & Preparation Stability

**Layer:** model · **As-of cutoff:** strictly prior to fight date; some fields
LLM-extracted (deferred) · **Status:** confirmed for deterministic fields; LLM fields
deferred (gated on eval)

Designed to capture preparation quality before it shows up in outcomes (camp
instability lags results by ~1–3 fights).

| Feature | Definition |
|---|---|
| `n_camps_last_5yr` | Number of distinct gyms in prior 5 years |
| `current_camp_tenure_days` | Days at current gym as of fight date |
| `camp_volatility_per_year` | `n_camps / years active` (serial-switching rate) |
| `days_since_last_camp_switch` | Recency of most recent gym change |
| `recent_camp_disruption` | Boolean: switched gym in last 6 months |
| `n_countries_trained_in` | Career count of countries trained in |
| `fight_requires_travel` | Boolean: fight country differs from current training base |
| `head_coach_tenure_years` | How long current head coach has been in corner |
| `corner_consistency` | Boolean: same corner as last fight |
| `full_camp` | Boolean: full camp vs short notice (<3 weeks) |
| `sparring_disruption_reported` | LLM: lost key sparring partner during camp — **deferred** (gated on LLM eval) |
| `camp_stability_score` | Composite 0–1; spec in `src/features/camp_features.py` (not yet implemented) |

**Reference archetypes:** long-tenured single-gym fighter ≈ 0.95; multi-gym /
multi-country mover ≈ 0.40; fighter mid-switch at fight time ≈ 0.20–0.35.

**Note:** Camp-data coverage correlates with fighter profile (elite fighters get more
press than prelim fighters), so missingness is **not random** and correlates with
rank. XGBoost handles nulls via the learned default branch. Monitor missingness rates
by rank tier in model diagnostics.

---

## Section 15 — LLM-Extracted Pre-Fight Intelligence

**Layer:** model · **As-of cutoff:** extracted offline from pre-fight coverage only,
cached, treated as static; **never recomputed at inference and never in the gradient
path** · **Status:** deferred (gated on LLM eval thresholds)

> **Leakage gate (2026-06-21, redesign D7).** Being offline buys reproducibility, not
> leakage protection. Every extracted fact additionally gates on
> `source_published_at < fight_date` — an offline job can still read a *post*-fight
> article and stamp a flag onto a pre-fight row, so the guard is on the source, not
> the runtime. Pytest-asserted: no fact feeding fight T may cite a source published on
> or after T. Honest semantics: these mean "publicly-known-pre-fight" (a coverage-
> biased subsample), which is exactly the information a bettor had at the time.

Eval suite at `tests/eval/llm_pipeline_eval.py` (not yet implemented) measures
precision/recall per signal type on 50 labeled fights. Acceptance: precision ≥ 0.80,
recall ≥ 0.60, non-null rate ≥ 0.30. Cached to DuckDB tables `llm_fighter_intel`.

### Injury & Physical Readiness

| Feature | Definition |
|---|---|
| `injury_site` | Body part: leg / hand / knee / rib / unknown |
| `injury_severity_signal` | cleared / managing / first_camp_back / unknown |
| `surgery_in_last_12mo` | Boolean |
| `opponent_targets_injury_site` | Boolean: opponent's style attacks the injury site |
| `last_fight_was_injury_stoppage` | Boolean: last fight ended by doctor or NC-injury |

### Weight Cut

| Feature | Definition |
|---|---|
| `weight_cut_concern_flagged` | Boolean: concern reported in pre-fight coverage |
| `missed_weight_last_3` | Boolean: missed weight in any of last 3 fights |
| `missed_weight_career` | Count of career weight misses |
| `moving_down_in_weight` | Boolean: current fight at lower class than last fight |

**Context:** A missed weight is not merely a fine and a 1-pound edge — it signals a
failed 8–12 week preparation, compromised rehydration, and a physical deficit that
does not show in the tale of the tape. The market prices the miss superficially.

### Matchup-Specific Intelligence (per fight)

| Feature | Definition |
|---|---|
| `llm_style_advantage_signal` | Which fighter has the stylistic edge per LLM analysis |
| `llm_historically_struggles_with_style` | Boolean: documented style-specific weakness |
| `llm_opponent_style_exploits_weakness` | Boolean: this opponent's style matches that weakness |
| `llm_chin_decline_reported` | Boolean: analyst commentary + finish pattern suggests chin degradation |

---

## Section 16 — Market-Derived Features (strategy layer — not model features)

> **Relocated to the strategy layer (2026-06-21, redesign D6).** These are **not**
> model/XGBoost features and must never enter PREDICT or REPS. They are documented
> here for completeness but live strategy-side, where they drive convergence-trade
> logic and the slippage model. Per the redesign the model is structurally odds-free:
> no odds-derived signal, historical or live, reaches the predictor.

**Layer:** strategy (**never in model training** — market probability cannot be used
to predict market probability) · **As-of cutoff:** opening = first Polymarket
snapshot; closing = final snapshot before fight · **Status:** confirmed (strategy-side)

| Feature | Definition |
|---|---|
| `opening_implied_prob` | First Polymarket snapshot implied probability |
| `closing_implied_prob` | Final snapshot before fight |
| `line_movement_magnitude` | `abs(closing − opening)` |
| `line_movement_direction` | Positive = moved toward A, negative = toward B |
| `spread_at_close` | Bid-ask spread at closing snapshot |
| `depth_at_close` | Order-book depth within 2% of mid at closing snapshot |
| `volume_last_24hr` | Trading volume in final 24 hours |

These are sourced from the self-captured order-book snapshots (see
`src/ufc_edge/market/capture.py`).

---

## Section 17 — Features Considered and Excluded

Full rationale for each in [DECISIONS.md](DECISIONS.md).

| Feature | Status | Reason Excluded |
|---|---|---|
| Retirement signal | **abandoned** | Too unreliable for extraction; fighters retire and return; LLM false-positive rate unacceptably high. |
| Judges' scorecards pre-fight | abandoned | Not available as-of fight date. |
| Post-fight medical suspensions | abandoned | Available only after the outcome — hard leakage. |
| Raw win % without context | abandoned | Superseded by Elo and opponent-adjusted versions. |
| Height differential as standalone | abandoned | Reach captures it better and more specifically. |
| Fighter nationality as standalone | abandoned | Captured via `home_advantage_delta` and `regional_circuit_quality_tier`. |
| Social-media follower count | abandoned | Correlated with name recognition already priced into odds; no independent signal. |
| Performance vs Expectation (old §7) | **dropped** (2026-06-21, D6b) | Fed historical closing odds into the model; passes temporal-leakage but fails the economic-circularity test, muddying model-edge vs execution-edge attribution. Model is now structurally odds-free. |

---

## Missingness Policy

| Situation | Handling |
|---|---|
| Common-opponent delta when n=0 | Return NaN; XGBoost learns default branch |
| LLM features below confidence threshold | Return NaN |
| Fighter with <3 UFC fights for variance features | Return NaN |
| Elo for debut fighter | Initialize at 1500 (prior mean) |
| PageRank for isolated node (no graph wins/losses) | Return global minimum score |
| Camp features for fighters with no press coverage | Return NaN; flag in diagnostics |

---

## Versioning

Features are versioned via DVC. Any change to a feature function increments the
feature version tag and invalidates downstream caches. MLflow logs the feature
version alongside every training run. Walk-forward backtest windows are locked to the
feature version used.
