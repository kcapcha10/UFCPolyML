# V1 Feature Registry

> Source of truth for all features entering the `features_v{N}` table. Definitions
> are carried verbatim from the owner's original registry. Scope is filtered by D39;
> exclusions and gating are marked with their decision numbers. Nothing in this file
> is invented beyond owner-approved additions (Glicko-2, D31).

## Key Parameters

- **Elo:** variable K factor (method bonus + recency), decay toward 1500 during
  inactivity, injury stoppages Elo-neutral (K=0), DQ outcomes K × 0.1, debut
  initialization at 1500.
- **Glicko-2 (D31):** per-fighter μ, RD (uncertainty grows with inactivity), σ.
  Parameters in `configs/graph.yaml`.
- **PageRank:** directed win graph, edges loser → winner, edge weights encode
  finish-type bonus + exponential recency decay + early-finish bonus, damping
  α = 0.85, hyperparameters in `configs/graph.yaml`.
- **Common opponents:** 3-year lookback; performance scores weight opponent quality
  by Elo and PageRank at time of fight, recency decay within window; NaN when no
  common opponents.
- **Inactivity tiers:** 0 = <6mo, 1 = 6–12mo, 2 = 1–2yr, 3 = 2yr+.
- **Injury stoppage:** fight ended by doctor stoppage or NC-injury.
- **Damage ratio:** `sig_strikes_landed / sig_strikes_absorbed`.
- **Grappling dominance:** `(TD landed + control time) / (TD absorbed + control time absorbed)`.
- **Weight bully:** `is_large_for_class` (reach AND height both top-quartile for
  weight class) × `grappling_utilization_rate` (TD attempts + control time,
  normalized by fight count).
- **Rematch competitiveness:** first meeting went to decision or was split/majority.
- **Missingness:** NaN → XGBoost default branch; <3 UFC fights → NaN for variance
  features; Elo debut = 1500; PageRank isolated node = global minimum.
- **Matchup convention:** deltas are `fighter_A_value − fighter_B_value`; positive
  favors fighter A.

---

## Section 1 — Fighter Physical Profile (v1: IN)

| Feature | Definition | Source |
|---|---|---|
| `height_cm` | Fighter height | ufcstats |
| `reach_cm` | Fighter reach | ufcstats |
| `reach_to_height_ratio` | Normalized reach-advantage proxy | derived |
| `stance` | Orthodox / Southpaw / Switch | ufcstats |
| `age_at_fight` | Age on fight date | derived |
| `weight_class` | Current weight class | ufcstats |
| `natural_weight_class` | Weight class where fighter started career | derived — **Kaggle-gated (D30)**; TODO(human) derivation source |
| `weight_class_delta` | Signed difference from natural class (cutting down = negative) | derived — Kaggle-gated (D30) |

---

## Section 2 — Activity & Inactivity (v1: IN)

| Feature | Definition |
|---|---|
| `days_since_last_fight` | Days between previous fight and current fight date |
| `fights_last_12mo` | Fight count in prior 12 months |
| `fights_last_3yr` | Fight count in prior 3 years |
| `fights_last_5yr` | Fight count in prior 5 years |
| `total_pro_fights` | Total professional fights as of fight date (pre-UFC part Kaggle-gated, D30) |
| `total_ufc_fights` | UFC fights only as of fight date |
| `last_fight_injury_stoppage` | Boolean: last fight ended by doctor stoppage or NC-injury |
| `age_x_inactivity` | `age_at_fight * days_since_last_fight` (interaction term) |
| `inactivity_tier` | Bucketed: 0=<6mo, 1=6–12mo, 2=1–2yr, 3=2yr+ |

---

## Section 3 — Win/Loss Record (v1: IN)

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
| `ufc_record_fights_count` | Fights used to compute `ufc_win_pct` |

### 3a — Debut Fighter Adjustment

| Feature | Definition | v1 status |
|---|---|---|
| `is_ufc_debut` | Boolean | IN |
| `debut_opponent_ufc_experience` | Opponent's UFC fight count (hostile vs soft debut) | IN |
| `debut_opponent_ufc_win_pct` | Debut opponent's UFC win % | IN |
| `contender_series_win` | Won a DWCS bout to get contract | IN if derivable from ufcstats event names |
| `pre_ufc_record_wins` / `pre_ufc_record_losses` | Record before UFC | Kaggle-gated (D30) |
| `pre_ufc_opponent_avg_win_pct` | Avg win % of pre-UFC opponents | Kaggle-gated (D30) |
| `pre_ufc_finish_rate` | Pre-UFC finish rate | Kaggle-gated (D30) |
| `regional_circuit_quality_tier` | LLM-extracted tier | **EXCLUDED** from model (D32 — due-diligence layer) |

---

## Section 4 — Finishing Profile (v1: IN)

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

### 4a — Never Been Finished (conditional interaction; v1: IN)

| Feature | Definition |
|---|---|
| `never_been_finished` | Boolean: 0 career stoppages |
| `never_been_finished_x_opp_finish_rate` | `never_finished * opponent finish rate` |

---

## Section 5 — Output & Efficiency (v1: IN; rolling averages over last N fights, as-of fight date)

`sig_strikes_per_min`, `sig_strikes_absorbed_per_min`, `striking_accuracy_pct`,
`striking_defense_pct`, `td_per_15min`, `td_accuracy_pct`, `td_defense_pct`,
`sub_attempts_per_15min`, `knockdown_rate` (knockdowns per sig strike thrown),
`damage_ratio`, `grappling_dominance`, `control_time_per_fight`,
`distance_strike_pct`, `clinch_strike_pct`, `ground_strike_pct`,
`head_target_pct`, `body_target_pct`, `leg_target_pct`.

### 5a — Output Variance by Card Position (v1: IN, gated on scraper capturing bout_order)

> **Status:** bout_order is DERIVABLE from UFCStats DOM position (card-position
> check verdict). A scraper extension task (FE-5.1) persists the index. Until
> completed, §5a features emit NaN.

`sig_strikes_main_card_avg`, `sig_strikes_prelim_avg`, `td_rate_main_card_avg`,
`td_rate_prelim_avg`, `grappling_abandonment_delta` (`td_rate_prelim − td_rate_main`),
`output_variance_by_position`.

Requires 3+ UFC fights; else NaN.

---

## Section 6 — Experience & Championship Context (v1: IN)

`title_fight_experience`, `has_been_champion`, `days_as_champion`,
`main_event_experience`, `five_round_experience`, `five_round_win_pct`.

---

## Section 8 — Weight Class & Physical Dominance (v1: IN)

### 8a — Migration

`is_weight_class_change`, `direction_of_change`, `fights_at_current_class`,
`win_pct_at_current_class`, `prior_class_win_pct`.

### 8b — Weight Bully (product term, not standalone)

`is_large_for_class` (reach + height both top-quartile for class),
`grappling_utilization_rate` (TD attempts + control time, normalized by fight
count), `weight_bully_score` = product.

---

## Section 9 — Graph-Derived (v1: IN except 9d)

Graph recomputed as-of each fight date, no future results.

### 9a — Elo

`elo_rating` (pre-fight), `elo_trajectory_last5` (linear-regression slope of last
5 values), `elo_peak`, `elo_current_vs_peak`. Config in `configs/graph.yaml`.

### 9a-bis — Glicko-2 (added by D31)

`glicko2_rating`, `glicko2_rd` (uncertainty; grows with inactivity). Matchup
deltas computed at §10 level. Parameters in `configs/graph.yaml`.

### 9b — PageRank

`pagerank_score`. Config in `configs/graph.yaml`.

### 9c — Common Opponents

`n_common_opponents`, `common_opp_score_a`, `common_opp_score_b`,
`common_opp_score_delta`, `common_opp_a_win_rate`, `common_opp_b_win_rate`.
3-year lookback, quality weighting by Elo/PageRank at time of fight, recency
decay, NaN when none.

### 9d — Opponent Trajectory (**EXCLUDED** from v1 — D39)

> Uses opponents' post-fight (future) results; unresolved as-of semantics;
> requires dedicated leakage tests. V2 only.

`opp_avg_post_fight_win_pct`, `opp_trajectory_score`,
`loss_to_future_contender`, `win_over_declining_opp`.

---

## Section 10 — Matchup-Level (v1: IN; deltas positive favors A)

`reach_delta`, `height_delta`, `age_delta`, `stance_matchup` (ortho_v_ortho /
ortho_v_south / south_v_south / switch_involved), `southpaw_matchup`,
`elo_delta`, `glicko2_rating_delta`, `glicko2_rd_delta`, `pagerank_delta`,
`finish_rate_delta`, `striking_efficiency_delta`, `td_accuracy_delta`,
`damage_ratio_delta`, `avg_fight_duration_delta` (pace mismatch),
`fight_duration_variance_delta` (chaos vs consistency),
`five_round_experience_delta`, `ufc_experience_delta`, `title_fight_exp_delta`.

### 10a — Grappling Sub-Type Matchup (v1: IN)

`wrestler_score_a/b` = `td_accuracy * td_per_15 * td_defense`;
`submission_score_a/b` = `sub_attempts_per_15 * submission_rate`;
`wrestling_delta`, `submission_delta`, `grappling_type_mismatch` (high wrestler
score one side, high submission score other side).

### 10b — Style Interaction (v1: IN, deterministic fields only)

`striker_vs_grappler`, `pressure_vs_counter`, `pace_mismatch_score`
(`sig_strikes_per_min delta * fight_duration_variance delta`),
`southpaw_orthodox_history` (fighter-specific historical win % vs southpaws —
NOT a generic stance premium).

---

## Section 11 — Rematch (v1: IN minus LLM field)

`is_rematch`, `fights_since_first_meeting`, `result_of_first_meeting`,
`first_meeting_method`, `first_meeting_competitive` (decision or
split/majority), `first_meeting_score_delta`.

`style_change_since_first_meeting` — **EXCLUDED** from model (LLM; D32 — due-diligence layer).

---

## Section 12 — Home Advantage & Geography (v1: partial)

`event_country` — IN (from ufcstats event location).

`fighter_*_home_country_fight`, `home_advantage_delta`,
`fighter_*_travel_required` — require fighter nationality / training base:
Kaggle-gated (D30) or dropped.

---

## Section 15 — Deterministic Weight-Cut & Short-Notice (v1: IN where sourceable)

`missed_weight_last_3`, `missed_weight_career`, `moving_down_in_weight`,
`full_camp` / short-notice — IN where computable from structured ufcstats data.

Camp/gym history fields (`n_camps_last_5yr`, `current_camp_tenure_days`, etc.)
— Kaggle-gated (D30) / deferred (no ufcstats source).

---

## Excluded Sections

### Sections 13–14 — LLM-gated (**EXCLUDED** from v1 model — D32, D39)

Behavioral profiles (§13), camp LLM fields (§14). Superseded by the annotate-only
due-diligence layer (D32) and its v2 promotion path.

### Section 16 — Market-Derived (strategy layer ONLY — D24, D39)

`opening_implied_prob`, `closing_implied_prob`, `line_movement_magnitude`,
`line_movement_direction`, `spread_at_close`, `depth_at_close`,
`volume_last_24hr`. Never model inputs; data preserved via D36 capture extension;
consumed by the report/strategy side only.

### Section 17 — Considered and Excluded

Retirement signal (LLM false-positive rate), judges' scorecards pre-fight (not
available as-of), post-fight medical suspensions (hard leakage), raw win %
without context (superseded by Elo/opponent-adjusted), height differential
standalone (reach captures it), nationality standalone (captured via home
advantage), social-media followers (already priced), performance-vs-expectation
old §7 (odds into model = economic circularity; dropped).
