# APEX QUANT ENGINE — ARCHITECTURE

Version: 0.1.0 (foundation)
Sport scope at launch: MMA / UFC (the domain the repo already serves). The core
(odds, EV, Kelly, calibration, CLV, backtest, risk) is sport-agnostic; only the
feature pipeline, simulator and priors are sport-specific.

---

## 0. Honest statement of current capability

This document describes the target system. The code in `engine/src` implements
the sport-agnostic mathematical core and the MMA simulator. **It is not yet a
source of betting recommendations**, for one reason:

> There is no historical fight database and no live odds feed connected to this
> repository. Without them the model cannot be calibrated, cannot be backtested,
> and cannot verify a current price.

Per the operating rules (data-quality gate, "verify the current price", "never
invent odds"), the recommendation engine is wired to **return `NO_BET` with
reason `INSUFFICIENT_DATA` whenever calibration evidence or a verified live
price is missing.** That is the correct output, not a limitation to work around.
Sections 2 and 19 define exactly what must be connected before the gate opens.

Everything shipped is deterministic and unit-tested, so that a claim of "the
simulation was run" is verifiable rather than asserted.

---

## 1. System overview

```
                        ┌────────────────────────────────┐
                        │        MODEL REGISTRY          │
                        │  versions, weights, provenance │
                        └───────────┬────────────────────┘
                                    │ (reads/writes)
 ┌──────────┐   ┌──────────┐   ┌────┴─────┐   ┌──────────┐   ┌──────────┐
 │  DATA    │──▶│ FEATURE  │──▶│  MODEL   │──▶│  MONTE   │──▶│  EDGE /  │
 │ PIPELINE │   │ PIPELINE │   │ ENSEMBLE │   │  CARLO   │   │    EV    │
 └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
      │              │              │              │              │
      ▼              ▼              ▼              ▼              ▼
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │  DATA    │   │ LEAKAGE  │   │  MODEL   │   │SENSITIVITY│  │ CANDIDATE│
 │ QUALITY  │   │  GUARD   │   │DISAGREE- │   │ / ADVERSE │  │  SCORING │
 │  SCORE   │   │(as-of ts)│   │  MENT    │   │ SCENARIOS │  │  (0-100) │
 └────┬─────┘   └──────────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
      │                             │              │              │
      └─────────────┬───────────────┴──────────────┴──────────────┘
                    ▼
             ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
             │   MARKET    │◀────▶│    RISK     │◀────▶│RECOMMENDATION│
             │   ENGINE    │      │   ENGINE    │      │   ENGINE     │
             │ devig, move,│      │ Kelly, caps,│      │ BET / PASS   │
             │ consensus   │      │ correlation,│      │ final report │
             └──────┬──────┘      │ daily halt  │      └──────┬───────┘
                    │             └──────┬──────┘             │
                    │                    │                    ▼
                    │                    │             ┌─────────────┐
                    └────────────────────┴────────────▶│  BET LEDGER │
                                                       └──────┬──────┘
                                                              │ results
                    ┌─────────────────────────────────────────┘
                    ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │  CALIBRATION ENGINE   │──▶│   BACKTEST / WALK-    │
        │ Brier, logloss, ECE,  │   │   FORWARD HARNESS     │
        │ reliability curve     │   │  (retrains ensemble   │
        └──────────┬────────────┘   │   weights, thresholds)│
                   │                └───────────┬───────────┘
                   ▼                            │
        ┌───────────────────────┐               │
        │     CLV TRACKER       │───────────────┘
        │ bet px vs closing px  │   feeds model-health monitor
        └───────────────────────┘
```

The loop is closed: results flow back into calibration → calibration and CLV
feed the model-health monitor → the monitor gates whether a version stays
deployed (§15) and whether the day's betting is halted (§14).

---

## 2. Data pipeline

**Contract.** Every record entering the system carries `(value, source,
observed_at, valid_as_of)`. No component may read a field whose `observed_at`
is after the decision timestamp. This single rule is what prevents look-ahead
bias in backtests and is enforced mechanically by the leakage guard, not by
discipline.

**Layers.**

| Layer | Responsibility |
|---|---|
| `adapters/*` | One module per external source. Converts a raw feed to the canonical schema. Never does analysis. |
| `schema.js` | Canonical `Fighter`, `Bout`, `Event`, `MarketQuote`, `Result` records + validators. Rejects malformed records loudly. |
| `store` | Append-only fact store. Snapshots are immutable; corrections are new rows with a later `observed_at`, never edits. |
| `quality.js` | Scores each bout's data 0–100 and returns the specific defects. |

**Sources required before the engine may issue a pick** (none are connected
today):

1. *Fight history* — bout-level results, method, round, time, per-round
   significant strikes/takedowns/control. Needed for: model fitting,
   calibration, backtesting.
2. *Fighter attributes* — DOB, reach, height, stance, weight class history.
3. *Odds* — opening line, timestamped movement, closing line, across ≥3 books.
   Needed for: price verification, devigging, consensus, CLV.
4. *News* — injuries, weight-miss, short-notice replacement, withdrawal, with
   timestamps.

**Adapter interface** (`data/adapters/index.js`): every adapter exposes
`fetchEvents(range)`, `fetchBout(id)`, `fetchOdds(boutId)`, `fetchResults(range)`,
each returning canonical records with provenance. Swapping a paid feed for a
scraper changes one file.

**Data quality score.** A weighted deduction model over defect classes:
staleness of odds, missing per-round stats, unknown/late lineup or weight-class
change, unresolved injury/short-notice flags, thin sample (< N pro bouts),
book coverage, and internal contradictions between sources. Output:
`{score, defects[], gate}` where `gate ∈ {PASS, DEGRADE, BLOCK}`.
`BLOCK` ⇒ NO BET, unconditionally, regardless of computed edge.
`DEGRADE` ⇒ edge is shrunk and stake is cut (§8, §13).

---

## 3. Feature engineering pipeline

Computed strictly from facts with `observed_at ≤ decision_time`.

- **Rate stats with shrinkage.** Raw per-minute rates from small samples are
  unusable. Every rate is shrunk toward its weight-class prior:
  `θ̂ = (n·x̄ + k·μ_prior) / (n + k)`, where `n` is exposure (minutes, attempts)
  and `k` is the prior strength fitted per statistic. This is §1's "regression
  toward the mean," implemented rather than asserted.
- **Opponent adjustment.** Raw stats are contaminated by schedule. Each stat is
  adjusted by an iterative opponent-strength pass (ridge-regularised, run to
  convergence) so that "landed 5.0/min against elite defense" outranks the same
  rate against journeymen.
- **Time decay.** Exponential recency weight `exp(-Δt/τ)` with τ fitted, not
  guessed. Layoff and age interact: `τ` shortens after long inactivity.
- **Aging curves.** Per-archetype, fitted from data — not hand-coded ladders.
  Until fitted, the hand-coded curves from the existing analyzer are used and
  explicitly flagged `unfitted: true` in the model card.
- **Style representation.** The current hard classifier (7 buckets) is replaced
  by a soft archetype vector from clustering on adjusted stats, so a fighter can
  be 0.6 wrestler / 0.4 boxer. Hard buckets discard information at the boundary.
- **Matchup interactions.** Offense-vs-defense pairings (TD offense × TD
  defense, pressure × footwork), reach × archetype, cardio × pace-of-opponent.
- **Situational.** Short notice, weight miss, altitude, travel/time-zone delta,
  cage size, title/5-round, layoff, post-title-fight letdown, new camp.

**Leakage guard.** The pipeline takes `asOf` as a required argument and throws
if any input carries a later `observed_at`. Backtests run through the identical
code path as live — there is no separate "historical" path to drift out of sync.

---

## 4. Statistical models

`MODEL A — Statistical.` Regularised logistic regression on the engineered
feature vector, predicting P(A wins). Ridge penalty tuned by walk-forward CV.
Linear, inspectable, hard to overfit — the workhorse and the benchmark that any
fancier model must beat out-of-sample.

`MODEL B — Bayesian / hierarchical.` Fighter latent strength with partial
pooling across weight class and archetype; posterior over strength rather than a
point estimate. Yields *uncertainty*, which the edge calculator needs: a 6-point
edge on a fighter with 3 fights and a wide posterior is not the same bet as a
6-point edge on a 20-fight veteran.

`MODEL C — Machine learning.` Gradient-boosted trees on the same features,
capped depth, monotonic constraints where sign is known. Deployed **only if it
beats Model A out-of-sample on log loss across a full walk-forward** — per the
governing rule that a simpler model that performs better out-of-sample wins.

`MODEL D — Simulation.` The Monte Carlo engine (§5). Its win probability is a
model in its own right, and it is the only component that can price method,
round, and total markets coherently.

`MODEL E — Market.` Devigged multi-book consensus. Not a "cheat" — a
well-devigged consensus close is the single hardest baseline in sports
forecasting, and treating it as a model forces every other component to justify
its existence by beating it.

`ENSEMBLE.` Log-odds pooling with weights fitted on out-of-sample performance
(§9), **not** a blind average:

```
logit(p_ens) = Σ wᵢ · logit(pᵢ),   Σ wᵢ = 1,  wᵢ ≥ 0
```

Weights come from stacking on walk-forward folds. Model **disagreement**
(spread of the `pᵢ`) is retained as a first-class signal: high disagreement
shrinks the edge and cuts the stake (§8, §10). Until walk-forward evidence
exists, the ensemble runs in **market-anchored mode**: the market receives the
dominant weight and the engine is not permitted to issue picks — it may only
record paper predictions to build the calibration record.

---

## 5. Monte Carlo engine

A continuous-time competing-risks simulator, not a coin flip with a probability.

**Per simulation:**

1. **Draw parameters, not just outcomes.** Each iteration samples fighter
   parameters from the model's posterior (striking hazard, sub threat, chin,
   cardio decay, judging tendency). This makes the resulting interval a genuine
   *parameter uncertainty* interval rather than pure Monte Carlo noise, which
   would shrink to zero with enough iterations and tell you nothing.
2. **Tick the fight** in 5-second increments across the scheduled rounds.
   At each tick four competing hazards are active: A-by-KO, A-by-sub,
   B-by-KO, B-by-sub. Hazards are modulated by:
   - **fatigue** — a cardio-driven multiplier that raises the opponent's finish
     hazard as the fight lengthens;
   - **accumulated damage** — a state variable that raises KO hazard after
     absorbed volume, which is what makes late-round finishes behave like real
     fights instead of a memoryless process;
   - **positional state** — striking vs. grappling exchange mix, driven by
     takedown offense/defense.
3. **If no finish**, score it. Per-round control scores from simulated
   differential output plus judging noise; majority of rounds wins; 10-8s and
   draws emerge naturally from the scoring model.

**Outputs from a single run** (all markets priced coherently, so parlay
correlation is measured rather than guessed):
P(A), P(B), method distribution per fighter, round distribution,
P(goes the distance), P(over/under X.5 rounds), joint outcome table.

**Reported for every candidate:** model probability, simulation standard error,
parameter-uncertainty interval, and:

**Sensitivity / adversarial re-runs** (§23) — the sim is automatically re-run
under: chin −1σ, cardio −1σ, takedown defense −1σ for the model's side, an
injury-adjustment scenario, and a recency-stripped scenario (last 2 fights
removed). **If the edge does not survive all of them, the candidate is
rejected.** This is a hard gate, not a note in the report.

---

## 6. Market & odds engine

- **Devigging.** Vig removal is a modelling choice with real consequences,
  so four estimators are implemented — multiplicative (proportional), additive,
  power, and Shin (insider-trading model). Favourite-longshot bias makes
  multiplicative systematically wrong on longshots; Shin/power are the honest
  defaults for two-way markets. The engine reports the **spread across methods**
  as market-price uncertainty and uses the least favourable one when computing
  edge.
- **Consensus.** Weighted across books by liquidity/sharpness, not a plain mean.
- **Line movement.** Opening → current path, velocity, and whether movement is
  toward or against the model. Steam and reverse line movement are recorded as
  *features with fitted coefficients*, never as proof of anything (§4 of the
  operating rules).
- **Timing.** Records the timestamp of every quote so CLV is computable and so
  the engine can detect "the edge existed an hour ago and is now gone" — a
  rejection condition, not a reason to bet the stale number.
- **Price verification.** A recommendation carries the exact quote and its
  timestamp; if the quote is older than the freshness threshold, the pick is
  suppressed pending re-verification.

---

## 7. Edge & EV calculation

For a candidate at decimal odds `d`:

```
p_mkt  = devig(market quotes)              // least-favourable estimator
p_mod  = ensemble probability (calibrated)
edge   = p_mod − p_mkt
EV     = p_mod·(d − 1) − (1 − p_mod)       // per unit staked
```

**A raw probability difference is not an edge.** Before it is treated as one it
is shrunk for everything that could be producing it spuriously:

```
p_eff = p_mkt + shrink · (p_mod − p_mkt)

shrink = f_calibration × f_disagreement × f_dataquality × f_sample × f_sensitivity
```

each factor in `[0,1]`: calibration reliability of the model in this probability
band, ensemble disagreement, data-quality score, effective sample behind the
fighters' features, and survival margin under §5's adverse scenarios. Every
staking and scoring decision uses `p_eff`, never `p_mod`. Model uncertainty
therefore mechanically pulls the estimate toward the market, which is the
correct prior when you have no evidence you are better than it.

**Risk-adjusted EV** subtracts a variance penalty: `EV − λ·Var`, with `Var`
from the simulator and `λ` set by the bankroll policy.

---

## 8. Model calibration

Tracked continuously, per model and for the ensemble:
Brier score, log loss, reliability curve over probability deciles, expected
calibration error, ROI, CLV, hit rate, max drawdown, and realised variance vs.
predicted.

Calibration is applied (isotonic regression once sample permits; Platt/logistic
scaling before that) **and** consumed: the reliability of the band a prediction
falls into becomes `f_calibration` in §7. A model that is systematically
overconfident at 70% has its 70% edges shrunk automatically rather than after a
human notices.

ROI alone is explicitly not a fitness function — a high-ROI, badly-calibrated
model is treated as unvalidated.

---

## 9. Backtesting & walk-forward framework

```
TRAIN → VALIDATE → OUT-OF-SAMPLE → WALK-FORWARD → PAPER → LIMITED LIVE
```

- Strictly chronological splits. Purge + embargo around fold boundaries so a
  card's fights cannot straddle train and test.
- Every feature is recomputed through the live `asOf` code path; the harness
  fails loudly if any fact postdates the decision time.
- **Bets are simulated at the price actually available at decision time**, with
  vig, using the recorded quote — not at closing, and not at the best price in
  hindsight.
- Reported per fold: log loss, Brier, ECE, ROI, CLV, hit rate, drawdown,
  bet count, and the same metrics for the market baseline. A strategy that does
  not beat the market baseline on log loss **and** show positive CLV is not
  deployed, whatever its ROI.
- Thresholds (§10) are fitted on walk-forward folds and re-fitted as data
  accumulates, with the fold's own out-of-sample performance as the criterion.

---

## 10. Candidate scoring (0–100)

A transparent weighted score over normalised components:

| Component | Weight | Source |
|---|---|---|
| Shrunk edge (`p_eff − p_mkt`) | 22 | §7 |
| Expected value at the verified price | 18 | §7 |
| Model agreement (inverse disagreement) | 12 | §4 |
| Calibration reliability in this band | 10 | §8 |
| Data quality | 10 | §2 |
| Simulation stability (SE, interval width) | 8 | §5 |
| Sensitivity survival margin | 8 | §5 |
| Market quality (book count, spread, liquidity) | 6 | §6 |
| Historical CLV of this bet type | 6 | §16 |

Bands: 90+ extremely strong · 80–89 strong · 70–79 watchlist · 60–69 no bet ·
<60 reject. **These are unvalidated defaults**, carried as configuration and
overwritten by §9's fitted thresholds once folds exist.

Hard vetoes that bypass the score entirely: data gate `BLOCK`, unverified or
stale price, edge that dies under any adverse scenario, unresolved key-fighter
status, or negative EV at the current number.

---

## 11. Risk & bankroll engine

- **Staking:** fractional Kelly on `p_eff` (never `p_mod`), default ¼-Kelly,
  hard-capped per bet and per event.
- **Correlation:** simultaneous positions are decorrelated via the simulator's
  joint outcome table; total exposure is capped on the *correlation-adjusted*
  sum, so three bets driven by the same fighter's cardio do not masquerade as
  diversification.
- **Automatic de-risking:** stake is cut when data quality degrades, model
  disagreement rises, drawdown deepens, or book liquidity thins.
- **Forbidden by construction:** the API has no path to increase stake after a
  loss. Martingale/chase sizing is not a policy choice that is discouraged — it
  is unrepresentable.

## 12. Daily loss protocol

A state machine: `NORMAL → CAUTION → REVIEW → HALT`.

Transitions fire on *statistical abnormality*, not on a bad feeling: realised
loss beyond the pre-computed simulated daily-loss distribution (e.g. worse than
the 1st percentile of the day's own Monte Carlo P&L distribution), or a
calibration break. In `REVIEW`, new recommendations stop, and the engine
produces a diagnostic partition of the loss: model error vs. data error vs.
news error vs. ordinary variance — because a day inside the simulated
distribution requires no action at all, and treating it as a signal is itself a
modelling error.

No new model is spawned because of a losing day. Version changes require §15's
criteria.

## 13. Model versioning

Immutable model cards: `MODEL_Vn` with training window, feature list, fitted
parameters, hyperparameters, all fold metrics, calibration curve, ROI, CLV, hit
rate, drawdown, and a content hash of the training data. Predictions are stored
with the version hash that produced them. Retro-editing a deployed version is
impossible by construction — a change produces a new version, which must pass
walk-forward validation independently before deployment. This is what makes
"never secretly modify the model after seeing the result" enforceable.

**Retirement criteria** (`RESET`): sustained out-of-sample log-loss degradation
against the market baseline, persistent calibration failure, negative CLV over a
significant sample, feature drift beyond threshold, or a data-pipeline break.

## 14. CLV engine

Every wager records: timestamp, price taken, book, devigged implied probability
at bet, closing price, devigged implied probability at close, CLV in probability
and in price terms, and the result. CLV is the primary short-horizon diagnostic:
it accumulates signal far faster than P&L, so it is what tells you whether a
losing month is variance or breakage.

## 15. Event scanner

For a slate: enumerate every bout × every available market, build features,
run the ensemble and the simulator, price every market from the same simulation,
devig, compute edge and EV, score, drop correlated and dominated candidates,
rank, and return only what clears the gates. It fills no quota. When nothing
clears, the output is `NO QUALIFYING EDGE.`

## 16. Recommendation engine

Emits the §18 report format for every surviving candidate — price and its
timestamp, model vs. market probability, edge, EV, simulation results with
interval, confidence, data quality, risk level, stake, why the model likes it,
**what would make it wrong**, final score, and `BET` or `PASS`. Parlays are
refused by default and only ever constructed from legs that individually clear
the gates with correlation taken from the simulator's joint table (§12 of the
operating rules).

---

## 17. How components communicate

Everything is a pure function over explicit inputs, returning a record with its
provenance. No hidden state, no globals, no ambient mutation — which is what
makes the backtest and the live path provably identical.

```
adapter → canonical records (+observed_at)
  → quality.assess(bout)                → {score, defects, gate}
  → features.build(bout, asOf)          → featureVector      [throws on leakage]
  → models.*.predict(featureVector)     → {p, uncertainty, version}
  → ensemble.pool(predictions, weights) → {p_mod, disagreement}
  → sim.run(params, seed, iters)        → {p, methods, rounds, joint, se, ci}
  → market.devig(quotes)                → {p_mkt, method spread, consensus}
  → edge.compute(p_mod, p_mkt, shrinks) → {p_eff, edge, EV, riskAdjEV}
  → score.rate(all of the above)        → {score 0-100, vetoes[]}
  → risk.size(score, p_eff, odds, book) → {stake, exposure, caps}
  → recommend.emit(...)                 → BET | PASS report
  → ledger.record(...)                  → bet row
       ↓ (after the event)
  → calibration.update(ledger, results) → metrics, reliability
  → clv.update(ledger, closingQuotes)   → CLV series
  → health.evaluate(metrics, clv)       → NORMAL|CAUTION|REVIEW|HALT, RESET flag
       ↓ feeds back into ensemble weights, shrink factors, thresholds
```

Data flows forward; only *fitted parameters* flow backward, and only through the
walk-forward harness. A result can never reach a prediction for an event that
had not yet occurred.

---

## 18. Implementation plan

| Phase | Deliverable | Gate to advance |
|---|---|---|
| **1 — Core (this commit)** | odds/devig math, EV & edge shrinkage, Kelly & exposure caps, Monte Carlo simulator, data-quality gate, scoring, calibration metrics, CLV tracker, risk state machine, scanner + recommender, walk-forward harness skeleton. Unit-tested, zero dependencies. | Tests green; every module pure and deterministic. |
| **2 — Data** | Adapters for fight history, fighter attributes, multi-book odds with timestamps, news. Fact store with `observed_at`. | Enough history to fit and validate; odds snapshots including closes. |
| **3 — Fit** | Fit Model A, priors for Model B, Elo, simulator hazard parameters. Feature pipeline with shrinkage + opponent adjustment. | Walk-forward log loss beats the devigged market baseline. |
| **4 — Validate** | Full walk-forward, calibration curves, threshold fitting, ensemble stacking weights. | Positive CLV and calibrated probabilities out-of-sample. |
| **5 — Paper** | Live paper trading, no money. CLV tracked on every paper bet. | Positive CLV over a meaningful sample. |
| **6 — Limited live** | Small fractional-Kelly deployment, daily protocol armed. | Reviewed at fixed intervals against §15 retirement criteria. |

The gate between phases 4 and 5, and between 5 and 6, is the entire point of the
system. Until phase 4 completes, the engine's honest output for any real slate
is `NO QUALIFYING EDGE — INSUFFICIENT DATA`.
