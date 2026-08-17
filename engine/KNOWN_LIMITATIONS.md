# KNOWN LIMITATIONS

Recorded deliberately. A system whose limitations are only in its author's head
is one where the next person to touch it will mistake an unfitted constant for a
validated finding.

## 0. The existing training data carries a label leak

Found by auditing this project's own Supabase database on 2026-08-17. Recorded
first because it invalidates work that already exists.

| Table | Finding |
|---|---|
| `ufc_fights` (5,703 decided bouts, 2015+) | `fighter_a` wins **58.2%** |
| `fight_predictions` (8,701 rows) | A-side wins **64.8%** |

Scraped fight records conventionally list the winner first, so the A slot is
correlated with the outcome. Any model trained on "A's features minus B's"
learns *the slot* rather than the fighters, and its backtest accuracy does not
survive contact with a live card, where there is no winner to sort by.

Two further measurements on `fight_predictions`:

- **Predictions are nearly constant**: sd = 0.062, mean = 0.520, with 86% of all
  predictions between 0.40 and 0.60 and exactly one above 0.80 — against a
  market that routinely prices fights at 0.20/0.80. The model has very little
  discriminating power.
- **Reliability is off by ~6 points** in every populated bin (predicted 0.54 →
  observed 0.60). Expected calibration error ≈ 0.06, which fails this engine's
  0.03 deployment gate.

**Consequences.** `fight_predictions`, `model_train` and `model_params` cannot
be used to validate anything as they stand, and the engine's Supabase adapter
deliberately does not read them. `schema.canonicalSides()` fixes the leak at the
source by assigning sides from a hash of the two names, which cannot encode the
result; `audit.auditDataset()` fails a dataset that still exhibits it. Rebuild
the training set from `ufc_fights` through the canonical assignment and re-fit.

## 0b. Odds history is the real blocker — recording now started

`picks` holds 126 rows with an opening price and **15** with a closing price.
That is far too few to measure CLV, which is the fastest honest signal available
and a hard requirement in the deployment criteria.

`odds_snapshots` and the hourly `/api/odds-snapshot` cron now exist to fix this
going forward. **The history starts accumulating the moment `ODDS_API_KEY` is
set — not before**, and it can never be backfilled. Until several months of
cards have been captured, CLV remains unmeasurable and `deployReady` stays
false.

## 0c. The walk-forward result: real, honest, and not good enough

`node engine/scripts/backtest.js` on 7,428 UFC bouts (2010-2026), 5,223
out-of-sample predictions across 11 annual folds:

| metric | value |
|---|---|
| log loss (Elo + logistic) | 0.6612 |
| log loss (Elo alone) | 0.6790 |
| log loss (coin flip) | 0.6931 |
| Brier | 0.2344 |
| ECE | 0.0286 (limit 0.03) |
| accuracy | 61.1% |
| prediction range | 16%–82%, sd 0.102 |

The leak audit inside the run reports the raw table at **58.9% A-side wins** and
the canonicalised set at **49.1%** (z = −1.49, OK), so the fix is verified on
the real data every time the backtest runs.

**Read this result honestly.** Beating a coin flip is a low bar. Sports betting
markets typically achieve log loss around 0.62–0.65 on MMA moneylines, and
simply backing the favourite wins about 62–65% of UFC fights — so at 0.661 and
61.1% this model is **probably no better than the closing line, and quite
possibly worse.** Nothing here demonstrates an edge. It demonstrates a working,
leak-free pipeline that produces calibrated probabilities, which is a
prerequisite for finding an edge, not evidence of one.

The comparison that decides it — model versus devigged closing line on the same
fights — cannot be run until §0b's odds history exists.

## 1. The engine cannot currently issue a pick — by design

No historical fight database and no odds feed are connected. Therefore:

- The model has **no out-of-sample calibration record**, so `f_calibration` in
  the edge-shrinkage formula is 0, so the effective probability collapses onto
  the market, so every EV is negative by exactly the vig, so everything is
  vetoed. This chain is tested (`core.test.js`), and it is the intended
  behaviour, not a bug to route around.
- `quality.assess` additionally raises a blocking `UNCALIBRATED_MODEL` defect
  whenever `bout.modelCalibrated` is not true.

Anything that removes these gates without first supplying real data and a real
walk-forward result converts a disciplined system into a random number
generator with good manners.

## 2. Simulator finish rates are now fitted; `TUNING` is not

**Resolved for method rates.** `FIGHTER_DEFAULTS.koRate` and `.subRate` were
fitted with `sensitivity.fitMethodRates` against 5,807 UFC bouts
(2015-01-01 → 2026-08-15) from `ufc_fights`:

| | observed | simulated |
|---|---|---|
| KO/TKO | 31.7% | 31.5% |
| Submission | 17.7% | 18.1% |
| Decision | 49.3% | 50.5% |

**Still unfitted:** everything in `TUNING` — `damageCoef`, `fatigueVulnCoef`,
`cardioTau`, `judgeNoise`, `controlScoreWeight`. These govern *how* the fight
evolves, and only the aggregate outcome has been anchored. Two different
fatigue/damage configurations can both reproduce the marginals while disagreeing
sharply about a specific matchup, so per-fighter parameters and the round
distribution remain unvalidated.

Note also that the fit was performed on an *average-vs-average* matchup. It
anchors the population base rate, not the response to parameter differences.

## 3. The cardio/fight-length interaction is counterintuitive and unresolved

With the shipped constants, a large cardio advantage is worth **less** over five
rounds than over three (≈62% → ≈58% in the test fixture). The mechanism is
explicit and reproducible:

- The fatigue differential between a well- and poorly-conditioned fighter peaks
  around the 10-minute mark and then narrows, because the tired fighter
  saturates near total fatigue while the fit one keeps degrading (pinned in
  `sim.test.js`).
- Extra rounds therefore convert near-certain decision wins for the better
  fighter into finish scrambles, which are shared more evenly.

Conventional MMA wisdom says cardio advantages compound in championship rounds.
The model disagrees. One of them is wrong. **This must be settled with data, not
by tuning constants until the output matches a prior** — that is the overfitting
the system exists to prevent. Until then, treat five-round cardio mismatches as
a known blind spot.

## 4. Draws are impossible in odd-round fights

Judges score every round for one fighter, so with 3 or 5 rounds an individual
card cannot be even and a majority draw cannot arise. Real draws (~0.4% of UFC
bouts) come from 10-8 rounds and point deductions, neither of which is modelled.
This slightly inflates both fighters' win probabilities. It matters most for
three-way moneyline markets, which the engine should not price until fixed.

## 5. Only the simulator is implemented among the five models

The architecture specifies five independent models (statistical, Bayesian, ML,
simulation, market). Implemented today: the **simulator** and the **market**
model. `ensemble.pool` exists, is tested, and refuses to pool unvalidated
components — but with one real model there is no genuine ensemble, and the
"model disagreement" signal is currently the gap between the simulator and the
market rather than a spread across independent modelling approaches.

## 5b. Scheduled rounds are not stored

`ufc_fights` records the round a fight *ended* in, not how many were scheduled.
A five-round fight finishing in round 2 is indistinguishable from a three-round
one. The adapter infers 5 only when a fight reaches round 4+, and leaves
`rounds` null otherwise rather than guessing. Until scheduled length is
available, main-event modelling and any five-round base rate are unreliable.

## 6. Feature engineering is not built

`ARCHITECTURE.md` §3 specifies opponent adjustment, shrinkage, time decay,
fitted aging curves and soft style archetypes. None of it exists yet: the
simulator currently consumes hand-supplied parameters. The shrinkage primitive
(`prob.shrinkToPrior`) is implemented and tested; the pipeline that would use it
is not.

## 7. Correlation handling is partial

- Within one bout, correlation is **measured** on the shared simulation
  (`jointMarket`), which is correct.
- Across bouts, the scanner keeps one selection per bout-side and the exposure
  cap accepts a correlation matrix, but nothing yet **estimates** cross-bout
  correlation (shared camps, shared judges, card-wide conditions). The matrix
  must be supplied by the caller; absent it, positions are treated as
  independent, which understates true risk.

## 8. Untested at scale

The engine has been exercised on synthetic bouts and unit fixtures only. It has
never processed a real card, and no claim about its live behaviour is supported.
