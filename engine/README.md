# Apex Quant Engine

A quantitative decision engine for MMA betting markets. Zero runtime
dependencies, pure functions throughout, deterministic and seeded.

**Its most important behaviour is refusing to bet.** Run the demo and watch it
decline a candidate with a 100/100 data-quality score, a favourable simulation,
and 6/6 adverse scenarios survived — because the model has no calibration record
and therefore has not earned the right to disagree with the market.

```bash
npm --prefix engine test          # 123 tests

# Real walk-forward on 7,428 UFC bouts (needs the two env vars once):
SUPABASE_URL=... SUPABASE_KEY=... node engine/scripts/fetch-data.js
node engine/scripts/backtest.js

node engine/demo.js               # end-to-end walkthrough on a synthetic bout
```

## Documents

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | Full system design, all 18 components, data flow, phase plan |
| `KNOWN_LIMITATIONS.md` | What is unfitted, unbuilt, or known to be wrong |

## Why it currently issues no picks

```
no usable price history
  → no CLV measurement
    → no walk-forward validation
      → no calibration record
        → f_calibration = 0
          → effective probability collapses onto the market price
            → EV is negative by exactly the vig
              → every candidate is vetoed
```

That chain is enforced in code and covered by tests. It is the design working,
not a gap to be patched.

Fight *results* are plentiful — 8,854 UFC bouts in the project's database, and
the simulator's finish rates are fitted against 5,807 of them. **Prices are the
blocker**: 15 closing lines exist in total, no multi-book history, no
timestamps. See `KNOWN_LIMITATIONS.md` §0b.

The same audit found a label leak in the existing training data — the A slot
wins 58-65% because scraped records list winners first. `schema.canonicalSides()`
fixes it at the source and `audit.auditDataset()` fails any dataset that still
shows it. Details in `KNOWN_LIMITATIONS.md` §0.

## Layout

```
src/
  core/prob.js        logit/sigmoid, log-odds pooling, seeded RNG, shrinkage
  core/odds.js        conversions, 4 devig estimators, multi-book consensus
  core/ev.js          EV, variance, and the edge-shrinkage model
  data/quality.js     defect catalogue and the PASS/DEGRADE/BLOCK gate
  data/schema.js      canonical records, validators, leak-free side assignment
  data/audit.js       dataset auditing: label leak, degenerate spread, coverage
  data/adapters/      read-only source adapters (Supabase, The Odds API)
  features/pipeline.js  as-of features, antisymmetric by construction
  models/elo.js       chronological ratings
  models/logistic.js  ridge logistic regression with side-bias detection
  scripts/           fetch-data.js, backtest.js
  sim/montecarlo.js   competing-risks fight simulator
  sim/sensitivity.js  adversarial re-runs, base-rate fitter
  models/ensemble.js  log-odds pooling, market-anchored mode, stacking
  models/registry.js  immutable hashed model cards, retirement criteria
  scoring/score.js    0-100 candidate score and hard vetoes
  risk/bankroll.js    fractional Kelly, caps, correlation-adjusted exposure
  risk/protocol.js    daily loss protocol as a state machine
  calibration/        Brier, log loss, ECE, band reliability, CLV
  backtest/           walk-forward folds, leakage guard, deployment criteria
  scan/               slate scanner, parlay evaluator, report generator
```

## The three ideas that carry the system

**1. A probability difference is not an edge.** The gap between model and market
is shrunk by calibration reliability, model disagreement, data quality, sample
size, adverse-scenario survival, and market quality. Each factor defaults to its
*pessimistic* value when evidence is missing, so an uninstrumented system
shrinks its edge to nothing rather than betting on unmeasured confidence.

```js
p_eff = p_mkt + (f_cal × f_disagree × f_data × f_sample × f_sens × f_mkt) × (p_mod − p_mkt)
```

**2. Uncertainty is simulated, not asserted.** Simulation iterations are grouped
into batches; each batch draws fresh fighter parameters from their posteriors.
The spread *across* batches is genuine epistemic uncertainty. Pure outcome noise
would shrink toward zero with more iterations and tell you nothing about how
much you actually know.

**3. A bad day is only informative if it was improbable.** Before a card, the
engine simulates the P&L distribution of the exact positions it took, including
correlation. Afterwards it asks which percentile the result landed in. A day at
the 20th percentile requires no action at all, and treating it as a signal is
itself a modelling error — the most common one in betting.

## Operating commands

The commands from the operating spec map onto the API:

| Command | Entry point |
|---|---|
| `SCAN` / `DEEP SCAN` | `scanner.scan({bouts, now, bankroll, modelState})` |
| `PARLAY` | `parlay.evaluate({legs, simulation})` — refuses by default |
| `REVIEW` | `metrics.report()` + `clv.aggregate()` + `clv.diagnose()` |
| `RESET` | `registry.evaluateRetirement(record)` |
| `BACKTEST` | `backtest.run({events, fit, predict})` |
| `TRACK` | `clv.betCLV()` per wager into the ledger |

`SAFE` and `VALUE` are staking-policy variants: pass a `policy` with a different
`kellyFraction` and `varianceLambda`. Neither is permitted to bypass the
positive-EV requirement.

## Current status

Walk-forward on 7,428 real bouts, 5,223 out-of-sample predictions:
log loss **0.6612** (Elo alone 0.6790, coin flip 0.6931), ECE **0.0286**,
accuracy **61.1%**, predictions spanning 16%–82%.

Calibrated and leak-free — but backing the favourite wins ~62–65% of UFC fights
and markets typically price them at log loss 0.62–0.65, so **this model is
probably not better than the closing line.** That comparison is the one that
matters and it cannot be run yet. See `KNOWN_LIMITATIONS.md` §0c.

## What to build next

1. **Set `ODDS_API_KEY`** so `/api/odds-snapshot` starts recording. The cron and
   table are built and tested; they capture nothing until the key exists, and
   odds history can never be backfilled.
2. **Wait for coverage** — a few months of cards, then run the model-vs-closing-
   line comparison. That is the first real evidence either way.
3. **Improve the model** meanwhile: opponent-adjusted striking/grappling
   features from per-round data, and the remaining components in
   `ARCHITECTURE.md` §4.
