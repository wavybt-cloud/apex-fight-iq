# Apex Quant Engine

A quantitative decision engine for MMA betting markets. Zero runtime
dependencies, pure functions throughout, deterministic and seeded.

**Its most important behaviour is refusing to bet.** Run the demo and watch it
decline a candidate with a 100/100 data-quality score, a favourable simulation,
and 6/6 adverse scenarios survived — because the model has no calibration record
and therefore has not earned the right to disagree with the market.

```bash
node engine/demo.js          # end-to-end walkthrough on a synthetic bout
npm --prefix engine test     # 74 tests
```

## Documents

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | Full system design, all 18 components, data flow, phase plan |
| `KNOWN_LIMITATIONS.md` | What is unfitted, unbuilt, or known to be wrong |

## Why it currently issues no picks

```
no historical data
  → no walk-forward validation
    → no calibration record
      → f_calibration = 0
        → effective probability collapses onto the market price
          → EV is negative by exactly the vig
            → every candidate is vetoed
```

That chain is enforced in code and covered by tests. It is the design working,
not a gap to be patched. Removing the gates without connecting real data and
completing a walk-forward would produce confident output with nothing behind it.

## Layout

```
src/
  core/prob.js        logit/sigmoid, log-odds pooling, seeded RNG, shrinkage
  core/odds.js        conversions, 4 devig estimators, multi-book consensus
  core/ev.js          EV, variance, and the edge-shrinkage model
  data/quality.js     defect catalogue and the PASS/DEGRADE/BLOCK gate
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

## What to build next

Phase 2 in `ARCHITECTURE.md`: connect a fight-history source, a fighter-attribute
source, and timestamped multi-book odds including closing lines. Everything
downstream is written and tested; it is waiting on data.
