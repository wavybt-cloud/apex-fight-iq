# KNOWN LIMITATIONS

Recorded deliberately. A system whose limitations are only in its author's head
is one where the next person to touch it will mistake an unfitted constant for a
validated finding.

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

## 2. Simulator constants are priors, not findings

`TUNING` in `sim/montecarlo.js` and the values in `FIGHTER_DEFAULTS` are
plausible starting points chosen by hand. They have not been fitted to observed
MMA data. Consequences:

- **Absolute** method and round probabilities should not be trusted yet. With
  the shipped defaults, evenly matched three-round fights reach the scorecards
  about 37% of the time; the real UFC figure is closer to 50%. Use
  `sensitivity.fitFinishRate()` against a real base rate before trusting method
  markets.
- **Relative** comparisons are better behaved: monotonicity in every parameter
  is tested, so "who is favoured and roughly by how much" is more reliable than
  "how often this ends in round 2".

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
