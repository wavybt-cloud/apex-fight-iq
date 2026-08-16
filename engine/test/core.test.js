'use strict';
const test = require('node:test');
const assert = require('node:assert');

const odds = require('../src/core/odds');
const ev = require('../src/core/ev');
const prob = require('../src/core/prob');
const bankroll = require('../src/risk/bankroll');

test('american/decimal/implied conversions round-trip', () => {
  for (const a of [-300, -142, -110, 100, 120, 250, 900]) {
    const d = odds.americanToDecimal(a);
    assert.strictEqual(odds.decimalToAmerican(d), a, `round-trip failed for ${a}`);
  }
  assert.ok(Math.abs(odds.americanToImplied(-110) - 0.5238) < 0.001);
  assert.ok(Math.abs(odds.americanToImplied(+100) - 0.5) < 1e-9);
});

test('overround is computed correctly for a standard -110/-110 market', () => {
  const d = [-110, -110].map(odds.americanToDecimal);
  assert.ok(Math.abs(odds.overround(d) - 0.0476) < 0.001);
});

test('every devig estimator produces probabilities summing to 1', () => {
  const cases = [[-300, 250], [-110, -110], [-1000, 650], [150, -180]];
  for (const c of cases) {
    const d = c.map(odds.americanToDecimal);
    const r = odds.devig(d);
    for (const [name, ps] of Object.entries(r.probs)) {
      const s = ps.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(s - 1) < 1e-6, `${name} on ${c} summed to ${s}`);
      for (const p of ps) assert.ok(p > 0 && p < 1, `${name} produced out-of-range ${p}`);
    }
  }
});

test('devig estimators disagree on longshots — which is why the spread is reported', () => {
  const d = [-1000, 650].map(odds.americanToDecimal);
  const r = odds.devig(d);
  // The longshot is index 1; multiplicative should price it higher than Shin,
  // which is the favourite-longshot correction doing its job.
  assert.ok(r.probs.multiplicative[1] > r.probs.shin[1],
    'expected multiplicative to assign the longshot a higher probability than Shin');
  assert.ok(r.spread[1] > 0.001, 'expected a material spread across estimators on a longshot');
});

test('conservative devig reading is the least favourable to the bettor', () => {
  const d = [-300, 250].map(odds.americanToDecimal);
  const r = odds.devig(d);
  for (let i = 0; i < 2; i++) {
    const all = Object.values(r.probs).map((ps) => ps[i]);
    assert.strictEqual(r.conservative(i), Math.max(...all));
  }
});

test('consensusMarket picks the best available price per outcome', () => {
  const m = odds.consensusMarket([
    { book: 'X', decimals: [1.90, 2.00] },
    { book: 'Y', decimals: [1.95, 1.95] },
  ]);
  assert.strictEqual(m.bestDecimal[0], 1.95);
  assert.strictEqual(m.bestBook[0], 'Y');
  assert.strictEqual(m.bestDecimal[1], 2.00);
  assert.strictEqual(m.bestBook[1], 'X');
  assert.strictEqual(m.bookCount, 2);
});

test('EV is zero at the break-even probability and positive above it', () => {
  const d = 2.0;
  assert.ok(Math.abs(ev.expectedValue(0.5, d)) < 1e-12);
  assert.ok(ev.expectedValue(0.55, d) > 0);
  assert.ok(ev.expectedValue(0.45, d) < 0);
});

test('log-odds pooling stays between its inputs and respects weights', () => {
  const p = prob.poolLogOdds([0.4, 0.8], [1, 1]);
  assert.ok(p > 0.4 && p < 0.8);
  const heavyLow = prob.poolLogOdds([0.4, 0.8], [9, 1]);
  assert.ok(heavyLow < p, 'weighting the low estimate should pull the pool down');
});

test('shrinkToPrior regresses small samples toward the prior', () => {
  const heavy = prob.shrinkToPrior(1.0, 2, 0.4, 10);   // 2 observations
  const light = prob.shrinkToPrior(1.0, 200, 0.4, 10); // 200 observations
  assert.ok(heavy < 0.55, 'a 2-sample observation should sit near the prior');
  assert.ok(light > 0.95, 'a 200-sample observation should sit near the observed value');
});

// --- The central safety property of the whole engine ---

test('edge shrinks to ZERO when the model has no calibration record', () => {
  const r = ev.computeEdge({
    modelProb: 0.65,
    marketProb: 0.50,
    decimalOdds: 2.0,
    context: {
      // calibrationReliability omitted entirely — the uninstrumented case
      disagreement: 0, dataQuality: 100, effectiveSample: 100,
      sensitivitySurvival: 1, bookCount: 5,
    },
  });
  assert.strictEqual(r.shrink.calibration, 0);
  assert.strictEqual(r.shrink.product, 0);
  assert.ok(Math.abs(r.effectiveProb - 0.50) < 1e-12,
    'with no calibration record the estimate must collapse onto the market');
  assert.strictEqual(r.effectiveEdge, 0);
  assert.strictEqual(r.positive, false,
    'a 15-point raw edge must NOT be bettable without a calibration record');
});

test('edge survives proportionally when every input is strong', () => {
  const r = ev.computeEdge({
    modelProb: 0.65, marketProb: 0.50, decimalOdds: 2.0,
    context: {
      calibrationReliability: 0.9, disagreement: 0.01, dataQuality: 95,
      effectiveSample: 40, sensitivitySurvival: 1, bookCount: 6,
    },
  });
  assert.ok(r.shrink.product > 0.5, `expected substantial retained edge, got ${r.shrink.product}`);
  assert.ok(r.effectiveEdge > 0 && r.effectiveEdge < r.rawEdge,
    'effective edge must be positive but strictly smaller than the raw difference');
  assert.ok(r.positive);
});

test('poor data quality alone can extinguish the edge', () => {
  const base = {
    modelProb: 0.65, marketProb: 0.50, decimalOdds: 2.0,
    context: {
      calibrationReliability: 0.9, disagreement: 0.01, dataQuality: 95,
      effectiveSample: 40, sensitivitySurvival: 1, bookCount: 6,
    },
  };
  const good = ev.computeEdge(base);
  const bad = ev.computeEdge(Object.assign({}, base, {
    context: Object.assign({}, base.context, { dataQuality: 50 }),
  }));
  assert.strictEqual(bad.shrink.dataQuality, 0);
  assert.strictEqual(bad.effectiveEdge, 0);
  assert.ok(good.effectiveEdge > 0);
});

// --- Staking ---

test('Kelly refuses a negative-EV price', () => {
  const r = bankroll.sizeBet({ bankroll: 10000, probability: 0.45, decimalOdds: 2.0 });
  assert.strictEqual(r.stake, 0);
  assert.match(r.reason, /NON_POSITIVE_KELLY/);
});

test('fractional Kelly is capped and scales down with drawdown', () => {
  const flat = bankroll.sizeBet({ bankroll: 10000, probability: 0.60, decimalOdds: 2.0 });
  const drawn = bankroll.sizeBet({ bankroll: 10000, probability: 0.60, decimalOdds: 2.0, drawdown: 0.12 });
  assert.ok(flat.stake > 0);
  assert.ok(drawn.stake < flat.stake, 'drawdown must reduce stake');
  assert.ok(flat.fraction <= bankroll.DEFAULT_POLICY.maxStakePctBankroll + 1e-12,
    'per-bet cap must bind');
});

test('correlated positions produce higher effective exposure than independent ones', () => {
  const stakes = [100, 100, 100];
  const independent = bankroll.portfolioExposure(stakes, [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
  ]);
  const correlated = bankroll.portfolioExposure(stakes, [
    [1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1],
  ]);
  assert.ok(correlated.effective > independent.effective);
  assert.ok(correlated.concentrationRatio > independent.concentrationRatio);
  assert.ok(correlated.concentrationRatio <= 1.0001);
});

test('portfolio cap scales stakes down when exposure breaches the limit', () => {
  const bets = [{ stake: 400, units: 4 }, { stake: 400, units: 4 }];
  const corr = [[1, 0.95], [0.95, 1]];
  const out = bankroll.applyPortfolioCaps(bets, 10000, corr, { maxEventExposurePct: 0.05 });
  assert.ok(out.breached);
  assert.ok(out.bets[0].stake < 400);
  assert.strictEqual(out.bets[0].capped, 'PORTFOLIO_EXPOSURE_CAP');
});
