'use strict';
const test = require('node:test');
const assert = require('node:assert');

const mc = require('../src/sim/montecarlo');
const sens = require('../src/sim/sensitivity');
const parlay = require('../src/scan/parlay');

const EVEN = {
  koRate: 0.45, subRate: 0.20, durability: 1.0, subDefense: 1.0,
  cardio: 1.0, output: 4.0, accuracy: 0.45, grappling: 1.0,
  takedownDefense: 1.0, control: 0.3,
};
const f = (name, over) => Object.assign({}, EVEN, over || {}, { name });

test('simulation is reproducible from its seed', () => {
  const args = { a: f('A'), b: f('B'), rounds: 3, iterations: 4000, batches: 20, seed: 99 };
  const r1 = mc.run(args);
  const r2 = mc.run(args);
  assert.strictEqual(r1.probA, r2.probA);
  assert.strictEqual(r1.goesDistance, r2.goesDistance);
  const r3 = mc.run(Object.assign({}, args, { seed: 100 }));
  assert.notStrictEqual(r1.probA, r3.probA, 'a different seed should give a different path');
});

test('identical fighters produce a symmetric result', () => {
  const r = mc.run({ a: f('A'), b: f('B'), rounds: 3, iterations: 30000, batches: 60, seed: 5 });
  assert.ok(Math.abs(r.probA - 0.5) < 0.02, `expected ~50/50, got ${r.probA}`);
});

test('all outcome probabilities form a valid distribution', () => {
  const r = mc.run({ a: f('A'), b: f('B', { koRate: 0.8 }), rounds: 5, iterations: 8000, batches: 20, seed: 3 });
  assert.ok(Math.abs(r.probA + r.probB + r.probDraw - 1) < 1e-9);
  const methodSum = r.method.a.ko + r.method.a.sub + r.method.a.dec
    + r.method.b.ko + r.method.b.sub + r.method.b.dec + r.method.draw;
  assert.ok(Math.abs(methodSum - 1) < 1e-9, `method probabilities summed to ${methodSum}`);
  const finishSum = r.finishRound.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(finishSum + r.goesDistance - 1) < 1e-9);
  assert.strictEqual(r.finishRound[0], 0, 'there is no round 0');
});

test('a stronger fighter wins more often — monotonicity in each parameter', () => {
  const base = { rounds: 3, iterations: 8000, batches: 20, seed: 11 };
  const even = mc.run(Object.assign({ a: f('A'), b: f('B') }, base)).probA;
  const better = mc.run(Object.assign({ a: f('A', { koRate: 1.2 }), b: f('B') }, base)).probA;
  const tougher = mc.run(Object.assign({ a: f('A', { durability: 1.6 }), b: f('B') }, base)).probA;
  const fitter = mc.run(Object.assign({ a: f('A', { cardio: 1.8 }), b: f('B') }, base)).probA;
  assert.ok(better > even, 'higher finishing rate must raise win probability');
  assert.ok(tougher > even, 'higher durability must raise win probability');
  assert.ok(fitter > even, 'better cardio must raise win probability');
});

test('the fatigue differential widens early, then peaks and narrows', () => {
  // The differential PEAKS around the 10-minute mark and narrows after it,
  // because the poorly-conditioned fighter saturates near total fatigue while
  // the well-conditioned one keeps degrading. This is the mechanism behind the
  // model's counterintuitive five-round behaviour — see KNOWN_LIMITATIONS.md.
  // Pinned here so that a change to the fatigue curve cannot pass unnoticed.
  const gap = (min) => mc.fatigueAt(min, 0.6) - mc.fatigueAt(min, 2.0);
  assert.ok(gap(5) > gap(0));
  assert.ok(gap(10) > gap(5));
  assert.ok(gap(25) < gap(10), 'saturation must narrow the differential late');
  assert.ok(gap(10) > 0.4, 'the peak differential should be material');
});

test('a cardio advantage is worth more than nothing at both fight lengths', () => {
  const mk = (rounds) => mc.run({
    a: f('A', { cardio: 2.0 }), b: f('B', { cardio: 0.6 }),
    rounds, iterations: 12000, batches: 30, seed: 21,
  }).probA;
  assert.ok(mk(3) > 0.5, 'better cardio must be an advantage over three rounds');
  assert.ok(mk(5) > 0.5, 'better cardio must be an advantage over five rounds');
});

test('more scheduled rounds means fewer fights reach the scorecards', () => {
  const three = mc.run({ a: f('A'), b: f('B'), rounds: 3, iterations: 8000, batches: 20, seed: 4 });
  const five = mc.run({ a: f('A'), b: f('B'), rounds: 5, iterations: 8000, batches: 20, seed: 4 });
  assert.ok(five.goesDistance < three.goesDistance);
});

test('parameter interval reflects epistemic uncertainty, not iteration count', () => {
  const wide = mc.run({
    a: f('A'), b: f('B'), rounds: 3, iterations: 12000, batches: 40, seed: 8,
    uncertainty: { a: { koRate: 0.9, durability: 0.5 }, b: { koRate: 0.9, durability: 0.5 } },
  });
  const narrow = mc.run({
    a: f('A'), b: f('B'), rounds: 3, iterations: 12000, batches: 40, seed: 8,
    uncertainty: { a: { koRate: 0.02, durability: 0.02 }, b: { koRate: 0.02, durability: 0.02 } },
  });
  const w = wide.paramInterval[1] - wide.paramInterval[0];
  const n = narrow.paramInterval[1] - narrow.paramInterval[0];
  assert.ok(w > n, `wider priors must give a wider interval (${w} vs ${n})`);
});

test('accumulated damage makes later rounds relatively more dangerous', () => {
  // With state-dependent hazards, per-round finish probability conditional on
  // reaching that round should not be flat.
  const r = mc.run({ a: f('A'), b: f('B'), rounds: 5, iterations: 30000, batches: 40, seed: 17 });
  let survived = 1;
  const conditional = [];
  for (let i = 1; i <= 5; i++) {
    conditional.push(r.finishRound[i] / survived);
    survived -= r.finishRound[i];
  }
  assert.ok(conditional[4] > conditional[0],
    `expected conditional finish rate to rise with damage: ${conditional.map((x) => x.toFixed(3))}`);
});

test('market predicates are mutually consistent', () => {
  const r = mc.run({ a: f('A'), b: f('B', { koRate: 0.9 }), rounds: 3, iterations: 8000, batches: 20, seed: 6 });
  const dist = mc.priceMarket(r, mc.MARKETS.goesDistance).probability;
  const inside = mc.priceMarket(r, mc.MARKETS.endsInside).probability;
  assert.ok(Math.abs(dist + inside - 1) < 1e-9);
  const aWin = mc.priceMarket(r, mc.MARKETS.winnerA).probability;
  const aParts = mc.MARKETS.aByKo, aS = mc.MARKETS.aBySub, aD = mc.MARKETS.aByDec;
  const sum = mc.priceMarket(r, aParts).probability
    + mc.priceMarket(r, aS).probability + mc.priceMarket(r, aD).probability;
  assert.ok(Math.abs(aWin - sum) < 1e-9, 'method probabilities must partition the win probability');
});

test('winning and method are correlated — measured, not assumed', () => {
  const r = mc.run({
    a: f('A', { koRate: 1.4, output: 5.5 }), b: f('B', { durability: 0.8 }),
    rounds: 3, iterations: 20000, batches: 30, seed: 12,
  });
  const j = mc.jointMarket(r, mc.MARKETS.winnerA, mc.MARKETS.endsInside);
  assert.ok(Math.abs(j.pXY - j.independentPXY) > 0.005,
    'a favourite winning and the fight ending inside must not be independent');
  // The sign is NEGATIVE even for a heavy finisher: a dominant fighter converts
  // the scorecards at a far higher rate than they convert scrambles, so the
  // underdog's live chance is concentrated in inside finishes. This is exactly
  // the kind of relationship a book's independence assumption misprices, and
  // the reason correlation is measured rather than assumed.
  assert.ok(j.correlation < 0, `expected negative correlation, got ${j.correlation}`);
  const pAGivenInside = j.pXY / j.pY;
  const pAGivenDistance = (j.pX - j.pXY) / (1 - j.pY);
  assert.ok(pAGivenDistance > pAGivenInside);
});

test('sensitivity analysis reduces the backed side probability under adverse assumptions', () => {
  const base = { a: f('A', { koRate: 1.1, durability: 1.3 }), b: f('B'), rounds: 3, iterations: 6000, batches: 20, seed: 31 };
  const s = sens.runSensitivity(base, 'A', 0.50);
  assert.strictEqual(s.scenarios.length, sens.ADVERSE_SCENARIOS.length);
  assert.ok(s.worst.probability <= s.baseline + 0.02,
    'the worst adverse scenario should not exceed the baseline');
  assert.ok(s.survival >= 0 && s.survival <= 1);
});

test('sensitivity survival goes to zero when the market price already exceeds the model', () => {
  const base = { a: f('A'), b: f('B'), rounds: 3, iterations: 4000, batches: 16, seed: 33 };
  const s = sens.runSensitivity(base, 'A', 0.95); // market says 95%, model says ~50%
  assert.strictEqual(s.survival, 0);
  assert.strictEqual(s.allSurvive, false);
});

test('finish-rate fitter hits a target base rate', () => {
  const sample = { a: f('A'), b: f('B'), rounds: 3, iterations: 4000, batches: 16, seed: 41 };
  const fit = sens.fitFinishRate(sample, 0.55, { tol: 0.02, maxIter: 18 });
  assert.ok(Math.abs(fit.achieved - 0.55) < 0.03,
    `fitter should reach the target distance rate, got ${fit.achieved}`);
  assert.ok(fit.multiplier > 0);
});

test('parlay is refused when a leg is not independently +EV', () => {
  const sim = mc.run({ a: f('A'), b: f('B'), rounds: 3, iterations: 4000, batches: 16, seed: 51 });
  const r = parlay.evaluate({
    simulation: sim,
    legs: [
      { selection: 'A ML', decimalOdds: 2.0, test: mc.MARKETS.winnerA, legEdge: { positive: true } },
      { selection: 'Inside', decimalOdds: 2.0, test: mc.MARKETS.endsInside, legEdge: { positive: false } },
    ],
  });
  assert.strictEqual(r.recommend, false);
  assert.strictEqual(r.verdict, 'PASS');
  assert.ok(r.refusals.some((x) => /not independently \+EV/.test(x)));
});

test('parlay refuses when correlation cannot be measured', () => {
  const r = parlay.evaluate({
    legs: [
      { selection: 'x', decimalOdds: 2, test: () => true, legEdge: { positive: true } },
      { selection: 'y', decimalOdds: 2, test: () => true, legEdge: { positive: true } },
    ],
  });
  assert.strictEqual(r.recommend, false);
  assert.ok(r.refusals.some((x) => /Correlation may not be guessed/.test(x)));
});

test('parlay measures positive correlation rather than assuming independence', () => {
  // A control-heavy grinder against a durable opponent: "A wins" and "goes the
  // distance" genuinely co-occur, which is the structurally sound parlay case.
  const sim = mc.run({
    a: f('A', { koRate: 0.15, subRate: 0.05, grappling: 2.0, control: 0.7, output: 4.5 }),
    b: f('B', { koRate: 0.15, subRate: 0.05, durability: 1.6, takedownDefense: 0.6 }),
    rounds: 3, iterations: 20000, batches: 30, seed: 61,
  });
  const r = parlay.evaluate({
    simulation: sim,
    legs: [
      { selection: 'A ML', decimalOdds: 1.5, test: mc.MARKETS.winnerA, legEdge: { positive: true } },
      { selection: 'Goes the distance', decimalOdds: 1.8, test: mc.MARKETS.goesDistance, legEdge: { positive: true } },
    ],
  });
  assert.ok(r.jointProbability > r.independenceProbability,
    `positively correlated legs must beat the independence assumption (${r.jointProbability} vs ${r.independenceProbability})`);
  assert.ok(r.correlationBenefit > 0);
  assert.strictEqual(r.pairwiseCorrelations.length, 1);
  assert.ok(r.pairwiseCorrelations[0].correlation > 0);
});

test('parlay refuses negatively correlated legs even when both are +EV', () => {
  const sim = mc.run({
    a: f('A', { koRate: 1.5, output: 6 }), b: f('B', { durability: 0.7 }),
    rounds: 3, iterations: 20000, batches: 30, seed: 61,
  });
  const r = parlay.evaluate({
    simulation: sim,
    legs: [
      { selection: 'A ML', decimalOdds: 1.5, test: mc.MARKETS.winnerA, legEdge: { positive: true } },
      { selection: 'Inside', decimalOdds: 1.8, test: mc.MARKETS.endsInside, legEdge: { positive: true } },
    ],
  });
  assert.strictEqual(r.recommend, false);
  assert.ok(r.refusals.some((x) => /negatively correlated/.test(x)),
    'the book\'s independence assumption favours the book on negatively correlated legs');
});
