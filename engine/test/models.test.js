'use strict';
const test = require('node:test');
const assert = require('node:assert');

const elo = require('../src/models/elo');
const logistic = require('../src/models/logistic');
const features = require('../src/features/pipeline');
const metrics = require('../src/calibration/metrics');
const { mulberry32 } = require('../src/core/prob');

const DAY = 86400000;
const T0 = Date.UTC(2020, 0, 1);

// --- Elo ---

test('an unrated pair is a coin flip', () => {
  const book = elo.createBook();
  assert.ok(Math.abs(book.predict('A', 'B', T0) - 0.5) < 1e-12);
});

test('winning raises your rating and lowers your opponent\'s, symmetrically', () => {
  const book = elo.createBook();
  book.update({ a: 'A', b: 'B', winnerSide: 'a', method: 'ko', date: T0 });
  const ra = book.snapshot('A', T0).rating;
  const rb = book.snapshot('B', T0).rating;
  assert.ok(ra > 1500 && rb < 1500);
  assert.ok(Math.abs((ra - 1500) - (1500 - rb)) < 1e-9, 'zero-sum update');
  assert.ok(book.predict('A', 'B', T0) > 0.5);
});

test('a finish moves ratings more than a decision', () => {
  const ko = elo.createBook();
  const dec = elo.createBook();
  ko.update({ a: 'A', b: 'B', winnerSide: 'a', method: 'ko', date: T0 });
  dec.update({ a: 'A', b: 'B', winnerSide: 'a', method: 'dec', date: T0 });
  assert.ok(ko.snapshot('A', T0).rating > dec.snapshot('A', T0).rating);
});

test('ratings regress toward the mean during long layoffs', () => {
  const book = elo.createBook();
  for (let i = 0; i < 5; i++) {
    book.update({ a: 'A', b: `Opp${i}`, winnerSide: 'a', method: 'ko', date: T0 + i * DAY });
  }
  const fresh = book.snapshot('A', T0 + 5 * DAY).rating;
  const stale = book.snapshot('A', T0 + 5 * DAY + 4 * 365 * DAY).rating;
  assert.ok(stale < fresh, 'an idle rating should decay toward 1500');
  assert.ok(stale > 1500, 'but not past the mean');
});

test('records track wins, losses and streaks', () => {
  const book = elo.createBook();
  book.update({ a: 'A', b: 'B', winnerSide: 'a', method: 'ko', date: T0 });
  book.update({ a: 'A', b: 'C', winnerSide: 'a', method: 'dec', date: T0 + DAY });
  book.update({ a: 'A', b: 'D', winnerSide: 'b', method: 'sub', date: T0 + 2 * DAY });
  const s = book.snapshot('A', T0 + 3 * DAY);
  assert.strictEqual(s.fights, 3);
  assert.strictEqual(s.wins, 2);
  assert.strictEqual(s.losses, 1);
  assert.strictEqual(s.streak, -1, 'a loss resets the streak to -1');
  assert.strictEqual(s.finishes, 1);
  assert.strictEqual(s.finishedAgainst, 1);
});

// --- Features ---

function bookWithHistory() {
  const book = elo.createBook();
  for (let i = 0; i < 8; i++) {
    book.update({ a: 'Winner', b: `Opp${i}`, winnerSide: 'a', method: 'ko', date: T0 + i * 30 * DAY });
  }
  for (let i = 0; i < 4; i++) {
    book.update({ a: 'Loser', b: `Foe${i}`, winnerSide: 'b', method: 'dec', date: T0 + i * 30 * DAY });
  }
  return book;
}

test('feature vectors are antisymmetric — a side bias is unrepresentable', () => {
  const book = bookWithHistory();
  const tape = new Map([
    ['Winner', { reachIn: 74, dob: Date.UTC(1994, 0, 1) }],
    ['Loser', { reachIn: 70, dob: Date.UTC(1990, 0, 1) }],
  ]);
  const args = { book, a: 'Winner', b: 'Loser', asOf: T0 + 300 * DAY, tape };
  assert.strictEqual(features.isAntisymmetric(args), true);
});

test('antisymmetry holds even when attributes are missing', () => {
  const book = bookWithHistory();
  const args = { book, a: 'Winner', b: 'Loser', asOf: T0 + 300 * DAY, tape: new Map() };
  assert.strictEqual(features.isAntisymmetric(args), true);
});

test('the stronger fighter produces a positive elo feature', () => {
  const book = bookWithHistory();
  const fv = features.build({ book, a: 'Winner', b: 'Loser', asOf: T0 + 300 * DAY });
  assert.ok(fv.parts.eloDiff > 0);
  assert.ok(fv.parts.experienceDiff > 0);
  assert.ok(fv.parts.formDiff > 0);
  assert.strictEqual(fv.vector.length, features.FEATURES.length);
});

test('coverage reports how much attribute data was available', () => {
  const book = bookWithHistory();
  const full = features.build({
    book, a: 'Winner', b: 'Loser', asOf: T0 + 300 * DAY,
    tape: new Map([['Winner', { reachIn: 74, dob: 1 }], ['Loser', { reachIn: 70, dob: 2 }]]),
  });
  const bare = features.build({ book, a: 'Winner', b: 'Loser', asOf: T0 + 300 * DAY });
  assert.ok(full.coverage > bare.coverage);
  assert.strictEqual(bare.parts.reachDiff, 0, 'absent attributes contribute nothing rather than guessing');
});

test('layoff penalty is zero for a normal turnaround and grows with inactivity', () => {
  assert.strictEqual(features.layoffPenalty(120), 0);
  assert.strictEqual(features.layoffPenalty(180), 0);
  assert.ok(features.layoffPenalty(545) > 0.9 && features.layoffPenalty(545) < 1.1);
  assert.strictEqual(features.layoffPenalty(null), 0);
});

// --- Logistic regression ---

test('logistic recovers a known separating direction', () => {
  const rng = mulberry32(1);
  const X = [], y = [];
  for (let i = 0; i < 800; i++) {
    const x1 = rng() * 4 - 2;
    const x2 = rng() * 4 - 2;
    X.push([x1, x2]);
    y.push(1 / (1 + Math.exp(-(1.5 * x1))) > rng() ? 1 : 0);
  }
  const m = logistic.fit(X, y, { l2: 0.5, iterations: 600 });
  assert.ok(m.weights[0] > 0.5, `expected a strong positive weight on x1, got ${m.weights[0]}`);
  assert.ok(Math.abs(m.weights[1]) < Math.abs(m.weights[0]) / 2, 'the noise feature should be small');
  assert.ok(Math.abs(m.intercept) < 0.2);
});

test('ridge shrinks weights toward zero', () => {
  const rng = mulberry32(2);
  const X = [], y = [];
  for (let i = 0; i < 400; i++) {
    const x = rng() * 4 - 2;
    X.push([x]);
    y.push(x > 0 ? 1 : 0);
  }
  const light = logistic.fit(X, y, { l2: 0.1, iterations: 400 });
  const heavy = logistic.fit(X, y, { l2: 500, iterations: 400 });
  assert.ok(Math.abs(heavy.weights[0]) < Math.abs(light.weights[0]));
});

test('logistic reports a side bias when the labels carry one', () => {
  // Antisymmetric features, but A wins 70% regardless — exactly the artifact
  // canonical side assignment is there to prevent.
  const rng = mulberry32(3);
  const X = [], y = [];
  for (let i = 0; i < 1000; i++) {
    X.push([rng() * 2 - 1]);
    y.push(rng() < 0.7 ? 1 : 0);
  }
  const m = logistic.fit(X, y, { l2: 1, iterations: 600 });
  assert.strictEqual(m.sideBiasDetected, true);
  assert.ok(m.intercept > 0.5);
});

test('a balanced dataset produces no side bias', () => {
  // Probabilistic labels, not a hard threshold: perfectly separable data drives
  // the weight to infinity and lets sampling noise in the class balance leak
  // into the intercept. Real fight outcomes are nowhere near separable.
  const rng = mulberry32(4);
  const X = [], y = [];
  for (let i = 0; i < 4000; i++) {
    const x = rng() * 2 - 1;
    X.push([x]);
    y.push(1 / (1 + Math.exp(-2 * x)) > rng() ? 1 : 0);
  }
  const m = logistic.fit(X, y, { l2: 1, iterations: 600 });
  assert.ok(m.weights[0] > 0.5, 'the real signal should still be found');
  assert.strictEqual(m.sideBiasDetected, false);
  assert.ok(Math.abs(m.intercept) < 0.05);
});

test('fit rejects malformed input', () => {
  assert.throws(() => logistic.fit([], []), /empty design matrix/);
  assert.throws(() => logistic.fit([[1]], [1, 0]), /length mismatch/);
  assert.throws(() => logistic.fit([[1, 2], [1]], [1, 0]), /ragged/);
});

// --- Placebo: the strongest available check that the pipeline does not leak ---

test('PLACEBO — with shuffled labels the model cannot beat a coin flip', () => {
  // If any future information reached the features, a model trained on
  // randomised outcomes would still find signal. It must not.
  const rng = mulberry32(99);
  const names = Array.from({ length: 120 }, (_, i) => `F${i}`);
  const fights = [];
  for (let i = 0; i < 3000; i++) {
    const a = names[Math.floor(rng() * names.length)];
    let b = names[Math.floor(rng() * names.length)];
    if (a === b) b = names[(names.indexOf(a) + 1) % names.length];
    fights.push({ a, b, date: T0 + i * DAY, winnerSide: rng() < 0.5 ? 'a' : 'b', method: 'dec' });
  }

  const book = elo.createBook();
  const X = [], y = [];
  for (const f of fights) {
    const fv = features.build({ book, a: f.a, b: f.b, asOf: f.date });
    X.push(fv.vector);
    y.push(f.winnerSide === 'a' ? 1 : 0);
    book.update(f);
  }

  const split = Math.floor(X.length * 0.7);
  const m = logistic.fit(X.slice(0, split), y.slice(0, split), { l2: 2, iterations: 400 });
  const preds = X.slice(split).map((x, i) => ({
    probability: logistic.predict(m, x), outcome: y[split + i],
  }));
  const ll = metrics.logLoss(preds);

  // Coin-flip log loss is ln 2 = 0.6931. Random labels must not beat it
  // by more than sampling noise.
  assert.ok(ll > 0.685, `shuffled labels produced log loss ${ll.toFixed(4)} — that indicates leakage`);
});

test('features built before an update never see that fight', () => {
  const book = elo.createBook();
  const before = features.build({ book, a: 'X', b: 'Y', asOf: T0 });
  book.update({ a: 'X', b: 'Y', winnerSide: 'a', method: 'ko', date: T0 });
  const after = features.build({ book, a: 'X', b: 'Y', asOf: T0 + DAY });
  assert.strictEqual(before.parts.eloDiff, 0, 'no history means no rating difference');
  assert.ok(after.parts.eloDiff > 0, 'the fight is only visible once absorbed');
});
