'use strict';
const test = require('node:test');
const assert = require('node:assert');

const quality = require('../src/data/quality');
const metrics = require('../src/calibration/metrics');
const clv = require('../src/calibration/clv');
const protocol = require('../src/risk/protocol');
const registry = require('../src/models/registry');
const ensemble = require('../src/models/ensemble');
const scoring = require('../src/scoring/score');
const scanner = require('../src/scan/scanner');
const walkforward = require('../src/backtest/walkforward');
const { americanToDecimal } = require('../src/core/odds');

const NOW = Date.UTC(2026, 7, 16, 18, 0, 0);

function goodFighter(name, over) {
  return Object.assign({
    name,
    confirmed: true, bouts: 15, statsUpdatedAt: NOW - 5 * 86400000, hasRoundDetail: true,
    koRate: 0.5, subRate: 0.2, durability: 1.0, subDefense: 1.0, cardio: 1.0,
    output: 4.0, accuracy: 0.45, grappling: 1.0, takedownDefense: 1.0, control: 0.3,
  }, over || {});
}

function goodBout(over) {
  return Object.assign({
    id: 'bout-1', event: 'Test Card',
    a: goodFighter('Fighter A'), b: goodFighter('Fighter B'),
    rounds: 3, modelCalibrated: true,
    quotes: [
      { market: 'moneyline', book: 'X', decimals: [1.95, 1.95], observedAt: NOW - 60000 },
      { market: 'moneyline', book: 'Y', decimals: [1.93, 1.98], observedAt: NOW - 30000 },
    ],
  }, over || {});
}

// --- Data quality gate ---

test('a complete bout passes the data quality gate', () => {
  const q = quality.assess(goodBout(), NOW);
  assert.strictEqual(q.gate, 'PASS');
  assert.ok(q.score >= 70, `score was ${q.score}`);
});

test('an unconfirmed fighter blocks the bout outright', () => {
  const q = quality.assess(goodBout({ a: goodFighter('A', { confirmed: false }) }), NOW);
  assert.strictEqual(q.gate, 'BLOCK');
  assert.ok(q.blocking.some((d) => d.key === 'FIGHTER_STATUS_UNKNOWN'));
});

test('an uncalibrated model blocks every bout', () => {
  const q = quality.assess(goodBout({ modelCalibrated: false }), NOW);
  assert.strictEqual(q.gate, 'BLOCK');
  assert.ok(q.blocking.some((d) => d.key === 'UNCALIBRATED_MODEL'));
});

test('very stale odds block; merely stale odds only degrade', () => {
  const stale = quality.assess(goodBout({
    quotes: [
      { market: 'moneyline', book: 'X', decimals: [1.95, 1.95], observedAt: NOW - 30 * 60000 },
      { market: 'moneyline', book: 'Y', decimals: [1.93, 1.98], observedAt: NOW - 30 * 60000 },
    ],
  }), NOW);
  assert.notStrictEqual(stale.gate, 'BLOCK');
  assert.ok(stale.defects.some((d) => d.key === 'STALE_ODDS'));

  const veryStale = quality.assess(goodBout({
    quotes: [{ market: 'moneyline', book: 'X', decimals: [1.95, 1.95], observedAt: NOW - 5 * 3600000 }],
  }), NOW);
  assert.strictEqual(veryStale.gate, 'BLOCK');
});

test('missing odds entirely blocks the bout', () => {
  const q = quality.assess(goodBout({ quotes: [] }), NOW);
  assert.strictEqual(q.gate, 'BLOCK');
  assert.ok(q.blocking.some((d) => d.key === 'NO_ODDS'));
});

// --- Calibration ---

test('log loss rewards calibrated confidence and punishes confident errors', () => {
  const good = [{ probability: 0.9, outcome: 1 }, { probability: 0.1, outcome: 0 }];
  const bad = [{ probability: 0.9, outcome: 0 }, { probability: 0.1, outcome: 1 }];
  assert.ok(metrics.logLoss(good) < metrics.logLoss(bad));
  assert.ok(metrics.brierScore(good) < metrics.brierScore(bad));
});

test('a perfectly calibrated set has near-zero calibration error', () => {
  const preds = [];
  for (let i = 0; i < 1000; i++) preds.push({ probability: 0.7, outcome: i % 10 < 7 ? 1 : 0 });
  assert.ok(metrics.expectedCalibrationError(preds) < 0.01);
});

test('band reliability returns zero without enough history', () => {
  const thin = [{ probability: 0.65, outcome: 1 }, { probability: 0.66, outcome: 1 }];
  assert.strictEqual(metrics.bandReliability(thin, 0.65), 0);
  assert.strictEqual(metrics.bandReliability([], 0.65), 0);
});

test('band reliability rewards a well-calibrated band and punishes an overconfident one', () => {
  const calibrated = [];
  for (let i = 0; i < 200; i++) calibrated.push({ probability: 0.65, outcome: i % 100 < 65 ? 1 : 0 });
  const overconfident = [];
  for (let i = 0; i < 200; i++) overconfident.push({ probability: 0.65, outcome: i % 100 < 45 ? 1 : 0 });
  const good = metrics.bandReliability(calibrated, 0.65);
  const bad = metrics.bandReliability(overconfident, 0.65);
  assert.ok(good > 0.7, `well-calibrated band scored ${good}`);
  assert.strictEqual(bad, 0, 'a 20-point calibration gap must earn no credit at all');
});

test('beating the market is measured, not assumed', () => {
  // Model is closer to the truth than the market on every observation.
  const preds = [];
  for (let i = 0; i < 200; i++) {
    const outcome = i % 10 < 6 ? 1 : 0;
    preds.push({ probability: 0.6, marketProb: 0.5, outcome });
  }
  const r = metrics.report(preds, null);
  assert.strictEqual(r.beatsMarket, true);
  assert.ok(r.logLossVsMarket > 0);
});

// --- CLV ---

test('CLV is positive when the market moves toward the side taken', () => {
  const r = clv.betCLV({
    betQuote: [2.10, 1.80],   // took A at 2.10
    closeQuote: [1.85, 2.00], // A shortened: the market agreed
    outcomeIndex: 0,
    decimalOdds: 2.10,
  });
  assert.ok(r.clvProbability > 0);
  assert.ok(r.beatClose);
  assert.ok(r.clvPercent > 0);
});

test('CLV is negative when the market moves away from the side taken', () => {
  const r = clv.betCLV({
    betQuote: [1.85, 2.00], closeQuote: [2.10, 1.80],
    outcomeIndex: 0, decimalOdds: 1.85,
  });
  assert.ok(r.clvProbability < 0);
  assert.ok(!r.beatClose);
});

test('CLV aggregate refuses to call a small sample significant', () => {
  const bets = [{ betQuote: [2.10, 1.80], closeQuote: [1.85, 2.00], outcomeIndex: 0, decimalOdds: 2.10 }];
  const agg = clv.aggregate(bets);
  assert.strictEqual(agg.significant, false);
  assert.strictEqual(clv.diagnose(agg, { profit: 500 }).verdict, 'INSUFFICIENT_SAMPLE');
});

test('CLV diagnosis distinguishes bad luck from a broken process', () => {
  const strong = { n: 200, significant: true, meanClvProbability: 0.02 };
  const weak = { n: 200, significant: true, meanClvProbability: -0.02 };
  assert.strictEqual(clv.diagnose(strong, { profit: -300 }).verdict, 'GOOD_PROCESS_BAD_LUCK');
  assert.strictEqual(clv.diagnose(weak, { profit: 300 }).verdict, 'LUCKY');
  assert.strictEqual(clv.diagnose(weak, { profit: -300 }).verdict, 'BROKEN');
  assert.strictEqual(clv.diagnose(strong, { profit: 300 }).verdict, 'HEALTHY');
});

// --- Daily loss protocol ---

test('an ordinary losing day does NOT trigger review', () => {
  const positions = Array.from({ length: 5 }, () => ({ stake: 100, probability: 0.55, decimalOdds: 2.0 }));
  // Losing 3 of 5 is entirely unremarkable.
  const day = protocol.evaluateDay({
    positions, realisedPnL: -100, bankroll: 9900, peakBankroll: 10000,
  });
  assert.strictEqual(day.state, 'NORMAL');
  assert.ok(day.percentile > 0.05);
  assert.strictEqual(day.permits.newBets, true);
});

test('a statistically abnormal day triggers review and suspends new bets', () => {
  const positions = Array.from({ length: 8 }, () => ({ stake: 100, probability: 0.60, decimalOdds: 2.0 }));
  const day = protocol.evaluateDay({
    positions, realisedPnL: -800, bankroll: 9200, peakBankroll: 10000, // lost every bet
  });
  assert.strictEqual(day.state, 'REVIEW');
  assert.strictEqual(day.permits.newBets, false);
  assert.strictEqual(day.permits.stakeMultiplier, 0);
});

test('a deep drawdown halts betting regardless of the day', () => {
  const day = protocol.evaluateDay({
    positions: [], realisedPnL: 0, bankroll: 7500, peakBankroll: 10000,
  });
  assert.strictEqual(day.state, 'HALT');
  assert.strictEqual(day.permits.newBets, false);
});

test('too few positions yields no inference from the result', () => {
  const day = protocol.evaluateDay({
    positions: [{ stake: 500, probability: 0.6, decimalOdds: 2.0 }],
    realisedPnL: -500, bankroll: 9500, peakBankroll: 10000,
  });
  assert.strictEqual(day.state, 'NORMAL');
  assert.ok(day.reasons.some((r) => /too few to infer/i.test(r)));
});

test('correlated positions produce a fatter loss tail than independent ones', () => {
  const mk = (group) => Array.from({ length: 6 }, () => ({
    stake: 100, probability: 0.6, decimalOdds: 2.0, group,
  }));
  const indep = protocol.simulateDayPnL(mk(null), { seed: 5, iterations: 20000 });
  const corr = protocol.simulateDayPnL(mk('same-card'), { seed: 5, iterations: 20000 });
  assert.ok(corr.p01 < indep.p01, 'correlated exposure must widen the left tail');
});

test('loss attribution refuses to blame the model for a single day', () => {
  const a = protocol.attributeLoss({ percentile: 0.03 });
  assert.strictEqual(a.conclusion, null);
  assert.match(a.warning, /single day is not a sample/i);
  assert.ok(a.checks.some((c) => c.cause === 'VARIANCE'));
});

// --- Ensemble ---

test('ensemble refuses to pool unvalidated models', () => {
  assert.throws(() => ensemble.pool([
    { key: 'x', probability: 0.6, weight: 1, validated: false },
  ]), /no validated models/);
});

test('market-anchored mode caps how far an unvalidated model may deviate', () => {
  const r = ensemble.marketAnchored(0.85, 0.50, { modelWeight: 0.15, maxDeviation: 0.05 });
  assert.ok(r.probability <= 0.55 + 1e-9);
  assert.strictEqual(r.mode, 'MARKET_ANCHORED');
  assert.match(r.note, /paper predictions only/);
});

test('stacking weights favour the model that fits out-of-sample', () => {
  const rows = [];
  for (let i = 0; i < 400; i++) {
    const outcome = i % 10 < 7 ? 1 : 0;
    rows.push({ probs: { good: 0.7, noise: 0.35 }, outcome });
  }
  const fit = ensemble.fitWeights(rows, ['good', 'noise']);
  assert.ok(fit.weights.good > fit.weights.noise,
    `expected the accurate model to be weighted higher: ${JSON.stringify(fit.weights)}`);
});

// --- Scoring ---

test('a veto zeroes the score no matter how good the rest looks', () => {
  const r = scoring.rate({
    edge: {
      positive: true, effectiveEdge: 0.08, evEffective: 0.15, effectiveProb: 0.6,
      shrink: { disagreement: 1, calibration: 1, product: 0.9 }, variance: 1,
    },
    simulation: { standardError: 0.001, paramInterval: [0.55, 0.60], iterations: 20000 },
    sensitivity: { survival: 1, allSurvive: true, scenarios: [] },
    quality: { gate: 'BLOCK', score: 20, defects: [], blocking: [{ label: 'Unconfirmed fighter' }] },
    market: { bookCount: 5, meanOverround: 0.04 },
  });
  assert.strictEqual(r.score, 0);
  assert.strictEqual(r.action, 'PASS');
  assert.ok(r.vetoes.some((v) => v.key === 'DATA_QUALITY_BLOCK'));
});

test('an edge that dies under adverse assumptions is vetoed', () => {
  const r = scoring.rate({
    edge: {
      positive: true, effectiveEdge: 0.05, evEffective: 0.10, effectiveProb: 0.6,
      shrink: { disagreement: 1, calibration: 1, product: 0.9 }, variance: 1,
    },
    simulation: { standardError: 0.001, paramInterval: [0.55, 0.60], iterations: 20000 },
    sensitivity: {
      survival: 0.5, allSurvive: false,
      scenarios: [{ key: 'chin_minus_1sd', survives: false }, { key: 'cardio_minus_1sd', survives: true }],
    },
    quality: { gate: 'PASS', score: 90, defects: [], blocking: [] },
    market: { bookCount: 5, meanOverround: 0.04 },
  });
  assert.ok(r.vetoed);
  assert.ok(r.vetoes.some((v) => v.key === 'EDGE_DIES_UNDER_ADVERSE_ASSUMPTIONS'));
});

test('missing sensitivity analysis is itself a veto', () => {
  const r = scoring.rate({
    edge: { positive: true, effectiveEdge: 0.05, evEffective: 0.1, shrink: { disagreement: 1, product: 1 }, variance: 1 },
    simulation: { standardError: 0.001, paramInterval: [0.55, 0.6], iterations: 1000 },
    quality: { gate: 'PASS', score: 90, defects: [], blocking: [] },
    market: { bookCount: 3, meanOverround: 0.04 },
  });
  assert.ok(r.vetoes.some((v) => v.key === 'NO_SENSITIVITY_ANALYSIS'));
});

test('score thresholds are never marked validated by default', () => {
  const r = scoring.rate({
    edge: { positive: true, effectiveEdge: 0.05, evEffective: 0.1, shrink: { disagreement: 1, product: 1 }, variance: 1 },
    simulation: { standardError: 0.001, paramInterval: [0.55, 0.6], iterations: 1000 },
    sensitivity: { survival: 1, allSurvive: true, scenarios: [] },
    quality: { gate: 'PASS', score: 90, defects: [], blocking: [] },
    market: { bookCount: 3, meanOverround: 0.04 },
  });
  assert.strictEqual(r.thresholdsValidated, false);
});

// --- Model registry ---

test('a model card is frozen and hashed', () => {
  const card = registry.createCard({
    version: 'MODEL_V1', createdAt: NOW, features: ['b', 'a'],
    parameters: { w: 1 }, trainingWindow: { from: 0, to: NOW, n: 500 },
  });
  assert.ok(card.hash);
  assert.strictEqual(card.deployable, false, 'a card without validation is not deployable');
  assert.throws(() => { 'use strict'; card.parameters = { w: 2 }; });
  assert.deepStrictEqual(card.features, ['a', 'b'], 'features are canonicalised for hashing');
});

test('editing a model produces a different hash, so history cannot be rewritten', () => {
  const base = { version: 'MODEL_V1', createdAt: NOW, features: ['a'], trainingWindow: { from: 0, to: NOW, n: 5 } };
  const c1 = registry.createCard(Object.assign({}, base, { parameters: { w: 1 } }));
  const c2 = registry.createCard(Object.assign({}, base, { parameters: { w: 2 } }));
  assert.notStrictEqual(c1.hash, c2.hash);
  const pred = registry.stampPrediction(c1, { probability: 0.6 });
  assert.strictEqual(registry.verifyPrediction(c1, pred), true);
  assert.strictEqual(registry.verifyPrediction(c2, pred), false,
    'a prediction must not validate against a modified version');
});

test('retirement requires a meaningful sample, not a losing run', () => {
  const small = registry.evaluateRetirement({ n: 20, ece: 0.9, clv: { significant: true, meanClvProbability: -0.05 } });
  assert.strictEqual(small.retire, false);
  assert.strictEqual(small.verdict, 'INSUFFICIENT_SAMPLE');

  const large = registry.evaluateRetirement({
    n: 400, ece: 0.09, clv: { significant: true, meanClvProbability: -0.03 },
  });
  assert.strictEqual(large.retire, true);
  assert.ok(large.triggered.some((t) => t.key === 'calibrationFailure'));
  assert.ok(large.triggered.some((t) => t.key === 'negativeCLV'));
  assert.match(large.note, /validate it independently/);
});

// --- Walk-forward ---

test('walk-forward reports insufficient history rather than inventing a result', () => {
  const r = walkforward.run({
    events: [{ date: 1 }, { date: 2 }],
    fit: () => ({}), predict: () => null,
  });
  assert.strictEqual(r.verdict, 'INSUFFICIENT_HISTORY');
  assert.strictEqual(r.aggregate, null);
});

test('leakage guard throws when a fact postdates the decision', () => {
  assert.throws(
    () => walkforward.assertNoLeakage([{ observedAt: NOW + 1000 }], NOW, 'odds'),
    /LEAKAGE/,
  );
  assert.strictEqual(walkforward.assertNoLeakage([{ observedAt: NOW - 1000 }], NOW), true);
});

test('folds are chronological and embargoed', () => {
  const events = Array.from({ length: 400 }, (_, i) => ({ date: i * 86400000, id: i }));
  const folds = walkforward.makeFolds(events, { trainMin: 100, testSize: 50, embargoMs: 7 * 86400000 });
  assert.ok(folds.length >= 2);
  for (const f of folds) {
    const maxTrain = Math.max(...f.train.map((e) => e.date));
    const minTest = Math.min(...f.test.map((e) => e.date));
    assert.ok(maxTrain < minTest, 'training data must precede test data');
    assert.ok(minTest - maxTrain >= 7 * 86400000, 'the embargo gap must be respected');
  }
});

test('walk-forward refuses to deploy a model that does not beat the market', () => {
  const events = Array.from({ length: 400 }, (_, i) => ({ date: i * 86400000, id: i, outcome: i % 2 }));
  const r = walkforward.run({
    events,
    foldOptions: { trainMin: 100, testSize: 50 },
    fit: () => ({}),
    // A coin-flip model against a market that is also a coin flip, with no CLV.
    predict: (m, ev) => ({ probability: 0.5, marketProb: 0.5, outcome: ev.outcome, stake: 0 }),
  });
  assert.strictEqual(r.deployable, false);
  assert.ok(r.failures.length > 0);
  assert.strictEqual(r.verdict, 'NOT_DEPLOYABLE');
});

// --- End-to-end scan ---

test('the scanner returns NO QUALIFYING EDGE on a clean slate with no calibration record', () => {
  const out = scanner.scan({
    bouts: [goodBout()],
    now: NOW,
    bankroll: 10000,
    modelState: {},   // no calibration history — the realistic current state
    options: { iterations: 3000, batches: 12, minScore: 80 },
  });
  assert.strictEqual(out.qualifying, 0);
  assert.strictEqual(out.summary, 'NO QUALIFYING EDGE.');
  assert.ok(out.marketsEvaluated > 0, 'markets should still be evaluated and reported');
  for (const c of out.allCandidates) {
    assert.strictEqual(c.stake.stake, 0, 'no stake may be recommended without calibration');
  }
});

test('the scanner blocks everything when the data gate fails, and says why', () => {
  const out = scanner.scan({
    bouts: [goodBout({ a: goodFighter('A', { confirmed: false }) })],
    now: NOW, bankroll: 10000,
    options: { iterations: 2000, batches: 8 },
  });
  assert.strictEqual(out.qualifying, 0);
  assert.ok(out.blocked && out.blocked.some((b) => /DATA_QUALITY_BLOCK/.test(b)));
});

test('the scanner is deterministic for a given slate and seed', () => {
  const args = {
    bouts: [goodBout()], now: NOW, bankroll: 10000,
    options: { iterations: 2000, batches: 8, seed: 4242 },
  };
  const a = scanner.scan(args);
  const b = scanner.scan(args);
  assert.strictEqual(a.allCandidates[0].simulatedProbability, b.allCandidates[0].simulatedProbability);
  assert.strictEqual(a.allCandidates[0].scored.score, b.allCandidates[0].scored.score);
});

test('the scanner keeps only one selection per bout side', () => {
  const dropped = scanner.dropCorrelated([
    { boutId: 'x', side: 'A', scored: { score: 95 }, selection: 'best' },
    { boutId: 'x', side: 'A', scored: { score: 90 }, selection: 'duplicate opinion' },
    { boutId: 'x', side: 'B', scored: { score: 88 }, selection: 'other side' },
  ]);
  assert.strictEqual(dropped.length, 2);
  assert.strictEqual(dropped[0].selection, 'best');
});

test('a report is emitted for every market, including passes', () => {
  const out = scanner.scan({
    bouts: [goodBout()], now: NOW, bankroll: 10000,
    options: { iterations: 2000, batches: 8 },
  });
  const c = out.allCandidates[0];
  assert.ok(c.emitted.text.includes('FINAL SCORE'));
  assert.ok(c.emitted.text.includes('WHAT COULD MAKE THE MODEL WRONG'));
  assert.strictEqual(c.emitted.verdict, 'PASS');
  assert.ok(c.emitted.report.whatCouldMakeTheModelWrong.length > 0);
});

test('a verified price is required — the American price round-trips into the report', () => {
  const out = scanner.scan({
    bouts: [goodBout()], now: NOW, bankroll: 10000,
    options: { iterations: 2000, batches: 8 },
  });
  const c = out.allCandidates.find((x) => x.marketKey === 'ML_A');
  assert.strictEqual(c.decimalOdds, 1.95, 'the best available price must be used');
  assert.strictEqual(c.emitted.report.currentPrice.american, -105);
  assert.ok(Math.abs(americanToDecimal(-105) - 1.952) < 0.002);
  assert.strictEqual(c.emitted.report.currentPrice.book, 'X');
});
