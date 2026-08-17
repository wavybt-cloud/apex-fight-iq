'use strict';
const test = require('node:test');
const assert = require('node:assert');

const schema = require('../src/data/schema');
const audit = require('../src/data/audit');
const supabase = require('../src/data/adapters/supabase');
const mc = require('../src/sim/montecarlo');
const sens = require('../src/sim/sensitivity');

// --- Canonical side assignment (the leak fix) ---

test('canonical sides are deterministic and order-independent', () => {
  const x = schema.canonicalSides('Jon Jones', 'Stipe Miocic');
  const y = schema.canonicalSides('Stipe Miocic', 'Jon Jones');
  assert.strictEqual(x.a, y.a);
  assert.strictEqual(x.b, y.b);
  assert.notStrictEqual(x.swapped, y.swapped);
});

test('canonical sides ignore case and surrounding whitespace', () => {
  const x = schema.canonicalSides('Jon Jones', 'Stipe Miocic');
  const y = schema.canonicalSides('  JON JONES ', 'stipe miocic');
  assert.strictEqual(x.swapped, y.swapped);
});

test('canonical side assignment is independent of who won', () => {
  // The property that matters: assignment cannot encode the label, because it
  // never sees it. Across many pairs the A slot should land near 50%.
  let aWins = 0;
  const n = 2000;
  for (let i = 0; i < n; i++) {
    const f1 = `Fighter ${i}`;
    const f2 = `Opponent ${i}`;
    // Let f1 always be the winner — the worst case, mimicking a scraper that
    // lists winners first.
    const sides = schema.canonicalSides(f1, f2);
    if (sides.a === f1) aWins++;
  }
  const rate = aWins / n;
  assert.ok(Math.abs(rate - 0.5) < 0.05,
    `A-slot should carry no information about the winner, got ${rate}`);
});

// --- Leak auditing ---

test('audit detects a side-assignment leak', () => {
  // 58% A-side wins over 5,000 bouts, the pattern measured in this repo's data.
  const rows = Array.from({ length: 5000 }, (_, i) => ({ winnerSide: i % 100 < 58 ? 'a' : 'b' }));
  const r = audit.sideBalance(rows);
  assert.strictEqual(r.leak, true);
  assert.strictEqual(r.verdict, 'SIDE_ASSIGNMENT_LEAK');
  assert.ok(r.aWinRate > 0.57 && r.aWinRate < 0.59);
  assert.ok(Math.abs(r.z) > 3);
  assert.match(r.remedy, /canonicalSides/);
});

test('audit passes a balanced dataset', () => {
  const rows = Array.from({ length: 5000 }, (_, i) => ({ winnerSide: i % 2 ? 'a' : 'b' }));
  const r = audit.sideBalance(rows);
  assert.strictEqual(r.leak, false);
  assert.strictEqual(r.verdict, 'OK');
});

test('audit detects a degenerate near-constant model', () => {
  // sd ~0.06 with everything in 0.4-0.6 — the pattern in fight_predictions.
  const preds = Array.from({ length: 1000 }, (_, i) => ({ probability: 0.52 + ((i % 20) - 10) * 0.006 }));
  const r = audit.predictionSpread(preds);
  assert.strictEqual(r.degenerate, true);
  assert.strictEqual(r.verdict, 'DEGENERATE_SPREAD');
  assert.ok(r.fractionBetween40And60 > 0.9);
});

test('audit passes a model with real spread', () => {
  const preds = Array.from({ length: 1000 }, (_, i) => ({ probability: 0.10 + (i % 80) * 0.01 }));
  const r = audit.predictionSpread(preds);
  assert.strictEqual(r.degenerate, false);
});

test('audit reports insufficient odds history when closing prices are scarce', () => {
  const bouts = Array.from({ length: 500 }, (_, i) => ({
    quotes: [{ book: 'X', observedAt: 1 }],
    closingQuote: i < 15 ? [1.9, 1.9] : null, // 15 closing prices, as in `picks`
  }));
  const r = audit.oddsCoverage(bouts);
  assert.strictEqual(r.clvCapable, false);
  assert.strictEqual(r.verdict, 'INSUFFICIENT_ODDS_HISTORY');
  assert.strictEqual(r.withClosingPrice, 15);
});

test('a leaky dataset is not fit-ready and never deploy-ready', () => {
  const r = audit.auditDataset({
    results: Array.from({ length: 5000 }, (_, i) => ({ winnerSide: i % 100 < 58 ? 'a' : 'b' })),
    bouts: Array.from({ length: 100 }, () => ({ quotes: [], closingQuote: null })),
  });
  assert.strictEqual(r.fitReady, false);
  assert.strictEqual(r.deployReady, false);
  assert.ok(r.blocking.includes('SIDE_ASSIGNMENT_LEAK'));
  assert.ok(r.warnings.includes('INSUFFICIENT_ODDS_HISTORY'));
});

test('a clean dataset without odds is fit-ready but not deploy-ready', () => {
  const r = audit.auditDataset({
    results: Array.from({ length: 5000 }, (_, i) => ({ winnerSide: i % 2 ? 'a' : 'b' })),
    bouts: Array.from({ length: 100 }, () => ({ quotes: [], closingQuote: null })),
  });
  assert.strictEqual(r.fitReady, true);
  assert.strictEqual(r.deployReady, false,
    'no closing prices means no CLV, which gates deployment even with clean data');
});

test('audit detects duplicate bouts', () => {
  const rows = [{ id: 'a' }, { id: 'b' }, { id: 'a' }];
  const r = audit.duplicates(rows, (x) => x.id);
  assert.strictEqual(r.verdict, 'DUPLICATES_PRESENT');
  assert.strictEqual(r.unique, 2);
});

// --- Schema validation ---

test('a quote without a timestamp is rejected', () => {
  const r = schema.validateQuote({ market: 'moneyline', book: 'X', decimals: [1.9, 1.9] });
  assert.strictEqual(r.valid, false);
  assert.ok(r.errors.some((e) => /observedAt/.test(e)));
});

test('a quote with an impossible price is rejected', () => {
  const r = schema.validateQuote({ market: 'moneyline', book: 'X', decimals: [1.0, 1.9], observedAt: 1 });
  assert.strictEqual(r.valid, false);
});

test('method strings normalise onto the engine outcomes', () => {
  assert.strictEqual(schema.normaliseMethod('KO/TKO'), 'ko');
  assert.strictEqual(schema.normaliseMethod('Submission (rear-naked choke)'), 'sub');
  assert.strictEqual(schema.normaliseMethod('Decision - Unanimous'), 'dec');
  assert.strictEqual(schema.normaliseMethod('Decision - Split'), 'dec');
  assert.strictEqual(schema.normaliseMethod('Overturned - No Contest'), 'nc');
  assert.strictEqual(schema.normaliseMethod('DQ'), 'dq');
  assert.strictEqual(schema.normaliseMethod(null), null);
});

test('only clean win/loss results are gradeable', () => {
  assert.strictEqual(schema.isGradeable({ winnerSide: 'a', method: 'ko' }), true);
  assert.strictEqual(schema.isGradeable({ winnerSide: null, method: 'draw' }), false);
  assert.strictEqual(schema.isGradeable({ winnerSide: 'a', method: 'nc' }), false);
  assert.strictEqual(schema.isGradeable({ winnerSide: 'a', method: 'dq' }), false);
});

test('facts require provenance', () => {
  assert.throws(() => schema.fact(1, null, 5), /source is required/);
  assert.throws(() => schema.fact(1, 'src', null), /observedAt is required/);
  const f = schema.fact(1, 'src', 5);
  assert.strictEqual(f.validAsOf, 5);
});

// --- Supabase row mapping ---

test('a raw fight row is canonicalised and the winner mapped to a side', () => {
  const row = {
    id: 42, event: 'UFC 300', event_date: '2024-04-13',
    fighter_a: 'Alex Pereira', fighter_b: 'Jamahal Hill',
    winner: 'Alex Pereira', method: 'KO/TKO', round: 1, weight_class: 'Light Heavyweight',
  };
  const bout = supabase.toHistoricalBout(row);
  const sides = schema.canonicalSides('Alex Pereira', 'Jamahal Hill');
  assert.strictEqual(bout.a.name, sides.a);
  assert.strictEqual(bout.b.name, sides.b);
  assert.strictEqual(bout.result.method, 'ko');
  // The winner must follow the canonical assignment, not the raw column order.
  const expected = sides.a === 'Alex Pereira' ? 'a' : 'b';
  assert.strictEqual(bout.result.winnerSide, expected);
});

test('a winner naming neither participant yields no side rather than a guess', () => {
  const bout = supabase.toHistoricalBout({
    id: 1, fighter_a: 'A Person', fighter_b: 'B Person', winner: 'Someone Else',
    method: 'Decision', round: 3, event_date: '2024-01-01',
  });
  assert.strictEqual(bout.result.winnerSide, null);
  assert.strictEqual(schema.isGradeable(bout.result), false);
});

test('five-round fights are inferred only when the fight proves it', () => {
  const long = supabase.toHistoricalBout({
    id: 1, fighter_a: 'A', fighter_b: 'B', winner: 'A', method: 'KO/TKO', round: 4, event_date: '2024-01-01',
  });
  const short = supabase.toHistoricalBout({
    id: 2, fighter_a: 'A', fighter_b: 'B', winner: 'A', method: 'KO/TKO', round: 2, event_date: '2024-01-01',
  });
  assert.strictEqual(long.rounds, 5);
  assert.strictEqual(short.rounds, null, 'a round-2 finish does not reveal the scheduled length');
});

test('the adapter refuses to build without credentials', () => {
  assert.throws(() => supabase.createAdapter({ url: null, key: null, fetchImpl: () => {} }),
    /url and key are required/);
});

test('the adapter maps history end-to-end through an injected fetch', async () => {
  const rows = [
    { id: 1, event: 'UFC 1', event_date: '2024-01-01', fighter_a: 'Ann Smith', fighter_b: 'Bea Jones', winner: 'Ann Smith', method: 'Submission (armbar)', round: 2 },
    { id: 2, event: 'UFC 1', event_date: '2024-01-01', fighter_a: 'Cal Ray', fighter_b: 'Dee Fox', winner: 'Dee Fox', method: 'Decision - Unanimous', round: 3 },
  ];
  let calledUrl = null;
  const adapter = supabase.createAdapter({
    url: 'https://example.test', key: 'k',
    fetchImpl: async (u) => { calledUrl = u; return { ok: true, json: async () => rows }; },
  });
  const hist = await adapter.fetchHistory({ from: '2024-01-01' });
  assert.strictEqual(hist.length, 2);
  assert.match(calledUrl, /ufc_fights/);
  assert.match(calledUrl, /event_date=gte\.2024-01-01/);
  assert.strictEqual(hist[0].result.method, 'sub');
  assert.strictEqual(hist[1].result.method, 'dec');
  // And the mapped dataset must itself pass the leak audit's mechanics.
  const bal = audit.sideBalance(hist.map((h) => h.result));
  assert.strictEqual(bal.n, 2);
});

test('the adapter surfaces HTTP failures rather than returning empty data', async () => {
  const adapter = supabase.createAdapter({
    url: 'https://example.test', key: 'k',
    fetchImpl: async () => ({ ok: false, status: 401, json: async () => ({}) }),
  });
  await assert.rejects(() => adapter.fetchHistory(), /returned 401/);
});

// --- Simulator now fitted to observed base rates ---

test('the fitted defaults reproduce observed UFC method rates', () => {
  // Anchored to 5,807 UFC bouts (2015-01-01 to 2026-08-15):
  // KO/TKO 31.7% · submission 17.7% · decision 49.3%.
  const even = (name) => Object.assign({}, mc.FIGHTER_DEFAULTS, { name });
  const r = mc.run({ a: even('A'), b: even('B'), rounds: 3, iterations: 40000, batches: 50, seed: 7 });
  const ko = r.method.a.ko + r.method.b.ko;
  const sub = r.method.a.sub + r.method.b.sub;
  assert.ok(Math.abs(ko - 0.317) < 0.03, `KO rate ${ko.toFixed(3)} should sit near the observed 0.317`);
  assert.ok(Math.abs(sub - 0.177) < 0.03, `sub rate ${sub.toFixed(3)} should sit near the observed 0.177`);
  assert.ok(Math.abs(r.goesDistance - 0.493) < 0.04, `decision rate ${r.goesDistance.toFixed(3)} should sit near the observed 0.493`);
});

test('the two-parameter method fitter converges on its targets', () => {
  const even = (name) => Object.assign({}, mc.FIGHTER_DEFAULTS, { name });
  const fit = sens.fitMethodRates(
    { a: even('A'), b: even('B'), rounds: 3, iterations: 8000, batches: 20, seed: 7 },
    { ko: 0.25, sub: 0.25 },
    { tol: 0.02, maxIter: 20 },
  );
  assert.strictEqual(fit.converged, true);
  assert.ok(Math.abs(fit.achieved.ko - 0.25) < 0.03);
  assert.ok(Math.abs(fit.achieved.sub - 0.25) < 0.03);
});
