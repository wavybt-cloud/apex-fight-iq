'use strict';
// Real walk-forward backtest on the project's UFC history.
//
//   node engine/scripts/fetch-data.js     # once, to populate engine/data
//   node engine/scripts/backtest.js
//
// Protocol:
//   * Sides are assigned by schema.canonicalSides — a hash of the two names,
//     which cannot encode the result. Without this the A slot wins 58% and
//     every metric below is fiction.
//   * Fights are processed in strict chronological order. A fight is PREDICTED
//     using ratings built only from earlier fights, and only then absorbed.
//   * Folds are chronological with an embargo; the model is refitted per fold
//     on the data available at that point, never on the whole history.

const fs = require('fs');
const path = require('path');

const schema = require('../src/data/schema');
const audit = require('../src/data/audit');
const elo = require('../src/models/elo');
const logistic = require('../src/models/logistic');
const features = require('../src/features/pipeline');
const metrics = require('../src/calibration/metrics');

const DATA = path.join(__dirname, '..', 'data');
const TRAIN_START = Date.UTC(2010, 0, 1);
const TEST_START = Date.UTC(2016, 0, 1);
const FOLD_MONTHS = 12;

function load(name) {
  const p = path.join(DATA, name);
  if (!fs.existsSync(p)) {
    console.error(`Missing ${p}. Run: node engine/scripts/fetch-data.js`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function prepare() {
  const raw = load('ufc_fights.json');
  const tottRows = load('fighter_tott.json');
  const tape = new Map(tottRows.map((r) => [r.fighter, {
    reachIn: r.reach_in != null ? Number(r.reach_in) : null,
    dob: r.dob ? Date.parse(r.dob) : null,
  }]));

  const bouts = [];
  let rawAWins = 0, rawDecided = 0;

  for (const r of raw) {
    if (!r.fighter_a || !r.fighter_b || r.fighter_a === r.fighter_b) continue;
    if (!r.event_date) continue;
    const date = Date.parse(r.event_date);
    if (!Number.isFinite(date) || date < TRAIN_START) continue;
    const method = schema.normaliseMethod(r.method);
    if (!['ko', 'sub', 'dec'].includes(method)) continue;
    if (!r.winner || (r.winner !== r.fighter_a && r.winner !== r.fighter_b)) continue;

    rawDecided++;
    if (r.winner === r.fighter_a) rawAWins++;

    const sides = schema.canonicalSides(r.fighter_a, r.fighter_b);
    bouts.push({
      id: String(r.id),
      date,
      a: sides.a,
      b: sides.b,
      winnerSide: r.winner === sides.a ? 'a' : 'b',
      method,
      round: r.round != null ? Number(r.round) : null,
    });
  }

  bouts.sort((x, y) => x.date - y.date || (x.id < y.id ? -1 : 1));
  return { bouts, tape, rawAWinRate: rawDecided ? rawAWins / rawDecided : null, rawDecided };
}

function foldBoundaries(bouts) {
  const marks = [];
  let t = TEST_START;
  const last = bouts[bouts.length - 1].date;
  while (t < last) {
    const d = new Date(t);
    marks.push(t);
    t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + FOLD_MONTHS, 1);
  }
  return marks;
}

function run() {
  const { bouts, tape, rawAWinRate, rawDecided } = prepare();

  console.log('='.repeat(74));
  console.log('WALK-FORWARD BACKTEST — UFC moneyline (fighter A wins)');
  console.log('='.repeat(74));
  console.log(`Usable bouts: ${bouts.length}  (${new Date(bouts[0].date).toISOString().slice(0, 10)} → ${new Date(bouts[bouts.length - 1].date).toISOString().slice(0, 10)})`);

  // --- Leak audit, before and after canonical assignment ---
  const rawAudit = audit.sideBalance(
    Array.from({ length: rawDecided }, (_, i) => ({ winnerSide: i < Math.round(rawAWinRate * rawDecided) ? 'a' : 'b' })),
  );
  const fixedAudit = audit.sideBalance(bouts);
  console.log('\n[LEAK AUDIT]');
  console.log(`  raw table order:        A wins ${(rawAWinRate * 100).toFixed(1)}%  → ${rawAudit.verdict}`);
  console.log(`  canonical assignment:   A wins ${(fixedAudit.aWinRate * 100).toFixed(1)}%  (z = ${fixedAudit.z.toFixed(2)}) → ${fixedAudit.verdict}`);

  const marks = foldBoundaries(bouts);
  const book = elo.createBook();
  const allPreds = [];
  const foldRows = [];

  let cursor = 0;
  const trainingRows = { X: [], y: [] };

  for (let fi = 0; fi < marks.length; fi++) {
    const foldStart = marks[fi];
    const foldEnd = fi + 1 < marks.length ? marks[fi + 1] : Infinity;

    // Absorb everything strictly before the fold, accumulating training rows.
    while (cursor < bouts.length && bouts[cursor].date < foldStart) {
      const f = bouts[cursor];
      const fv = features.build({ book, a: f.a, b: f.b, asOf: f.date, tape });
      if (f.date >= TEST_START - 6 * 365.25 * 86400000) {
        trainingRows.X.push(fv.vector);
        trainingRows.y.push(f.winnerSide === 'a' ? 1 : 0);
      }
      book.update({ a: f.a, b: f.b, winnerSide: f.winnerSide, method: f.method, date: f.date });
      cursor++;
    }

    if (trainingRows.X.length < 500) continue;
    const model = logistic.fit(trainingRows.X, trainingRows.y, { l2: 2.0, iterations: 500 });

    // Predict the fold, then absorb it.
    const preds = [];
    while (cursor < bouts.length && bouts[cursor].date < foldEnd) {
      const f = bouts[cursor];
      const fv = features.build({ book, a: f.a, b: f.b, asOf: f.date, tape });
      const outcome = f.winnerSide === 'a' ? 1 : 0;
      preds.push({
        probability: logistic.predict(model, fv.vector),
        eloProbability: book.predict(f.a, f.b, f.date),
        outcome,
      });
      trainingRows.X.push(fv.vector);
      trainingRows.y.push(outcome);
      book.update({ a: f.a, b: f.b, winnerSide: f.winnerSide, method: f.method, date: f.date });
      cursor++;
    }
    if (!preds.length) continue;

    allPreds.push(...preds);
    foldRows.push({
      from: new Date(foldStart).toISOString().slice(0, 7),
      n: preds.length,
      trainN: model.n,
      logLoss: metrics.logLoss(preds),
      eloLogLoss: metrics.logLoss(preds.map((p) => ({ probability: p.eloProbability, outcome: p.outcome }))),
      brier: metrics.brierScore(preds),
      acc: preds.filter((p) => (p.probability >= 0.5 ? 1 : 0) === p.outcome).length / preds.length,
      intercept: model.intercept,
      sideBias: model.sideBiasDetected,
    });
  }

  // --- Per-fold table ---
  console.log('\n[FOLDS]  (out-of-sample; model refitted before each)');
  console.log('  period    n    train   logLoss   elo_LL   brier    acc    intercept');
  for (const f of foldRows) {
    console.log(`  ${f.from}  ${String(f.n).padStart(4)}  ${String(f.trainN).padStart(5)}   `
      + `${f.logLoss.toFixed(4)}   ${f.eloLogLoss.toFixed(4)}  ${f.brier.toFixed(4)}  `
      + `${(f.acc * 100).toFixed(1)}%   ${f.intercept >= 0 ? ' ' : ''}${f.intercept.toFixed(4)}${f.sideBias ? '  <-- SIDE BIAS' : ''}`);
  }

  // --- Aggregate ---
  const report = metrics.report(allPreds, null);
  const eloPreds = allPreds.map((p) => ({ probability: p.eloProbability, outcome: p.outcome }));
  const coinLL = metrics.logLoss(allPreds.map((p) => ({ probability: 0.5, outcome: p.outcome })));

  console.log('\n[AGGREGATE OUT-OF-SAMPLE]');
  console.log(`  predictions          ${report.n}`);
  console.log(`  log loss (logistic)  ${report.logLoss.toFixed(4)}`);
  console.log(`  log loss (elo only)  ${metrics.logLoss(eloPreds).toFixed(4)}`);
  console.log(`  log loss (coin flip) ${coinLL.toFixed(4)}`);
  console.log(`  brier                ${report.brier.toFixed(4)}`);
  console.log(`  ECE                  ${report.ece.toFixed(4)}   (deployment limit 0.03)`);
  console.log(`  accuracy             ${(allPreds.filter((p) => (p.probability >= 0.5 ? 1 : 0) === p.outcome).length / allPreds.length * 100).toFixed(1)}%`);

  console.log('\n[RELIABILITY]');
  console.log('   bin        n   predicted  observed     gap');
  for (const b of report.reliability) {
    if (!b.n) continue;
    console.log(`  ${b.lo.toFixed(1)}-${b.hi.toFixed(1)}  ${String(b.n).padStart(5)}     `
      + `${b.predicted.toFixed(3)}     ${b.observed.toFixed(3)}  ${b.gap >= 0 ? ' ' : ''}${b.gap.toFixed(3)}`);
  }

  const spread = audit.predictionSpread(allPreds);
  console.log('\n[SPREAD]');
  console.log(`  ${spread.detail}`);

  console.log('\n[VERDICT]');
  const beatsCoin = report.logLoss < coinLL;
  const calibrated = report.ece < 0.03;
  console.log(`  beats a coin flip:            ${beatsCoin ? 'yes' : 'NO'}`);
  console.log(`  calibrated (ECE < 0.03):      ${calibrated ? 'yes' : 'NO'}`);
  console.log('  beats the market:             UNKNOWN — no odds history to compare against');
  console.log('  positive CLV:                 UNKNOWN — 15 closing prices in the database');
  console.log('\n  DEPLOYABLE: NO. Beating a coin flip is not the bar; beating the closing');
  console.log('  line is, and that cannot be evaluated until odds are recorded.');
  console.log('='.repeat(74));
}

run();
