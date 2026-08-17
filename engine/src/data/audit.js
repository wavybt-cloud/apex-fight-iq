'use strict';
// Dataset auditing.
//
// This module exists because of a defect found in this project's own data
// rather than as a theoretical precaution. Before any model is fitted, the
// training set is audited for artifacts that make a model look good in
// backtest and fail live. Each check returns a verdict and, where it can, the
// remedy.

const { mean, binomialSE } = require('../core/prob');

/**
 * Side-assignment leak.
 *
 * If the A slot wins materially more than half the time, the slot itself is
 * correlated with the label. A model fed A-minus-B features will learn the slot
 * rather than the fighters, and its backtest accuracy is fictitious.
 *
 * Measured on this repo's data at the time of writing: A-side won 58.2% of
 * 5,703 UFC bouts since 2015 — about 12 standard errors from 50%, which is not
 * chance.
 *
 * @param {Array<{winnerSide:'a'|'b'|null}>} rows
 */
function sideBalance(rows, opts) {
  const o = Object.assign({ tolerance: 0.02 }, opts || {});
  const decided = rows.filter((r) => r.winnerSide === 'a' || r.winnerSide === 'b');
  if (!decided.length) {
    return { n: 0, verdict: 'NO_DATA', aWinRate: null, leak: false };
  }
  const p = mean(decided.map((r) => (r.winnerSide === 'a' ? 1 : 0)));
  const se = binomialSE(0.5, decided.length);
  const z = se > 0 ? (p - 0.5) / se : 0;
  const leak = Math.abs(p - 0.5) > o.tolerance && Math.abs(z) > 3;
  return {
    n: decided.length,
    aWinRate: p,
    z,
    leak,
    verdict: leak ? 'SIDE_ASSIGNMENT_LEAK' : 'OK',
    detail: leak
      ? `The A slot wins ${(p * 100).toFixed(1)}% of bouts (z = ${z.toFixed(1)}). The slot is correlated with the outcome, so any model trained on A-vs-B differences will learn the slot rather than the fighters.`
      : `A-side win rate ${(p * 100).toFixed(1)}%, consistent with balanced assignment.`,
    remedy: leak
      ? 'Re-assign sides with schema.canonicalSides(), which orders by a hash of the names and is therefore independent of the result. Then re-fit; the previous model\'s metrics are void.'
      : null,
  };
}

/**
 * Predictive spread. A model whose probabilities barely move is not forecasting;
 * it is emitting a constant near the base rate. It can still post a respectable
 * log loss while being useless for finding mispriced games, so this is checked
 * separately from calibration.
 *
 * Measured on this repo's `fight_predictions`: sd = 0.062 with 86% of all
 * predictions inside 0.4-0.6, against a market that routinely prices fights
 * at 0.20 and 0.80.
 */
function predictionSpread(preds, opts) {
  const o = Object.assign({ minStdev: 0.10, minRange: 0.40 }, opts || {});
  if (!preds.length) return { n: 0, verdict: 'NO_DATA' };
  const ps = preds.map((p) => p.probability);
  const m = mean(ps);
  const sd = Math.sqrt(mean(ps.map((x) => (x - m) * (x - m))));
  const lo = Math.min(...ps), hi = Math.max(...ps);
  const inMiddle = ps.filter((x) => x >= 0.4 && x <= 0.6).length / ps.length;
  const degenerate = sd < o.minStdev || (hi - lo) < o.minRange;
  return {
    n: preds.length,
    mean: m, stdev: sd, min: lo, max: hi, fractionBetween40And60: inMiddle,
    verdict: degenerate ? 'DEGENERATE_SPREAD' : 'OK',
    degenerate,
    detail: degenerate
      ? `Predictions span ${(lo * 100).toFixed(0)}%-${(hi * 100).toFixed(0)}% with sd ${sd.toFixed(3)}; ${(inMiddle * 100).toFixed(0)}% sit between 40% and 60%. The model is close to a constant and cannot identify mispriced fights.`
      : `Predictions span ${(lo * 100).toFixed(0)}%-${(hi * 100).toFixed(0)}%, sd ${sd.toFixed(3)}.`,
  };
}

/**
 * Odds coverage. Without timestamped prices there is no verifiable edge, and
 * without closing prices there is no CLV — which is the fastest honest signal
 * available. A dataset can be rich in fight results and still be unable to
 * support a betting model for this reason alone.
 */
function oddsCoverage(bouts, opts) {
  const o = Object.assign({ minClosingForCLV: 100, minBooks: 2 }, opts || {});
  const n = bouts.length;
  const withOpen = bouts.filter((b) => (b.quotes || []).length > 0).length;
  const withClose = bouts.filter((b) => b.closingQuote != null).length;
  const withTimestamps = bouts.filter((b) => (b.quotes || []).every((q) => q.observedAt != null) && (b.quotes || []).length).length;
  const multiBook = bouts.filter((b) => new Set((b.quotes || []).map((q) => q.book)).size >= o.minBooks).length;
  const clvCapable = withClose >= o.minClosingForCLV;
  return {
    n,
    withAnyPrice: withOpen,
    withClosingPrice: withClose,
    withTimestamps,
    multiBook,
    clvCapable,
    verdict: clvCapable ? 'OK' : 'INSUFFICIENT_ODDS_HISTORY',
    detail: clvCapable
      ? `${withClose} bouts carry a closing price; CLV can be measured.`
      : `Only ${withClose} bouts carry a closing price (${o.minClosingForCLV} needed). CLV cannot be measured, so the process cannot be validated on the fastest available signal.`,
  };
}

/** Duplicate detection — the same bout appearing twice inflates apparent sample. */
function duplicates(rows, keyFn) {
  const seen = new Map();
  for (const r of rows) {
    const k = keyFn(r);
    seen.set(k, (seen.get(k) || 0) + 1);
  }
  const dupes = [...seen.entries()].filter(([, c]) => c > 1);
  return {
    n: rows.length,
    unique: seen.size,
    duplicateKeys: dupes.length,
    verdict: dupes.length ? 'DUPLICATES_PRESENT' : 'OK',
    examples: dupes.slice(0, 5).map(([k, c]) => ({ key: k, count: c })),
  };
}

/**
 * Full audit. Returns `fitReady: false` if any check would invalidate a model
 * fitted on this data.
 */
function auditDataset(args) {
  const checks = {};
  if (args.results) checks.sideBalance = sideBalance(args.results);
  if (args.predictions) checks.predictionSpread = predictionSpread(args.predictions);
  if (args.bouts) checks.oddsCoverage = oddsCoverage(args.bouts);
  if (args.results && args.duplicateKey) checks.duplicates = duplicates(args.results, args.duplicateKey);

  const blocking = [];
  if (checks.sideBalance && checks.sideBalance.leak) blocking.push('SIDE_ASSIGNMENT_LEAK');
  if (checks.duplicates && checks.duplicates.duplicateKeys > 0) blocking.push('DUPLICATES_PRESENT');

  const warnings = [];
  if (checks.predictionSpread && checks.predictionSpread.degenerate) warnings.push('DEGENERATE_SPREAD');
  if (checks.oddsCoverage && !checks.oddsCoverage.clvCapable) warnings.push('INSUFFICIENT_ODDS_HISTORY');

  return {
    checks,
    blocking,
    warnings,
    fitReady: blocking.length === 0,
    // CLV is not optional for deployment, so odds coverage gates live betting
    // even though it does not block fitting.
    deployReady: blocking.length === 0 && !warnings.includes('INSUFFICIENT_ODDS_HISTORY'),
  };
}

module.exports = { sideBalance, predictionSpread, oddsCoverage, duplicates, auditDataset };
