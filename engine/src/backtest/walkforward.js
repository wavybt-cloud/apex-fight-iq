'use strict';
// Walk-forward harness.
//
// Two rules make this trustworthy rather than decorative:
//
// 1. Every feature is recomputed through the SAME `asOf` code path the live
//    engine uses. There is no separate historical path that can silently drift.
// 2. The leakage guard throws — loudly — if any fact used by a decision carries
//    an `observedAt` later than that decision. Look-ahead bias does not survive
//    a thrown exception, whereas it easily survives a code review.

const metrics = require('../calibration/metrics');
const clvLib = require('../calibration/clv');

/**
 * Assert that no input postdates the decision. Call this from every feature
 * builder; it is cheap and it is the whole defence against leakage.
 */
function assertNoLeakage(facts, asOf, label) {
  for (const f of facts || []) {
    if (f && f.observedAt != null && f.observedAt > asOf) {
      throw new Error(
        `LEAKAGE: ${label || 'fact'} observed at ${new Date(f.observedAt).toISOString()} ` +
        `used for a decision at ${new Date(asOf).toISOString()}`,
      );
    }
  }
  return true;
}

/**
 * Build chronological folds with a purge/embargo gap.
 *
 * The embargo matters more in MMA than it looks: fights on the same card share
 * conditions and news, so a card straddling a fold boundary leaks information
 * across it.
 *
 * @param {Array<{date:number}>} events sorted or unsorted
 * @param {object} opts {trainMin, testSize, embargoMs}
 */
function makeFolds(events, opts) {
  const o = Object.assign({ trainMin: 200, testSize: 50, embargoMs: 7 * 86400000 }, opts || {});
  const sorted = events.slice().sort((a, b) => a.date - b.date);
  const folds = [];
  let start = o.trainMin;
  while (start + o.testSize <= sorted.length) {
    const trainEnd = start;
    const testStart = start;
    const testEnd = Math.min(sorted.length, start + o.testSize);
    const boundary = sorted[testStart].date;
    // Purge: drop training rows inside the embargo window before the test set.
    const train = sorted.slice(0, trainEnd).filter((e) => e.date <= boundary - o.embargoMs);
    const test = sorted.slice(testStart, testEnd);
    if (train.length >= o.trainMin * 0.5) {
      folds.push({ index: folds.length, train, test, boundary });
    }
    start += o.testSize;
  }
  return folds;
}

/**
 * Run a walk-forward evaluation.
 *
 * @param {object} args
 * @param {Array} args.events           historical events, each with `date`
 * @param {(train:Array)=>object} args.fit          returns a fitted model
 * @param {(model:object, event:object)=>object|null} args.predict
 *        returns {probability, marketProb, outcome, decimalOdds, stake, ...} or null to skip
 * @param {object} [args.foldOptions]
 */
function run(args) {
  const folds = makeFolds(args.events, args.foldOptions);
  if (!folds.length) {
    return {
      folds: [], aggregate: null,
      verdict: 'INSUFFICIENT_HISTORY',
      note: 'Not enough historical events to build a single walk-forward fold. No claim about performance can be made.',
    };
  }

  const foldResults = [];
  const allPreds = [];
  const allBets = [];

  for (const fold of folds) {
    const model = args.fit(fold.train);
    const preds = [];
    const bets = [];
    for (const ev of fold.test) {
      const p = args.predict(model, ev);
      if (!p) continue;
      if (p.decidedAt != null && p.decidedAt > ev.date) {
        throw new Error(`LEAKAGE: decision timestamp postdates the event for ${ev.id || ev.date}`);
      }
      preds.push({ probability: p.probability, outcome: p.outcome, marketProb: p.marketProb });
      if (p.stake > 0) {
        bets.push({
          stake: p.stake,
          decimalOdds: p.decimalOdds,
          result: p.outcome === 1 ? 'win' : 'loss',
          betQuote: p.betQuote,
          closeQuote: p.closeQuote,
          outcomeIndex: p.outcomeIndex,
        });
      }
    }
    allPreds.push(...preds);
    allBets.push(...bets);
    foldResults.push({
      index: fold.index,
      trainSize: fold.train.length,
      testSize: fold.test.length,
      metrics: metrics.report(preds, bets),
      clv: bets.length ? clvLib.aggregate(bets) : null,
    });
  }

  const agg = metrics.report(allPreds, allBets);
  const aggClv = allBets.length ? clvLib.aggregate(allBets) : { n: 0, significant: false, meanClvProbability: null };

  // Deployment criteria. Both must hold; ROI is deliberately not among them.
  const beatsMarket = agg.beatsMarket === true;
  const positiveCLV = aggClv.significant && aggClv.meanClvProbability > 0;
  const calibrated = agg.ece != null && agg.ece < 0.03;

  const failures = [];
  if (!beatsMarket) failures.push('Does not beat the devigged market baseline on log loss.');
  if (!positiveCLV) failures.push(aggClv.significant ? 'Closing line value is not positive.' : 'Closing line value is not statistically distinguishable from zero.');
  if (!calibrated) failures.push(`Expected calibration error ${agg.ece == null ? 'unavailable' : agg.ece.toFixed(4)} exceeds the 0.03 limit.`);

  return {
    folds: foldResults,
    aggregate: agg,
    clv: aggClv,
    criteria: { beatsMarket, positiveCLV, calibrated },
    deployable: failures.length === 0,
    failures,
    verdict: failures.length === 0 ? 'DEPLOYABLE' : 'NOT_DEPLOYABLE',
  };
}

module.exports = { run, makeFolds, assertNoLeakage };
