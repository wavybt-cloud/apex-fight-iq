'use strict';
// Calibration measurement.
//
// ROI is deliberately not the fitness function. A model can post a strong ROI
// over a small sample while being badly miscalibrated, and that model will
// eventually give the bankroll back. Log loss and calibration error are the
// metrics that move first and lie least.

const { clamp, mean } = require('../core/prob');

/**
 * @typedef {{probability:number, outcome:0|1, marketProb?:number}} Prediction
 */

function brierScore(preds) {
  if (!preds.length) return null;
  return mean(preds.map((p) => Math.pow(p.probability - p.outcome, 2)));
}

function logLoss(preds) {
  if (!preds.length) return null;
  return mean(preds.map((p) => {
    const q = clamp(p.probability, 1e-9, 1 - 1e-9);
    return -(p.outcome * Math.log(q) + (1 - p.outcome) * Math.log(1 - q));
  }));
}

/** Log loss of always predicting the base rate — the floor any model must beat. */
function baseRateLogLoss(preds) {
  if (!preds.length) return null;
  const base = mean(preds.map((p) => p.outcome));
  return logLoss(preds.map((p) => ({ probability: base, outcome: p.outcome })));
}

/** Log loss of the market on the same predictions — the benchmark that matters. */
function marketLogLoss(preds) {
  const withMarket = preds.filter((p) => p.marketProb != null);
  if (!withMarket.length) return null;
  return logLoss(withMarket.map((p) => ({ probability: p.marketProb, outcome: p.outcome })));
}

/**
 * Reliability curve over probability bins.
 * @returns {Array<{lo,hi,n,predicted,observed,gap}>}
 */
function reliabilityCurve(preds, bins = 10) {
  const out = [];
  for (let i = 0; i < bins; i++) {
    const lo = i / bins, hi = (i + 1) / bins;
    const inBin = preds.filter((p) => p.probability >= lo && (i === bins - 1 ? p.probability <= hi : p.probability < hi));
    out.push({
      lo, hi,
      n: inBin.length,
      predicted: inBin.length ? mean(inBin.map((p) => p.probability)) : null,
      observed: inBin.length ? mean(inBin.map((p) => p.outcome)) : null,
      gap: inBin.length ? mean(inBin.map((p) => p.probability)) - mean(inBin.map((p) => p.outcome)) : null,
    });
  }
  return out;
}

/** Expected calibration error: sample-weighted mean absolute gap. */
function expectedCalibrationError(preds, bins = 10) {
  if (!preds.length) return null;
  const curve = reliabilityCurve(preds, bins);
  let acc = 0;
  for (const b of curve) if (b.n) acc += (b.n / preds.length) * Math.abs(b.gap);
  return acc;
}

/**
 * Reliability of the model in the probability band a given prediction falls in.
 * This is `f_calibration` in the edge-shrinkage formula: it is how the system
 * automatically distrusts its own 70% predictions if its past 70% predictions
 * only came in 58% of the time.
 *
 * Returns 0 when the band has too little history — no record means no credit.
 *
 * @param {Prediction[]} history
 * @param {number} probability
 * @param {{bins?:number, minSample?:number, tolerance?:number}} [opts]
 */
function bandReliability(history, probability, opts) {
  const o = Object.assign({ bins: 10, minSample: 25, tolerance: 0.10 }, opts || {});
  if (!history || !history.length) return 0;
  const width = 1 / o.bins;
  const idx = Math.min(o.bins - 1, Math.floor(clamp(probability, 0, 0.999999) / width));
  const lo = idx * width, hi = lo + width;
  const inBand = history.filter((p) => p.probability >= lo && p.probability < hi);
  if (inBand.length < o.minSample) return 0;
  const gap = Math.abs(mean(inBand.map((p) => p.probability)) - mean(inBand.map((p) => p.outcome)));
  // Full credit at a perfect match, zero credit once the gap reaches tolerance.
  const accuracy = clamp(1 - gap / o.tolerance, 0, 1);
  // Confidence grows with sample size in the band.
  const confidence = inBand.length / (inBand.length + o.minSample);
  return accuracy * confidence;
}

/** P&L metrics from a settled ledger. */
function performance(bets) {
  const settled = bets.filter((b) => b.result === 'win' || b.result === 'loss' || b.result === 'push');
  if (!settled.length) {
    return { n: 0, staked: 0, profit: 0, roi: null, hitRate: null, maxDrawdown: 0, peak: 0 };
  }
  let staked = 0, profit = 0, wins = 0, decided = 0;
  let running = 0, peak = 0, maxDD = 0;
  for (const b of settled) {
    staked += b.stake;
    let p = 0;
    if (b.result === 'win') { p = b.stake * (b.decimalOdds - 1); wins++; decided++; }
    else if (b.result === 'loss') { p = -b.stake; decided++; }
    profit += p;
    running += p;
    peak = Math.max(peak, running);
    maxDD = Math.max(maxDD, peak - running);
  }
  return {
    n: settled.length,
    staked,
    profit,
    roi: staked > 0 ? profit / staked : null,
    hitRate: decided > 0 ? wins / decided : null,
    maxDrawdown: maxDD,
    maxDrawdownPct: peak > 0 ? maxDD / peak : null,
    peak,
  };
}

/** Full report for a model version. */
function report(preds, bets) {
  const ll = logLoss(preds);
  const mll = marketLogLoss(preds);
  return {
    n: preds.length,
    brier: brierScore(preds),
    logLoss: ll,
    baseRateLogLoss: baseRateLogLoss(preds),
    marketLogLoss: mll,
    // The only comparison that establishes the model knows something.
    logLossVsMarket: ll != null && mll != null ? mll - ll : null,
    beatsMarket: ll != null && mll != null ? ll < mll : null,
    ece: expectedCalibrationError(preds),
    reliability: reliabilityCurve(preds),
    performance: bets ? performance(bets) : null,
  };
}

module.exports = {
  brierScore, logLoss, baseRateLogLoss, marketLogLoss,
  reliabilityCurve, expectedCalibrationError, bandReliability,
  performance, report,
};
