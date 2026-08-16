'use strict';
// Edge and expected value.
//
// The governing idea: a difference between the model's probability and the
// market's is NOT an edge. It is a hypothesis that the model knows something
// the market does not — and the market is a strong, liquid, information-rich
// estimator. So the difference is shrunk by everything that could plausibly be
// generating it spuriously, and only what survives is treated as edge.

const { clamp } = require('./prob');

/** Expected profit per 1 unit staked at decimal odds d with true probability p. */
function expectedValue(p, decimalOdds) {
  if (!(decimalOdds > 1)) throw new Error('expectedValue: decimal odds must exceed 1');
  const q = clamp(p, 0, 1);
  return q * (decimalOdds - 1) - (1 - q);
}

/** Variance of profit per 1 unit staked. Needed for risk-adjusted EV and exposure caps. */
function betVariance(p, decimalOdds) {
  const q = clamp(p, 0, 1);
  const win = decimalOdds - 1;
  const ev = expectedValue(q, decimalOdds);
  return q * Math.pow(win - ev, 2) + (1 - q) * Math.pow(-1 - ev, 2);
}

/**
 * Shrinkage factors, each in [0,1]. Their product is how much of the raw
 * model-vs-market difference the engine is willing to believe.
 *
 * Every factor defaults to a PESSIMISTIC value when its evidence is absent,
 * so an uninstrumented system shrinks the edge to nothing rather than betting
 * on unmeasured confidence.
 */
function shrinkageFactors(ctx) {
  const f = {};

  // How reliable has the model been in this probability band? No record ⇒ 0.
  f.calibration = ctx.calibrationReliability != null
    ? clamp(ctx.calibrationReliability, 0, 1) : 0;

  // Ensemble spread. Full weight at zero disagreement, none at `maxDisagreement`.
  const dis = ctx.disagreement != null ? Math.abs(ctx.disagreement) : 1;
  const maxDis = ctx.maxDisagreement != null ? ctx.maxDisagreement : 0.15;
  f.disagreement = clamp(1 - dis / maxDis, 0, 1);

  // Data quality 0-100 mapped so that "merely adequate" still costs you.
  const dq = ctx.dataQuality != null ? clamp(ctx.dataQuality, 0, 100) : 0;
  f.dataQuality = clamp((dq - 50) / 50, 0, 1);

  // Effective sample behind the features, via a saturating curve.
  const n = ctx.effectiveSample != null ? Math.max(0, ctx.effectiveSample) : 0;
  const nHalf = ctx.sampleHalfPoint != null ? ctx.sampleHalfPoint : 10;
  f.sample = n / (n + nHalf);

  // Fraction of adverse scenarios in which the edge kept its sign.
  f.sensitivity = ctx.sensitivitySurvival != null
    ? clamp(ctx.sensitivitySurvival, 0, 1) : 0;

  // Market quality: a lone book quoting a wide number is weak evidence of a price.
  const books = ctx.bookCount != null ? ctx.bookCount : 1;
  f.marketQuality = clamp((books - 1) / 2, 0, 1) * 0.5 + 0.5;

  f.product = f.calibration * f.disagreement * f.dataQuality * f.sample
    * f.sensitivity * f.marketQuality;
  return f;
}

/**
 * Compute the full edge picture for one candidate.
 *
 * @param {object} args
 * @param {number} args.modelProb     ensemble probability for this outcome
 * @param {number} args.marketProb    devigged market probability (conservative reading)
 * @param {number} args.decimalOdds   the price actually available
 * @param {object} args.context       inputs to shrinkageFactors
 * @param {number} [args.varianceLambda] risk aversion for risk-adjusted EV
 */
function computeEdge(args) {
  const { modelProb, marketProb, decimalOdds } = args;
  if (!(decimalOdds > 1)) throw new Error('computeEdge: decimal odds must exceed 1');
  const pMod = clamp(modelProb, 0, 1);
  const pMkt = clamp(marketProb, 0, 1);

  const shrink = shrinkageFactors(args.context || {});
  const pEff = pMkt + shrink.product * (pMod - pMkt);

  const rawEdge = pMod - pMkt;
  const effEdge = pEff - pMkt;

  const evRaw = expectedValue(pMod, decimalOdds);
  const evEff = expectedValue(pEff, decimalOdds);
  const varEff = betVariance(pEff, decimalOdds);
  const lambda = args.varianceLambda != null ? args.varianceLambda : 0.25;

  // Break-even probability at this price: below it the bet is negative EV.
  const breakEven = 1 / decimalOdds;

  return {
    modelProb: pMod,
    marketProb: pMkt,
    effectiveProb: pEff,
    rawEdge,
    effectiveEdge: effEdge,
    shrink,
    evRaw,
    evEffective: evEff,
    variance: varEff,
    riskAdjustedEV: evEff - lambda * varEff,
    breakEvenProb: breakEven,
    // The only EV that is allowed to justify a bet is the shrunk one.
    positive: evEff > 0,
  };
}

module.exports = { expectedValue, betVariance, shrinkageFactors, computeEdge };
