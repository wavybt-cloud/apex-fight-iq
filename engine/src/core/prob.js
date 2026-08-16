'use strict';
// Numeric primitives shared by every component. Pure, deterministic, no globals.

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function logit(p) {
  const q = clamp(p, 1e-9, 1 - 1e-9);
  return Math.log(q / (1 - q));
}

function sigmoid(x) {
  // Branchless-overflow-safe form.
  if (x >= 0) { const e = Math.exp(-x); return 1 / (1 + e); }
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * Pool probabilities in log-odds space with non-negative weights.
 * Log-odds pooling is externally Bayesian and, unlike a linear average, does
 * not drag confident agreeing models toward the middle.
 */
function poolLogOdds(probs, weights) {
  if (!probs.length) throw new Error('poolLogOdds: no probabilities');
  const w = weights || probs.map(() => 1 / probs.length);
  if (w.length !== probs.length) throw new Error('poolLogOdds: weight/prob length mismatch');
  let sw = 0;
  for (const x of w) {
    if (!(x >= 0)) throw new Error('poolLogOdds: negative or NaN weight');
    sw += x;
  }
  if (sw <= 0) throw new Error('poolLogOdds: weights sum to zero');
  let acc = 0;
  for (let i = 0; i < probs.length; i++) acc += (w[i] / sw) * logit(probs[i]);
  return sigmoid(acc);
}

/** Seeded RNG. Seeding is mandatory: an unreproducible simulation cannot be audited. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller standard normal from a uniform generator. */
function randNormal(rng) {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Lognormal multiplicative noise with median 1 — for scaling positive parameters. */
function jitter(rng, sigma) {
  if (sigma <= 0) return 1;
  return Math.exp(randNormal(rng) * sigma);
}

function mean(xs) {
  if (!xs.length) return NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function variance(xs) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return s / (xs.length - 1);
}

function stdev(xs) { return Math.sqrt(variance(xs)); }

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * clamp(q, 0, 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Standard error of a proportion from n independent trials. */
function binomialSE(p, n) {
  if (n <= 0) return NaN;
  return Math.sqrt(Math.max(0, p * (1 - p)) / n);
}

/**
 * Shrink an observed rate toward a prior. This is the mechanism behind every
 * "regression toward the mean" claim in the system: k is the number of
 * pseudo-observations of prior strength.
 */
function shrinkToPrior(observed, n, prior, k) {
  if (!(n >= 0)) throw new Error('shrinkToPrior: bad n');
  if (!(k > 0)) throw new Error('shrinkToPrior: k must be > 0');
  return (n * observed + k * prior) / (n + k);
}

module.exports = {
  clamp, logit, sigmoid, poolLogOdds, mulberry32, randNormal, jitter,
  mean, variance, stdev, quantile, binomialSE, shrinkToPrior,
};
