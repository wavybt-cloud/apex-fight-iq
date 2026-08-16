'use strict';
// Odds conversion and vig removal.
//
// Devigging is not a formality. The choice of estimator moves the implied
// probability of a longshot by several points, which is the same order as any
// edge worth betting. So the engine computes several estimators, reports the
// spread as market-price uncertainty, and uses the least favourable one when
// deciding whether an edge exists.

const { clamp } = require('./prob');

function americanToDecimal(a) {
  if (a == null || !Number.isFinite(a)) return null;
  if (a === 0) throw new Error('americanToDecimal: 0 is not a valid American price');
  return a > 0 ? 1 + a / 100 : 1 + 100 / Math.abs(a);
}

function decimalToAmerican(d) {
  if (!(d > 1)) throw new Error('decimalToAmerican: decimal odds must exceed 1');
  return d >= 2 ? Math.round((d - 1) * 100) : -Math.round(100 / (d - 1));
}

function decimalToImplied(d) {
  if (!(d > 1)) throw new Error('decimalToImplied: decimal odds must exceed 1');
  return 1 / d;
}

function americanToImplied(a) {
  const d = americanToDecimal(a);
  return d == null ? null : decimalToImplied(d);
}

function impliedToDecimal(p) {
  if (!(p > 0 && p < 1)) throw new Error('impliedToDecimal: probability out of range');
  return 1 / p;
}

/** Total booked probability minus 1. 0.045 means a 4.5% overround. */
function overround(decimals) {
  let s = 0;
  for (const d of decimals) s += decimalToImplied(d);
  return s - 1;
}

// --- Devig estimators. Each takes raw implied probabilities summing to > 1. ---

/** Proportional/multiplicative: divide by the booksum. Known to under-price longshots. */
function devigMultiplicative(raw) {
  const s = raw.reduce((a, b) => a + b, 0);
  return raw.map((p) => p / s);
}

/** Additive: subtract the overround equally. Distorts short prices instead. */
function devigAdditive(raw) {
  const s = raw.reduce((a, b) => a + b, 0);
  const excess = (s - 1) / raw.length;
  return raw.map((p) => clamp(p - excess, 1e-6, 1 - 1e-6));
}

/**
 * Power method: find k such that Σ pᵢ^k = 1. Handles favourite-longshot bias
 * better than either linear method because the correction is multiplicative in
 * log space.
 */
function devigPower(raw, tol = 1e-12, maxIter = 200) {
  let lo = 0.5, hi = 2.0;
  const f = (k) => raw.reduce((a, p) => a + Math.pow(p, k), 0) - 1;
  // Σp^k decreases in k for p<1, so expand hi until the sum drops below 1.
  let guard = 0;
  while (f(hi) > 0 && guard++ < 60) hi *= 1.5;
  for (let i = 0; i < maxIter; i++) {
    const mid = (lo + hi) / 2;
    const v = f(mid);
    if (Math.abs(v) < tol) { lo = hi = mid; break; }
    if (v > 0) lo = mid; else hi = mid;
  }
  const k = (lo + hi) / 2;
  const out = raw.map((p) => Math.pow(p, k));
  const s = out.reduce((a, b) => a + b, 0);
  return out.map((p) => p / s); // renormalise away residual solver error
}

/**
 * Shin (1993): models the overround as compensation for a fraction z of
 * insider money. Two-way markets only, which covers moneylines and every
 * two-sided total/method market the engine prices.
 */
function devigShin(raw) {
  if (raw.length !== 2) return devigPower(raw);
  const s = raw.reduce((a, b) => a + b, 0);
  const z = shinZ(raw, s);
  if (!(z > 0)) return devigMultiplicative(raw);
  const adj = raw.map((p) => shinProb(p, s, z));
  const t = adj.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(t) || t <= 0) return devigMultiplicative(raw);
  return adj.map((p) => p / t);
}

/** Shin's inversion of a booked price into a true probability given insider fraction z. */
function shinProb(p, s, z) {
  return (Math.sqrt(z * z + 4 * (1 - z) * (p * p) / s) - z) / (2 * (1 - z));
}

/** Bisect for the z that makes Shin-adjusted probabilities sum to 1. */
function shinZ(raw, s) {
  const g = (z) => raw.reduce((t, p) => t + shinProb(p, s, z), 0) - 1;
  const lo0 = 0, hi0 = 0.4999;
  if (g(lo0) * g(hi0) > 0) return 0;
  let lo = lo0, hi = hi0;
  for (let i = 0; i < 120; i++) {
    const mid = (lo + hi) / 2;
    if (g(lo) * g(mid) <= 0) hi = mid; else lo = mid;
  }
  return (lo + hi) / 2;
}

const ESTIMATORS = {
  multiplicative: devigMultiplicative,
  additive: devigAdditive,
  power: devigPower,
  shin: devigShin,
};

/**
 * Devig a market with every estimator.
 *
 * @param {number[]} decimals decimal prices for each outcome of one market
 * @returns {{probs:Object, consensus:number[], spread:number[], overround:number,
 *            conservative:(i:number)=>number}}
 *   `conservative(i)` returns the HIGHEST devigged probability any estimator
 *   assigns to outcome i — the least favourable reading for a bettor backing
 *   it, since a higher market probability means a smaller claimed edge.
 */
function devig(decimals) {
  if (!Array.isArray(decimals) || decimals.length < 2) {
    throw new Error('devig: need at least two outcome prices');
  }
  const raw = decimals.map(decimalToImplied);
  const sum = raw.reduce((a, b) => a + b, 0);
  if (sum <= 1) {
    // No vig (or an arbitrage). Normalising is still the right move, but flag it.
    const norm = raw.map((p) => p / sum);
    return {
      probs: Object.fromEntries(Object.keys(ESTIMATORS).map((k) => [k, norm])),
      consensus: norm,
      spread: norm.map(() => 0),
      overround: sum - 1,
      noVig: true,
      conservative: (i) => norm[i],
    };
  }
  const probs = {};
  for (const [name, fn] of Object.entries(ESTIMATORS)) probs[name] = fn(raw);
  const names = Object.keys(probs);
  const consensus = raw.map((_, i) => {
    let s = 0;
    for (const n of names) s += probs[n][i];
    return s / names.length;
  });
  const spread = raw.map((_, i) => {
    const vals = names.map((n) => probs[n][i]);
    return Math.max(...vals) - Math.min(...vals);
  });
  return {
    probs,
    consensus,
    spread,
    overround: sum - 1,
    noVig: false,
    conservative: (i) => Math.max(...names.map((n) => probs[n][i])),
  };
}

/**
 * Combine quotes from several books into one market view.
 * Weighted by book weight (a liquidity/sharpness proxy), never a plain mean:
 * a recreational book's number is not evidence of the same quality as a
 * sharp book's.
 *
 * @param {Array<{book:string, decimals:number[], weight?:number, observedAt?:number}>} quotes
 */
function consensusMarket(quotes) {
  if (!Array.isArray(quotes) || !quotes.length) throw new Error('consensusMarket: no quotes');
  const n = quotes[0].decimals.length;
  for (const q of quotes) {
    if (q.decimals.length !== n) throw new Error('consensusMarket: outcome count mismatch across books');
  }
  const devigged = quotes.map((q) => ({ quote: q, dv: devig(q.decimals) }));
  let wsum = 0;
  const acc = new Array(n).fill(0);
  const consAcc = new Array(n).fill(0);
  for (const { quote, dv } of devigged) {
    const w = quote.weight != null ? quote.weight : 1;
    if (!(w >= 0)) throw new Error('consensusMarket: negative book weight');
    wsum += w;
    for (let i = 0; i < n; i++) {
      acc[i] += w * dv.conservative(i);
      consAcc[i] += w * dv.consensus[i];
    }
  }
  if (wsum <= 0) throw new Error('consensusMarket: book weights sum to zero');
  const conservative = acc.map((x) => x / wsum);
  const central = consAcc.map((x) => x / wsum);
  // Best available price per outcome — what a bettor would actually take.
  const bestDecimal = new Array(n).fill(0);
  const bestBook = new Array(n).fill(null);
  for (const q of quotes) {
    for (let i = 0; i < n; i++) {
      if (q.decimals[i] > bestDecimal[i]) { bestDecimal[i] = q.decimals[i]; bestBook[i] = q.book; }
    }
  }
  const disagreement = devigged.length < 2 ? 0 : Math.max(...Array.from({ length: n }, (_, i) => {
    const vals = devigged.map(({ dv }) => dv.consensus[i]);
    return Math.max(...vals) - Math.min(...vals);
  }));
  return {
    bookCount: quotes.length,
    conservative,     // used for edge: least favourable to the bettor
    central,          // used for reporting
    bestDecimal,
    bestBook,
    bookDisagreement: disagreement,
    meanOverround: devigged.reduce((a, { dv }) => a + dv.overround, 0) / devigged.length,
  };
}

module.exports = {
  americanToDecimal, decimalToAmerican, decimalToImplied, americanToImplied,
  impliedToDecimal, overround,
  devigMultiplicative, devigAdditive, devigPower, devigShin,
  devig, consensusMarket, ESTIMATORS,
};
