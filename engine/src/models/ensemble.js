'use strict';
// Ensemble pooling and model registry.

const { poolLogOdds, logit, sigmoid, clamp } = require('../core/prob');

/**
 * Pool component model probabilities.
 *
 * Weights must come from validated out-of-sample performance (stacking on
 * walk-forward folds). Blind averaging is explicitly not supported: it hands a
 * model that has never been validated the same authority as one that has.
 *
 * @param {Array<{key:string, probability:number, weight:number, validated?:boolean}>} predictions
 * @param {{requireValidated?:boolean}} [opts]
 */
function pool(predictions, opts) {
  const o = Object.assign({ requireValidated: true }, opts || {});
  if (!predictions || !predictions.length) throw new Error('ensemble.pool: no predictions');

  const usable = o.requireValidated
    ? predictions.filter((p) => p.validated !== false)
    : predictions.slice();
  const dropped = predictions.filter((p) => !usable.includes(p)).map((p) => p.key);

  if (!usable.length) {
    throw new Error('ensemble.pool: no validated models available — the engine must not issue a probability');
  }

  const probs = usable.map((p) => clamp(p.probability, 1e-6, 1 - 1e-6));
  const weights = usable.map((p) => {
    if (p.weight == null) throw new Error(`ensemble.pool: model "${p.key}" has no weight`);
    return p.weight;
  });

  const pooled = poolLogOdds(probs, weights);

  // Disagreement: weighted spread in probability space. Retained as a signal,
  // because agreement among independent models is itself evidence and its
  // absence is a reason to bet less.
  const maxP = Math.max(...probs);
  const minP = Math.min(...probs);
  const wsum = weights.reduce((a, b) => a + b, 0);
  const wmean = probs.reduce((a, p, i) => a + p * weights[i], 0) / wsum;
  const wvar = probs.reduce((a, p, i) => a + weights[i] * Math.pow(p - wmean, 2), 0) / wsum;

  return {
    probability: pooled,
    range: maxP - minP,
    stdev: Math.sqrt(wvar),
    disagreement: maxP - minP,
    components: usable.map((p, i) => ({
      key: p.key, probability: probs[i], weight: weights[i] / wsum,
    })),
    droppedUnvalidated: dropped,
  };
}

/**
 * Market-anchored mode: the operating mode before any walk-forward evidence
 * exists. The market gets dominant weight and the deviation the model is
 * permitted to express is explicitly capped, because an unvalidated model that
 * disagrees strongly with the market is far more likely to be wrong than right.
 *
 * Output from this mode is for building a paper record. It is not sufficient
 * to justify a wager — the recommendation engine blocks on the missing
 * calibration record regardless.
 */
function marketAnchored(modelProb, marketProb, opts) {
  const o = Object.assign({ modelWeight: 0.15, maxDeviation: 0.05 }, opts || {});
  const pooled = sigmoid((1 - o.modelWeight) * logit(marketProb) + o.modelWeight * logit(modelProb));
  const capped = clamp(pooled, marketProb - o.maxDeviation, marketProb + o.maxDeviation);
  return {
    probability: capped,
    uncapped: pooled,
    wasCapped: Math.abs(pooled - capped) > 1e-9,
    mode: 'MARKET_ANCHORED',
    note: 'Unvalidated mode. Suitable for paper predictions only.',
  };
}

/**
 * Fit stacking weights on out-of-sample predictions by minimising log loss over
 * the simplex. Coordinate ascent on a coarse grid — the weight vector is short
 * and the objective is smooth, so this is adequate and has no dependencies.
 *
 * @param {Array<{probs:Object<string,number>, outcome:0|1}>} rows
 * @param {string[]} keys
 */
function fitWeights(rows, keys, opts) {
  const o = Object.assign({ steps: 20, rounds: 6 }, opts || {});
  if (!rows.length) throw new Error('fitWeights: no out-of-sample rows');
  let w = keys.map(() => 1 / keys.length);

  const loss = (ws) => {
    let acc = 0;
    for (const r of rows) {
      const ps = keys.map((k) => clamp(r.probs[k], 1e-6, 1 - 1e-6));
      const p = poolLogOdds(ps, ws);
      const q = clamp(p, 1e-9, 1 - 1e-9);
      acc += -(r.outcome * Math.log(q) + (1 - r.outcome) * Math.log(1 - q));
    }
    return acc / rows.length;
  };

  let best = loss(w);
  for (let round = 0; round < o.rounds; round++) {
    let improved = false;
    for (let i = 0; i < keys.length; i++) {
      for (let s = 0; s <= o.steps; s++) {
        const cand = w.slice();
        cand[i] = s / o.steps;
        const sum = cand.reduce((a, b) => a + b, 0);
        if (sum <= 0) continue;
        const norm = cand.map((x) => x / sum);
        const l = loss(norm);
        if (l < best - 1e-9) { best = l; w = norm; improved = true; }
      }
    }
    if (!improved) break;
  }
  return {
    weights: Object.fromEntries(keys.map((k, i) => [k, w[i]])),
    logLoss: best,
    n: rows.length,
  };
}

module.exports = { pool, marketAnchored, fitWeights };
