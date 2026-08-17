'use strict';
// Ridge-regularised logistic regression.
//
// Fitted by full-batch gradient descent with an adaptive step. No dependencies,
// deterministic, and small enough to read — which matters more than speed here,
// because this model is the benchmark every fancier component must beat.
//
// NOTE ON THE INTERCEPT: it is fitted but expected to sit at ~0, because the
// feature vectors are antisymmetric (see features/pipeline.js). A materially
// non-zero intercept means the training set carries a side bias, so `fit`
// reports it and `sideBiasDetected` flags it rather than letting it pass.

const { sigmoid, clamp } = require('../core/prob');

const DEFAULTS = {
  l2: 1.0,
  iterations: 400,
  learningRate: 0.5,
  tolerance: 1e-7,
  fitIntercept: true,
};

/**
 * @param {number[][]} X  design matrix
 * @param {number[]} y    0/1 labels
 * @param {object} [opts]
 */
function fit(X, y, opts) {
  const cfg = Object.assign({}, DEFAULTS, opts || {});
  if (!X.length) throw new Error('logistic.fit: empty design matrix');
  if (X.length !== y.length) throw new Error('logistic.fit: X/y length mismatch');
  const n = X.length;
  const d = X[0].length;
  for (const row of X) if (row.length !== d) throw new Error('logistic.fit: ragged design matrix');

  let w = new Array(d).fill(0);
  let b = 0;
  let lr = cfg.learningRate;
  let prev = Infinity;

  const objective = (w_, b_) => {
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const z = dot(w_, X[i]) + b_;
      const p = clamp(sigmoid(z), 1e-12, 1 - 1e-12);
      loss += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
    }
    loss /= n;
    for (const wi of w_) loss += (cfg.l2 / (2 * n)) * wi * wi;
    return loss;
  };

  let iterations = 0;
  for (let it = 0; it < cfg.iterations; it++) {
    iterations = it + 1;
    const gw = new Array(d).fill(0);
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(dot(w, X[i]) + b);
      const err = p - y[i];
      const row = X[i];
      for (let j = 0; j < d; j++) gw[j] += err * row[j];
      gb += err;
    }
    for (let j = 0; j < d; j++) gw[j] = gw[j] / n + (cfg.l2 / n) * w[j];
    gb /= n;

    const cand = w.map((wj, j) => wj - lr * gw[j]);
    const candB = cfg.fitIntercept ? b - lr * gb : 0;
    const loss = objective(cand, candB);

    if (loss > prev) {
      lr *= 0.5;            // overshot — back off and retry from the same point
      if (lr < 1e-8) break;
      continue;
    }
    if (Math.abs(prev - loss) < cfg.tolerance) { w = cand; b = candB; prev = loss; break; }
    w = cand; b = candB; prev = loss;
  }

  return {
    weights: w,
    intercept: b,
    loss: prev,
    n,
    iterations,
    l2: cfg.l2,
    // With antisymmetric features an honest intercept is ~0.
    sideBiasDetected: Math.abs(b) > 0.05,
    predict: (x) => sigmoid(dot(w, x) + b),
  };
}

function predict(model, x) {
  return sigmoid(dot(model.weights, x) + model.intercept);
}

function dot(w, x) {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i] * x[i];
  return s;
}

module.exports = { fit, predict, DEFAULTS };
