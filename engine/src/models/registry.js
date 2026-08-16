'use strict';
// Model versioning.
//
// The point of this module is to make one specific failure impossible:
// adjusting a model after seeing how its picks did, and then reporting the
// adjusted model's historical record as if it had been live. Cards are frozen
// on creation, predictions carry the hash of the version that produced them,
// and any change produces a new version that must be validated independently.

const crypto = require('crypto');

function hash(obj) {
  return crypto.createHash('sha256').update(JSON.stringify(obj)).digest('hex').slice(0, 16);
}

/**
 * Create an immutable model card.
 *
 * @param {object} spec
 * @param {string} spec.version        e.g. 'MODEL_V1'
 * @param {number} spec.createdAt
 * @param {string[]} spec.features
 * @param {object} spec.parameters
 * @param {object} spec.hyperparameters
 * @param {{from:number, to:number, n:number}} spec.trainingWindow
 * @param {object} [spec.validation]   walk-forward output; absent = unvalidated
 */
function createCard(spec) {
  for (const k of ['version', 'createdAt', 'features', 'parameters', 'trainingWindow']) {
    if (spec[k] == null) throw new Error(`createCard: missing required field "${k}"`);
  }
  const core = {
    version: spec.version,
    createdAt: spec.createdAt,
    features: spec.features.slice().sort(),
    parameters: spec.parameters,
    hyperparameters: spec.hyperparameters || {},
    trainingWindow: spec.trainingWindow,
    trainingDataHash: spec.trainingDataHash || null,
  };
  const card = Object.assign({}, core, {
    hash: hash(core),
    validation: spec.validation || null,
    deployable: spec.validation ? spec.validation.deployable === true : false,
    deployedAt: null,
    retiredAt: null,
    notes: spec.notes || null,
  });
  return Object.freeze(card);
}

/** A prediction is only meaningful alongside the exact version that made it. */
function stampPrediction(card, prediction) {
  return Object.freeze(Object.assign({}, prediction, {
    modelVersion: card.version,
    modelHash: card.hash,
    stampedAt: Date.now(),
  }));
}

/**
 * Verify that a stored prediction was produced by the card it claims.
 * Detects a version being edited underneath its own history.
 */
function verifyPrediction(card, prediction) {
  return prediction.modelVersion === card.version && prediction.modelHash === card.hash;
}

const RETIREMENT_CRITERIA = {
  logLossDegradation: {
    test: (r) => r.recentLogLoss != null && r.baselineLogLoss != null
      && r.recentLogLoss > r.baselineLogLoss,
    label: 'Out-of-sample log loss no longer beats the market baseline',
  },
  calibrationFailure: {
    test: (r) => r.ece != null && r.ece > 0.05,
    label: 'Persistent calibration failure (ECE > 0.05)',
  },
  negativeCLV: {
    test: (r) => r.clv != null && r.clv.significant && r.clv.meanClvProbability < 0,
    label: 'Significantly negative closing line value',
  },
  featureDrift: {
    test: (r) => r.featureDrift != null && r.featureDrift > (r.driftThreshold || 0.25),
    label: 'Feature distribution drift beyond threshold',
  },
  pipelineFailure: {
    test: (r) => r.pipelineHealthy === false,
    label: 'Data pipeline failure',
  },
};

/**
 * Evaluate whether a deployed version should be retired.
 * Deliberately requires a minimum sample: a losing stretch is not a criterion.
 */
function evaluateRetirement(record, opts) {
  const o = Object.assign({ minSample: 100 }, opts || {});
  if (!record || (record.n || 0) < o.minSample) {
    return {
      retire: false,
      verdict: 'INSUFFICIENT_SAMPLE',
      triggered: [],
      note: `${record ? record.n || 0 : 0} settled predictions; ${o.minSample} required before any retirement judgement. A losing run alone is never a criterion.`,
    };
  }
  const triggered = Object.entries(RETIREMENT_CRITERIA)
    .filter(([, c]) => c.test(record))
    .map(([key, c]) => ({ key, label: c.label }));
  return {
    retire: triggered.length > 0,
    verdict: triggered.length ? 'RETIRE' : 'KEEP',
    triggered,
    note: triggered.length
      ? 'Build a successor version and validate it independently before deploying. Do not patch this version in place.'
      : 'No retirement criterion met.',
  };
}

module.exports = { createCard, stampPrediction, verifyPrediction, evaluateRetirement, RETIREMENT_CRITERIA, hash };
