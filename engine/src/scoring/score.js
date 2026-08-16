'use strict';
// Candidate scoring (0-100) and hard vetoes.
//
// The score is a ranking device, not a licence. Vetoes are checked first and
// cannot be outvoted by a high score — an unverified price or an edge that dies
// under adverse assumptions ends the conversation regardless of how good the
// rest of the picture looks.

const { clamp } = require('../core/prob');

const WEIGHTS = {
  effectiveEdge: 22,
  expectedValue: 18,
  modelAgreement: 12,
  calibration: 10,
  dataQuality: 10,
  simulationStability: 8,
  sensitivitySurvival: 8,
  marketQuality: 6,
  historicalCLV: 6,
};

const BANDS = [
  { min: 90, label: 'EXTREMELY STRONG CANDIDATE', action: 'BET' },
  { min: 80, label: 'STRONG CANDIDATE', action: 'BET' },
  { min: 70, label: 'WATCHLIST / POSSIBLE BET', action: 'PASS' },
  { min: 60, label: 'NO BET', action: 'PASS' },
  { min: 0, label: 'REJECT', action: 'PASS' },
];

// Normalisation scales. UNVALIDATED defaults — replaced by walk-forward fitting.
const SCALES = {
  edgeFull: 0.06,      // 6 points of shrunk edge scores full marks
  evFull: 0.10,        // 10% EV per unit staked scores full marks
  seGood: 0.005,       // simulation SE at or below this is fully stable
  intervalGood: 0.10,  // parameter interval width at or below this is stable
  clvFull: 0.02,       // 2 points of mean CLV scores full marks
};

/**
 * @param {object} c candidate
 * @param {object} c.edge          output of core/ev.computeEdge
 * @param {object} c.simulation    output of sim/montecarlo.run
 * @param {object} c.sensitivity   output of sim/sensitivity.runSensitivity
 * @param {object} c.quality       output of data/quality.assess
 * @param {object} c.market        output of core/odds.consensusMarket
 * @param {object} [c.clv]         output of calibration/clv.aggregate
 * @param {number} [c.calibrationReliability]
 * @param {boolean} [c.priceVerified]
 */
function rate(c) {
  const vetoes = [];

  // --- Hard vetoes ---
  if (!c.quality || c.quality.gate === 'BLOCK') {
    const detail = c.quality && c.quality.blocking.length
      ? c.quality.blocking.map((d) => d.label).join('; ')
      : 'data quality gate failed';
    vetoes.push({ key: 'DATA_QUALITY_BLOCK', detail });
  }
  if (c.priceVerified === false) {
    vetoes.push({ key: 'PRICE_UNVERIFIED', detail: 'No fresh, verified price for this market' });
  }
  if (!c.edge || !c.edge.positive) {
    vetoes.push({
      key: 'NON_POSITIVE_EV',
      detail: c.edge
        ? `Shrunk EV ${(c.edge.evEffective * 100).toFixed(2)}% at the current number`
        : 'no edge computed',
    });
  }
  if (c.sensitivity && !c.sensitivity.allSurvive) {
    const failed = c.sensitivity.scenarios.filter((s) => !s.survives).map((s) => s.key);
    vetoes.push({
      key: 'EDGE_DIES_UNDER_ADVERSE_ASSUMPTIONS',
      detail: `Edge does not survive: ${failed.join(', ')}`,
    });
  }
  if (!c.sensitivity) {
    vetoes.push({ key: 'NO_SENSITIVITY_ANALYSIS', detail: 'Adversarial re-runs were not performed' });
  }

  // --- Components ---
  const parts = {};
  parts.effectiveEdge = c.edge ? clamp(c.edge.effectiveEdge / SCALES.edgeFull, 0, 1) : 0;
  parts.expectedValue = c.edge ? clamp(c.edge.evEffective / SCALES.evFull, 0, 1) : 0;
  parts.modelAgreement = c.edge ? clamp(c.edge.shrink.disagreement, 0, 1) : 0;
  parts.calibration = c.calibrationReliability != null ? clamp(c.calibrationReliability, 0, 1) : 0;
  parts.dataQuality = c.quality ? clamp(c.quality.score / 100, 0, 1) : 0;

  if (c.simulation) {
    const seScore = clamp(1 - c.simulation.standardError / SCALES.seGood, 0, 1);
    const width = c.simulation.paramInterval[1] - c.simulation.paramInterval[0];
    const ciScore = clamp(1 - width / SCALES.intervalGood, 0, 1);
    parts.simulationStability = 0.4 * seScore + 0.6 * ciScore;
  } else {
    parts.simulationStability = 0;
  }

  parts.sensitivitySurvival = c.sensitivity ? clamp(c.sensitivity.survival, 0, 1) : 0;

  if (c.market) {
    const bookScore = clamp((c.market.bookCount - 1) / 3, 0, 1);
    const vigScore = clamp(1 - c.market.meanOverround / 0.10, 0, 1);
    parts.marketQuality = 0.6 * bookScore + 0.4 * vigScore;
  } else {
    parts.marketQuality = 0;
  }

  parts.historicalCLV = c.clv && c.clv.significant && c.clv.meanClvProbability != null
    ? clamp(c.clv.meanClvProbability / SCALES.clvFull, 0, 1)
    : 0;

  let score = 0;
  const contributions = {};
  for (const [k, w] of Object.entries(WEIGHTS)) {
    const v = (parts[k] || 0) * w;
    contributions[k] = { normalised: parts[k] || 0, weight: w, points: v };
    score += v;
  }

  const band = BANDS.find((b) => score >= b.min);
  const vetoed = vetoes.length > 0;

  return {
    score: vetoed ? 0 : score,
    rawScore: score,
    band: vetoed ? 'REJECT' : band.label,
    action: vetoed ? 'PASS' : band.action,
    vetoes,
    vetoed,
    contributions,
    thresholdsValidated: false, // set true only by the walk-forward harness
  };
}

module.exports = { rate, WEIGHTS, BANDS, SCALES };
