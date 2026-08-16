'use strict';
// Daily loss protocol and model-health monitor.
//
// The central idea: a losing day is only informative if it was IMPROBABLE under
// the day's own risk profile. Before the card runs, the engine simulates the
// P&L distribution of the exact positions it took. Afterwards it asks where the
// realised result fell in that distribution. A day at the 20th percentile is
// the system working as designed and requires no action whatsoever — reacting
// to it is itself a modelling error, and the most common one in betting.

const { mulberry32, quantile } = require('../core/prob');

const STATES = ['NORMAL', 'CAUTION', 'REVIEW', 'HALT'];

const DEFAULT_THRESHOLDS = {
  cautionPercentile: 0.05,   // worse than the 5th percentile of the simulated day
  reviewPercentile: 0.01,    // worse than the 1st percentile
  haltDrawdownPct: 0.20,     // peak-to-trough bankroll drawdown
  reviewDrawdownPct: 0.12,
  consecutiveAbnormalDays: 2,
  minBetsForInference: 3,
};

/**
 * Simulate the P&L distribution of a set of positions.
 *
 * Correlation matters: five bets on one card driven by the same fighters do not
 * produce the tidy distribution independence would suggest, and it is precisely
 * the fat left tail that the protocol needs to measure.
 *
 * @param {Array<{stake:number, probability:number, decimalOdds:number, group?:string}>} positions
 * @param {{iterations?:number, seed?:number, intraGroupCorrelation?:number}} [opts]
 */
function simulateDayPnL(positions, opts) {
  const o = Object.assign({ iterations: 20000, seed: 12345, intraGroupCorrelation: 0.5 }, opts || {});
  const rng = mulberry32(o.seed);
  const results = [];
  const groups = [...new Set(positions.map((p) => p.group || null))].filter(Boolean);

  for (let i = 0; i < o.iterations; i++) {
    // One shared shock per correlated group, blended with each bet's own draw.
    const shocks = {};
    for (const g of groups) shocks[g] = rng();
    let pnl = 0;
    for (const p of positions) {
      const own = rng();
      const shared = p.group ? shocks[p.group] : null;
      const u = shared == null ? own
        : o.intraGroupCorrelation * shared + (1 - o.intraGroupCorrelation) * own;
      pnl += u < p.probability ? p.stake * (p.decimalOdds - 1) : -p.stake;
    }
    results.push(pnl);
  }
  results.sort((a, b) => a - b);
  return {
    samples: results,
    p01: quantile(results, 0.01),
    p05: quantile(results, 0.05),
    p25: quantile(results, 0.25),
    median: quantile(results, 0.50),
    p95: quantile(results, 0.95),
    expected: results.reduce((a, b) => a + b, 0) / results.length,
    /** Fraction of simulated days at or below a realised P&L. */
    percentileOf: (actual) => {
      let lo = 0, hi = results.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (results[mid] < actual) lo = mid + 1; else hi = mid;
      }
      return lo / results.length;
    },
  };
}

/**
 * Evaluate the day and return the resulting risk state.
 *
 * @param {object} args
 * @param {Array} args.positions      positions taken (with stake/probability/odds)
 * @param {number} args.realisedPnL
 * @param {number} args.bankroll
 * @param {number} args.peakBankroll
 * @param {number} [args.priorAbnormalDays]
 * @param {object} [args.thresholds]
 */
function evaluateDay(args) {
  const T = Object.assign({}, DEFAULT_THRESHOLDS, args.thresholds || {});
  const positions = args.positions || [];
  const drawdown = args.peakBankroll > 0
    ? Math.max(0, (args.peakBankroll - args.bankroll) / args.peakBankroll) : 0;

  const reasons = [];
  let state = 'NORMAL';
  let percentile = null;
  let dist = null;

  if (positions.length >= T.minBetsForInference) {
    dist = simulateDayPnL(positions, args.simOptions);
    percentile = dist.percentileOf(args.realisedPnL);
    if (percentile <= T.reviewPercentile) {
      state = 'REVIEW';
      reasons.push(`Day's result at the ${(percentile * 100).toFixed(2)}th percentile of its own simulated distribution — beyond ordinary variance.`);
    } else if (percentile <= T.cautionPercentile) {
      state = 'CAUTION';
      reasons.push(`Day's result at the ${(percentile * 100).toFixed(1)}th percentile — poor but not yet abnormal.`);
    }
  } else if (args.realisedPnL < 0) {
    reasons.push(`Only ${positions.length} position(s): too few to infer anything from the result. No action.`);
  }

  if (drawdown >= T.haltDrawdownPct) {
    state = 'HALT';
    reasons.push(`Drawdown ${(drawdown * 100).toFixed(1)}% breaches the halt limit.`);
  } else if (drawdown >= T.reviewDrawdownPct && rank(state) < rank('REVIEW')) {
    state = 'REVIEW';
    reasons.push(`Drawdown ${(drawdown * 100).toFixed(1)}% breaches the review limit.`);
  }

  const abnormal = percentile != null && percentile <= T.cautionPercentile;
  const streak = abnormal ? (args.priorAbnormalDays || 0) + 1 : 0;
  if (streak >= T.consecutiveAbnormalDays && rank(state) < rank('REVIEW')) {
    state = 'REVIEW';
    reasons.push(`${streak} consecutive statistically abnormal days.`);
  }

  return {
    state,
    percentile,
    drawdown,
    abnormalDayStreak: streak,
    reasons,
    distribution: dist ? {
      expected: dist.expected, p01: dist.p01, p05: dist.p05,
      median: dist.median, p95: dist.p95,
    } : null,
    // What the state permits. There is no path here that raises stake size.
    permits: permissions(state),
  };
}

function rank(s) { return STATES.indexOf(s); }

function permissions(state) {
  switch (state) {
    case 'NORMAL': return { newBets: true, stakeMultiplier: 1.0, note: 'Normal operation.' };
    case 'CAUTION': return { newBets: true, stakeMultiplier: 0.5, note: 'Sizing halved pending clarity.' };
    case 'REVIEW': return {
      newBets: false, stakeMultiplier: 0,
      note: 'New recommendations suspended. Run the loss attribution before resuming.',
    };
    case 'HALT': return {
      newBets: false, stakeMultiplier: 0,
      note: 'Betting halted. Version must be evaluated against the retirement criteria before any resumption.',
    };
    default: return { newBets: false, stakeMultiplier: 0, note: 'Unknown state — fail closed.' };
  }
}

/**
 * Partition a bad day into candidate causes. Deliberately returns a checklist
 * rather than a verdict: attributing a loss automatically is exactly the kind
 * of overconfident inference the system exists to avoid.
 */
function attributeLoss(day) {
  const checks = [];
  const push = (cause, question, evidence) => checks.push({ cause, question, evidence });

  push('VARIANCE', 'Was the result inside the simulated distribution?',
    day.percentile != null
      ? `Result at the ${(day.percentile * 100).toFixed(1)}th percentile. ${day.percentile > 0.01 ? 'Consistent with ordinary variance — no model change warranted.' : 'Beyond the 1st percentile — variance alone is an unlikely explanation.'}`
      : 'Too few positions to say.');
  push('DATA', 'Did any position rest on data that turned out to be wrong or stale?',
    'Check quality scores and defect lists recorded with each position.');
  push('NEWS', 'Did late news (injury, weight, corner stoppage, replacement) postdate the decision?',
    'Compare each position timestamp against the news timeline.');
  push('MARKET', 'Did the market move against the positions before close?',
    'Negative CLV on the day points at the process; positive CLV points at variance.');
  push('CALIBRATION', 'Are recent probabilities in these bands still calibrated?',
    'Run the reliability curve on the trailing window, not on the day alone.');
  push('CORRELATION', 'Were the positions more correlated than the sizing assumed?',
    'Compare the realised joint outcome against the correlation matrix used for sizing.');
  push('MODEL', 'Only if the above are cleared: is there evidence of genuine degradation?',
    'Requires the §15 retirement criteria over a meaningful sample — never a single day.');
  return {
    checks,
    conclusion: null, // filled in by a human or a later automated pass, never assumed here
    warning: 'A single day is not a sample. Do not retune weights on this evidence.',
  };
}

module.exports = {
  STATES, DEFAULT_THRESHOLDS, simulateDayPnL, evaluateDay, permissions, attributeLoss,
};
