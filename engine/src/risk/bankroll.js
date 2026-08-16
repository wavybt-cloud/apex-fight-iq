'use strict';
// Staking and exposure.
//
// Note the shape of this API: there is no function anywhere that takes a
// previous result as an input to a stake. Martingale and chase-sizing are not
// discouraged here — they are unrepresentable.

const { clamp } = require('../core/prob');

const DEFAULT_POLICY = {
  kellyFraction: 0.25,       // quarter Kelly
  maxStakePctBankroll: 0.02, // hard per-bet cap
  maxEventExposurePct: 0.05, // correlation-adjusted cap across one event
  maxDailyExposurePct: 0.08,
  minStakeUnits: 0.1,        // below this the bet is not worth placing
  drawdownDerisk: [          // cut size as drawdown deepens
    { drawdown: 0.05, multiplier: 0.75 },
    { drawdown: 0.10, multiplier: 0.50 },
    { drawdown: 0.20, multiplier: 0.25 },
  ],
};

/** Full-Kelly fraction of bankroll. Negative means the bet is -EV: do not bet. */
function kellyFull(p, decimalOdds) {
  if (!(decimalOdds > 1)) throw new Error('kellyFull: decimal odds must exceed 1');
  const b = decimalOdds - 1;
  const q = 1 - p;
  return (b * p - q) / b;
}

/** Drawdown multiplier from the policy ladder (worst applicable rung wins). */
function drawdownMultiplier(drawdown, policy) {
  const rungs = policy.drawdownDerisk || [];
  let m = 1;
  for (const r of rungs) if (drawdown >= r.drawdown) m = Math.min(m, r.multiplier);
  return m;
}

/**
 * Size a single bet.
 *
 * @param {object} args
 * @param {number} args.bankroll
 * @param {number} args.probability   effective (shrunk) probability — never the raw model one
 * @param {number} args.decimalOdds
 * @param {number} [args.drawdown]    current drawdown as a fraction of peak bankroll
 * @param {number} [args.confidenceMultiplier] extra de-risking in [0,1]
 * @param {object} [args.policy]
 */
function sizeBet(args) {
  const policy = Object.assign({}, DEFAULT_POLICY, args.policy || {});
  const { bankroll, probability, decimalOdds } = args;
  if (!(bankroll > 0)) throw new Error('sizeBet: bankroll must be positive');

  const full = kellyFull(probability, decimalOdds);
  if (!(full > 0)) {
    return {
      stake: 0, fraction: 0, kellyFull: full, capped: null,
      reason: 'NON_POSITIVE_KELLY — the price does not compensate the risk',
    };
  }

  const ddMult = drawdownMultiplier(args.drawdown || 0, policy);
  const confMult = args.confidenceMultiplier != null ? clamp(args.confidenceMultiplier, 0, 1) : 1;

  // The cap is applied to the Kelly fraction FIRST, then the de-risking
  // multipliers. Doing it the other way round means that whenever the cap binds
  // — which is exactly when the bet is largest — drawdown and low-confidence
  // de-risking silently stop having any effect.
  let fraction = full * policy.kellyFraction;
  let capped = null;
  if (fraction > policy.maxStakePctBankroll) {
    fraction = policy.maxStakePctBankroll;
    capped = 'PER_BET_CAP';
  }
  fraction *= ddMult * confMult;

  const stake = bankroll * fraction;
  const unitSize = bankroll * 0.01;
  if (stake < unitSize * policy.minStakeUnits) {
    return {
      stake: 0, fraction: 0, kellyFull: full, capped,
      reason: 'BELOW_MIN_STAKE — edge too small to be worth the exposure',
    };
  }

  return {
    stake,
    fraction,
    units: stake / unitSize,
    kellyFull: full,
    kellyFractionApplied: policy.kellyFraction,
    drawdownMultiplier: ddMult,
    confidenceMultiplier: confMult,
    capped,
    reason: null,
  };
}

/**
 * Correlation-adjusted portfolio exposure.
 *
 * Three bets on the same fighter are one bet wearing three hats. The effective
 * exposure uses the quadratic form sqrt(wᵀΣw) rather than the naive sum, so
 * correlated positions cannot masquerade as diversification.
 *
 * @param {number[]} stakes
 * @param {number[][]} correlation square matrix, 1s on the diagonal
 */
function portfolioExposure(stakes, correlation) {
  const n = stakes.length;
  if (!n) return { naive: 0, effective: 0, concentrationRatio: 1 };
  if (correlation) {
    if (correlation.length !== n) throw new Error('portfolioExposure: correlation matrix size mismatch');
    for (const row of correlation) {
      if (row.length !== n) throw new Error('portfolioExposure: correlation matrix not square');
    }
  }
  const naive = stakes.reduce((a, b) => a + b, 0);
  let quad = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const rho = correlation ? correlation[i][j] : (i === j ? 1 : 0);
      quad += stakes[i] * stakes[j] * rho;
    }
  }
  const effective = Math.sqrt(Math.max(0, quad));
  return {
    naive,
    effective,
    // 1 = perfectly correlated (no diversification), lower = genuine spread.
    concentrationRatio: naive > 0 ? effective / naive : 1,
  };
}

/**
 * Apply portfolio caps to a set of already-sized bets, scaling them down
 * proportionally if correlation-adjusted exposure breaches a limit.
 */
function applyPortfolioCaps(bets, bankroll, correlation, policy) {
  const pol = Object.assign({}, DEFAULT_POLICY, policy || {});
  const stakes = bets.map((b) => b.stake);
  const exp = portfolioExposure(stakes, correlation);
  const limit = bankroll * pol.maxEventExposurePct;
  if (exp.effective <= limit || exp.effective === 0) {
    return { bets, exposure: exp, scaled: 1, breached: false };
  }
  const scale = limit / exp.effective;
  return {
    bets: bets.map((b) => Object.assign({}, b, {
      stake: b.stake * scale,
      units: b.units != null ? b.units * scale : undefined,
      capped: 'PORTFOLIO_EXPOSURE_CAP',
    })),
    exposure: exp,
    scaled: scale,
    breached: true,
  };
}

module.exports = {
  DEFAULT_POLICY, kellyFull, drawdownMultiplier, sizeBet,
  portfolioExposure, applyPortfolioCaps,
};
