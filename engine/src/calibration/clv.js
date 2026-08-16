'use strict';
// Closing Line Value.
//
// CLV is the fastest-accumulating honest signal available. P&L over 50 bets is
// almost pure noise; CLV over 50 bets already says something about whether the
// process finds prices before the market does. It is therefore the primary
// short-horizon diagnostic, and a strategy with good CLV and bad P&L is treated
// very differently from one with good P&L and bad CLV.

const { devig } = require('../core/odds');
const { mean, stdev } = require('../core/prob');

/**
 * CLV for a single wager.
 *
 * Both prices are devigged before comparison. Comparing raw prices conflates
 * beating the market with the book's margin changing, which are different
 * things.
 *
 * @param {object} bet
 * @param {number[]} bet.betQuote      decimal prices of every outcome at bet time
 * @param {number[]} bet.closeQuote    decimal prices of every outcome at close
 * @param {number} bet.outcomeIndex    which outcome was backed
 * @param {number} bet.decimalOdds     the price actually taken
 */
function betCLV(bet) {
  const { betQuote, closeQuote, outcomeIndex } = bet;
  if (!betQuote || !closeQuote) return null;
  const atBet = devig(betQuote);
  const atClose = devig(closeQuote);
  const pBet = atBet.consensus[outcomeIndex];
  const pClose = atClose.consensus[outcomeIndex];

  // Positive when the market moved toward the side we took, i.e. our price was
  // better than the closing price.
  const probCLV = pClose - pBet;
  const priceTaken = bet.decimalOdds;
  const priceClose = closeQuote[outcomeIndex];

  return {
    impliedAtBet: pBet,
    impliedAtClose: pClose,
    clvProbability: probCLV,
    // Percentage return advantage over betting the same side at the close.
    clvPercent: priceClose > 1 ? (priceTaken / priceClose - 1) : null,
    beatClose: priceTaken > priceClose,
    priceTaken,
    priceClose,
  };
}

/**
 * Aggregate CLV across a ledger, with a significance check so a positive
 * average over a handful of bets is not mistaken for evidence.
 */
function aggregate(bets) {
  const rows = bets.map((b) => ({ bet: b, clv: betCLV(b) })).filter((r) => r.clv);
  if (!rows.length) {
    return { n: 0, meanClvProbability: null, beatCloseRate: null, significant: false };
  }
  const probs = rows.map((r) => r.clv.clvProbability);
  const pcts = rows.map((r) => r.clv.clvPercent).filter((x) => x != null);
  const m = mean(probs);
  const sd = stdev(probs);
  const se = rows.length > 1 ? sd / Math.sqrt(rows.length) : null;
  const t = se && se > 0 ? m / se : null;

  return {
    n: rows.length,
    meanClvProbability: m,
    meanClvPercent: pcts.length ? mean(pcts) : null,
    stdev: sd,
    standardError: se,
    tStat: t,
    // Two-sided ~95% threshold. Deliberately conservative about calling it real.
    significant: t != null && Math.abs(t) > 1.96,
    beatCloseRate: rows.filter((r) => r.clv.beatClose).length / rows.length,
    rows,
  };
}

/**
 * Interpretation guide used by the model-health monitor. Kept explicit because
 * the CLV/P&L cross-tabulation is the single most useful diagnostic the system
 * produces, and it is easy to read backwards.
 */
function diagnose(clvAgg, perf) {
  const clvPositive = clvAgg.n > 0 && clvAgg.meanClvProbability > 0;
  const profitable = perf && perf.profit > 0;
  if (!clvAgg.significant) {
    return { verdict: 'INSUFFICIENT_SAMPLE', action: 'Keep recording. No conclusion is available yet.' };
  }
  if (clvPositive && profitable) {
    return { verdict: 'HEALTHY', action: 'Process and results agree. Continue at current sizing.' };
  }
  if (clvPositive && !profitable) {
    return {
      verdict: 'GOOD_PROCESS_BAD_LUCK',
      action: 'CLV is positive: the prices were right and results lag. Do not retune on this. Continue, reduced size if drawdown limits bind.',
    };
  }
  if (!clvPositive && profitable) {
    return {
      verdict: 'LUCKY',
      action: 'Profit without CLV is not evidence of edge. Do not scale up. Investigate before increasing exposure.',
    };
  }
  return {
    verdict: 'BROKEN',
    action: 'Negative CLV and negative P&L over a significant sample. Halt new bets and evaluate the version for retirement.',
  };
}

module.exports = { betCLV, aggregate, diagnose };
