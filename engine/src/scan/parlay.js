'use strict';
// Parlays.
//
// Default answer: no. A parlay is only built when every leg independently
// clears the gates AND the correlation, measured on the simulation rather than
// assumed from a fixed haircut, makes the combined price favourable. That is a
// rare combination, and the honest output most of the time is a refusal.
//
// The one structurally sound case is POSITIVE correlation inside a single
// bout — e.g. "fighter wins" and "fight does not go the distance" for a heavy
// finisher — where the book prices the legs as independent but the outcomes are
// not. Cross-bout parlays of independent legs are, mathematically, a way of
// paying the vig several times.

const { jointMarket } = require('../sim/montecarlo');
const { expectedValue } = require('../core/ev');

/**
 * True joint probability of a set of legs within one bout, measured directly on
 * the simulated outcomes.
 */
function jointProbability(simulation, tests) {
  let hits = 0;
  for (const o of simulation.outcomes) {
    let all = true;
    for (const t of tests) { if (!t(o)) { all = false; break; } }
    if (all) hits++;
  }
  return hits / simulation.outcomes.length;
}

/**
 * Evaluate a candidate parlay.
 *
 * @param {object} args
 * @param {Array<{selection:string, decimalOdds:number, test:Function, legEdge:object, legScore:object}>} args.legs
 * @param {object} args.simulation  a single run covering all legs (same bout)
 * @param {number} [args.parlayDecimalOdds] the book's actual combined price, if quoted
 */
function evaluate(args) {
  const legs = args.legs || [];
  const refusals = [];

  if (legs.length < 2) refusals.push('A parlay needs at least two legs.');

  for (const leg of legs) {
    if (!leg.legEdge || !leg.legEdge.positive) {
      refusals.push(`Leg "${leg.selection}" is not independently +EV. Every leg must stand on its own.`);
    }
    if (leg.legScore && leg.legScore.vetoed) {
      refusals.push(`Leg "${leg.selection}" is vetoed: ${leg.legScore.vetoes.map((v) => v.key).join(', ')}.`);
    }
  }

  if (!args.simulation) {
    refusals.push('No simulation available, so leg correlation cannot be measured. Correlation may not be guessed.');
    return { recommend: false, refusals, verdict: 'PASS' };
  }

  const tests = legs.map((l) => l.test);
  const joint = jointProbability(args.simulation, tests);
  const independent = legs.reduce((p, l) => {
    const single = jointProbability(args.simulation, [l.test]);
    return p * single;
  }, 1);

  const combinedOdds = args.parlayDecimalOdds
    || legs.reduce((d, l) => d * l.decimalOdds, 1);

  const ev = expectedValue(joint, combinedOdds);
  const evIfIndependent = expectedValue(independent, combinedOdds);

  // Pairwise correlations, for the report.
  const pairs = [];
  for (let i = 0; i < legs.length; i++) {
    for (let j = i + 1; j < legs.length; j++) {
      const jm = jointMarket(args.simulation, legs[i].test, legs[j].test);
      pairs.push({ a: legs[i].selection, b: legs[j].selection, correlation: jm.correlation });
    }
  }

  if (joint <= 0) refusals.push('The legs never co-occurred in simulation — the combination is effectively impossible.');
  if (ev <= 0) {
    refusals.push(`Measured joint probability ${(joint * 100).toFixed(2)}% gives EV ${(ev * 100).toFixed(2)}% at the offered price.`);
  }
  if (joint < independent) {
    refusals.push('Legs are negatively correlated: the book\'s independence assumption favours the book here, not you.');
  }

  const recommend = refusals.length === 0 && ev > 0;

  return {
    recommend,
    verdict: recommend ? 'BET' : 'PASS',
    refusals,
    jointProbability: joint,
    independenceProbability: independent,
    correlationBenefit: joint - independent,
    combinedDecimalOdds: combinedOdds,
    expectedValue: ev,
    expectedValueUnderIndependence: evIfIndependent,
    pairwiseCorrelations: pairs,
    note: recommend
      ? 'Legs are positively correlated and the combined price does not reflect it — this is edge singles cannot capture.'
      : 'Default position retained: no parlay.',
  };
}

module.exports = { evaluate, jointProbability };
