'use strict';
// Event scanner — the orchestrator.
//
// Scans every bout and every market on a slate, prices them from one simulation
// per bout (so correlations between markets are measured rather than assumed),
// applies the gates, and returns only what survives. It fills no quota. When
// nothing clears, it says so.

const quality = require('../data/quality');
const mc = require('../sim/montecarlo');
const sens = require('../sim/sensitivity');
const { consensusMarket } = require('../core/odds');
const { computeEdge } = require('../core/ev');
const ensemble = require('../models/ensemble');
const scoring = require('../scoring/score');
const bankroll = require('../risk/bankroll');
const recommend = require('./recommend');
const { bandReliability } = require('../calibration/metrics');

/**
 * Markets derived from a single simulation. Each entry names the outcome, the
 * predicate that prices it, and which side of the bout it backs (for the
 * adversarial re-runs).
 */
function marketCatalogue(bout) {
  const nameA = bout.a.name, nameB = bout.b.name;
  return [
    { key: 'ML_A', market: 'Moneyline', selection: nameA, test: mc.MARKETS.winnerA, side: 'A', quoteKey: 'moneyline', outcomeIndex: 0 },
    { key: 'ML_B', market: 'Moneyline', selection: nameB, test: mc.MARKETS.winnerB, side: 'B', quoteKey: 'moneyline', outcomeIndex: 1 },
    { key: 'DISTANCE_YES', market: 'Goes the distance', selection: 'Yes', test: mc.MARKETS.goesDistance, side: null, quoteKey: 'distance', outcomeIndex: 0 },
    { key: 'DISTANCE_NO', market: 'Goes the distance', selection: 'No', test: mc.MARKETS.endsInside, side: null, quoteKey: 'distance', outcomeIndex: 1 },
    { key: 'A_KO', market: 'Method', selection: `${nameA} by KO/TKO`, test: mc.MARKETS.aByKo, side: 'A', quoteKey: 'method_a_ko', outcomeIndex: 0 },
    { key: 'A_SUB', market: 'Method', selection: `${nameA} by submission`, test: mc.MARKETS.aBySub, side: 'A', quoteKey: 'method_a_sub', outcomeIndex: 0 },
    { key: 'A_DEC', market: 'Method', selection: `${nameA} by decision`, test: mc.MARKETS.aByDec, side: 'A', quoteKey: 'method_a_dec', outcomeIndex: 0 },
    { key: 'B_KO', market: 'Method', selection: `${nameB} by KO/TKO`, test: mc.MARKETS.bByKo, side: 'B', quoteKey: 'method_b_ko', outcomeIndex: 0 },
    { key: 'B_SUB', market: 'Method', selection: `${nameB} by submission`, test: mc.MARKETS.bBySub, side: 'B', quoteKey: 'method_b_sub', outcomeIndex: 0 },
    { key: 'B_DEC', market: 'Method', selection: `${nameB} by decision`, test: mc.MARKETS.bByDec, side: 'B', quoteKey: 'method_b_dec', outcomeIndex: 0 },
  ];
}

/**
 * Scan one slate.
 *
 * @param {object} args
 * @param {object[]} args.bouts     canonical bout records with `quotes`
 * @param {number} args.now         decision timestamp (ms)
 * @param {number} args.bankroll
 * @param {object} [args.modelState] {calibrationHistory, ensembleWeights, validated}
 * @param {object} [args.riskState]  output of risk/protocol.evaluateDay
 * @param {object} [args.clv]        output of calibration/clv.aggregate
 * @param {object} [args.options]    {iterations, batches, seed, minScore}
 */
function scan(args) {
  const o = Object.assign({ iterations: 20000, batches: 40, seed: 20260816, minScore: 80 }, args.options || {});
  const modelState = args.modelState || {};
  const riskState = args.riskState || { permits: { newBets: true, stakeMultiplier: 1 }, state: 'NORMAL' };

  const evaluated = [];
  const rejected = [];

  for (const bout of args.bouts) {
    const q = quality.assess(bout, args.now);

    // Even a blocked bout is simulated and reported — knowing WHY a bout was
    // skipped is more useful than a silent omission.
    let sim = null;
    try {
      sim = mc.run({
        a: bout.a, b: bout.b, rounds: bout.rounds || 3,
        iterations: o.iterations, batches: o.batches,
        seed: hashSeed(o.seed, bout.id || bout.event || ''),
        uncertainty: bout.uncertainty,
      });
    } catch (err) {
      rejected.push({ bout: boutLabel(bout), reason: 'SIMULATION_FAILED', detail: err.message });
      continue;
    }

    const catalogue = marketCatalogue(bout);
    for (const m of catalogue) {
      const quotes = (bout.quotes || []).filter((x) => x.market === m.quoteKey);
      if (!quotes.length) {
        rejected.push({ bout: boutLabel(bout), market: m.market, selection: m.selection, reason: 'NO_PRICE' });
        continue;
      }
      const market = consensusMarket(quotes.map((x) => ({
        book: x.book, decimals: x.decimals, weight: x.weight, observedAt: x.observedAt,
      })));

      const marketProb = market.conservative[m.outcomeIndex];
      const decimalOdds = market.bestDecimal[m.outcomeIndex];
      const simPrice = mc.priceMarket(sim, m.test);

      // Ensemble: the simulator is one component. Others join once validated.
      let modelProb;
      let disagreement;
      const components = (modelState.components || []).filter((c) => c.marketKey === m.key);
      if (components.length) {
        const pooled = ensemble.pool(
          components.concat([{ key: 'simulation', probability: simPrice.probability, weight: modelState.simulationWeight || 1, validated: modelState.simulationValidated !== false }]),
          { requireValidated: true },
        );
        modelProb = pooled.probability;
        disagreement = pooled.disagreement;
      } else if (modelState.validated) {
        modelProb = simPrice.probability;
        disagreement = 0;
      } else {
        // No validated model: the simulator may not speak on its own authority.
        const anchored = ensemble.marketAnchored(simPrice.probability, marketProb);
        modelProb = anchored.probability;
        disagreement = Math.abs(simPrice.probability - marketProb);
      }

      const calRel = bandReliability(modelState.calibrationHistory || [], modelProb);

      // Adversarial re-runs, only where a side is identifiable.
      let sensitivity = null;
      if (m.side) {
        sensitivity = sens.runSensitivity({
          a: bout.a, b: bout.b, rounds: bout.rounds || 3,
          iterations: Math.min(o.iterations, 6000), batches: Math.min(o.batches, 20),
          seed: hashSeed(o.seed + 7, bout.id || ''), uncertainty: bout.uncertainty,
        }, m.side, marketProb);
      }

      const edge = computeEdge({
        modelProb,
        marketProb,
        decimalOdds,
        context: {
          calibrationReliability: calRel,
          disagreement,
          dataQuality: q.score,
          effectiveSample: Math.min(bout.a.bouts || 0, bout.b.bouts || 0),
          sensitivitySurvival: sensitivity ? sensitivity.survival : 0,
          bookCount: market.bookCount,
        },
      });

      const scored = scoring.rate({
        edge, simulation: sim, sensitivity, quality: q, market,
        clv: args.clv, calibrationReliability: calRel,
        priceVerified: q.gate !== 'BLOCK' && !q.defects.some((d) => d.key === 'NO_ODDS' || d.key === 'VERY_STALE_ODDS'),
      });

      const stake = scored.action === 'BET' && riskState.permits.newBets
        ? bankroll.sizeBet({
          bankroll: args.bankroll,
          probability: edge.effectiveProb,
          decimalOdds,
          drawdown: riskState.drawdown || 0,
          confidenceMultiplier: (riskState.permits.stakeMultiplier || 1) * edge.shrink.product,
          policy: args.policy,
        })
        : { stake: 0, fraction: 0, units: 0, reason: scored.vetoed ? 'VETOED' : (!riskState.permits.newBets ? `RISK_STATE_${riskState.state}` : 'BELOW_THRESHOLD') };

      const candidate = {
        boutId: bout.id,
        event: bout.event || boutLabel(bout),
        sport: 'MMA',
        market: m.market,
        selection: m.selection,
        marketKey: m.key,
        side: m.side,
        decimalOdds,
        book: market.bestBook[m.outcomeIndex],
        priceObservedAt: Math.max(...quotes.map((x) => x.observedAt || 0)),
        priceVerified: scored.vetoes.every((v) => v.key !== 'PRICE_UNVERIFIED'),
        simulation: sim,
        simulatedProbability: simPrice.probability,
        marketAnchored: !modelState.validated && !components.length,
        sensitivity, quality: q, marketView: market, edge, scored, stake,
      };
      candidate.emitted = recommend.emit(candidate);
      evaluated.push(candidate);
    }
  }

  // Rank, then drop dominated/correlated duplicates.
  const passing = evaluated
    .filter((c) => !c.scored.vetoed && c.scored.score >= o.minScore && c.stake.stake > 0)
    .sort((x, y) => y.scored.score - x.scored.score);

  const selected = dropCorrelated(passing);

  return {
    generatedAt: new Date(args.now).toISOString(),
    riskState: riskState.state,
    boutsScanned: args.bouts.length,
    marketsEvaluated: evaluated.length,
    qualifying: selected.length,
    selections: selected,
    allCandidates: evaluated,
    rejected,
    summary: selected.length
      ? `${selected.length} qualifying selection(s).`
      : 'NO QUALIFYING EDGE.',
    // The single most important line in the output when data is missing.
    blocked: evaluated.length && evaluated.every((c) => c.scored.vetoed)
      ? summariseBlocks(evaluated)
      : null,
  };
}

/**
 * Keep the best candidate per bout-side. Two markets driven by the same fighter
 * winning are one opinion expressed twice; sizing them independently would
 * understate the true exposure.
 */
function dropCorrelated(candidates) {
  const seen = new Set();
  const out = [];
  for (const c of candidates) {
    const key = `${c.boutId}:${c.side || 'neutral'}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(c);
  }
  return out;
}

function summariseBlocks(candidates) {
  const counts = new Map();
  for (const c of candidates) {
    for (const v of c.scored.vetoes) counts.set(v.key, (counts.get(v.key) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([key, n]) => `${key} (${n} market${n === 1 ? '' : 's'})`);
}

function boutLabel(b) {
  return `${(b.a && b.a.name) || '?'} vs ${(b.b && b.b.name) || '?'}`;
}

/** Deterministic per-bout seed so a rescan of the same slate reproduces exactly. */
function hashSeed(base, str) {
  let h = base >>> 0;
  for (let i = 0; i < String(str).length; i++) {
    h = (Math.imul(h ^ String(str).charCodeAt(i), 16777619)) >>> 0;
  }
  return h;
}

module.exports = { scan, marketCatalogue, dropCorrelated, hashSeed };
