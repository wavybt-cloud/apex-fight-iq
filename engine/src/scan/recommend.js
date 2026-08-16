'use strict';
// Recommendation engine — produces the mandated report and the BET/PASS verdict.

const { decimalToAmerican } = require('../core/odds');

function pct(x, dp = 1) { return x == null ? '—' : (x * 100).toFixed(dp) + '%'; }
function signed(x, dp = 2) { return x == null ? '—' : (x >= 0 ? '+' : '') + (x * 100).toFixed(dp) + '%'; }

/**
 * Build the full report for one candidate.
 *
 * @param {object} c candidate assembled by the scanner
 * @returns {{verdict:'BET'|'PASS', report:object, text:string}}
 */
function emit(c) {
  const s = c.scored;
  const e = c.edge;
  const sim = c.simulation;
  const verdict = s.action === 'BET' && c.stake && c.stake.stake > 0 ? 'BET' : 'PASS';

  const risk = riskLevel(c);
  const reasons = whyModelLikesIt(c);
  const wrong = whatCouldMakeItWrong(c);

  const report = {
    sport: c.sport || 'MMA',
    event: c.event,
    market: c.market,
    selection: c.selection,
    currentPrice: {
      decimal: c.decimalOdds,
      american: decimalToAmerican(c.decimalOdds),
      book: c.book,
      observedAt: c.priceObservedAt ? new Date(c.priceObservedAt).toISOString() : null,
      verified: c.priceVerified === true,
    },
    modelProbability: e ? e.modelProb : null,
    rawSimulationProbability: c.simulatedProbability != null ? c.simulatedProbability : null,
    marketAnchored: c.marketAnchored === true,
    effectiveProbability: e ? e.effectiveProb : null,
    marketImpliedProbability: e ? e.marketProb : null,
    estimatedEdge: e ? e.effectiveEdge : null,
    rawEdgeBeforeShrinkage: e ? e.rawEdge : null,
    expectedValue: e ? e.evEffective : null,
    riskAdjustedEV: e ? e.riskAdjustedEV : null,
    simulation: sim ? {
      iterations: sim.iterations,
      seed: sim.seed,
      // The simulated probability of THIS market, not merely of the fighter
      // winning — method and total markets are priced from the same run.
      hitRate: c.simulatedProbability != null
        ? c.simulatedProbability
        : (c.side === 'A' ? sim.probA : sim.probB),
      standardError: sim.standardError,
      parameterInterval: sim.paramInterval,
      goesDistance: sim.goesDistance,
      method: sim.method,
    } : null,
    sensitivity: c.sensitivity ? {
      survival: c.sensitivity.survival,
      worstCase: c.sensitivity.worst,
      bestCase: c.sensitivity.best,
    } : null,
    modelConfidence: confidenceLabel(c),
    dataQuality: c.quality ? { score: c.quality.score, gate: c.quality.gate, defects: c.quality.defects.map((d) => d.label) } : null,
    riskLevel: risk,
    recommendedStake: c.stake || null,
    whyTheModelLikesIt: reasons,
    whatCouldMakeTheModelWrong: wrong,
    finalScore: s.score,
    band: s.band,
    vetoes: s.vetoes,
    verdict,
  };

  return { verdict, report, text: renderText(report) };
}

function confidenceLabel(c) {
  if (!c.simulation || !c.edge) return 'NONE — inputs incomplete';
  const width = c.simulation.paramInterval[1] - c.simulation.paramInterval[0];
  const shrink = c.edge.shrink.product;
  if (shrink === 0) return 'NONE — no validated calibration record; the model is not permitted to claim an edge';
  if (shrink < 0.2 || width > 0.20) return 'LOW';
  if (shrink < 0.5 || width > 0.12) return 'MODERATE';
  return 'HIGH';
}

function riskLevel(c) {
  if (!c.simulation || !c.edge) return 'UNQUANTIFIED';
  const width = c.simulation.paramInterval[1] - c.simulation.paramInterval[0];
  const varr = c.edge.variance;
  if (width > 0.20 || varr > 2.5) return 'HIGH VARIANCE';
  if (width > 0.12 || varr > 1.2) return 'MODERATE VARIANCE';
  return 'LOWER VARIANCE';
}

function whyModelLikesIt(c) {
  const out = [];
  if (!c.edge) return ['No edge computed.'];
  out.push(`Model ${pct(c.edge.modelProb)}${c.marketAnchored ? ' (market-anchored: unvalidated models are not allowed to deviate freely)' : ''} vs market ${pct(c.edge.marketProb)}; after shrinkage the engine is willing to claim ${signed(c.edge.effectiveEdge)} of edge.`);
  if (c.simulation) {
    out.push(`${c.simulation.iterations.toLocaleString()} simulated fights (seed ${c.simulation.seed}) put the selection at ${pct(c.simulatedProbability != null ? c.simulatedProbability : (c.side === 'A' ? c.simulation.probA : c.simulation.probB))} before anchoring, parameter interval ${pct(c.simulation.paramInterval[0])}–${pct(c.simulation.paramInterval[1])}.`);
  }
  if (c.sensitivity) {
    out.push(`Edge held in ${c.sensitivity.scenarios.filter((s) => s.survives).length}/${c.sensitivity.scenarios.length} adverse scenarios; worst case "${c.sensitivity.worst.label}" left ${signed(c.sensitivity.worst.edge)}.`);
  }
  if (c.drivers && c.drivers.length) out.push(...c.drivers);
  return out;
}

function whatCouldMakeItWrong(c) {
  const out = [];
  if (c.sensitivity) {
    for (const s of c.sensitivity.scenarios.slice().sort((x, y) => x.probability - y.probability).slice(0, 3)) {
      out.push(`${s.label}: probability falls to ${pct(s.probability)} (${signed(s.edge)} edge).`);
    }
  }
  if (c.quality && c.quality.defects.length) {
    out.push(`Data defects present: ${c.quality.defects.map((d) => d.label).join('; ')}.`);
  }
  if (c.edge && c.edge.shrink.calibration < 0.5) {
    out.push('The model has little or no verified calibration history in this probability band — the stated probability may simply be wrong.');
  }
  if (c.edge && c.edge.shrink.disagreement < 0.5) {
    out.push('Component models disagree materially; the ensemble number is masking a genuine dispute.');
  }
  if (c.marketView && c.marketView.bookCount < 2) {
    out.push('Only one book is quoting, so the "market price" is one operator\'s opinion rather than a market.');
  }
  out.push('The market may already know something the model does not — late news, camp reports, or money from people closer to the fight.');
  return out;
}

function renderText(r) {
  const L = [];
  L.push(`SPORT: ${r.sport}`);
  L.push(`EVENT: ${r.event}`);
  L.push(`MARKET: ${r.market} — ${r.selection}`);
  L.push(`CURRENT PRICE: ${r.currentPrice.american > 0 ? '+' : ''}${r.currentPrice.american} (${r.currentPrice.decimal.toFixed(3)}) @ ${r.currentPrice.book || 'n/a'}${r.currentPrice.verified ? '' : '  [UNVERIFIED]'}`);
  L.push('');
  L.push(`MODEL PROBABILITY: ${pct(r.modelProbability)}${r.marketAnchored ? ' [market-anchored]' : ''}  (after shrinkage: ${pct(r.effectiveProbability)})`);
  if (r.rawSimulationProbability != null) {
    L.push(`  raw simulation before anchoring/shrinkage: ${pct(r.rawSimulationProbability)}`);
  }
  L.push(`MARKET IMPLIED PROBABILITY: ${pct(r.marketImpliedProbability)}`);
  L.push(`ESTIMATED EDGE: ${signed(r.estimatedEdge)}   (raw, before shrinkage: ${signed(r.rawEdgeBeforeShrinkage)})`);
  L.push(`EXPECTED VALUE: ${signed(r.expectedValue)} per unit staked   |   RISK-ADJUSTED: ${signed(r.riskAdjustedEV)}`);
  L.push('');
  if (r.simulation) {
    L.push(`SIMULATION RESULTS: ${r.simulation.iterations.toLocaleString()} fights, seed ${r.simulation.seed}`);
    L.push(`  hit rate ${pct(r.simulation.hitRate)} · SE ${pct(r.simulation.standardError, 2)} · parameter interval ${pct(r.simulation.parameterInterval[0])}–${pct(r.simulation.parameterInterval[1])}`);
    L.push(`  goes the distance ${pct(r.simulation.goesDistance)}`);
  } else {
    L.push('SIMULATION RESULTS: none run');
  }
  if (r.sensitivity) {
    L.push(`  adverse-scenario survival ${pct(r.sensitivity.survival, 0)} · worst: ${r.sensitivity.worstCase.label} → ${signed(r.sensitivity.worstCase.edge)}`);
  }
  L.push(`MODEL CONFIDENCE: ${r.modelConfidence}`);
  L.push(`DATA QUALITY: ${r.dataQuality ? `${r.dataQuality.score}/100 (${r.dataQuality.gate})` : '—'}`);
  if (r.dataQuality && r.dataQuality.defects.length) L.push(`  defects: ${r.dataQuality.defects.join('; ')}`);
  L.push('');
  L.push(`RISK LEVEL: ${r.riskLevel}`);
  L.push(`RECOMMENDED STAKE: ${r.recommendedStake && r.recommendedStake.stake > 0
    ? `${r.recommendedStake.units.toFixed(2)}u (${(r.recommendedStake.fraction * 100).toFixed(2)}% of bankroll)`
    : '0 — no stake'}`);
  L.push('');
  L.push('WHY THE MODEL LIKES IT:');
  for (const x of r.whyTheModelLikesIt) L.push(`  · ${x}`);
  L.push('');
  L.push('WHAT COULD MAKE THE MODEL WRONG:');
  for (const x of r.whatCouldMakeTheModelWrong) L.push(`  · ${x}`);
  L.push('');
  L.push(`FINAL SCORE: ${r.finalScore.toFixed(1)} / 100 — ${r.band}`);
  if (r.vetoes.length) {
    L.push('VETOES:');
    for (const v of r.vetoes) L.push(`  ✗ ${v.key}: ${v.detail}`);
  }
  L.push('');
  L.push(r.verdict);
  return L.join('\n');
}

module.exports = { emit, renderText, confidenceLabel, riskLevel };
