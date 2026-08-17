'use strict';
// MMA Monte Carlo engine — continuous-time competing risks.
//
// Two design decisions distinguish this from a coin-flip-with-a-probability:
//
// 1. PARAMETER UNCERTAINTY. Iterations are grouped into batches; each batch
//    draws a fresh parameter set from the fighters' posteriors and then
//    simulates many fights under it. The spread ACROSS batches is genuine
//    epistemic uncertainty. Pure outcome noise would shrink to zero with more
//    iterations and would tell you nothing about how much you actually know.
//
// 2. STATE. Damage accumulates and fatigue builds, so finish hazards are not
//    memoryless. This is what makes late-round finishes and cardio-driven
//    collapses emerge from the model rather than being asserted by a rule.
//
// Every run is seeded and therefore reproducible: a claim that a simulation was
// run is checkable by re-running it.

const { mulberry32, jitter, clamp, mean, stdev, quantile } = require('../core/prob');

const ROUND_MINUTES = 5;
const TICK_SECONDS = 5;
const TICK_MINUTES = TICK_SECONDS / 60;

/**
 * Canonical simulator parameters for one fighter.
 * All are "per 15 minutes vs. a division-average opponent" or unitless
 * multipliers where 1.0 is division-average.
 *
 * FITTED, not guessed. koRate and subRate were solved with
 * `sensitivity.fitMethodRates` so that two average fighters over three rounds
 * reproduce the observed UFC marginals:
 *
 *   source:   5,807 UFC bouts, 2015-01-01 to 2026-08-15 (ufc_fights)
 *   observed: KO/TKO 31.7% · submission 17.7% · decision 49.3%
 *   achieved: KO/TKO 31.5% · submission 18.1% · decision 50.5%
 *
 * Re-run the fit whenever the hazard model or TUNING changes — these constants
 * are only meaningful together with the rest of the model.
 */
const FIGHTER_DEFAULTS = {
  name: 'Fighter',
  koRate: 0.2603,      // expected KO/TKO finishes per 15 min vs average opposition
  subRate: 0.1839,     // expected submission finishes per 15 min
  durability: 1.0,     // >1 = harder to finish (divides opponent's KO hazard)
  subDefense: 1.0,     // >1 = harder to submit
  cardio: 1.0,         // >1 = fatigues more slowly
  output: 4.0,         // significant strikes landed per minute at full freshness
  accuracy: 0.45,
  grappling: 1.0,      // takedown/control offense
  takedownDefense: 1.0,
  control: 0.3,        // share of time in controlling position when grappling
};

/** Per-fighter parameter uncertainty (lognormal sigma). Wider = less is known. */
const DEFAULT_UNCERTAINTY = {
  koRate: 0.30, subRate: 0.35, durability: 0.15,
  subDefense: 0.15, cardio: 0.12, output: 0.12, grappling: 0.18,
};

const TUNING = {
  damageCoef: 0.85,      // how much accumulated damage raises KO hazard
  fatigueVulnCoef: 0.55, // how much fatigue raises vulnerability to finishes
  fatigueOutputCoef: 0.30,
  cardioTau: 9.0,        // minutes; fatigue time-constant at cardio = 1.0
  judgeNoise: 0.18,      // per-judge per-round scoring noise
  controlScoreWeight: 1.6,
};

function withDefaults(f) {
  const out = Object.assign({}, FIGHTER_DEFAULTS, f || {});
  for (const k of ['koRate', 'subRate', 'durability', 'subDefense', 'cardio', 'output', 'grappling', 'takedownDefense']) {
    if (!(out[k] >= 0)) throw new Error(`simulator: fighter "${out.name}" has invalid ${k}`);
  }
  return out;
}

/** Draw one parameter realisation for a fighter from its posterior. */
function drawParams(f, unc, rng) {
  const u = Object.assign({}, DEFAULT_UNCERTAINTY, unc || {});
  return {
    name: f.name,
    koRate: f.koRate * jitter(rng, u.koRate),
    subRate: f.subRate * jitter(rng, u.subRate),
    durability: f.durability * jitter(rng, u.durability),
    subDefense: f.subDefense * jitter(rng, u.subDefense),
    cardio: f.cardio * jitter(rng, u.cardio),
    output: f.output * jitter(rng, u.output),
    accuracy: f.accuracy,
    grappling: f.grappling * jitter(rng, u.grappling),
    takedownDefense: f.takedownDefense,
    control: f.control,
  };
}

/** Fatigue in [0,1): approaches 1 as the fight wears on, slower for better cardio. */
function fatigueAt(minutes, cardio) {
  const tau = TUNING.cardioTau * Math.max(0.2, cardio);
  return 1 - Math.exp(-minutes / tau);
}

/**
 * Share of exchange time each fighter spends in their preferred phase.
 * A grappler who cannot get the fight down does not get to use their grappling.
 */
function grapplePressure(att, def) {
  const ratio = att.grappling / Math.max(0.2, def.takedownDefense);
  return clamp(ratio / (ratio + 1.6), 0, 0.85);
}

/**
 * Simulate one fight to completion under a fixed parameter set.
 * @returns {{winner:'A'|'B'|null, method:'ko'|'sub'|'dec'|'draw', round:number, timeSec:number}}
 */
function simulateOne(A, B, rounds, rng) {
  let dmgA = 0, dmgB = 0;          // damage absorbed by each fighter
  const roundsWonA = [];
  const gpA = grapplePressure(A, B);
  const gpB = grapplePressure(B, A);

  for (let r = 1; r <= rounds; r++) {
    let scoreA = 0, scoreB = 0;
    for (let t = 0; t < (ROUND_MINUTES * 60) / TICK_SECONDS; t++) {
      const elapsed = (r - 1) * ROUND_MINUTES + t * TICK_MINUTES;
      const fatA = fatigueAt(elapsed, A.cardio);
      const fatB = fatigueAt(elapsed, B.cardio);

      const effOutA = A.output * (1 - TUNING.fatigueOutputCoef * fatA);
      const effOutB = B.output * (1 - TUNING.fatigueOutputCoef * fatB);

      const vulnA = (1 + TUNING.fatigueVulnCoef * fatA) * (1 + TUNING.damageCoef * dmgA);
      const vulnB = (1 + TUNING.fatigueVulnCoef * fatB) * (1 + TUNING.damageCoef * dmgB);

      // Striking hazards apply mostly on the feet; grappling hazards on the mat.
      const standShareA = 1 - gpB;   // A strikes when B is not grappling them
      const standShareB = 1 - gpA;

      const hKoA = (A.koRate / 15) * (effOutA / Math.max(0.5, A.output)) * standShareA * vulnB / Math.max(0.2, B.durability);
      const hKoB = (B.koRate / 15) * (effOutB / Math.max(0.5, B.output)) * standShareB * vulnA / Math.max(0.2, A.durability);
      const hSubA = (A.subRate / 15) * gpA * vulnB / Math.max(0.2, B.subDefense);
      const hSubB = (B.subRate / 15) * gpB * vulnA / Math.max(0.2, A.subDefense);

      const total = hKoA + hKoB + hSubA + hSubB;
      const pEvent = 1 - Math.exp(-total * TICK_MINUTES);

      if (rng() < pEvent) {
        const pick = rng() * total;
        const timeSec = Math.round(t * TICK_SECONDS + rng() * TICK_SECONDS);
        if (pick < hKoA) return { winner: 'A', method: 'ko', round: r, timeSec };
        if (pick < hKoA + hSubA) return { winner: 'A', method: 'sub', round: r, timeSec };
        if (pick < hKoA + hSubA + hKoB) return { winner: 'B', method: 'ko', round: r, timeSec };
        return { winner: 'B', method: 'sub', round: r, timeSec };
      }

      // No finish: accrue damage and round score.
      dmgA += (effOutB * B.accuracy * TICK_MINUTES) / 45;
      dmgB += (effOutA * A.accuracy * TICK_MINUTES) / 45;
      scoreA += effOutA * A.accuracy * TICK_MINUTES + gpA * A.control * TUNING.controlScoreWeight * TICK_MINUTES;
      scoreB += effOutB * B.accuracy * TICK_MINUTES + gpB * B.control * TUNING.controlScoreWeight * TICK_MINUTES;
    }
    roundsWonA.push({ scoreA, scoreB });
  }

  // Went the distance: three independent judges, each with their own noise.
  let cardsA = 0, cardsB = 0, cardsDraw = 0;
  for (let j = 0; j < 3; j++) {
    let a = 0, b = 0;
    for (const rd of roundsWonA) {
      const margin = (rd.scoreA - rd.scoreB) + (rng() * 2 - 1) * TUNING.judgeNoise * (rd.scoreA + rd.scoreB + 1);
      if (margin > 0) a++; else b++;
    }
    if (a > b) cardsA++; else if (b > a) cardsB++; else cardsDraw++;
  }
  if (cardsA >= 2) return { winner: 'A', method: 'dec', round: rounds, timeSec: ROUND_MINUTES * 60 };
  if (cardsB >= 2) return { winner: 'B', method: 'dec', round: rounds, timeSec: ROUND_MINUTES * 60 };
  return { winner: null, method: 'draw', round: rounds, timeSec: ROUND_MINUTES * 60 };
}

/**
 * Run the full simulation.
 *
 * @param {object} args
 * @param {object} args.a  fighter A parameters
 * @param {object} args.b  fighter B parameters
 * @param {number} [args.rounds=3]
 * @param {number} [args.iterations=20000] total fights simulated
 * @param {number} [args.batches=40] parameter draws; iterations are split across them
 * @param {number} [args.seed=1] required for reproducibility
 * @param {object} [args.uncertainty] {a:{...}, b:{...}} lognormal sigmas
 */
function run(args) {
  const a = withDefaults(args.a);
  const b = withDefaults(args.b);
  const rounds = args.rounds || 3;
  const iterations = args.iterations || 20000;
  const batches = Math.max(2, args.batches || 40);
  const perBatch = Math.max(1, Math.floor(iterations / batches));
  const rng = mulberry32(args.seed != null ? args.seed : 1);
  const unc = args.uncertainty || {};

  const outcomes = [];
  const batchPA = [];

  for (let batch = 0; batch < batches; batch++) {
    const pa = drawParams(a, unc.a, rng);
    const pb = drawParams(b, unc.b, rng);
    let winsA = 0;
    for (let i = 0; i < perBatch; i++) {
      const o = simulateOne(pa, pb, rounds, rng);
      outcomes.push(o);
      if (o.winner === 'A') winsA++;
    }
    batchPA.push(winsA / perBatch);
  }

  const n = outcomes.length;
  const tally = {
    aKo: 0, aSub: 0, aDec: 0, bKo: 0, bSub: 0, bDec: 0, draw: 0,
  };
  const finishRound = new Array(rounds + 1).fill(0);
  let distance = 0;
  let totalSeconds = 0;

  for (const o of outcomes) {
    if (o.winner === null) tally.draw++;
    else if (o.winner === 'A') tally['a' + cap(o.method)]++;
    else tally['b' + cap(o.method)]++;
    if (o.method === 'dec' || o.method === 'draw') {
      distance++;
      totalSeconds += rounds * ROUND_MINUTES * 60;
    } else {
      finishRound[o.round]++;
      totalSeconds += (o.round - 1) * ROUND_MINUTES * 60 + o.timeSec;
    }
  }

  const pA = (tally.aKo + tally.aSub + tally.aDec) / n;
  const pB = (tally.bKo + tally.bSub + tally.bDec) / n;
  const sortedBatch = batchPA.slice().sort((x, y) => x - y);

  return {
    seed: args.seed != null ? args.seed : 1,
    iterations: n,
    batches,
    rounds,
    probA: pA,
    probB: pB,
    probDraw: tally.draw / n,
    method: {
      a: { ko: tally.aKo / n, sub: tally.aSub / n, dec: tally.aDec / n },
      b: { ko: tally.bKo / n, sub: tally.bSub / n, dec: tally.bDec / n },
      draw: tally.draw / n,
    },
    finishRound: finishRound.map((c) => c / n),
    goesDistance: distance / n,
    meanDurationSec: totalSeconds / n,
    // Parameter-uncertainty interval: the spread of per-batch win rates.
    // This is the honest interval; it does NOT shrink with more iterations.
    paramInterval: [quantile(sortedBatch, 0.05), quantile(sortedBatch, 0.95)],
    paramStdev: stdev(batchPA),
    // Monte Carlo standard error of the mean across batches.
    standardError: stdev(batchPA) / Math.sqrt(batches),
    batchMean: mean(batchPA),
    outcomes,
  };
}

function cap(m) { return m.charAt(0).toUpperCase() + m.slice(1); }

/**
 * Price an arbitrary market from a completed run.
 * @param {object} result run() output
 * @param {(o:object)=>boolean} test predicate over a single outcome
 */
function priceMarket(result, test) {
  let hits = 0;
  for (const o of result.outcomes) if (test(o)) hits++;
  const p = hits / result.outcomes.length;
  return { probability: p, hits, n: result.outcomes.length };
}

/**
 * Joint probability and correlation of two markets, measured on the SAME
 * simulated fights. This is how parlay correlation is obtained — measured,
 * never assumed from a fixed haircut.
 */
function jointMarket(result, testX, testY) {
  let x = 0, y = 0, both = 0;
  for (const o of result.outcomes) {
    const hx = testX(o), hy = testY(o);
    if (hx) x++;
    if (hy) y++;
    if (hx && hy) both++;
  }
  const n = result.outcomes.length;
  const px = x / n, py = y / n, pxy = both / n;
  const denom = Math.sqrt(px * (1 - px) * py * (1 - py));
  return {
    pX: px, pY: py, pXY: pxy,
    independentPXY: px * py,
    correlation: denom > 0 ? (pxy - px * py) / denom : 0,
  };
}

/** Standard market predicates. */
const MARKETS = {
  winnerA: (o) => o.winner === 'A',
  winnerB: (o) => o.winner === 'B',
  goesDistance: (o) => o.method === 'dec' || o.method === 'draw',
  endsInside: (o) => o.method === 'ko' || o.method === 'sub',
  aByKo: (o) => o.winner === 'A' && o.method === 'ko',
  aBySub: (o) => o.winner === 'A' && o.method === 'sub',
  aByDec: (o) => o.winner === 'A' && o.method === 'dec',
  bByKo: (o) => o.winner === 'B' && o.method === 'ko',
  bBySub: (o) => o.winner === 'B' && o.method === 'sub',
  bByDec: (o) => o.winner === 'B' && o.method === 'dec',
  /** Over X.5 rounds: the fight is still going after round ceil(X.5) minus half. */
  overRounds: (x) => (o) => {
    const secs = o.method === 'dec' || o.method === 'draw'
      ? Infinity
      : (o.round - 1) * ROUND_MINUTES * 60 + o.timeSec;
    return secs > x * ROUND_MINUTES * 60;
  },
};

module.exports = {
  run, priceMarket, jointMarket, MARKETS,
  simulateOne, fatigueAt, grapplePressure, withDefaults, drawParams,
  FIGHTER_DEFAULTS, DEFAULT_UNCERTAINTY, TUNING, ROUND_MINUTES,
};
