'use strict';
// Adversarial testing (§23) and base-rate fitting.
//
// A model probability that only survives under its own favourite assumptions is
// not evidence. Before any candidate is allowed through, the simulation is
// re-run under scenarios chosen to be hostile to the side the model likes, and
// the candidate must keep its edge in all of them.

const mc = require('./montecarlo');

/**
 * Scenarios applied to the fighter the model is backing. Each degrades an
 * assumption the edge might be resting on.
 */
const ADVERSE_SCENARIOS = [
  {
    key: 'chin_minus_1sd',
    label: 'Backed fighter is more fragile than estimated',
    apply: (f, u) => ({ f: Object.assign({}, f, { durability: f.durability * Math.exp(-(u.durability || 0.15)) }), u }),
  },
  {
    key: 'cardio_minus_1sd',
    label: 'Backed fighter fades earlier than estimated',
    apply: (f, u) => ({ f: Object.assign({}, f, { cardio: f.cardio * Math.exp(-(u.cardio || 0.12)) }), u }),
  },
  {
    key: 'td_defense_minus_1sd',
    label: 'Backed fighter is more takeable-down than estimated',
    apply: (f, u) => ({ f: Object.assign({}, f, { takedownDefense: f.takedownDefense * 0.85 }), u }),
  },
  {
    key: 'finishing_regressed',
    label: 'Backed fighter finishing rate regressed toward division mean',
    apply: (f, u) => ({
      f: Object.assign({}, f, {
        koRate: f.koRate * 0.75 + mc.FIGHTER_DEFAULTS.koRate * 0.25,
        subRate: f.subRate * 0.75 + mc.FIGHTER_DEFAULTS.subRate * 0.25,
      }),
      u,
    }),
  },
  {
    key: 'output_regressed',
    label: 'Backed fighter output regressed (recent form was noise)',
    apply: (f, u) => ({
      f: Object.assign({}, f, { output: f.output * 0.9 + mc.FIGHTER_DEFAULTS.output * 0.1 }),
      u,
    }),
  },
  {
    key: 'wider_uncertainty',
    label: 'Parameter uncertainty is 50% wider than assumed',
    apply: (f, u) => {
      const wider = {};
      for (const [k, v] of Object.entries(Object.assign({}, mc.DEFAULT_UNCERTAINTY, u))) wider[k] = v * 1.5;
      return { f, u: wider };
    },
  },
];

/**
 * Re-run the simulation under every adverse scenario.
 *
 * @param {object} base           arguments accepted by montecarlo.run
 * @param {'A'|'B'} backedSide    the side the model wants to bet
 * @param {number} marketProb     devigged market probability for that side
 * @returns {{survival:number, scenarios:Array, worst:object, best:object, baseline:number}}
 *   `survival` is the fraction of scenarios in which the edge stayed positive.
 */
function runSensitivity(base, backedSide, marketProb) {
  const baseline = mc.run(base);
  const baseProb = backedSide === 'A' ? baseline.probA : baseline.probB;

  const scenarios = ADVERSE_SCENARIOS.map((sc) => {
    const key = backedSide === 'A' ? 'a' : 'b';
    const uKey = backedSide === 'A' ? 'a' : 'b';
    const unc = Object.assign({}, base.uncertainty || {});
    const applied = sc.apply(mc.withDefaults(base[key]), Object.assign({}, mc.DEFAULT_UNCERTAINTY, unc[uKey] || {}));
    const args = Object.assign({}, base, { uncertainty: Object.assign({}, unc, { [uKey]: applied.u }) });
    args[key] = applied.f;
    const r = mc.run(args);
    const p = backedSide === 'A' ? r.probA : r.probB;
    return {
      key: sc.key,
      label: sc.label,
      probability: p,
      edge: p - marketProb,
      survives: p - marketProb > 0,
      delta: p - baseProb,
    };
  });

  const survived = scenarios.filter((s) => s.survives).length;
  const sorted = scenarios.slice().sort((x, y) => x.probability - y.probability);

  return {
    baseline: baseProb,
    baselineEdge: baseProb - marketProb,
    scenarios,
    survival: scenarios.length ? survived / scenarios.length : 0,
    allSurvive: survived === scenarios.length,
    worst: sorted[0],
    best: sorted[sorted.length - 1],
  };
}

/**
 * Fit a global finish-hazard multiplier so the simulator reproduces an observed
 * base rate (e.g. "51% of UFC 3-round fights go to decision").
 *
 * The shipped TUNING constants are UNFITTED priors. This is the mechanical step
 * that replaces them with something anchored to reality, and it must be run
 * against real base rates before the simulator's absolute method/round
 * probabilities are trusted. Win probability is far less sensitive to this
 * multiplier than method probability is, because it scales both fighters.
 *
 * @param {object} sample  {a, b, rounds, iterations, batches, seed}
 * @param {number} targetDistanceRate observed fraction of fights going to decision
 * @returns {{multiplier:number, achieved:number, iterations:number}}
 */
function fitFinishRate(sample, targetDistanceRate, opts) {
  const o = Object.assign({ tol: 0.004, maxIter: 30, lo: 0.05, hi: 6 }, opts || {});
  const evaluate = (m) => {
    const scale = (f) => Object.assign({}, mc.withDefaults(f), {
      koRate: mc.withDefaults(f).koRate * m,
      subRate: mc.withDefaults(f).subRate * m,
    });
    const r = mc.run(Object.assign({}, sample, { a: scale(sample.a), b: scale(sample.b) }));
    return r.goesDistance;
  };
  let lo = o.lo, hi = o.hi, mid = 1, achieved = evaluate(1), iter = 0;
  // goesDistance decreases monotonically in the multiplier.
  for (; iter < o.maxIter; iter++) {
    mid = (lo + hi) / 2;
    achieved = evaluate(mid);
    if (Math.abs(achieved - targetDistanceRate) < o.tol) break;
    if (achieved > targetDistanceRate) lo = mid; else hi = mid;
  }
  return { multiplier: mid, achieved, iterations: iter + 1, target: targetDistanceRate };
}

/**
 * Fit BOTH finish hazards so the simulator reproduces observed KO and
 * submission base rates, not just the overall finish rate.
 *
 * Nominal koRate/subRate are not the realised rates: the KO hazard is scaled by
 * the share of time spent standing and the submission hazard by grappling
 * pressure. So the mapping from parameter to outcome is solved numerically
 * rather than assumed.
 *
 * Damped multiplicative fixed point — the objective is smooth and monotone in
 * each parameter, so this converges in a handful of iterations.
 *
 * @param {object} sample {a, b, rounds, iterations, batches, seed}
 * @param {{ko:number, sub:number}} targets observed marginal rates
 */
function fitMethodRates(sample, targets, opts) {
  const o = Object.assign({ tol: 0.006, maxIter: 25, damping: 0.7 }, opts || {});
  const baseA = mc.withDefaults(sample.a);
  const baseB = mc.withDefaults(sample.b);
  let koScale = 1, subScale = 1;
  let achieved = null;

  for (let i = 0; i < o.maxIter; i++) {
    const scale = (f) => Object.assign({}, f, {
      koRate: f.koRate * koScale,
      subRate: f.subRate * subScale,
    });
    const r = mc.run(Object.assign({}, sample, { a: scale(baseA), b: scale(baseB) }));
    const ko = r.method.a.ko + r.method.b.ko;
    const sub = r.method.a.sub + r.method.b.sub;
    achieved = { ko, sub, distance: r.goesDistance };
    if (Math.abs(ko - targets.ko) < o.tol && Math.abs(sub - targets.sub) < o.tol) {
      return { koScale, subScale, achieved, iterations: i + 1, converged: true, targets };
    }
    if (ko > 0) koScale *= Math.pow(targets.ko / ko, o.damping);
    if (sub > 0) subScale *= Math.pow(targets.sub / sub, o.damping);
  }
  return { koScale, subScale, achieved, iterations: o.maxIter, converged: false, targets };
}

module.exports = { ADVERSE_SCENARIOS, runSensitivity, fitFinishRate, fitMethodRates };
