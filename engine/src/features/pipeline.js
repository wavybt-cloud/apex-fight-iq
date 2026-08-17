'use strict';
// Feature engineering with an as-of contract.
//
// Every feature is antisymmetric: swapping A and B negates the vector. This is
// deliberate. It means the model physically cannot express "the A slot tends to
// win" — the very bias that contaminates this project's existing training data
// — because there is no intercept-like asymmetry for it to attach to.

const { clamp, shrinkToPrior } = require('../core/prob');

const FEATURES = [
  'eloDiff',
  'experienceDiff',
  'formDiff',
  'layoffDiff',
  'finishRateDiff',
  'durabilityDiff',
  'ageDiff',
  'reachDiff',
];

/**
 * Build the feature vector for one bout.
 *
 * @param {object} args
 * @param {object} args.book     Elo book, updated only with fights before `asOf`
 * @param {string} args.a        canonical A-side name
 * @param {string} args.b        canonical B-side name
 * @param {number} args.asOf     decision timestamp
 * @param {Map} [args.tape]      name -> {reachIn, dob}
 * @returns {{vector:number[], parts:object, coverage:number}}
 */
function build(args) {
  const { book, a, b, asOf } = args;
  const tape = args.tape || new Map();
  const sa = book.snapshot(a, asOf);
  const sb = book.snapshot(b, asOf);
  const ta = tape.get(a) || {};
  const tb = tape.get(b) || {};

  const parts = {};

  // Rating difference, scaled so a typical gap is order 1.
  parts.eloDiff = (sa.rating - sb.rating) / 200;

  // Experience on a log scale: fight 3 vs 4 matters, fight 23 vs 24 does not.
  parts.experienceDiff = Math.log1p(sa.fights) - Math.log1p(sb.fights);

  // Recent form, capped so a long streak does not dominate.
  parts.formDiff = (clamp(sa.streak, -5, 5) - clamp(sb.streak, -5, 5)) / 5;

  // Layoff, in years past a normal ~6-month turnaround.
  parts.layoffDiff = layoffPenalty(sa.layoffDays) - layoffPenalty(sb.layoffDays);

  // Finish rates, shrunk toward the division mean so 1-for-1 is not 100%.
  parts.finishRateDiff = shrinkToPrior(rate(sa.finishes, sa.fights), sa.fights, 0.45, 6)
    - shrinkToPrior(rate(sb.finishes, sb.fights), sb.fights, 0.45, 6);

  // How often each has been finished — the durability signal, sign flipped so
  // that positive still favours A.
  parts.durabilityDiff = shrinkToPrior(rate(sb.finishedAgainst, sb.fights), sb.fights, 0.45, 6)
    - shrinkToPrior(rate(sa.finishedAgainst, sa.fights), sa.fights, 0.45, 6);

  // Physical attributes, present for about half the roster.
  const ageA = ageAt(ta.dob, asOf), ageB = ageAt(tb.dob, asOf);
  parts.ageDiff = (ageA != null && ageB != null) ? (ageB - ageA) / 5 : 0;
  parts.reachDiff = (ta.reachIn != null && tb.reachIn != null) ? (ta.reachIn - tb.reachIn) / 4 : 0;

  const covered = [
    ta.dob != null && tb.dob != null,
    ta.reachIn != null && tb.reachIn != null,
    sa.fights > 0 && sb.fights > 0,
  ].filter(Boolean).length;

  return {
    vector: FEATURES.map((k) => parts[k]),
    parts,
    coverage: covered / 3,
    snapshots: { a: sa, b: sb },
  };
}

function rate(n, d) { return d > 0 ? n / d : 0; }

function layoffPenalty(days) {
  if (days == null) return 0;
  return clamp((days - 180) / 365, 0, 3);
}

function ageAt(dobMs, atMs) {
  if (dobMs == null || atMs == null) return null;
  return (atMs - dobMs) / (365.25 * 86400000);
}

/**
 * Verify the antisymmetry that keeps a side bias unrepresentable. Exported so
 * it can be asserted in tests rather than assumed.
 */
function isAntisymmetric(args, tol = 1e-9) {
  const fwd = build(args);
  const rev = build(Object.assign({}, args, { a: args.b, b: args.a }));
  return fwd.vector.every((v, i) => Math.abs(v + rev.vector[i]) < tol);
}

module.exports = { build, FEATURES, isAntisymmetric, layoffPenalty, ageAt };
