'use strict';
// Canonical records and validators.
//
// The most important function here is `canonicalSides`. See the note on it:
// this repository's own historical data carries a side-ordering artifact that
// leaks the label, and every model trained on it inherits the leak.

const crypto = require('crypto');

/** Every fact carries where it came from and when it was observed. */
function fact(value, source, observedAt, validAsOf) {
  if (source == null) throw new Error('fact: source is required');
  if (observedAt == null) throw new Error('fact: observedAt is required');
  return Object.freeze({ value, source, observedAt, validAsOf: validAsOf != null ? validAsOf : observedAt });
}

const REQUIRED_FIGHTER_FIELDS = ['name'];
const SIM_FIGHTER_FIELDS = [
  'koRate', 'subRate', 'durability', 'subDefense', 'cardio',
  'output', 'accuracy', 'grappling', 'takedownDefense', 'control',
];

function validateFighter(f, opts) {
  const o = Object.assign({ requireSimParams: false }, opts || {});
  const errors = [];
  if (!f || typeof f !== 'object') return { valid: false, errors: ['fighter is not an object'] };
  for (const k of REQUIRED_FIGHTER_FIELDS) {
    if (f[k] == null || f[k] === '') errors.push(`missing ${k}`);
  }
  if (o.requireSimParams) {
    for (const k of SIM_FIGHTER_FIELDS) {
      if (f[k] == null) errors.push(`missing simulator parameter ${k}`);
      else if (!Number.isFinite(f[k]) || f[k] < 0) errors.push(`${k} must be a non-negative number`);
    }
  }
  if (f.bouts != null && (!Number.isInteger(f.bouts) || f.bouts < 0)) errors.push('bouts must be a non-negative integer');
  if (f.dob != null && !Number.isFinite(f.dob)) errors.push('dob must be an epoch timestamp');
  return { valid: errors.length === 0, errors };
}

function validateQuote(q) {
  const errors = [];
  if (!q || typeof q !== 'object') return { valid: false, errors: ['quote is not an object'] };
  if (!q.market) errors.push('missing market');
  if (!q.book) errors.push('missing book');
  if (q.observedAt == null) errors.push('missing observedAt — a price without a timestamp cannot be verified or used for CLV');
  if (!Array.isArray(q.decimals) || q.decimals.length < 2) errors.push('decimals must list at least two outcomes');
  else {
    for (const d of q.decimals) {
      if (!Number.isFinite(d) || d <= 1) { errors.push(`invalid decimal price ${d}`); break; }
    }
  }
  return { valid: errors.length === 0, errors };
}

function validateBout(b, opts) {
  const errors = [];
  if (!b || typeof b !== 'object') return { valid: false, errors: ['bout is not an object'] };
  if (!b.id) errors.push('missing id');
  for (const side of ['a', 'b']) {
    const r = validateFighter(b[side], opts);
    if (!r.valid) errors.push(...r.errors.map((e) => `${side}: ${e}`));
  }
  if (b.rounds != null && ![3, 5].includes(b.rounds)) errors.push('rounds must be 3 or 5');
  for (const q of b.quotes || []) {
    const r = validateQuote(q);
    if (!r.valid) errors.push(...r.errors.map((e) => `quote(${q && q.book}): ${e}`));
  }
  return { valid: errors.length === 0, errors };
}

/**
 * Assign fighters to the A and B slots deterministically and independently of
 * the outcome.
 *
 * WHY THIS EXISTS — a real defect found in this project's own data:
 *
 *   In `ufc_fights` (2015+), `fighter_a` wins 58.2% of bouts, and in
 *   `fight_predictions` the A-side wins 64.8%. Scraped fight records
 *   conventionally list the winner first, so the A slot is correlated with the
 *   label. Any model trained on "A's features minus B's features" learns
 *   "A tends to win" and reports inflated accuracy that evaporates live, where
 *   there is no winner to sort by.
 *
 * Ordering by a hash of the two names is stable, reproducible, and provably
 * independent of who won, which removes the leak at the source rather than
 * hoping downstream code compensates.
 */
function canonicalSides(nameA, nameB) {
  const key = (n) => crypto.createHash('sha1').update(String(n).trim().toLowerCase()).digest('hex');
  const ka = key(nameA), kb = key(nameB);
  // Tie-break on the raw name so identical hashes (never expected) stay deterministic.
  const aFirst = ka < kb || (ka === kb && String(nameA) <= String(nameB));
  return aFirst
    ? { a: nameA, b: nameB, swapped: false }
    : { a: nameB, b: nameA, swapped: true };
}

/** Canonical result record. `winnerSide` is 'a' | 'b' | null (draw/NC). */
function makeResult(spec) {
  const method = normaliseMethod(spec.method);
  return Object.freeze({
    boutId: spec.boutId,
    winnerSide: spec.winnerSide != null ? spec.winnerSide : null,
    method,
    round: spec.round != null ? spec.round : null,
    timeSec: spec.timeSec != null ? spec.timeSec : null,
    observedAt: spec.observedAt,
    source: spec.source,
  });
}

/** Map the many free-text method spellings onto the engine's three outcomes. */
function normaliseMethod(raw) {
  if (raw == null) return null;
  const s = String(raw).toLowerCase();
  if (s.includes('dq') || s.includes('disqualification')) return 'dq';
  if (s.includes('no contest') || s === 'nc') return 'nc';
  if (s.includes('draw')) return 'draw';
  if (s.includes('sub')) return 'sub';
  if (s.includes('ko') || s.includes('tko') || s.includes('stoppage') || s.includes('retirement')) return 'ko';
  if (s.includes('dec')) return 'dec';
  return null;
}

/** Bouts whose result is not a clean win/loss must not train or grade a model. */
function isGradeable(result) {
  return result != null
    && result.winnerSide != null
    && ['ko', 'sub', 'dec'].includes(result.method);
}

module.exports = {
  fact, validateFighter, validateQuote, validateBout,
  canonicalSides, makeResult, normaliseMethod, isGradeable,
  SIM_FIGHTER_FIELDS, REQUIRED_FIGHTER_FIELDS,
};
