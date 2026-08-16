'use strict';
// Data quality gate.
//
// This runs BEFORE any modelling and can veto a bet outright. A model fed stale
// odds and an unconfirmed lineup produces a confident number that means nothing;
// the failure mode is silent, so the check has to be structural.

const { clamp } = require('../core/prob');

/**
 * Defect catalogue. `blocks: true` means the defect alone forces NO BET,
 * regardless of the computed edge.
 */
const DEFECTS = {
  NO_ODDS:            { penalty: 100, blocks: true,  label: 'No verified price available' },
  STALE_ODDS:         { penalty: 35,  blocks: false, label: 'Price older than freshness limit' },
  VERY_STALE_ODDS:    { penalty: 100, blocks: true,  label: 'Price too old to act on' },
  SINGLE_BOOK:        { penalty: 12,  blocks: false, label: 'Only one book quoting — weak price evidence' },
  WIDE_BOOK_SPREAD:   { penalty: 15,  blocks: false, label: 'Books disagree materially on the price' },
  HIGH_OVERROUND:     { penalty: 10,  blocks: false, label: 'Unusually high vig — illiquid or protected market' },
  FIGHTER_STATUS_UNKNOWN: { penalty: 100, blocks: true, label: 'Participation of a fighter is unconfirmed' },
  UNRESOLVED_INJURY:  { penalty: 100, blocks: true,  label: 'Unresolved injury report on a participant' },
  SHORT_NOTICE:       { penalty: 22,  blocks: false, label: 'Short-notice replacement — priors are unreliable' },
  WEIGHT_MISS:        { penalty: 18,  blocks: false, label: 'Weight miss changes the matchup' },
  MISSING_STATS:      { penalty: 25,  blocks: false, label: 'Core statistics missing for a participant' },
  THIN_SAMPLE:        { penalty: 20,  blocks: false, label: 'Too few professional bouts to estimate from' },
  STALE_STATS:        { penalty: 15,  blocks: false, label: 'Statistics not updated since a recent bout' },
  NO_ROUND_DETAIL:    { penalty: 10,  blocks: false, label: 'No per-round detail — pace estimates are coarse' },
  SOURCE_CONFLICT:    { penalty: 30,  blocks: false, label: 'Sources disagree on a material fact' },
  LATE_OPPONENT_CHANGE: { penalty: 100, blocks: true, label: 'Opponent changed — model inputs invalid' },
  UNCALIBRATED_MODEL: { penalty: 100, blocks: true,  label: 'Model has no out-of-sample calibration record' },
};

const DEFAULT_LIMITS = {
  oddsStaleSeconds: 900,        // 15 min
  oddsVeryStaleSeconds: 7200,   // 2 h
  statsStaleDays: 45,
  minBouts: 6,
  minBooks: 2,
  maxBookSpread: 0.04,          // 4 percentage points of devigged probability
  maxOverround: 0.08,
  passScore: 70,
  degradeScore: 50,
};

/**
 * Assess one bout.
 *
 * @param {object} bout canonical bout record
 * @param {number} nowMs decision timestamp
 * @param {object} [limits]
 * @returns {{score:number, gate:'PASS'|'DEGRADE'|'BLOCK', defects:Array, blocking:Array}}
 */
function assess(bout, nowMs, limits) {
  const L = Object.assign({}, DEFAULT_LIMITS, limits || {});
  const found = [];
  const add = (key, detail) => found.push({ key, detail, ...DEFECTS[key] });

  // --- Price ---
  const quotes = (bout.quotes || []).filter(Boolean);
  if (!quotes.length) {
    add('NO_ODDS');
  } else {
    const newest = Math.max(...quotes.map((q) => q.observedAt || 0));
    const age = (nowMs - newest) / 1000;
    if (age > L.oddsVeryStaleSeconds) add('VERY_STALE_ODDS', `${Math.round(age / 60)} min old`);
    else if (age > L.oddsStaleSeconds) add('STALE_ODDS', `${Math.round(age / 60)} min old`);
    if (quotes.length < L.minBooks) add('SINGLE_BOOK', `${quotes.length} book(s)`);
    if (bout.bookSpread != null && bout.bookSpread > L.maxBookSpread) {
      add('WIDE_BOOK_SPREAD', `${(bout.bookSpread * 100).toFixed(1)} pts`);
    }
    if (bout.overround != null && bout.overround > L.maxOverround) {
      add('HIGH_OVERROUND', `${(bout.overround * 100).toFixed(1)}%`);
    }
  }

  // --- Participants ---
  for (const side of ['a', 'b']) {
    const f = bout[side];
    if (!f) { add('FIGHTER_STATUS_UNKNOWN', `side ${side} missing`); continue; }
    if (f.confirmed === false || f.confirmed == null) {
      add('FIGHTER_STATUS_UNKNOWN', f.name || `side ${side}`);
    }
    if (f.injuryUnresolved) add('UNRESOLVED_INJURY', f.name);
    if (f.shortNotice) add('SHORT_NOTICE', f.name);
    if (f.weightMiss) add('WEIGHT_MISS', f.name);
    if (f.bouts != null && f.bouts < L.minBouts) add('THIN_SAMPLE', `${f.name}: ${f.bouts} bouts`);
    const required = ['output', 'accuracy', 'takedownDefense', 'grappling'];
    const missing = required.filter((k) => f[k] == null);
    if (missing.length) add('MISSING_STATS', `${f.name}: ${missing.join(', ')}`);
    if (f.statsUpdatedAt != null) {
      const days = (nowMs - f.statsUpdatedAt) / 86400000;
      if (days > L.statsStaleDays) add('STALE_STATS', `${f.name}: ${Math.round(days)}d`);
    }
    if (f.hasRoundDetail === false) add('NO_ROUND_DETAIL', f.name);
  }

  if (bout.opponentChangedAt != null) add('LATE_OPPONENT_CHANGE');
  if (bout.sourceConflicts && bout.sourceConflicts.length) {
    add('SOURCE_CONFLICT', bout.sourceConflicts.join('; '));
  }
  if (bout.modelCalibrated === false || bout.modelCalibrated == null) {
    add('UNCALIBRATED_MODEL', 'no walk-forward calibration record for the deployed version');
  }

  // Deduplicate by key, keeping the first detail seen.
  const seen = new Map();
  for (const d of found) if (!seen.has(d.key)) seen.set(d.key, d);
  const defects = [...seen.values()];

  const blocking = defects.filter((d) => d.blocks);
  const penalty = defects.reduce((s, d) => s + d.penalty, 0);
  const score = clamp(100 - penalty, 0, 100);

  let gate;
  if (blocking.length) gate = 'BLOCK';
  else if (score >= L.passScore) gate = 'PASS';
  else if (score >= L.degradeScore) gate = 'DEGRADE';
  else gate = 'BLOCK';

  return { score, gate, defects, blocking, penalty };
}

module.exports = { assess, DEFECTS, DEFAULT_LIMITS };
