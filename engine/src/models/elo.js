'use strict';
// Elo-style ratings.
//
// The rating book is stateful, but the state is strictly a fold over past
// fights: `predict` reads only what `update` has already absorbed, and the
// backtest feeds fights in chronological order. That is what makes a rating
// usable as an as-of feature without leaking the future into it.

const DEFAULTS = {
  initial: 1500,
  scale: 400,
  kBase: 32,
  kNovice: 48,       // larger updates while a rating is still uncertain
  noviceFights: 5,
  finishBonus: 1.25, // a finish is stronger evidence than a split decision
  decisionFactor: 0.9,
  regressPerYear: 0.15, // idle ratings drift back toward the mean
};

function createBook(opts) {
  const cfg = Object.assign({}, DEFAULTS, opts || {});
  const ratings = new Map();   // name -> {rating, fights, wins, losses, lastFight, streak, finishes}

  function get(name) {
    let r = ratings.get(name);
    if (!r) {
      r = {
        rating: cfg.initial, fights: 0, wins: 0, losses: 0,
        lastFight: null, streak: 0, finishes: 0, finishedAgainst: 0,
      };
      ratings.set(name, r);
    }
    return r;
  }

  /** Rating regressed toward the mean for time spent inactive. */
  function ratingAsOf(name, atMs) {
    const r = get(name);
    if (r.lastFight == null || atMs == null) return r.rating;
    const years = Math.max(0, (atMs - r.lastFight) / (365.25 * 86400000));
    if (years <= 1) return r.rating;
    const pull = Math.min(0.6, cfg.regressPerYear * (years - 1));
    return r.rating + (cfg.initial - r.rating) * pull;
  }

  function expected(ra, rb) {
    return 1 / (1 + Math.pow(10, (rb - ra) / cfg.scale));
  }

  /** Probability that A beats B, using ratings as they stand right now. */
  function predict(nameA, nameB, atMs) {
    return expected(ratingAsOf(nameA, atMs), ratingAsOf(nameB, atMs));
  }

  function kFor(r) {
    return r.fights < cfg.noviceFights ? cfg.kNovice : cfg.kBase;
  }

  /**
   * Absorb one settled fight.
   * @param {object} f {a, b, winnerSide:'a'|'b', method:'ko'|'sub'|'dec', date}
   */
  function update(f) {
    if (f.winnerSide !== 'a' && f.winnerSide !== 'b') return;
    const ra = get(f.a), rb = get(f.b);
    const curA = ratingAsOf(f.a, f.date), curB = ratingAsOf(f.b, f.date);
    const eA = expected(curA, curB);
    const sA = f.winnerSide === 'a' ? 1 : 0;

    const decisive = f.method === 'ko' || f.method === 'sub';
    const mult = decisive ? cfg.finishBonus : cfg.decisionFactor;
    const k = ((kFor(ra) + kFor(rb)) / 2) * mult;
    const delta = k * (sA - eA);

    ra.rating = curA + delta;
    rb.rating = curB - delta;

    for (const [r, won] of [[ra, sA === 1], [rb, sA === 0]]) {
      r.fights += 1;
      r.lastFight = f.date;
      if (won) { r.wins += 1; r.streak = r.streak >= 0 ? r.streak + 1 : 1; }
      else { r.losses += 1; r.streak = r.streak <= 0 ? r.streak - 1 : -1; }
    }
    if (decisive) {
      (sA === 1 ? ra : rb).finishes += 1;
      (sA === 1 ? rb : ra).finishedAgainst += 1;
    }
  }

  function snapshot(name, atMs) {
    const r = get(name);
    return {
      rating: ratingAsOf(name, atMs),
      fights: r.fights, wins: r.wins, losses: r.losses,
      streak: r.streak, finishes: r.finishes, finishedAgainst: r.finishedAgainst,
      lastFight: r.lastFight,
      layoffDays: r.lastFight != null && atMs != null ? (atMs - r.lastFight) / 86400000 : null,
    };
  }

  return { predict, update, snapshot, ratingAsOf, get size() { return ratings.size; }, config: cfg };
}

module.exports = { createBook, DEFAULTS };
