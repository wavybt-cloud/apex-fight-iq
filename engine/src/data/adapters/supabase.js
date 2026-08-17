'use strict';
// Read-only adapter over the project's Supabase database.
//
// Reads only. Nothing in this module writes, and it deliberately has no code
// path that could: the engine's job is to form opinions about data, not to
// mutate the site's production tables.
//
// Tables consumed (row counts observed 2026-08-17):
//   ufc_fights      8,854  event, date, fighter_a/b, winner, method, round
//   pfl_fights      2,257  same shape, PFL
//   fighter_tott    4,489  height, reach, stance, dob, weight
//   fighter_ratings 2,748  Elo, fights, W/L, finishes, streak, last_fight
//   fighter_profiles 2,748 derived per-fighter scores
//   picks             136  the site's own published picks, with book/close odds
//
// NOT consumed, deliberately:
//   fight_predictions / model_train / model_params — these carry the A-side
//   ordering leak documented in data/audit.js (A wins 64.8% there). Fitting on
//   them reproduces the leak. Rebuild them from ufc_fights through
//   schema.canonicalSides() instead.

const schema = require('../schema');

const TABLES = {
  fights: 'ufc_fights',
  pflFights: 'pfl_fights',
  tott: 'fighter_tott',
  ratings: 'fighter_ratings',
  profiles: 'fighter_profiles',
  picks: 'picks',
};

/**
 * @param {object} config
 * @param {string} config.url           SUPABASE_URL
 * @param {string} config.key           service-role or anon key
 * @param {Function} [config.fetchImpl] injected for testing
 */
function createAdapter(config) {
  const url = config.url || process.env.SUPABASE_URL;
  const key = config.key || process.env.SUPABASE_SERVICE_ROLE_KEY;
  const doFetch = config.fetchImpl || globalThis.fetch;
  if (!url || !key) throw new Error('supabase adapter: url and key are required');
  if (typeof doFetch !== 'function') throw new Error('supabase adapter: no fetch implementation available');

  async function select(table, query) {
    const endpoint = `${url}/rest/v1/${table}?${query}`;
    const res = await doFetch(endpoint, {
      method: 'GET',
      headers: { apikey: key, Authorization: `Bearer ${key}`, Accept: 'application/json' },
    });
    if (!res.ok) throw new Error(`supabase adapter: ${table} returned ${res.status}`);
    const rows = await res.json();
    if (!Array.isArray(rows)) throw new Error(`supabase adapter: ${table} did not return an array`);
    return rows;
  }

  /**
   * Historical bouts, canonicalised.
   *
   * Every row is passed through schema.canonicalSides so that the A slot is a
   * hash of the names rather than "whoever the scraper listed first". Without
   * this the returned dataset carries the label leak.
   */
  async function fetchHistory(opts) {
    const o = Object.assign({ from: '2015-01-01', to: null, limit: 20000, promotion: 'ufc' }, opts || {});
    const table = o.promotion === 'pfl' ? TABLES.pflFights : TABLES.fights;
    const filters = [
      'select=id,event,event_date,fighter_a,fighter_b,winner,method,round,weight_class',
      `event_date=gte.${o.from}`,
      'order=event_date.asc',
      `limit=${o.limit}`,
    ];
    if (o.to) filters.push(`event_date=lte.${o.to}`);
    const rows = await select(table, filters.join('&'));
    return rows.map(toHistoricalBout).filter(Boolean);
  }

  async function fetchFighterTape(names) {
    if (!names || !names.length) return new Map();
    const list = names.map((n) => `"${String(n).replace(/"/g, '')}"`).join(',');
    const rows = await select(TABLES.tott,
      `select=fighter,height_in,reach_in,stance,dob,weight_lb&fighter=in.(${encodeURIComponent(list)})`);
    return new Map(rows.map((r) => [r.fighter, {
      name: r.fighter,
      heightIn: num(r.height_in),
      reachIn: num(r.reach_in),
      stance: r.stance || null,
      dob: r.dob ? Date.parse(r.dob) : null,
      weightLb: num(r.weight_lb),
    }]));
  }

  async function fetchRatings(names) {
    if (!names || !names.length) return new Map();
    const list = names.map((n) => `"${String(n).replace(/"/g, '')}"`).join(',');
    const rows = await select(TABLES.ratings,
      `select=fighter,rating,peak_rating,fights,wins,losses,finishes,last_fight,streak,updated_at&fighter=in.(${encodeURIComponent(list)})`);
    return new Map(rows.map((r) => [r.fighter, {
      name: r.fighter,
      rating: num(r.rating),
      peakRating: num(r.peak_rating),
      bouts: int(r.fights),
      wins: int(r.wins),
      losses: int(r.losses),
      finishes: int(r.finishes),
      streak: int(r.streak),
      lastFight: r.last_fight ? Date.parse(r.last_fight) : null,
      statsUpdatedAt: r.updated_at ? Date.parse(r.updated_at) : null,
    }]));
  }

  /**
   * The site's own published picks. Useful as a graded ledger and as the only
   * source of closing prices currently present — though there are very few, so
   * `audit.oddsCoverage` will correctly report that CLV is not yet measurable.
   */
  async function fetchPickLedger(opts) {
    const o = Object.assign({ limit: 1000 }, opts || {});
    const rows = await select(TABLES.picks,
      `select=id,event_name,event_date,fighter,opponent,win_pct,book_odds,close_odds,result,created_at&order=created_at.asc&limit=${o.limit}`);
    return rows.map((r) => ({
      id: r.id,
      event: r.event_name,
      eventDate: r.event_date ? Date.parse(r.event_date) : null,
      selection: r.fighter,
      opponent: r.opponent,
      modelProbability: r.win_pct != null ? Number(r.win_pct) / 100 : null,
      bookAmerican: int(r.book_odds),
      closeAmerican: int(r.close_odds),
      result: r.result,
      observedAt: r.created_at ? Date.parse(r.created_at) : null,
    }));
  }

  return { fetchHistory, fetchFighterTape, fetchRatings, fetchPickLedger, select, TABLES };
}

/** Convert one raw fight row into a canonical historical bout. */
function toHistoricalBout(row) {
  if (!row.fighter_a || !row.fighter_b) return null;
  const sides = schema.canonicalSides(row.fighter_a, row.fighter_b);
  const method = schema.normaliseMethod(row.method);
  let winnerSide = null;
  if (row.winner) {
    if (row.winner === sides.a) winnerSide = 'a';
    else if (row.winner === sides.b) winnerSide = 'b';
    // A winner naming neither participant is a data error; leave null so the
    // bout is excluded from grading rather than silently mis-assigned.
  }
  const date = row.event_date ? Date.parse(row.event_date) : null;
  return {
    id: `${row.id}`,
    event: row.event,
    eventDate: date,
    date,
    weightClass: row.weight_class || null,
    a: { name: sides.a },
    b: { name: sides.b },
    sidesSwapped: sides.swapped,
    // Scheduled rounds are not stored; inferred only where the fight proves it.
    rounds: row.round != null && row.round > 3 ? 5 : null,
    result: schema.makeResult({
      boutId: `${row.id}`,
      winnerSide,
      method,
      round: row.round != null ? Number(row.round) : null,
      observedAt: date,
      source: 'supabase:ufc_fights',
    }),
  };
}

function num(x) { return x == null ? null : Number(x); }
function int(x) { return x == null ? null : parseInt(x, 10); }

module.exports = { createAdapter, toHistoricalBout, TABLES };
