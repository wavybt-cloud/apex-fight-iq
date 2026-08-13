// Serves index.html / analyzer.html from GitHub (main by default; pin a commit
// via the SITE_REF env var on Vercel to freeze the deploy).
// - index.html: patches EVENT_NAME / EVENT_DATE from Supabase (next pending event)
// - analyzer.html: rebuilds the hardcoded NEXT_CARD from live picks + fighter tables
const REF = process.env.SITE_REF || 'main';
const RAW = 'https://raw.githubusercontent.com/wavybt-cloud/apex-fight-iq/' + REF + '/';
const SB_URL = 'https://whhbvglvtkqizfllxgtf.supabase.co';
const SB_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndoaGJ2Z2x2dGtxaXpmbGx4Z3RmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODI1ODM3ODEsImV4cCI6MjA5ODE1OTc4MX0.u1EM3q_FvLwPN_rulFNrTzfKAodUA74P4js748PV8i8';

function sb(path) {
  return fetch(SB_URL + '/rest/v1/' + path, {
    headers: { apikey: SB_KEY, Authorization: 'Bearer ' + SB_KEY },
  }).then(r => r.json());
}
const num = (v, d) => (v == null || isNaN(Number(v))) ? d : Number(v);
function ageFromDob(dob) {
  if (!dob) return 30;
  const ms = Date.now() - new Date(dob + 'T00:00:00Z').getTime();
  return Math.max(18, Math.min(50, Math.floor(ms / 3.15576e10)));
}
function oppOdds(a) {
  // derive opponent's american odds from the picked side's, ~4.5% overround
  const ia = a < 0 ? (-a) / (-a + 100) : 100 / (a + 100);
  let ib = Math.min(0.97, Math.max(0.03, 1.045 - ia));
  const am = ib >= 0.5 ? Math.round(-100 * ib / (1 - ib)) : Math.round(100 * (1 - ib) / ib);
  return Math.max(-2000, Math.min(2000, am));
}
function statBlock(name, prof, tott, rat) {
  const s = k => prof ? num(prof[k], null) : null;
  const st = s('s_striking'), gr = s('s_grappling'), ca = s('s_cardio'),
        du = s('s_durability'), iq = s('s_iq'), ac = s('s_activity');
  const fights = prof ? num(prof.fights, 0) : 0;
  return {
    name: name,
    strAcc: Math.round(38 + num(st, 40) * 0.14),
    strDef: Math.round(44 + num(iq, 40) * 0.14),
    slpm: +(2.8 + num(ac, 60) * 0.025).toFixed(1),
    tdAcc: Math.round(25 + num(gr, 35) * 0.3),
    tdDef: Math.round(40 + num(du, 60) * 0.3),
    tdAvg: +(num(gr, 35) / 30).toFixed(1),
    subAvg: +(num(gr, 35) / 60).toFixed(1),
    avgTime: Math.round(7 + num(ca, 50) * 0.08),
    reach: Math.round(num(tott && tott.reach_in, 72)),
    age: ageFromDob(tott && tott.dob),
    streak: Math.max(0, num(rat && rat.streak, 0)),
    research: Math.min(9, 3 + Math.floor(fights / 3)),
  };
}
function billingOf(b) {
  b = (b || '').toLowerCase();
  if (b.indexOf('main event') >= 0) return 'Main event';
  if (b.indexOf('co-main') >= 0) return 'Co-main event';
  if (b.indexOf('main') >= 0) return 'Main card';
  if (b.indexOf('early') >= 0) return 'Early prelim';
  return 'Prelim';
}
async function buildNextCard() {
  const ev = await sb('picks?select=event_name&result=eq.pending&event_name=not.ilike.ARCHIVED*&event_name=not.ilike.DWCS*&order=event_date.asc&limit=1');
  if (!Array.isArray(ev) || !ev[0]) return null;
  const evName = ev[0].event_name;
  const picks = await sb('picks?select=fighter,opponent,weight_class,billing,win_pct,book_odds,card_order&event_name=eq.' +
    encodeURIComponent(evName) + '&result=eq.pending&order=card_order.asc.nullslast,win_pct.desc');
  if (!Array.isArray(picks) || !picks.length) return null;
  const names = [];
  picks.forEach(p => { names.push(p.fighter, p.opponent); });
  const inList = 'in.(' + names.map(n => '"' + String(n).replace(/"/g, '') + '"').join(',') + ')';
  const [profiles, totts, ratings] = await Promise.all([
    sb('fighter_profiles?select=fighter,fights,s_striking,s_grappling,s_cardio,s_durability,s_iq,s_activity&fighter=' + encodeURIComponent(inList)),
    sb('fighter_tott?select=fighter,reach_in,dob&fighter=' + encodeURIComponent(inList)),
    sb('fighter_ratings?select=fighter,streak&fighter=' + encodeURIComponent(inList)),
  ]);
  const by = rows => { const m = {}; (Array.isArray(rows) ? rows : []).forEach(r => m[r.fighter] = r); return m; };
  const P = by(profiles), T = by(totts), R = by(ratings);
  const ord = { 'Main event': 0, 'Co-main event': 1, 'Main card': 2, 'Prelim': 3, 'Early prelim': 4 };
  const card = picks.map(p => {
    const billing = billingOf(p.billing);
    const aOdds = p.book_odds != null ? Number(p.book_odds) : null;
    return {
      billing: billing,
      weightClass: p.weight_class || '',
      odds: aOdds != null ? { a: aOdds, b: oppOdds(aOdds) } : null,
      pub: { a: Math.round(num(p.win_pct, 50)), b: 100 - Math.round(num(p.win_pct, 50)) },
      a: statBlock(p.fighter, P[p.fighter], T[p.fighter], R[p.fighter]),
      b: statBlock(p.opponent, P[p.opponent], T[p.opponent], R[p.opponent]),
      rounds: billing === 'Main event' ? 5 : 3,
      _ord: ord[billing] != null ? ord[billing] : 5,
    };
  });
  card.sort((x, y) => x._ord - y._ord);
  card.forEach(c => { delete c._ord; });
  return card;
}

module.exports = async (req, res) => {
  try {
    const isAnalyzer = (req.query && req.query.f) === 'analyzer';
    const file = isAnalyzer ? 'analyzer.html' : 'index.html';
    let html = await fetch(RAW + file).then(r => {
      if (!r.ok) throw new Error('raw fetch ' + r.status);
      return r.text();
    });

    if (isAnalyzer) {
      try {
        const card = await buildNextCard();
        if (card && card.length) {
          html = html.replace(/const NEXT_CARD = \[[\s\S]*?\n\];/,
            'const NEXT_CARD = ' + JSON.stringify(card, null, 1) + ';');
        }
      } catch (e) { /* serve original analyzer on failure */ }
    } else {
      try {
        const rows = await sb('picks?select=event_name,event_date&result=eq.pending&event_name=not.ilike.ARCHIVED*&event_name=not.ilike.DWCS*&order=event_date.asc&limit=1');
        if (Array.isArray(rows) && rows[0] && rows[0].event_name) {
          const name = String(rows[0].event_name).replace(/[\\']/g, '');
          const date = String(rows[0].event_date || '').slice(0, 10);
          html = html.replace(/const EVENT_NAME\s*=\s*'[^']*'/,
            "const EVENT_NAME    = '" + name + "'");
          if (/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            html = html.replace(/const EVENT_DATE\s*=\s*'[^']*'/,
              "const EVENT_DATE    = '" + date + "T13:00:00-04:00'");
          }
        }
        // Track record: exclude archived drafts and DWCS scouting picks from the public record
        html = html.replace(/result=neq\.pending/g,
          'result=neq.pending&event_name=not.ilike.ARCHIVED*&event_name=not.ilike.DWCS*');
      } catch (e) { /* fall through with original constants */ }
    }

    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.setHeader('Cache-Control', 's-maxage=120, stale-while-revalidate=600');
    res.status(200).send(html);
  } catch (err) {
    res.status(502).send('Upstream error: ' + err.message);
  }
};
