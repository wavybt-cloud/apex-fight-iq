// Cron endpoint: capture a snapshot of every open MMA market and store it.
//
// Odds history cannot be backfilled. This endpoint running on a schedule is the
// only way the engine will ever be able to measure closing line value, which is
// the gate on live deployment. Everything else in the pipeline can wait; this
// cannot.
//
// Env:
//   ODDS_API_KEY                 the-odds-api.com key
//   SUPABASE_URL
//   SUPABASE_SERVICE_ROLE_KEY    needs insert on odds_snapshots (RLS is on)
//   CRON_SECRET                  optional; when set, required as a bearer token

const oddsapi = require('./engine/src/data/adapters/oddsapi');

module.exports = async (req, res) => {
  // Vercel Cron sends `Authorization: Bearer $CRON_SECRET`. When the secret is
  // configured, refuse anything else so the endpoint cannot be used to burn the
  // odds-API quota.
  const secret = process.env.CRON_SECRET;
  if (secret) {
    const auth = req.headers.authorization || '';
    if (auth !== `Bearer ${secret}`) return res.status(401).json({ error: 'Unauthorized' });
  }

  const apiKey = process.env.ODDS_API_KEY;
  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!apiKey || !supabaseUrl || !serviceKey) {
    return res.status(500).json({ error: 'Missing ODDS_API_KEY, SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY' });
  }

  const capturedAt = Date.now();

  try {
    const url = oddsapi.oddsUrl(apiKey, { regions: 'us,eu', markets: 'h2h' });
    const r = await fetch(url);
    if (!r.ok) {
      const body = await r.text();
      return res.status(502).json({ error: `odds api ${r.status}`, detail: body.slice(0, 300) });
    }
    const events = await r.json();
    const rows = oddsapi.normalise(events, capturedAt);

    if (!rows.length) {
      return res.status(200).json({ captured: 0, events: Array.isArray(events) ? events.length : 0, note: 'no open two-way markets' });
    }

    const payload = rows.map((x) => ({
      captured_at: new Date(x.capturedAt).toISOString(),
      source: x.source,
      event_key: x.eventKey,
      commence_time: x.commenceTime ? new Date(x.commenceTime).toISOString() : null,
      fighter_a: x.fighterA,
      fighter_b: x.fighterB,
      book: x.book,
      market: x.market,
      outcome_a: x.outcomeA,
      outcome_b: x.outcomeB,
      price_a: x.priceA,
      price_b: x.priceB,
      point: x.point,
      book_updated_at: x.bookUpdatedAt ? new Date(x.bookUpdatedAt).toISOString() : null,
    }));

    const ins = await fetch(`${supabaseUrl}/rest/v1/odds_snapshots`, {
      method: 'POST',
      headers: {
        apikey: serviceKey,
        Authorization: `Bearer ${serviceKey}`,
        'Content-Type': 'application/json',
        // Duplicates are expected when a cron retries; ignore them rather than
        // failing the whole capture.
        Prefer: 'resolution=ignore-duplicates,return=minimal',
      },
      body: JSON.stringify(payload),
    });
    if (!ins.ok) {
      const body = await ins.text();
      return res.status(502).json({ error: `supabase insert ${ins.status}`, detail: body.slice(0, 300) });
    }

    const remaining = r.headers.get('x-requests-remaining');
    return res.status(200).json({
      captured: payload.length,
      events: events.length,
      books: [...new Set(rows.map((x) => x.book))].length,
      apiRequestsRemaining: remaining != null ? Number(remaining) : null,
    });
  } catch (err) {
    console.error('odds-snapshot failed', err);
    return res.status(500).json({ error: 'Server error' });
  }
};
