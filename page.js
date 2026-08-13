// Serves index.html (the whole app — odds board + Fight Lab) from GitHub
// (main by default; pin a commit via the SITE_REF env var on Vercel).
// - index.html: patches EVENT_NAME / EVENT_DATE from Supabase (next pending event)
// - /analyzer: legacy route — the Fight Lab now lives inside the app at /#lab
const REF = process.env.SITE_REF || 'main';
const RAW = 'https://raw.githubusercontent.com/wavybt-cloud/apex-fight-iq/' + REF + '/';
const SB_URL = 'https://whhbvglvtkqizfllxgtf.supabase.co';
const SB_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndoaGJ2Z2x2dGtxaXpmbGx4Z3RmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODI1ODM3ODEsImV4cCI6MjA5ODE1OTc4MX0.u1EM3q_FvLwPN_rulFNrTzfKAodUA74P4js748PV8i8';

function sb(path) {
  return fetch(SB_URL + '/rest/v1/' + path, {
    headers: { apikey: SB_KEY, Authorization: 'Bearer ' + SB_KEY },
  }).then(r => r.json());
}

module.exports = async (req, res) => {
  try {
    if ((req.query && req.query.f) === 'analyzer') {
      res.statusCode = 302;
      res.setHeader('Location', '/#lab');
      res.end();
      return;
    }
    let html = await fetch(RAW + 'index.html').then(r => {
      if (!r.ok) throw new Error('raw fetch ' + r.status);
      return r.text();
    });
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

    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.setHeader('Cache-Control', 's-maxage=120, stale-while-revalidate=600');
    res.status(200).send(html);
  } catch (err) {
    res.status(502).send('Upstream error: ' + err.message);
  }
};
