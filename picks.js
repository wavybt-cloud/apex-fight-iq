// POST { token, event } -> { picks: [...pro picks...] }  only if the token is valid.
// Pro picks are read with the service-role key (bypasses RLS) and never sit in the public table response.
const crypto = require('crypto');

function verify(token) {
  try {
    const [body, sig] = String(token).split('.');
    if (!body || !sig) return null;
    const expected = crypto.createHmac('sha256', process.env.SESSION_SECRET).update(body).digest('base64url');
    const a = Buffer.from(sig), b = Buffer.from(expected);
    if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
    const payload = JSON.parse(Buffer.from(body, 'base64url').toString());
    if (!payload.exp || payload.exp < Date.now()) return null;
    return payload;
  } catch (e) { return null; }
}

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { token, event } = req.body || {};
  if (!verify(token)) return res.status(403).json({ error: 'Invalid or expired session' });

  try {
    const url = process.env.SUPABASE_URL + '/rest/v1/picks?select=*'
      + '&event_name=eq.' + encodeURIComponent(event || '')
      + '&is_pro=eq.true&result=eq.pending&order=win_pct.desc';
    const r = await fetch(url, { headers: {
      'apikey': process.env.SUPABASE_SERVICE_ROLE_KEY,
      'Authorization': 'Bearer ' + process.env.SUPABASE_SERVICE_ROLE_KEY,
    }});
    const picks = await r.json();
    res.status(200).json({ picks: Array.isArray(picks) ? picks : [] });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
};
