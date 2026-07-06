// POST { email } -> { token }  if the email is an active Pro subscriber.
// Token is an HMAC-signed, expiring string verified by /api/picks. No external deps.
const crypto = require('crypto');

function sign(payload) {
  const body = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const sig  = crypto.createHmac('sha256', process.env.SESSION_SECRET).update(body).digest('base64url');
  return body + '.' + sig;
}

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { email } = req.body || {};
    if (!email || !email.includes('@')) return res.status(400).json({ error: 'Valid email required' });

    const r = await fetch(
      process.env.SUPABASE_URL + '/rest/v1/subscribers?select=tier,active&email=eq.' + encodeURIComponent(email.toLowerCase()),
      { headers: {
          'apikey': process.env.SUPABASE_SERVICE_ROLE_KEY,
          'Authorization': 'Bearer ' + process.env.SUPABASE_SERVICE_ROLE_KEY,
      }}
    );
    const rows = await r.json();
    const pro = Array.isArray(rows) && rows.some(x => x.tier === 'pro' && x.active === true);
    if (!pro) return res.status(403).json({ error: 'No active Pro subscription for this email.' });

    const token = sign({ e: email.toLowerCase(), exp: Date.now() + 1000 * 60 * 60 * 24 * 30 }); // 30 days
    res.status(200).json({ token });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
};
