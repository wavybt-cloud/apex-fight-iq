// Stripe webhook — marks a user Pro in Supabase after payment, and downgrades on cancel.
// Requires env vars: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

// Stripe signature verification needs the RAW request body, so disable Vercel's parser.
module.exports.config = { api: { bodyParser: false } };

function readRawBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => (data += c));
    req.on('end', () => resolve(Buffer.from(data)));
    req.on('error', reject);
  });
}

async function sbWrite(path, method, body) {
  return fetch(process.env.SUPABASE_URL + '/rest/v1/' + path, {
    method,
    headers: {
      'apikey': process.env.SUPABASE_SERVICE_ROLE_KEY,
      'Authorization': 'Bearer ' + process.env.SUPABASE_SERVICE_ROLE_KEY,
      'Content-Type': 'application/json',
      'Prefer': 'resolution=merge-duplicates',
    },
    body: JSON.stringify(body),
  });
}

module.exports = async (req, res) => {
  const sig = req.headers['stripe-signature'];
  let event;
  try {
    const raw = await readRawBody(req);
    event = stripe.webhooks.constructEvent(raw, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    return res.status(400).send('Webhook Error: ' + err.message);
  }

  try {
    if (event.type === 'checkout.session.completed') {
      const s = event.data.object;
      const email = (s.customer_email || (s.metadata && s.metadata.email) || '').toLowerCase().trim();
      if (email) {
        // on_conflict=email: update the existing subscriber row instead of duplicating
        await sbWrite('subscribers?on_conflict=email', 'POST', {
          email,
          stripe_customer_id: s.customer,
          stripe_subscription_id: s.subscription,
          tier: 'pro',
          active: true,
        });
      }
    }
    if (event.type === 'customer.subscription.deleted') {
      const sub = event.data.object;
      await sbWrite('subscribers?stripe_subscription_id=eq.' + sub.id, 'PATCH', {
        tier: 'free', active: false,
      });
    }
  } catch (e) {
    console.error('Supabase update failed', e);
  }

  res.json({ received: true });
};
