# Apex Fight IQ — Complete Site Code

Live at: https://apex-fight-iq.vercel.app and https://apexfightiq-v2.vercel.app
(both Vercel projects run the SAME deployed configuration)

## How the live site works (important)

The deployments do NOT serve the static files directly. Every page request runs
through `page.js` — a small server layer that:

1. Fetches `index.html` / `analyzer.html` from this GitHub repo (the `main`
   branch by default; set the `SITE_REF` env var on Vercel to pin a commit).
2. Patches `index.html` on the fly:
   - `EVENT_NAME` / `EVENT_DATE` are set from Supabase (next pending event in
     the `picks` table, skipping names starting with `ARCHIVED` or `DWCS`)
   - the track-record query gets `&event_name=not.ilike.ARCHIVED*&event_name=not.ilike.DWCS*`
     appended so archived drafts and DWCS scouting picks never appear in the
     public record
3. Rebuilds the analyzer's hardcoded `NEXT_CARD` array from live Supabase data
   (picks + fighter_profiles + fighter_tott + fighter_ratings).
4. Serves with a 2-minute CDN cache.

So: to change the site's LOOK, edit `index.html` / `analyzer.html` and push to
`main`. To change EVENTS/PICKS/RECORD, you never touch code — it all flows from
the Supabase `picks` table.

## Files

- `index.html`   — landing page: odds board, track record, pick'em game, paywall
- `analyzer.html` — **Fight Lab v5 (Quantum Engine)**: 17-factor cross-matchup
  model + 10,000-run Monte Carlo fight simulator. Live card analysis, best-bets
  strip, Matchup Lab (search the full Supabase roster or build custom fighters),
  method/round/distance distributions, prop pricing with fair odds, vig-free
  market edge + EV + fractional-Kelly staking, parlay desk, pick tracking with
  Platt self-calibration, and PNG share cards. The `NEXT_CARD` constant is
  rebuilt server-side by `page.js` on every request and also live-syncs
  client-side straight from Supabase.
- `page.js`      — the dynamic server layer described above (all HTML routes)
- `checkout.js`  — Vercel function: Stripe Checkout session
  (env: STRIPE_SECRET_KEY, STRIPE_PRICE_MONTHLY, STRIPE_PRICE_6MONTH, SITE_URL)
- `webhook.js`   — Vercel function: Stripe webhook → marks subscribers Pro in
  Supabase (env: STRIPE_WEBHOOK_SECRET, SUPABASE_SERVICE_ROLE_KEY)
- `picks.js`, `session.js` — legacy, not routed/deployed
- `vercel.json`  — routes all pages through page.js; checkout/webhook/API unchanged
- `package.json`, `robots.txt`, `sitemap.xml`

## Backend (Supabase project: APEXFIGHTIQ / whhbvglvtkqizfllxgtf)

Tables: picks (public record), scouting_picks (DWCS etc., off the record),
ufc_fights (history, 8.7k fights), fighter_ratings / fighter_profiles /
fighter_tott, fight_predictions + model_train + model_params (the v3 model:
Elo + experience + streak + finish rate + layoff + age diff + past-prime,
logistic regression, retrained weekly), pfl_fights + pfl_fighter_ratings
(separate PFL engine), ext_algo_picks (Ky Slider benchmark, private).

Key SQL functions: analyze_matchup(a,b) — the model; sync_pick_probabilities();
compute_fighter_ratings(); build_fighter_profiles(); fit_win_model(iters,lr);
eval_win_model(cutoff,use_age) — holdout harness; analyze_matchup_pfl(a,b);
compute_pfl_ratings().

Edge functions: sync-results (ESPN → ufc_fights, nightly cron),
ingest-pfl-history (ESPN → pfl_fights, weekly cron), next-card, plus scrapers.

Automation: nightly Supabase crons (grade picks, sync results, ratings, pick
probabilities), weekly Sunday cloud task (grade → ingest → retrain → next card →
Ky Slider benchmark), weekly PFL sync.

## The Fight Lab v5 engine (analyzer.html)

Three probability signals per bout, shown side by side and blended into a
consensus (60% client engine / 40% server model when both exist):

- **Engine v5 (client)** — 17 cross-matchup factors (striking volume/accuracy/
  defense, KO power, wrestling offense, TD defense, control matchup, submission
  threat, cardio & pace (5-round weighted), durability, reach conditioned on
  style, age curve, momentum, experience, a style rock-paper-scissors matrix,
  sustained output, finishing instinct) → logistic → Platt-calibrated on the
  user's own graded picks.
- **Server model (v3)** — `win_pct` from the `picks` table.
- **Market** — vig-free implied probability from the book odds.

The Monte Carlo simulator runs 10,000 fights per bout (round-level dominance,
fighter-specific KO/sub hazards vs opponent durability/grappling exposure,
fatigue drift, judge noise on close cards) to produce method-of-victory and
round-of-finish distributions, distance probability, and over/under round
totals — each priced to fair American odds.

## Anon key note

The Supabase anon key embedded in index.html / analyzer.html / page.js is
public by design (row-level security controls access). The service-role key
lives only in Vercel/Supabase env vars — never commit it.
