-- Odds snapshot history.
--
-- This table is the critical path for the quant engine. Odds history cannot be
-- backfilled — only accumulated — so every hour this is not running is an hour
-- of validation data permanently lost. Without a closing line there is no CLV,
-- and without CLV there is no way to tell a real edge from a lucky month.
--
-- Append-only by intent: a correction is a new row with a later captured_at,
-- never an UPDATE. That is what makes the history usable for as-of feature
-- construction without leaking the future.

create table if not exists public.odds_snapshots (
  id              bigint generated always as identity primary key,
  captured_at     timestamptz not null default now(),
  source          text        not null,
  event_key       text        not null,   -- stable event id from the source
  commence_time   timestamptz,
  fighter_a       text        not null,
  fighter_b       text        not null,
  book            text        not null,
  market          text        not null,   -- 'h2h', 'totals', ...
  outcome_a       text,
  outcome_b       text,
  price_a         numeric     not null check (price_a > 1),
  price_b         numeric     not null check (price_b > 1),
  point           numeric,
  book_updated_at timestamptz
);

-- Primary read pattern: every quote for one event/market, newest first.
create index if not exists odds_snapshots_event_idx
  on public.odds_snapshots (event_key, market, captured_at desc);

-- Secondary: "what did we capture recently", and closing-line lookups.
create index if not exists odds_snapshots_captured_idx
  on public.odds_snapshots (captured_at desc);

create index if not exists odds_snapshots_commence_idx
  on public.odds_snapshots (commence_time);

-- Deduplicate: one row per (event, market, book, capture). A cron that runs
-- twice must not double-count a price.
create unique index if not exists odds_snapshots_unique_idx
  on public.odds_snapshots (event_key, market, book, captured_at);

-- RLS on with no policies: readable and writable only by the service role.
-- Odds history is an asset; there is no reason to expose it to the browser.
alter table public.odds_snapshots enable row level security;
