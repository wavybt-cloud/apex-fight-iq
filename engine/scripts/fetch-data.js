'use strict';
// Pull the historical tables to local JSON so fitting and backtesting are
// reproducible offline. Read-only; writes nothing back.
//
//   SUPABASE_URL=... SUPABASE_KEY=... node engine/scripts/fetch-data.js
//
// The anon/publishable key is sufficient — these tables are readable by the
// site. Nothing here needs, or should be given, the service-role key.

const fs = require('fs');
const path = require('path');

const URL = process.env.SUPABASE_URL;
const KEY = process.env.SUPABASE_KEY;
const OUT = path.join(__dirname, '..', 'data');
const PAGE = 1000;

if (!URL || !KEY) {
  console.error('Set SUPABASE_URL and SUPABASE_KEY.');
  process.exit(1);
}

async function fetchAll(table, select, order) {
  const rows = [];
  for (let offset = 0; ; offset += PAGE) {
    const url = `${URL}/rest/v1/${table}?select=${select}&order=${order}&limit=${PAGE}&offset=${offset}`;
    const res = await fetch(url, { headers: { apikey: KEY, Authorization: `Bearer ${KEY}` } });
    if (!res.ok) throw new Error(`${table}: HTTP ${res.status} ${await res.text()}`);
    const batch = await res.json();
    rows.push(...batch);
    process.stdout.write(`\r  ${table}: ${rows.length}`);
    if (batch.length < PAGE) break;
  }
  process.stdout.write('\n');
  return rows;
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const jobs = [
    ['ufc_fights', 'id,event,event_date,fighter_a,fighter_b,winner,method,round,weight_class', 'event_date.asc,id.asc'],
    ['fighter_tott', 'fighter,height_in,reach_in,stance,dob,weight_lb', 'fighter.asc'],
  ];
  for (const [table, select, order] of jobs) {
    const rows = await fetchAll(table, select, order);
    fs.writeFileSync(path.join(OUT, `${table}.json`), JSON.stringify(rows));
    console.log(`  wrote ${rows.length} rows to data/${table}.json`);
  }
})().catch((e) => { console.error('\n' + e.message); process.exit(1); });
