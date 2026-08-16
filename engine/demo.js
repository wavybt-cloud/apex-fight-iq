'use strict';
// Demonstration of the engine on a synthetic bout.
//
// The fighter numbers below are INVENTED for the purpose of exercising the code
// path. They are not scouting data and no conclusion about any real fighter
// should be drawn from this output. Its purpose is to show what the engine does
// with a bout, and — importantly — to show it correctly refusing to bet.
//
//   node engine/demo.js

const engine = require('./src');

const NOW = Date.now();

const bout = {
  id: 'demo-1',
  event: 'DEMO CARD — synthetic fighters, invented numbers',
  rounds: 3,
  modelCalibrated: true, // pretend a calibration record exists for the data gate
  a: {
    name: 'Southpaw Sniper', confirmed: true, bouts: 18,
    statsUpdatedAt: NOW - 3 * 86400000, hasRoundDetail: true,
    koRate: 0.85, subRate: 0.10, durability: 0.95, subDefense: 1.1,
    cardio: 0.85, output: 5.2, accuracy: 0.52, grappling: 0.5, takedownDefense: 1.3, control: 0.2,
  },
  b: {
    name: 'Pressure Wrestler', confirmed: true, bouts: 22,
    statsUpdatedAt: NOW - 2 * 86400000, hasRoundDetail: true,
    koRate: 0.25, subRate: 0.55, durability: 1.35, subDefense: 1.2,
    cardio: 1.45, output: 3.6, accuracy: 0.44, grappling: 2.1, takedownDefense: 0.9, control: 0.65,
  },
  quotes: [
    { market: 'moneyline', book: 'BookOne', decimals: [2.10, 1.78], observedAt: NOW - 120000 },
    { market: 'moneyline', book: 'BookTwo', decimals: [2.05, 1.83], observedAt: NOW - 90000 },
    { market: 'moneyline', book: 'BookThree', decimals: [2.12, 1.76], observedAt: NOW - 45000 },
    { market: 'distance', book: 'BookOne', decimals: [1.95, 1.87], observedAt: NOW - 120000 },
    { market: 'distance', book: 'BookTwo', decimals: [1.90, 1.92], observedAt: NOW - 60000 },
  ],
};

console.log('='.repeat(78));
console.log('APEX QUANT ENGINE', engine.VERSION);
console.log('STATUS:', engine.STATUS);
console.log('='.repeat(78));

// --- 1. Data quality gate ---
const q = engine.quality.assess(bout, NOW);
console.log(`\n[1] DATA QUALITY: ${q.score}/100 — ${q.gate}`);
for (const d of q.defects) console.log(`      · ${d.label}${d.detail ? ` (${d.detail})` : ''}`);
if (!q.defects.length) console.log('      · no defects');

// --- 2. Simulation ---
const sim = engine.sim.run({
  a: bout.a, b: bout.b, rounds: bout.rounds,
  iterations: 40000, batches: 50, seed: 20260816,
});
console.log(`\n[2] MONTE CARLO — ${sim.iterations.toLocaleString()} fights, seed ${sim.seed}`);
console.log(`      ${bout.a.name}: ${(sim.probA * 100).toFixed(1)}%   ${bout.b.name}: ${(sim.probB * 100).toFixed(1)}%`);
console.log(`      parameter interval for A: ${(sim.paramInterval[0] * 100).toFixed(1)}% – ${(sim.paramInterval[1] * 100).toFixed(1)}%  (SE ${(sim.standardError * 100).toFixed(2)}%)`);
console.log(`      goes the distance: ${(sim.goesDistance * 100).toFixed(1)}%`);
console.log(`      ${bout.a.name}: KO ${(sim.method.a.ko * 100).toFixed(1)}%  SUB ${(sim.method.a.sub * 100).toFixed(1)}%  DEC ${(sim.method.a.dec * 100).toFixed(1)}%`);
console.log(`      ${bout.b.name}: KO ${(sim.method.b.ko * 100).toFixed(1)}%  SUB ${(sim.method.b.sub * 100).toFixed(1)}%  DEC ${(sim.method.b.dec * 100).toFixed(1)}%`);

// --- 3. Market ---
const mlQuotes = bout.quotes.filter((x) => x.market === 'moneyline');
const market = engine.odds.consensusMarket(mlQuotes.map((x) => ({ book: x.book, decimals: x.decimals })));
console.log(`\n[3] MARKET — ${market.bookCount} books, mean overround ${(market.meanOverround * 100).toFixed(2)}%`);
console.log(`      devigged (conservative): A ${(market.conservative[0] * 100).toFixed(1)}%  B ${(market.conservative[1] * 100).toFixed(1)}%`);
console.log(`      best price A: ${market.bestDecimal[0]} @ ${market.bestBook[0]}`);
const dv = engine.odds.devig(mlQuotes[0].decimals);
console.log('      estimator spread on A: '
  + Object.entries(dv.probs).map(([k, v]) => `${k} ${(v[0] * 100).toFixed(1)}%`).join(', '));

// --- 4. Adversarial testing ---
const sens = engine.sensitivity.runSensitivity(
  { a: bout.a, b: bout.b, rounds: bout.rounds, iterations: 12000, batches: 24, seed: 991 },
  'A', market.conservative[0],
);
console.log(`\n[4] ADVERSARIAL RE-RUNS — survival ${(sens.survival * 100).toFixed(0)}%`);
for (const s of sens.scenarios) {
  console.log(`      ${s.survives ? '✓' : '✗'} ${s.label}: ${(s.probability * 100).toFixed(1)}% (edge ${(s.edge * 100 >= 0 ? '+' : '')}${(s.edge * 100).toFixed(1)})`);
}

// --- 5 & 6. Full scan with the honest model state: no calibration history ---
console.log('\n[5] FULL SCAN — model state: no walk-forward calibration record\n');
const scan = engine.scanner.scan({
  bouts: [bout], now: NOW, bankroll: 10000,
  modelState: {},
  options: { iterations: 12000, batches: 24, seed: 20260816, minScore: 80 },
});

const ml = scan.allCandidates.find((c) => c.marketKey === 'ML_A');
console.log(ml.emitted.text);

console.log('\n' + '='.repeat(78));
console.log('SCAN RESULT:', scan.summary);
console.log(`Bouts scanned: ${scan.boutsScanned} · markets evaluated: ${scan.marketsEvaluated} · qualifying: ${scan.qualifying}`);
if (scan.blocked) console.log('Blocking reasons:', scan.blocked.join(', '));
console.log('='.repeat(78));
