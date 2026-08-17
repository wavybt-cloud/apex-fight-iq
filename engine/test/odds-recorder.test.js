'use strict';
const test = require('node:test');
const assert = require('node:assert');

const oddsapi = require('../src/data/adapters/oddsapi');
const { consensusMarket } = require('../src/core/odds');
const clv = require('../src/calibration/clv');

const T = Date.UTC(2026, 7, 20, 12, 0, 0);
const START = Date.UTC(2026, 7, 22, 2, 0, 0);

function apiEvent(over) {
  return Object.assign({
    id: 'evt-1',
    commence_time: new Date(START).toISOString(),
    home_team: 'Ann Smith',
    away_team: 'Bea Jones',
    bookmakers: [
      {
        key: 'pinnacle',
        last_update: new Date(T - 60000).toISOString(),
        markets: [{ key: 'h2h', outcomes: [{ name: 'Ann Smith', price: 1.95 }, { name: 'Bea Jones', price: 1.95 }] }],
      },
      {
        key: 'draftkings',
        last_update: new Date(T - 30000).toISOString(),
        markets: [{ key: 'h2h', outcomes: [{ name: 'Ann Smith', price: 2.05 }, { name: 'Bea Jones', price: 1.83 }] }],
      },
    ],
  }, over || {});
}

test('an API event flattens to one row per book per market', () => {
  const rows = oddsapi.normaliseEvent(apiEvent(), T);
  assert.strictEqual(rows.length, 2);
  assert.deepStrictEqual(rows.map((r) => r.book).sort(), ['draftkings', 'pinnacle']);
  const dk = rows.find((r) => r.book === 'draftkings');
  assert.strictEqual(dk.priceA, 2.05);
  assert.strictEqual(dk.priceB, 1.83);
  assert.strictEqual(dk.capturedAt, T);
  assert.strictEqual(dk.commenceTime, START);
  assert.strictEqual(dk.eventKey, 'evt-1');
});

test('malformed and one-sided markets are skipped, not guessed at', () => {
  assert.strictEqual(oddsapi.normaliseEvent(null, T).length, 0);
  assert.strictEqual(oddsapi.normaliseEvent({ id: 'x' }, T).length, 0);
  const bad = apiEvent({
    bookmakers: [
      { key: 'a', markets: [{ key: 'h2h', outcomes: [{ name: 'X', price: 1.9 }] }] },
      { key: 'b', markets: [{ key: 'h2h', outcomes: [{ name: 'X', price: 1.0 }, { name: 'Y', price: 2.0 }] }] },
      { key: 'c', markets: [{ key: 'h2h', outcomes: [{ name: 'X', price: 1.9 }, { name: 'Y', price: 2.1 }] }] },
    ],
  });
  const rows = oddsapi.normaliseEvent(bad, T);
  assert.strictEqual(rows.length, 1, 'only the well-formed two-way market survives');
  assert.strictEqual(rows[0].book, 'c');
});

test('normalise handles an empty or absent feed', () => {
  assert.deepStrictEqual(oddsapi.normalise([], T), []);
  assert.deepStrictEqual(oddsapi.normalise(null, T), []);
});

test('stored rows convert back into canonical quotes, newest per book', () => {
  const rows = [
    ...oddsapi.normaliseEvent(apiEvent(), T - 3600000),
    ...oddsapi.normaliseEvent(apiEvent(), T),
  ];
  const quotes = oddsapi.toQuotes(rows, 'h2h');
  assert.strictEqual(quotes.length, 2, 'one quote per book, not one per capture');
  for (const q of quotes) {
    assert.strictEqual(q.observedAt, T, 'the most recent capture wins');
    assert.strictEqual(q.decimals.length, 2);
  }
  // And they feed the market engine unchanged.
  const m = consensusMarket(quotes.map((q) => ({ book: q.book, decimals: q.decimals })));
  assert.strictEqual(m.bookCount, 2);
  assert.strictEqual(m.bestDecimal[0], 2.05);
  assert.strictEqual(m.bestBook[0], 'draftkings');
});

test('the closing line is the last quote BEFORE the fight starts', () => {
  const early = oddsapi.normaliseEvent(apiEvent(), START - 7200000);
  const late = oddsapi.normaliseEvent(apiEvent(), START - 600000);
  const after = oddsapi.normaliseEvent(apiEvent(), START + 600000);
  const close = oddsapi.closingQuote([...early, ...late, ...after], START, 'h2h');
  assert.strictEqual(close.length, 2);
  for (const q of close) {
    assert.strictEqual(q.observedAt, START - 600000,
      'a price quoted after the fight began must never count as a close');
  }
});

test('no eligible pre-fight capture yields no closing line rather than a wrong one', () => {
  const after = oddsapi.normaliseEvent(apiEvent(), START + 600000);
  assert.strictEqual(oddsapi.closingQuote(after, START, 'h2h'), null);
});

test('captured snapshots support an end-to-end CLV calculation', () => {
  // Bet early at 2.05, market closes shorter on that side: positive CLV.
  const openRows = oddsapi.normaliseEvent(apiEvent(), START - 86400000);
  const closeEvent = apiEvent({
    bookmakers: [{
      key: 'draftkings',
      markets: [{ key: 'h2h', outcomes: [{ name: 'Ann Smith', price: 1.75 }, { name: 'Bea Jones', price: 2.15 }] }],
    }],
  });
  const closeRows = oddsapi.normaliseEvent(closeEvent, START - 600000);

  const betQuote = oddsapi.toQuotes(openRows.filter((r) => r.book === 'draftkings'), 'h2h')[0];
  const closeQuote = oddsapi.closingQuote(closeRows, START, 'h2h')[0];

  const result = clv.betCLV({
    betQuote: betQuote.decimals,
    closeQuote: closeQuote.decimals,
    outcomeIndex: 0,
    decimalOdds: betQuote.decimals[0],
  });
  assert.ok(result.clvProbability > 0, 'the market moved toward the side taken');
  assert.ok(result.beatClose);
  assert.ok(result.clvPercent > 0.15, `expected a material price advantage, got ${result.clvPercent}`);
});

test('the request URL asks for decimal odds across multiple regions', () => {
  const url = oddsapi.oddsUrl('KEY123', { regions: 'us,eu', markets: 'h2h' });
  assert.ok(url.includes(oddsapi.SPORT));
  assert.ok(url.includes('oddsFormat=decimal'), 'American conversion loses precision on long prices');
  assert.ok(url.includes('regions=us%2Ceu'));
  assert.ok(url.includes('apiKey=KEY123'));
});
