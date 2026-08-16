'use strict';
// Apex Quant Engine — public surface.
//
// Every export is a pure function over explicit inputs. There is no hidden
// state and no global mutation anywhere in the engine, which is what makes the
// backtest path and the live path provably identical.

module.exports = {
  prob: require('./core/prob'),
  odds: require('./core/odds'),
  ev: require('./core/ev'),

  quality: require('./data/quality'),

  sim: require('./sim/montecarlo'),
  sensitivity: require('./sim/sensitivity'),

  ensemble: require('./models/ensemble'),
  registry: require('./models/registry'),

  scoring: require('./scoring/score'),

  bankroll: require('./risk/bankroll'),
  protocol: require('./risk/protocol'),

  metrics: require('./calibration/metrics'),
  clv: require('./calibration/clv'),

  backtest: require('./backtest/walkforward'),

  scanner: require('./scan/scanner'),
  parlay: require('./scan/parlay'),
  recommend: require('./scan/recommend'),

  VERSION: '0.1.0',
  STATUS: 'FOUNDATION — no data source connected; not authorised to issue picks',
};
