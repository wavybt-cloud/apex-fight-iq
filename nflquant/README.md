# nflquant — NFL quantitative prediction research platform

A leakage-safe, chronologically-validated NFL prediction engine: nflverse data
ingestion, walk-forward EPA features, a dynamic QB model, Elo + regression +
gradient-boosting members, an out-of-sample-weighted ensemble with probability
calibration, and a possession-level Monte Carlo simulator whose score
distributions are validated against real NFL structure (key numbers, one-score
rates, overtime).

Predictions are probability distributions, never certainties. The platform's
first loyalty is calibration: a 70% prediction should win about 70% of the
time, and the test results below are reported against that standard.

---

## Results — final test window (2023–2025, 854 games, fully out-of-sample)

Members retrain per season on prior seasons only; ensemble weights were fit on
2012–2018 out-of-sample predictions; calibrators on 2019–2022. The 2023–2025
window was untouched until the final evaluation.

| model                | Brier | log loss | accuracy | MAE (margin) |
|----------------------|-------|----------|----------|--------------|
| home-team baseline*  | .244  | .681     | 58%      | 11.1         |
| Elo (adjusted)       | .2207 | .6315    | 64.9%    | 10.24        |
| ridge (pure)         | .2190 | .6270    | 64.3%    | 10.17        |
| **ensemble (pure, calibrated)** | **.2189** | **.6270** | 65.7% | 10.17 |
| ensemble (market-aware) | .2107 | .6089  | 67.8%    | 9.82         |
| closing market       | .2102 | .6077    | 68.2%    | 9.79         |

*home baseline from the development window; test values comparable.

Totals: model+market blend MAE 10.11 vs market 10.12 — a statistical tie.

**Against the spread**: betting model-vs-closing-line edges at −110 produced
**negative ROI at every edge threshold** (−1.6% to −11.8%). This platform does
not beat closing lines, and no honest public-data backtest of this scope does.
Its value is calibrated probabilities, realistic distributions, and a
research harness that measures rather than asserts.

Calibration (pure ensemble, test window): predicted-vs-actual within ~2pp in
most bins across 0.50–0.86; largest deviation ~7pp in a 172-game bin.

### Ablations (ridge, test window)

| variant     | Brier  | Δ vs full |
|-------------|--------|-----------|
| full        | .2190  | —         |
| no EPA      | .2200  | +.0010    |
| no Elo      | .2203  | +.0013    |
| no QB model | .2202  | +.0012    |
| no context (rest/travel/weather) | .2185 | −.0005 |

EPA, Elo, and the QB model each carry real weight. The context group adds
nothing out-of-sample on this window — kept for report explanations, flagged
as a candidate for removal per the "keep what works" rule.

---

## Architecture

```
nflquant/
├── config/default.yaml        # every constant, with provenance notes
├── nflquant/
│   ├── config.py              # config + config-hash + git-commit metadata
│   ├── data/                  # ingest.py (nflverse games/PBP/injuries), validate.py
│   ├── features/              # team_game.py (PBP aggregates), build.py (walk-forward)
│   ├── models/                # elo, baselines, qb, gbm, ensemble, calibration
│   ├── market/lines.py        # de-vig, spread<->prob, American odds
│   ├── simulation/            # possession.py (drive-level MC), diagnostics.py
│   ├── evaluation/metrics.py  # Brier, log loss, ECE, calibration tables, ATS
│   ├── backtesting/rolling.py # rolling-origin engine + run registry
│   └── reports/               # game_report.py, weekly.py
├── scripts/
│   ├── build_features.py      # rebuild the feature matrix
│   ├── backtest_baselines.py  # phase 3/4 dev-window benchmark
│   ├── backtest_v2.py         # + QB, opponent adjustment, GBM
│   ├── fit_ensemble.py        # stacker/blender on dev, calibration on val
│   ├── sim_diagnostics.py     # simulator realism vs real NFL
│   ├── final_test_eval.py     # the untouched 2023-2025 evaluation + ablations
│   └── predict_week.py        # LIVE: predict the upcoming week, 50k sims/game
├── tests/                     # market math, leakage, simulation (17 tests)
├── runs/                      # experiment registry (config hash, git commit, metrics)
└── reports_out/               # per-game reports, weekly dashboard, predictions.json
```

## Data

- **Games spine** — `nflverse/nfldata games.csv`: every game 1999–present with
  final scores, closing spread/total/moneylines, rest days, roof/surface/
  temp/wind, starting QBs, coaches. Refreshed when stale.
- **Play-by-play** — nflverse-data releases, 2006–present, EPA/WP included;
  column-pruned parquet cache (~34MB for 20 seasons).
- **Injuries** — nflverse weekly injury reports (ingestion available;
  position-weighted features are future work — QB status, the largest single
  injury effect, is already carried via listed starters and the QB model).

## Leakage protocol

- The feature builder walks games chronologically and reads each team's EWMA
  state **before** updating it with that game — features cannot contain the
  game's own outcome, mechanically.
- Automated tests prove truncation invariance: deleting all future games
  changes no feature of a probe game (`tests/test_leakage.py`).
- Validation is season-chronological everywhere. No shuffling, anywhere.
- Ensemble weights, calibration and simulator constants are each fit on
  windows that end before the windows they are evaluated on.

## Model layers

1. **Elo** — margin-adjusted (538-style MOV multiplier), rest, HFA,
   QB-continuity penalty, between-season regression.
2. **QB model** — shrunk EPA/dropback per QB (200-dropback prior, debut
   penalty, per-game decay), keyed to each game's listed starter with
   continuity fallback for unannounced starters.
3. **Feature models** — ridge margin/total + logistic on ~40 walk-forward
   features: opponent-adjusted off/def EPA, pass/rush splits, success rate,
   explosive rate, third downs, sack rates, CPOE, drive TD/FG/turnover rates,
   turnover regression via EWMA + garbage-time filtering, rest, travel,
   weather (roof-gated), divisional flags.
4. **GBM** — LightGBM with chronological early stopping. Finding: it does not
   beat ridge alone here; it survives as ensemble diversity.
5. **Ensemble** — logit-space stacker (probabilities) and NNLS blends
   (margin/total), fit strictly on earlier out-of-sample predictions. Two
   modes: **pure** (no market inputs) and **market-aware**.
6. **Calibration** — Platt / beta / isotonic, fit on validation only;
   isotonic gated to n≥3000 to prevent small-sample overfitting (beta won).

## Simulator

Drive-level Monte Carlo (default 50,000 sims/game): per-sim team-strength
shocks (parameter uncertainty), correlated pace, TD/FG/empty/turnover drive
outcomes scaled to projected points net of defensive scores, halftime game
script, late-game convergence (FG-chase ties), regulation walk-offs,
backdoor-cover consolation scores, XP/2-pt mix, OT with regular-season ties.

Realism, validated on 2019–2022 (simulated vs actual): one-score rate
50.5%/52.5%, |margin|=3 mass 13.6%/14.3%, |margin|=7 9.4%/8.6%, blowouts
24.8%/24.7%, OT 4.8%/5.7%, margin σ 12.6/13.0. Constants live in config with
provenance; rerun `scripts/sim_diagnostics.py` before touching them.

## Usage

```bash
pip install -e .                       # from nflquant/
python3 -m pytest tests/ -q            # 17 tests
python3 scripts/build_features.py      # data + features (downloads on first run)
python3 scripts/backtest_baselines.py  # reproduce dev benchmarks
python3 scripts/fit_ensemble.py        # fit + freeze ensemble/calibration
python3 scripts/final_test_eval.py     # the honest numbers
python3 scripts/predict_week.py        # live: upcoming week, reports + dashboard
python3 scripts/predict_week.py --season 2026 --week 3 --sims 250000
```

Every run writes a record to `runs/` with config hash, git commit and metrics.

## Honest limitations

- **Preseason**: nflverse carries no preseason games or PBP, so no trained
  preseason model is possible from this data. Preseason handling (rating
  compression, depth/motivation priors) lives in the site's `/nfl` analyzer
  page; this engine covers regular season + playoffs.
- **Injuries beyond QB** are an integration layer, not a fitted model.
- **Market data** is closing lines only (no openers/line movement), so CLV is
  measured against close, and "market-aware" means close-aware.
- The pure model trails the market by ~0.009 Brier. That gap is the honest
  size of the market's information advantage at this feature set.
