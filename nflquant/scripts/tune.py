"""Staged tuning program. All objectives are scored on the DEVELOPMENT window
(2012-2018) only; the chosen configuration is then confirmed once on the
validation window (2019-2022). The 2023-2025 test window is not consulted -
it was spent on the final evaluation, and 2026 live results are the new test.

Stages:
  A. Elo: Optuna over k, rest, QB penalty, season regression, HFA
     (static value vs dynamic walk-forward halflife). Objective: dev log loss.
  B. Feature EWMA halflives: offense x defense x turnover grid.
     Objective: dev Brier of a rolling ridge (no-context feature set).
  C. QB model: prior strength x decay x CPOE weight grid. Same objective.
  D. Recency: season sample-weight halflife for ridge/logit/GBM. Same.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import optuna
import pandas as pd

from nflquant.backtesting.rolling import rolling_predictions
from nflquant.config import PACKAGE_ROOT, cache_dir, load_config
from nflquant.data.ingest import load_games, load_pbp
from nflquant.evaluation.metrics import brier, log_loss
from nflquant.features.build import build_features, feature_columns
from nflquant.features.team_game import team_game_stats
from nflquant.injuries.model import injury_features
from nflquant.models.baselines import LogisticModel, RidgeMarginModel
from nflquant.models.elo import run_elo
from nflquant.models.gbm import GBMModel
from nflquant.models.qb import qb_game_stats, run_qb_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEV = list(range(2012, 2019))
VAL = list(range(2019, 2023))
CONTEXT = {"rest_diff", "div_game", "temp", "wind", "tz_travel", "dome",
           "surface_grass", "neutral"}
QB_COLS = ["d_qb_points", "home_qb_n_eff", "away_qb_n_eff"]
INJ_COLS = ["d_inj", "home_inj", "away_inj"]


def tuned_cols():
    """Final feature set: pure minus the context group (ablation-justified)."""
    base = [c for c in feature_columns("pure") if c not in CONTEXT]
    return base + ["elo_diff_eff"] + QB_COLS + INJ_COLS


def dev_metric(preds: pd.DataFrame, metric=brier) -> float:
    d = preds[preds.home_win.isin([0.0, 1.0])]
    return metric(d.home_win.values.astype(float), d.p_home.values.astype(float))


def main():
    cfg = load_config()
    games = load_games(cfg)
    pbp = load_pbp(cfg)
    tg = team_game_stats(pbp, gt_band=tuple(cfg["features"]["garbage_time_wp"]))
    qg = qb_game_stats(pbp)
    inj = injury_features(cfg, games)

    # base features for QB-change flags (halflife-independent)
    feats0 = build_features(games, None, tg=tg)
    qb_flags = feats0[["game_id", "home_qb_change", "away_qb_change"]]
    dev_ids = set(feats0[(feats0.season.isin(DEV)) & feats0.result.notna()
                         & feats0.home_win.isin([0.0, 1.0])].game_id)
    y_by_id = feats0.set_index("game_id").home_win

    # ---------------- Stage A: Elo ----------------
    def elo_objective(trial):
        mode = trial.suggest_categorical("hfa_mode", ["static", "dynamic"])
        kwargs = dict(
            k=trial.suggest_float("k", 8, 35),
            rest_points_per_day=trial.suggest_float("rest", 0, 12),
            qb_change_penalty=trial.suggest_float("qb_pen", 0, 120),
            preseason_regress=trial.suggest_float("regress", 0.15, 0.5),
            hfa_points=trial.suggest_float("hfa", 25, 75),
            dynamic_hfa_halflife=(
                trial.suggest_float("hfa_hl", 80, 900, log=True) if mode == "dynamic" else None
            ),
            qb_change_flags=qb_flags,
        )
        elo = run_elo(games, **kwargs)
        e = elo[elo.game_id.isin(dev_ids)]
        y = y_by_id.loc[e.game_id].values.astype(float)
        return log_loss(y, e.elo_home_prob.values)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=11))
    study.optimize(elo_objective, n_trials=160, show_progress_bar=False)
    elo_best = study.best_params
    # baseline: current config values
    elo_cur = run_elo(games, qb_change_flags=qb_flags, **{k: v for k, v in cfg["elo"].items()})
    e = elo_cur[elo_cur.game_id.isin(dev_ids)]
    ll_cur = log_loss(y_by_id.loc[e.game_id].values.astype(float), e.elo_home_prob.values)
    print(f"\n[A] Elo dev logloss: current {ll_cur:.5f} -> tuned {study.best_value:.5f}")
    print("    best:", {k: round(v, 3) if isinstance(v, float) else v for k, v in elo_best.items()})

    elo_kwargs = dict(
        k=elo_best["k"], rest_points_per_day=elo_best["rest"],
        qb_change_penalty=elo_best["qb_pen"], preseason_regress=elo_best["regress"],
        hfa_points=elo_best["hfa"],
        dynamic_hfa_halflife=elo_best.get("hfa_hl"),
    )
    elo_frame = run_elo(games, qb_change_flags=qb_flags, **elo_kwargs)

    # helper: assemble enriched feats for given halflives/qb params
    def enriched(hl_off, hl_def, hl_to, qb_params=None):
        f = build_features(games, None, ewma_halflife=hl_off,
                           halflife_def=hl_def, halflife_to=hl_to, tg=tg)
        f = f.merge(elo_frame, on="game_id", how="left")
        q = run_qb_model(games, None, qg=qg, **(qb_params or {}))
        f = f.merge(q, on="game_id", how="left")
        return f.merge(inj, on="game_id", how="left")

    cols = tuned_cols()

    def ridge_dev_brier(f, season_halflife=None):
        p = rolling_predictions(
            f, lambda: RidgeMarginModel(cols, "r", season_halflife=season_halflife), DEV)
        return dev_metric(p)

    # ---------------- Stage B: EWMA halflives ----------------
    results_b = []
    for ho in (4, 6, 8):
        for hd in (6, 10, 14):
            for ht in (6, 20, 40):
                b = ridge_dev_brier(enriched(ho, hd, ht))
                results_b.append(((ho, hd, ht), b))
    results_b.sort(key=lambda x: x[1])
    print("\n[B] halflife grid (off, def, to) -> ridge dev brier; top 5:")
    for (combo, b) in results_b[:5]:
        print(f"    {combo}: {b:.5f}")
    base_b = dict(results_b)[(6, 6, 6)]
    print(f"    baseline (6,6,6): {base_b:.5f}")
    hl_best = results_b[0][0]

    # ---------------- Stage C: QB parameters ----------------
    results_c = []
    for ps in (100, 200, 400):
        for dec in (0.95, 0.97, 0.99):
            for cw in (0.0, 0.005, 0.01):
                qp = dict(prior_strength=ps, game_decay=dec, cpoe_weight=cw)
                b = ridge_dev_brier(enriched(*hl_best, qb_params=qp))
                results_c.append(((ps, dec, cw), b))
    results_c.sort(key=lambda x: x[1])
    print("\n[C] QB grid (prior, decay, cpoe_w) -> ridge dev brier; top 5:")
    for (combo, b) in results_c[:5]:
        print(f"    {combo}: {b:.5f}")
    qb_best = dict(prior_strength=results_c[0][0][0], game_decay=results_c[0][0][1],
                   cpoe_weight=results_c[0][0][2])

    feats_best = enriched(*hl_best, qb_params=qb_best)

    # ---------------- Stage D: recency halflife ----------------
    results_d = []
    for shl in (None, 3, 5, 8, 12):
        b_r = ridge_dev_brier(feats_best, season_halflife=shl)
        p_l = rolling_predictions(
            feats_best, lambda: LogisticModel(cols, "l", season_halflife=shl), DEV)
        results_d.append((shl, b_r, dev_metric(p_l)))
    print("\n[D] recency season-halflife -> dev brier (ridge, logit):")
    for shl, br, bl in results_d:
        print(f"    {shl}: {br:.5f}  {bl:.5f}")
    shl_best = min(results_d, key=lambda x: x[1] + x[2])[0]

    # ---------------- Validation confirmation (once) ----------------
    print("\n=== VALIDATION confirmation (2019-2022, one look) ===")
    old_cols = feature_columns("pure") + ["elo_diff_eff"] + QB_COLS + INJ_COLS
    feats_old = pd.read_parquet(cache_dir(cfg) / "features_enriched.parquet")
    rows = []
    for label, f, c, shl in [
        ("old ridge", feats_old, old_cols, None),
        ("tuned ridge", feats_best, cols, shl_best),
        ("old logit", feats_old, old_cols, None),
        ("tuned logit", feats_best, cols, shl_best),
        ("old gbm", feats_old, old_cols, None),
        ("tuned gbm", feats_best, cols, shl_best),
    ]:
        if "ridge" in label:
            fac = lambda c=c, s=shl: RidgeMarginModel(c, "m", season_halflife=s)
        elif "logit" in label:
            fac = lambda c=c, s=shl: LogisticModel(c, "m", season_halflife=s)
        else:
            fac = lambda c=c, s=shl: GBMModel(c, "m", season_halflife=s)
        p = rolling_predictions(f, fac, VAL)
        rows.append({"model": label, "val_brier": dev_metric(p),
                     "val_logloss": dev_metric(p, log_loss)})
    # elo old vs tuned on val
    val_ids = set(feats0[(feats0.season.isin(VAL)) & feats0.home_win.isin([0.0, 1.0])].game_id)
    for label, ef in [("old elo", elo_cur), ("tuned elo", elo_frame)]:
        e = ef[ef.game_id.isin(val_ids)]
        y = y_by_id.loc[e.game_id].values.astype(float)
        rows.append({"model": label, "val_brier": brier(y, e.elo_home_prob.values),
                     "val_logloss": log_loss(y, e.elo_home_prob.values)})
    print(pd.DataFrame(rows).round(5).to_string(index=False))

    out = {
        "elo": {**{k: round(v, 3) for k, v in elo_kwargs.items() if v is not None},
                "mode": elo_best["hfa_mode"]},
        "halflives": {"off": hl_best[0], "def": hl_best[1], "to": hl_best[2]},
        "qb": qb_best, "season_halflife": shl_best,
        "cols": "pure minus context + elo + qb + inj",
    }
    (PACKAGE_ROOT / "runs" / "tuning_result.json").write_text(json.dumps(out, indent=2))
    print("\nsaved runs/tuning_result.json")


if __name__ == "__main__":
    main()
