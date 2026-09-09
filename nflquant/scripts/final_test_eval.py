"""FINAL test-window evaluation (2023-2025). Everything upstream is frozen:
member hyperparameters, ensemble weights (fit on dev 2012-2018), calibrators
(fit on validation 2019-2022). Each member retrains per season on all data
before that season, as it would live.

Also runs the feature-group ablation study on the test window using the
ridge margin model (fast, stable, representative).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from nflquant.backtesting.rolling import record_run, rolling_predictions
from nflquant.config import cache_dir, load_config
from nflquant.evaluation.metrics import (ats_record, calibration_table,
                                         margin_errors, summarize)
from nflquant.features.build import EWMA_STATS, feature_columns
from nflquant.models.baselines import RidgeMarginModel
from backtest_v2 import load_enriched, QB_COLS
from fit_ensemble import frame_of, gather_oos, member_factories


def roi_at_110(result, pred_margin, spread_line, threshold):
    """Flat-stake ROI betting the model side when |model - market| >= threshold."""
    m = np.isfinite(result) & np.isfinite(pred_margin) & np.isfinite(spread_line)
    r, pm, sl = result[m], pred_margin[m], spread_line[m]
    edge = pm - sl
    bet = np.abs(edge) >= threshold
    if bet.sum() == 0:
        return {"bets": 0}
    r, pm, sl, edge = r[bet], pm[bet], sl[bet], edge[bet]
    push = r == sl
    win = np.where(edge > 0, r > sl, r < sl)[~push]
    profit = win.sum() * (100 / 110) - (~win).sum()
    return {"bets": int(len(win)), "win_pct": round(float(win.mean()), 4),
            "roi": round(float(profit / len(win)), 4)}


def main():
    cfg = load_config()
    feats = load_enriched(cfg)
    bundle = joblib.load(cache_dir(cfg) / "ensemble_bundle.joblib")
    test = list(range(cfg["seasons"]["test"][0], cfg["seasons"]["test"][1] + 1))

    facs = member_factories(cfg)
    oos = gather_oos(feats, facs, test, "test", cfg)
    ref = oos["elo"].set_index("game_id")

    tp = frame_of(oos).dropna()
    tm = frame_of(oos, "margin").dropna()
    tt = frame_of(oos, "total").dropna()

    rows = []
    for m in ["elo", "logit_pure", "ridge_pure", "gbm_pure", "market"]:
        rows.append({"model": m} | summarize(oos[m]))

    ens = {
        "ens_pure_raw": bundle["stack_pure"].predict(tp),
        "ens_pure_cal": bundle["cal_pure"].transform(bundle["stack_pure"].predict(tp)),
        "ens_mkt_raw": bundle["stack_mkt"].predict(tp),
        "ens_mkt_cal": bundle["cal_mkt"].transform(bundle["stack_mkt"].predict(tp)),
    }
    for name, p in ens.items():
        df = ref.loc[tp.index].assign(p_home=p, margin=np.nan).reset_index()
        rows.append({"model": name} | summarize(df))

    for name, blender, mem in [
        ("blend_margin_pure", bundle["blend_pure"], bundle["margin_members_pure"]),
        ("blend_margin_mkt", bundle["blend_mkt"], bundle["margin_members_mkt"]),
    ]:
        mm = blender.predict(tm[mem])
        d = ref.loc[tm.index]
        r = {"model": name} | margin_errors(d.result.values, mm)
        r |= ats_record(d.result.values, mm, d.spread_line.values)
        rows.append(r)

    # totals
    for name, blender, mem in [
        ("blend_total_pure", bundle["blend_total"], bundle["total_members"]),
        ("blend_total_mkt", bundle["blend_total_mkt"], bundle["total_members_mkt"]),
    ]:
        tot = blender.predict(tt[mem])
        d = ref.loc[tt.index]
        err = d.total_actual.values - tot
        keep = np.isfinite(err)
        rows.append({"model": name, "mae": float(np.mean(np.abs(err[keep]))),
                     "rmse": float(np.sqrt(np.mean(err[keep] ** 2)))})
    mkt_tot_err = (ref.total_actual - ref.total_line).dropna()
    rows.append({"model": "market_total", "mae": float(mkt_tot_err.abs().mean()),
                 "rmse": float(np.sqrt((mkt_tot_err ** 2).mean()))})

    res = pd.DataFrame(rows)
    print("=== FINAL TEST WINDOW", test[0], "-", test[-1], "===")
    print(res.round(4).to_string(index=False))

    # calibration of the flagship calibrated ensembles
    d = ref.loc[tp.index]
    y = d.home_win.values
    for name in ["ens_pure_cal", "ens_mkt_cal"]:
        print(f"\ncalibration table ({name}):")
        print(calibration_table(y, ens[name]).round(3).to_string(index=False))

    # honest ATS/ROI at thresholds (pure-model margins vs closing spread)
    mm = bundle["blend_pure"].predict(tm[bundle["margin_members_pure"]])
    d = ref.loc[tm.index]
    print("\nATS ROI by edge threshold (pure blend vs closing spread, -110):")
    for th in (0.5, 1.5, 2.5, 3.5, 5.0):
        print(f"  edge>={th}: {roi_at_110(d.result.values, mm, d.spread_line.values, th)}")

    # ---------- ablation study (ridge margin, test window) ----------
    groups = {
        "full": feature_columns("pure") + ["elo_diff_eff"] + QB_COLS,
        "no_epa": [c for c in feature_columns("pure") if not any(
            k in c for k in ["epa", "success", "explosive", "third", "sack", "cpoe",
                             "td_rate", "fg_rate", "to_rate", "n_drives", "turnovers", "plays"]
        )] + ["elo_diff_eff"] + QB_COLS,
        "no_elo": feature_columns("pure") + QB_COLS,
        "no_qb": feature_columns("pure") + ["elo_diff_eff"],
        "no_context": [c for c in feature_columns("pure") if c not in (
            "rest_diff", "div_game", "temp", "wind", "tz_travel", "dome",
            "surface_grass", "neutral")] + ["elo_diff_eff"] + QB_COLS,
    }
    ab_rows = []
    for gname, cols in groups.items():
        p = rolling_predictions(feats, lambda c=cols: RidgeMarginModel(c, f"ridge_{gname}"), test)
        ab_rows.append({"ablation": gname, "n_features": len(cols)} | summarize(p))
    ab = pd.DataFrame(ab_rows)
    print("\n=== ablations (ridge, test window) ===")
    print(ab.round(4).to_string(index=False))

    record_run("final_test", cfg, res, extra={"ablations": ab.to_dict("records")})


if __name__ == "__main__":
    main()
