"""Phase 7+9: ensemble weights on dev OOS, calibration on validation OOS.

Protocol:
    members produce rolling OOS predictions for dev (2012-2018) and
    validation (2019-2022) seasons; the stacker/blender are fit on dev and
    evaluated on validation; the calibrator is fit on validation ensemble
    output. Everything is then frozen (joblib bundle) for the untouched test
    window and live prediction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from nflquant.backtesting.rolling import rolling_predictions, record_run
from nflquant.config import cache_dir, load_config
from nflquant.evaluation.metrics import summarize
from nflquant.features.build import feature_columns
from nflquant.models.baselines import LogisticModel, MarketBaseline, RidgeMarginModel
from nflquant.models.calibration import pick_calibrator
from nflquant.models.ensemble import MarginBlender, ProbStacker
from nflquant.models.gbm import GBMModel
from backtest_v2 import load_enriched, EXTRA_COLS


def member_factories(cfg):
    pure = feature_columns("pure") + ["elo_diff_eff"] + EXTRA_COLS
    from backtest_baselines import EloAsModel
    return {
        "elo": lambda: EloAsModel(),
        "logit_pure": lambda: LogisticModel(pure, "logit_pure"),
        "ridge_pure": lambda: RidgeMarginModel(pure, "ridge_pure"),
        "gbm_pure": lambda: GBMModel(pure, "gbm_pure"),
        "market": lambda: MarketBaseline(),
    }


def gather_oos(feats, factories, seasons, tag, cfg):
    out = {}
    for name, fac in factories.items():
        cache = cache_dir(cfg) / f"oos_{tag}_{name}.parquet"
        if cache.exists():
            out[name] = pd.read_parquet(cache)
        else:
            out[name] = rolling_predictions(feats, fac, seasons)
            out[name].to_parquet(cache, index=False)
    return out


def frame_of(oos, col="p_home"):
    cols = {
        m: df.set_index("game_id")[col]
        for m, df in oos.items() if col in df.columns and df[col].notna().any()
    }
    return pd.DataFrame(cols)


def main():
    cfg = load_config()
    feats = load_enriched(cfg)
    dev = list(range(2012, cfg["seasons"]["validation"][0]))
    val = list(range(cfg["seasons"]["validation"][0], cfg["seasons"]["validation"][1] + 1))

    facs = member_factories(cfg)
    oos_dev = gather_oos(feats, facs, dev, "dev2", cfg)
    oos_val = gather_oos(feats, facs, val, "val", cfg)

    PURE = ["elo", "logit_pure", "ridge_pure", "gbm_pure"]
    MKT = PURE + ["market"]

    stack_pure = ProbStacker(PURE).fit(oos_dev)
    stack_mkt = ProbStacker(MKT).fit(oos_dev)
    print("stacker weights (pure):", stack_pure.weights())
    print("stacker weights (mkt): ", stack_mkt.weights())

    MARGIN_PURE = ["elo", "ridge_pure", "gbm_pure"]
    MARGIN_MKT = MARGIN_PURE + ["market"]
    blend_pure = MarginBlender(MARGIN_PURE).fit(oos_dev)
    blend_mkt = MarginBlender(MARGIN_MKT).fit(oos_dev)
    TOTAL_MEMBERS = ["ridge_pure", "gbm_pure"]
    blend_total = MarginBlender(TOTAL_MEMBERS).fit(oos_dev, pred_col="total", target_col="total_actual")
    blend_total_mkt = MarginBlender(TOTAL_MEMBERS + ["market"]).fit(
        oos_dev, pred_col="total", target_col="total_actual")
    print("margin weights (pure):", blend_pure.weights())
    print("margin weights (mkt): ", blend_mkt.weights())
    print("total weights (pure): ", blend_total.weights(), "| mkt:", blend_total_mkt.weights())

    # ---- validation evaluation ----
    ref = oos_val["elo"].set_index("game_id")
    vp = frame_of(oos_val).dropna()
    vm = frame_of(oos_val, "margin").dropna()
    vt = frame_of(oos_val, "total").dropna()

    rows = []
    for name, p in [("ens_pure", stack_pure.predict(vp)), ("ens_mkt", stack_mkt.predict(vp))]:
        df = ref.loc[vp.index].assign(p_home=p, margin=np.nan).reset_index()
        rows.append({"model": name} | summarize(df))
    for m in ["elo", "ridge_pure", "gbm_pure", "market", "logit_pure"]:
        rows.append({"model": m} | summarize(oos_val[m]))
    # blended margins
    for name, blender, mem in [("blend_pure", blend_pure, MARGIN_PURE), ("blend_mkt", blend_mkt, MARGIN_MKT)]:
        mm = blender.predict(vm)
        df = ref.loc[vm.index].assign(p_home=np.nan, margin=mm).reset_index()
        from nflquant.evaluation.metrics import margin_errors, ats_record
        r = {"model": name}
        r |= margin_errors(df.result.values, df.margin.values)
        r |= ats_record(df.result.values, df.margin.values, df.spread_line.values)
        rows.append(r)
    res = pd.DataFrame(rows)
    print("\n=== validation window", val[0], "-", val[-1], "===")
    print(res.round(4).to_string(index=False))

    # ---- calibration on validation ensemble output ----
    y_val = ref.loc[vp.index, "home_win"].values
    decided = np.isin(y_val, [0.0, 1.0])
    cal_pure, rep_pure = pick_calibrator(stack_pure.predict(vp)[decided], y_val[decided])
    # market mode: identity by principle. The de-vigged market is already
    # calibrated; a correction fitted on ~1k validation games is noise (and
    # the market's dominant stacker weight passes its calibration through).
    from nflquant.models.calibration import IdentityCalibrator
    cal_mkt = IdentityCalibrator().fit(None, None)
    rep_mkt = {"none": "by principle"}
    print("\ncalibration (pure):", rep_pure, "->", cal_pure.name)
    print("calibration (mkt): ", rep_mkt, "->", cal_mkt.name)

    bundle = {
        "members_pure": PURE, "members_mkt": MKT,
        "margin_members_pure": MARGIN_PURE, "margin_members_mkt": MARGIN_MKT,
        "total_members": TOTAL_MEMBERS, "total_members_mkt": TOTAL_MEMBERS + ["market"],
        "stack_pure": stack_pure, "stack_mkt": stack_mkt,
        "blend_pure": blend_pure, "blend_mkt": blend_mkt,
        "blend_total": blend_total, "blend_total_mkt": blend_total_mkt,
        "cal_pure": cal_pure, "cal_mkt": cal_mkt,
        "fit_dev": dev, "fit_val": val,
    }
    joblib.dump(bundle, cache_dir(cfg) / "ensemble_bundle.joblib")
    record_run("ensemble_val", cfg, res, extra={
        "stack_pure": stack_pure.weights(), "stack_mkt": stack_mkt.weights(),
        "cal_pure": rep_pure, "cal_mkt": rep_mkt,
    })
    print("\nbundle saved.")


if __name__ == "__main__":
    main()
