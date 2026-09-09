"""Live prediction pipeline: refresh data, retrain members on all completed
games, predict the target week, simulate every game 50k times, emit per-game
reports and the weekly dashboard.

Usage:
    python3 scripts/predict_week.py [--season 2026] [--week 1] [--sims 50000]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from nflquant.config import PACKAGE_ROOT, cache_dir, load_config
from nflquant.data.ingest import load_games, load_pbp, load_pbp_season
from nflquant.features.build import build_features, feature_columns
from nflquant.logging_utils import get_logger
from nflquant.market.lines import novig_home_prob
from nflquant.models.baselines import LogisticModel, MarketBaseline, RidgeMarginModel
from nflquant.models.elo import run_elo
from nflquant.models.gbm import GBMModel
from nflquant.models.qb import run_qb_model
from nflquant.reports.game_report import (confidence_grade, format_game_report,
                                          key_drivers)
from nflquant.simulation.possession import simulate_game, summarize_sims

from nflquant.injuries.model import injury_features

log = get_logger("predict")
QB_COLS = ["d_qb_points", "home_qb_n_eff", "away_qb_n_eff"]
INJ_COLS = ["d_inj", "home_inj", "away_inj"]


def fresh_features(cfg) -> pd.DataFrame:
    games = load_games(cfg, max_age_hours=6)
    lo, hi = cfg["data"]["pbp_seasons"]
    seasons = list(range(lo, hi + 1))
    # include the current season's PBP when the release carries it
    cur = int(games.season.max())
    if cur not in seasons:
        try:
            load_pbp_season(cfg, cur)
            seasons.append(cur)
            log.info("current-season PBP %s available", cur)
        except Exception as e:
            log.warning("no PBP for %s yet (%s); features carry over from %s",
                        cur, e.__class__.__name__, cur - 1)
    pbp = load_pbp(cfg, seasons=seasons)
    feats = build_features(
        games, pbp,
        ewma_halflife=cfg["features"]["ewma_halflife_games"],
        gt_band=tuple(cfg["features"]["garbage_time_wp"]),
    )
    elo = run_elo(games, qb_change_flags=feats[["game_id", "home_qb_change", "away_qb_change"]],
                  **{k: v for k, v in cfg["elo"].items()})
    feats = feats.merge(elo, on="game_id", how="left")
    qb = run_qb_model(games, pbp)
    feats = feats.merge(qb, on="game_id", how="left")
    inj = injury_features(cfg, games)
    return feats.merge(inj, on="game_id", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    n_sims = args.sims or cfg["simulation"]["n_sims"]
    bundle = joblib.load(cache_dir(cfg) / "ensemble_bundle.joblib")
    feats = fresh_features(cfg)

    season = args.season or int(feats.season.max())
    upcoming = feats[(feats.season == season) & feats.result.isna()]
    if args.week:
        target = upcoming[upcoming.week == args.week]
    else:
        wk = int(upcoming.week.min())
        target = upcoming[upcoming.week == wk]
    if not len(target):
        print(f"no unplayed games found for season {season}"
              + (f" week {args.week}" if args.week else ""))
        return
    week = int(target.week.iloc[0])
    log.info("predicting season %s week %s: %d games", season, week, len(target))

    # ---- train members on everything completed ----
    train = feats[feats.result.notna()]
    pure = feature_columns("pure") + ["elo_diff_eff"] + QB_COLS + INJ_COLS
    members = {
        "elo": None,  # sequential; read from columns
        "logit_pure": LogisticModel(pure, "logit_pure").fit(train),
        "ridge_pure": RidgeMarginModel(pure, "ridge_pure").fit(train),
        "gbm_pure": GBMModel(pure, "gbm_pure").fit(train),
        "market": MarketBaseline().fit(train),
    }

    preds = {}
    for name, mdl in members.items():
        if name == "elo":
            preds[name] = pd.DataFrame({"p_home": target.elo_home_prob.values,
                                        "margin": target.elo_margin.values},
                                       index=target.index)
        else:
            preds[name] = mdl.predict(target)

    prob_frame = pd.DataFrame({m: preds[m]["p_home"] for m in bundle["members_pure"]})
    p_pure = bundle["cal_pure"].transform(bundle["stack_pure"].predict(prob_frame))
    has_ml = target.home_moneyline.notna().values
    p_mkt = np.full(len(target), np.nan)
    if has_ml.any():
        pf_m = pd.DataFrame({m: preds[m]["p_home"] for m in bundle["members_mkt"]})
        p_mkt = bundle["cal_mkt"].transform(bundle["stack_mkt"].predict(pf_m))

    margin_frame = pd.DataFrame({m: preds[m]["margin"] for m in bundle["margin_members_pure"]})
    margins = bundle["blend_pure"].predict(margin_frame)
    total_frame = pd.DataFrame({m: preds[m]["total"] for m in bundle["total_members"]})
    totals = bundle["blend_total"].predict(total_frame)

    out_dir = PACKAGE_ROOT / "reports_out" / f"{season}_week{week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = cfg["simulation"]
    games_out = []
    for i, (_, row) in enumerate(target.iterrows()):
        exp_h = (totals[i] + margins[i]) / 2
        exp_a = (totals[i] - margins[i]) / 2
        sims = simulate_game(
            exp_h, exp_a,
            home_drive_mix=(row.home_td_rate or 0.24, row.home_fg_rate or 0.15),
            away_drive_mix=(row.away_td_rate or 0.24, row.away_fg_rate or 0.15),
            home_to_rate=row.home_to_rate or 0.11, away_to_rate=row.away_to_rate or 0.11,
            n_sims=n_sims,
            drives_mean=sim_cfg["drives_mean"], drives_sd=sim_cfg["drives_sd"],
            param_sd_pts=sim_cfg["param_sd_pts"],
            drive_var_shrink=sim_cfg["drive_var_shrink"],
            endgame_tie_prob=sim_cfg["endgame_tie_prob"],
            walkoff_prob=sim_cfg["walkoff_prob"],
            consolation_prob=sim_cfg["consolation_prob"],
            playoff=bool(row.playoff), seed=sim_cfg["seed"] + i,
        )
        s = summarize_sims(sims, row.spread_line if pd.notna(row.spread_line) else None,
                           row.total_line if pd.notna(row.total_line) else None)
        member_probs = {m: float(preds[m]["p_home"].iloc[i]) for m in bundle["members_pure"]}
        if pd.notna(row.home_moneyline):
            member_probs["market"] = float(novig_home_prob(row.away_moneyline, row.home_moneyline))
        qb_uncertain = not isinstance(row.get("home_qb_name"), str) or \
            not isinstance(row.get("away_qb_name"), str) or \
            bool(row.home_qb_change or row.away_qb_change)
        grade = confidence_grade(member_probs, week, qb_uncertain, s["sd_margin"])
        g = {
            "game_id": row.game_id, "away": row.away_team, "home": row.home_team,
            "gameday": str(pd.Timestamp(row.gameday).date()), "week": week,
            "p_home": float(p_pure[i]),
            "p_home_mkt": float(p_mkt[i]) if np.isfinite(p_mkt[i]) else None,
            "margin": float(margins[i]), "total": float(totals[i]),
            "spread_line": float(row.spread_line) if pd.notna(row.spread_line) else None,
            "total_line": float(row.total_line) if pd.notna(row.total_line) else None,
            "sim": s, "members": member_probs,
            "drivers": key_drivers(row, members["ridge_pure"], pure),
            "confidence": grade,
        }
        games_out.append(g)
        report = format_game_report(g)
        (out_dir / f"{row.game_id}.txt").write_text(report)
        print(report)

    # ---- weekly dashboard ----
    from nflquant.reports.weekly import write_dashboard
    dash = write_dashboard(games_out, out_dir, season, week)
    print(dash)
    (out_dir / "predictions.json").write_text(json.dumps(
        [{k: v for k, v in g.items() if k != "drivers"} |
         {"drivers": [[d, round(v, 3)] for d, v in g["drivers"]]} for g in games_out],
        indent=2, default=str))

    # ---- bridge file for the /nfl web page ----
    # nflverse uses LA for the Rams; the page (ESPN) uses LAR
    to_page = {"LA": "LAR"}
    bridge = {
        "generated": pd.Timestamp.now().isoformat(),
        "engine": "nflquant v0.1 (calibrated ensemble + 50k possession sims)",
        "season": season, "week": week,
        "games": [{
            "away": to_page.get(g["away"], g["away"]),
            "home": to_page.get(g["home"], g["home"]),
            "gameday": g["gameday"],
            "p_home": round(g["p_home"], 4),
            "margin": round(g["margin"], 2),
            "total": round(g["total"], 2),
            "p_home_cover": round(g["sim"].get("p_home_cover"), 4) if "p_home_cover" in g["sim"] else None,
            "p_over": round(g["sim"].get("p_over"), 4) if "p_over" in g["sim"] else None,
            "confidence": g["confidence"][0],
        } for g in games_out],
    }
    bridge_path = PACKAGE_ROOT.parent / "nfl-model.json"
    bridge_path.write_text(json.dumps(bridge, indent=1))
    log.info("reports written to %s; page bridge -> %s", out_dir, bridge_path)


if __name__ == "__main__":
    main()
