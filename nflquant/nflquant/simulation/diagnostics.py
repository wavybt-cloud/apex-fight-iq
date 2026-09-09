"""Simulation realism diagnostics.

Simulates a window of historical games from model projections and compares
aggregate distributional properties against what actually happened. If the
simulator produces unrealistic football, its constants get recalibrated
before anyone trusts a percentile from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflquant.simulation.possession import simulate_game, summarize_sims


def simulate_window(
    games: pd.DataFrame,
    n_sims: int = 4000,
    param_sd_pts: float = 2.6,
    drive_var_shrink: float = 0.72,
    endgame_tie_prob: float = 0.30,
    seed: int = 1,
) -> pd.DataFrame:
    """games needs: margin_pred, total_pred, home/away drive mixes + to rates,
    playoff flag, result, total_actual, spread_line, total_line."""
    rows = []
    for i, g in enumerate(games.itertuples(index=False)):
        exp_h = (g.total_pred + g.margin_pred) / 2.0
        exp_a = (g.total_pred - g.margin_pred) / 2.0
        sims = simulate_game(
            exp_h, exp_a,
            home_drive_mix=(g.h_td_rate, g.h_fg_rate),
            away_drive_mix=(g.a_td_rate, g.a_fg_rate),
            home_to_rate=g.h_to_rate, away_to_rate=g.a_to_rate,
            n_sims=n_sims, param_sd_pts=param_sd_pts,
            drive_var_shrink=drive_var_shrink, endgame_tie_prob=endgame_tie_prob,
            playoff=bool(g.playoff), seed=seed + i,
        )
        s = summarize_sims(sims, g.spread_line, g.total_line)
        rows.append({
            "game_id": g.game_id, "p_home": s["p_home"],
            "mean_margin": s["mean_margin"], "sd_margin": s["sd_margin"],
            "mean_total": s["mean_total"], "sd_total": s["sd_total"],
            "p_ot": s["p_ot"], "p_one_score": s["p_one_score"],
            "p_blowout": s["p_blowout_17"], "key3": s["key_mass_3"], "key7": s["key_mass_7"],
            "p_home_cover": s.get("p_home_cover", np.nan),
            "p_over": s.get("p_over", np.nan),
        })
    return pd.DataFrame(rows)


def realism_report(sim_stats: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Compare simulated aggregate properties with observed ones."""
    a = actual.copy()
    a["margin"] = a["result"]
    obs_margin_err = a["margin"] - sim_stats["mean_margin"].values
    obs_total_err = a["total_actual"] - sim_stats["mean_total"].values
    rows = [
        ("home win rate", sim_stats.p_home.mean(), (a.margin > 0).mean()),
        ("margin sd (vs model mean)", sim_stats.sd_margin.mean(), obs_margin_err.std()),
        ("total mean", sim_stats.mean_total.mean(), a.total_actual.mean()),
        ("total sd (vs model mean)", sim_stats.sd_total.mean(), obs_total_err.std()),
        ("OT rate", sim_stats.p_ot.mean(), a.overtime.mean()),
        ("one-score rate", sim_stats.p_one_score.mean(), (a.margin.abs() <= 8).mean()),
        ("blowout(17+) rate", sim_stats.p_blowout.mean(), (a.margin.abs() >= 17).mean()),
        ("|margin|=3 mass", sim_stats.key3.mean(), (a.margin.abs() == 3).mean()),
        ("|margin|=7 mass", sim_stats.key7.mean(), (a.margin.abs() == 7).mean()),
    ]
    return pd.DataFrame(rows, columns=["property", "simulated", "actual"])
