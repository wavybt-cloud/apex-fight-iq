"""Margin-adjusted Elo with rest, QB-continuity and home-field adjustments.

Sequential over the full games spine (1999+), so it needs no per-season
retraining: every game's prediction uses only prior results by construction.
Emits pregame ratings, win probability and expected margin per game.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

BASE = 1500.0


def run_elo(
    games: pd.DataFrame,
    k: float = 20.0,
    hfa_points: float = 48.0,
    preseason_regress: float = 0.33,
    mov_multiplier: bool = True,
    rest_points_per_day: float = 6.0,
    qb_change_penalty: float = 55.0,
    points_per_elo: float = 0.0402,
    qb_change_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-game pregame Elo columns aligned with `games` order.

    qb_change_flags: optional frame with game_id, home_qb_change, away_qb_change
    (from the feature builder). Without it the QB adjustment is skipped.
    """
    qb = {}
    if qb_change_flags is not None:
        qb = qb_change_flags.set_index("game_id")[
            ["home_qb_change", "away_qb_change"]
        ].to_dict("index")

    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []

    g = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    for gm in g.itertuples(index=False):
        for team in (gm.home_team, gm.away_team):
            ratings.setdefault(team, BASE)
            if last_season.get(team) is not None and last_season[team] != gm.season:
                ratings[team] = (1 - preseason_regress) * ratings[team] + preseason_regress * BASE
            last_season[team] = gm.season

        rh, ra = ratings[gm.home_team], ratings[gm.away_team]
        neutral = getattr(gm, "location", "Home") == "Neutral"
        adj = 0.0 if neutral else hfa_points
        hr = 7 if pd.isna(gm.home_rest) else gm.home_rest
        ar = 7 if pd.isna(gm.away_rest) else gm.away_rest
        adj += rest_points_per_day * float(np.clip(hr - ar, -7, 7))
        q = qb.get(gm.game_id)
        if q:
            adj -= qb_change_penalty * q.get("home_qb_change", 0)
            adj += qb_change_penalty * q.get("away_qb_change", 0)

        diff = rh + adj - ra
        p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        exp_margin = diff * points_per_elo

        rows.append({
            "game_id": gm.game_id,
            "elo_home_pre": rh, "elo_away_pre": ra, "elo_diff_eff": diff,
            "elo_home_prob": p_home, "elo_margin": exp_margin,
        })

        if pd.isna(gm.result):
            continue  # unplayed: prediction emitted, no update
        s_home = 1.0 if gm.result > 0 else (0.0 if gm.result < 0 else 0.5)
        mult = 1.0
        if mov_multiplier:
            winner_diff = diff if gm.result > 0 else -diff
            mult = math.log(abs(gm.result) + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = k * mult * (s_home - p_home)
        ratings[gm.home_team] = rh + delta
        ratings[gm.away_team] = ra - delta

    return pd.DataFrame(rows)
