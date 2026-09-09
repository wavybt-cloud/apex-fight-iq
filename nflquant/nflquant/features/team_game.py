"""PBP -> one row per (game_id, team) of offensive/defensive efficiency aggregates.

These are POST-game facts about that single game. They become predictive
features only after the walk-forward EWMA in build.py, which reads a team's
state strictly before each game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# fixed_drive_result values in nflverse PBP
DRIVE_TD = "Touchdown"
DRIVE_FG = "Field goal"

EXPLOSIVE_PASS_YDS = 20
EXPLOSIVE_RUSH_YDS = 10


def team_game_stats(pbp: pd.DataFrame, gt_band: tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
    """Aggregate PBP to team-game rows (offense and defense-allowed views merged)."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    p["is_pass"] = (p["play_type"] == "pass").astype(int)
    p["is_rush"] = (p["play_type"] == "run").astype(int)
    p["explosive"] = (
        (p["is_pass"] == 1) & (p["yards_gained"] >= EXPLOSIVE_PASS_YDS)
    ) | ((p["is_rush"] == 1) & (p["yards_gained"] >= EXPLOSIVE_RUSH_YDS))
    p["third_att"] = (p["down"] == 3).astype(int)
    p["third_conv"] = ((p["down"] == 3) & (p["yards_gained"] >= p["ydstogo"])).astype(int)
    p["turnover"] = (
        p["interception"].fillna(0).astype(float) + p["fumble_lost"].fillna(0).astype(float)
    ).clip(0, 1)
    # garbage-time filter on offense win probability
    wp = p["wp"].astype(float)
    p["meaningful"] = wp.between(gt_band[0], gt_band[1]) | wp.isna()

    def agg_side(df: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
        m = df[df["meaningful"]]
        g = m.groupby(["game_id", team_col])
        out = pd.DataFrame({
            f"{prefix}_epa": g["epa"].mean(),
            f"{prefix}_success": g["success"].mean(),
            f"{prefix}_explosive": g["explosive"].mean(),
            f"{prefix}_plays": g["epa"].size(),
        })
        pass_g = m[m["is_pass"] == 1].groupby(["game_id", team_col])
        rush_g = m[m["is_rush"] == 1].groupby(["game_id", team_col])
        out[f"{prefix}_pass_epa"] = pass_g["epa"].mean()
        out[f"{prefix}_rush_epa"] = rush_g["epa"].mean()
        third = m.groupby(["game_id", team_col])[["third_att", "third_conv"]].sum()
        out[f"{prefix}_third_conv"] = np.where(
            third["third_att"] > 0, third["third_conv"] / third["third_att"], np.nan
        )
        to = df.groupby(["game_id", team_col])["turnover"].sum()  # turnovers incl. garbage time
        out[f"{prefix}_turnovers"] = to
        if "sack" in df.columns and "qb_dropback" in df.columns:
            db = m[m["qb_dropback"] == 1].groupby(["game_id", team_col])
            out[f"{prefix}_sack_rate"] = db["sack"].mean()
            if "cpoe" in m.columns:
                out[f"{prefix}_cpoe"] = db["cpoe"].mean()
        out.index.names = ["game_id", "team"]
        return out.reset_index()

    off = agg_side(p, "posteam", "off")
    de = agg_side(p, "defteam", "def")

    # Drive outcome rates per team-game (offense), for the simulator and rz proxy
    drv = pbp[pbp["fixed_drive"].notna() & pbp["posteam"].notna()]
    dgrp = drv.groupby(["game_id", "posteam", "fixed_drive"])["fixed_drive_result"].first().reset_index()
    dagg = dgrp.groupby(["game_id", "posteam"])["fixed_drive_result"].agg(
        n_drives="size",
        td_rate=lambda s: (s == DRIVE_TD).mean(),
        fg_rate=lambda s: (s == DRIVE_FG).mean(),
        to_rate=lambda s: s.isin(["Turnover", "Turnover on downs", "Opp touchdown"]).mean(),
    ).reset_index().rename(columns={"posteam": "team"})

    out = off.merge(de, on=["game_id", "team"], how="outer")
    out = out.merge(dagg, on=["game_id", "team"], how="left")

    meta = pbp.groupby("game_id")[["season", "week", "home_team", "away_team"]].first().reset_index()
    out = out.merge(meta, on="game_id", how="left")
    return out
