"""Position-weighted injury burden from official weekly injury reports.

The weekly report is published before the game, so joining it to that game is
legitimately pregame information (same class as the closing line or the
listed starter).

Burden = sum over listed players of position_weight x P(misses game | status).
Not all injuries are equal: a tackle ruled Out outweighs a questionable
fullback. Weights are in rough points of team quality and deliberately
conservative - most reported players are role players, and the QB starter
effect is already carried by the QB model (the QB weight here mostly prices
backup-QB depth and late-week uncertainty).

Statuses map to historical miss rates: Out ~ certain, Doubtful ~85%,
Questionable ~35%, listed-without-status ~15% (practice-report only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflquant.data.ingest import TEAM_REMAP, load_injuries_season
from nflquant.logging_utils import get_logger

log = get_logger(__name__)

POSITION_WEIGHTS = {
    "QB": 1.5,
    "T": 0.60, "G": 0.40, "C": 0.50, "OL": 0.50, "OT": 0.60, "OG": 0.40,
    "WR": 0.50, "TE": 0.35, "RB": 0.35, "FB": 0.10,
    "CB": 0.50, "S": 0.40, "FS": 0.40, "SS": 0.40, "DB": 0.40,
    "DE": 0.50, "DT": 0.40, "NT": 0.40, "DL": 0.45,
    "LB": 0.35, "OLB": 0.45, "ILB": 0.35, "MLB": 0.35,
    "K": 0.15, "P": 0.10, "LS": 0.10,
}
DEFAULT_WEIGHT = 0.30

MISS_PROB = {"Out": 1.00, "Doubtful": 0.85, "Questionable": 0.35}
MISS_PROB_UNLISTED = 0.15   # on the report with practice status only

FIRST_INJURY_SEASON = 2009  # nflverse injuries coverage start


def team_week_burden(cfg, seasons: list[int]) -> pd.DataFrame:
    """One row per (season, week, team): expected points of absent talent."""
    frames = []
    for s in seasons:
        inj = load_injuries_season(cfg, s)
        if inj is None or not len(inj):
            continue
        df = inj[["season", "week", "team", "position", "report_status"]].copy()
        df["team"] = df["team"].replace(TEAM_REMAP)
        df["w"] = df["position"].map(POSITION_WEIGHTS).fillna(DEFAULT_WEIGHT)
        df["p_miss"] = df["report_status"].map(MISS_PROB).fillna(MISS_PROB_UNLISTED)
        df["burden"] = df["w"] * df["p_miss"]
        g = df.groupby(["season", "week", "team"], as_index=False).agg(
            inj_burden=("burden", "sum"),
            inj_out_count=("report_status", lambda s_: (s_ == "Out").sum()),
        )
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "inj_burden", "inj_out_count"])
    out = pd.concat(frames, ignore_index=True)
    log.info("injury burden built: %d team-weeks over %d seasons", len(out), len(frames))
    return out


def injury_features(cfg, games: pd.DataFrame) -> pd.DataFrame:
    """Per-game injury columns keyed by game_id.

    Games before injury coverage (or where a team filed no report - rare)
    get NaN and are median-imputed inside the model pipelines.
    """
    seasons = sorted(set(games.season.unique()) & set(range(FIRST_INJURY_SEASON, 2100)))
    tw = team_week_burden(cfg, seasons)
    key = tw.set_index(["season", "week", "team"])

    def look(season, week, team, col):
        try:
            return key.at[(season, week, team), col]
        except KeyError:
            return np.nan

    rows = []
    for gm in games.itertuples(index=False):
        h_b = look(gm.season, gm.week, gm.home_team, "inj_burden")
        a_b = look(gm.season, gm.week, gm.away_team, "inj_burden")
        h_o = look(gm.season, gm.week, gm.home_team, "inj_out_count")
        a_o = look(gm.season, gm.week, gm.away_team, "inj_out_count")
        rows.append({
            "game_id": gm.game_id,
            "home_inj": h_b, "away_inj": a_b,
            "d_inj": (a_b - h_b) if pd.notna(a_b) and pd.notna(h_b) else np.nan,
            "home_inj_out": h_o, "away_inj_out": a_o,
        })
    return pd.DataFrame(rows)
