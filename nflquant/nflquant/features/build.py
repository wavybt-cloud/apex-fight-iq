"""Walk-forward, leakage-safe game feature matrix.

The builder iterates games in strict chronological order. For each game it
FIRST reads both teams' current EWMA state (features), THEN updates the state
with that game's team-game stats. A feature therefore can only contain
information from games that finished before the one being predicted.
Between seasons, team state regresses toward the trailing league mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflquant.features.team_game import team_game_stats
from nflquant.logging_utils import get_logger

log = get_logger(__name__)

# Stats carried in per-team EWMA state (higher = better offense unless noted)
EWMA_STATS = [
    "off_epa", "off_pass_epa", "off_rush_epa", "off_success", "off_explosive",
    "off_third_conv", "off_sack_rate", "off_cpoe", "off_turnovers", "off_plays",
    "def_epa", "def_pass_epa", "def_rush_epa", "def_success", "def_explosive",
    "def_sack_rate", "td_rate", "fg_rate", "to_rate", "n_drives",
    # opponent-adjusted: raw game EPA corrected by the opponent's PREGAME
    # strength before entering the EWMA (a big day against a stingy defense
    # counts for more than the same day against a sieve)
    "adj_off_epa", "adj_def_epa",
]

# Approximate home-stadium timezone offset from ET (positive = further west)
TEAM_TZ = {
    "ARI": 2, "ATL": 0, "BAL": 0, "BUF": 0, "CAR": 0, "CHI": 1, "CIN": 0,
    "CLE": 0, "DAL": 1, "DEN": 2, "DET": 0, "GB": 1, "HOU": 1, "IND": 0,
    "JAX": 0, "KC": 1, "LA": 3, "LAC": 3, "LV": 3, "MIA": 0, "MIN": 1,
    "NE": 0, "NO": 1, "NYG": 0, "NYJ": 0, "PHI": 0, "PIT": 0, "SEA": 3,
    "SF": 3, "TB": 0, "TEN": 1, "WAS": 0,
}

DOME_ROOFS = {"dome", "closed"}


def build_features(
    games: pd.DataFrame,
    pbp: pd.DataFrame,
    ewma_halflife: float = 6.0,
    season_regress: float = 0.35,
    gt_band: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Return one row per game with pregame features for both teams.

    Only games whose season has PBP coverage get EPA features; earlier games
    get NaN (models trained on the EPA feature set filter those out).
    """
    tg = team_game_stats(pbp, gt_band=gt_band)
    tg_map: dict[tuple[str, str], pd.Series] = {
        (r.game_id, r.team): r for r in tg.itertuples(index=False)
    }

    alpha = 1.0 - 0.5 ** (1.0 / ewma_halflife)
    state: dict[str, dict] = {}          # team -> {stat: ewma, "_n": games, "_season": last}
    league: dict[str, float] = {}        # trailing league mean per stat (EWMA, slow)
    last_qb: dict[str, str] = {}         # team -> last starting QB name

    rows = []
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    for gm in games.itertuples(index=False):
        row: dict = {"game_id": gm.game_id, "season": gm.season, "week": gm.week,
                     "gameday": gm.gameday, "game_type": gm.game_type,
                     "home_team": gm.home_team, "away_team": gm.away_team}

        for side, team, qb in (("home", gm.home_team, gm.home_qb_name),
                               ("away", gm.away_team, gm.away_qb_name)):
            st = state.get(team)
            if st is not None and st["_season"] != gm.season:
                # between-season regression toward trailing league mean
                for s in EWMA_STATS:
                    if st.get(s) is not None and league.get(s) is not None:
                        st[s] = (1 - season_regress) * st[s] + season_regress * league[s]
                st["_n_season"] = 0
                st["_season"] = gm.season
            for s in EWMA_STATS:
                row[f"{side}_{s}"] = st.get(s) if st else np.nan
            row[f"{side}_n_games"] = st["_n"] if st else 0
            row[f"{side}_n_season"] = st.get("_n_season", 0) if st else 0
            # QB continuity: does the listed starter differ from the last start?
            prev_qb = last_qb.get(team)
            row[f"{side}_qb_change"] = int(
                prev_qb is not None and isinstance(qb, str) and qb != prev_qb
            )

        # game context (all known pregame)
        row["home_rest"] = gm.home_rest
        row["away_rest"] = gm.away_rest
        row["rest_diff"] = (gm.home_rest or 7) - (gm.away_rest or 7)
        row["div_game"] = gm.div_game
        row["playoff"] = int(gm.game_type != "REG")
        row["neutral"] = int(getattr(gm, "location", "Home") == "Neutral")
        row["dome"] = int(str(gm.roof) in DOME_ROOFS)
        row["surface_grass"] = int(str(getattr(gm, "surface", "")).startswith("grass"))
        temp = getattr(gm, "temp", np.nan)
        wind = getattr(gm, "wind", np.nan)
        row["temp"] = 68.0 if row["dome"] else (temp if pd.notna(temp) else 60.0)
        row["wind"] = 0.0 if row["dome"] else (wind if pd.notna(wind) else 8.0)
        row["tz_travel"] = TEAM_TZ.get(gm.home_team, 0) - TEAM_TZ.get(gm.away_team, 0)

        # market (kept for market-aware mode and evaluation; excluded in pure mode)
        row["spread_line"] = gm.spread_line
        row["total_line"] = gm.total_line
        row["away_moneyline"] = getattr(gm, "away_moneyline", np.nan)
        row["home_moneyline"] = getattr(gm, "home_moneyline", np.nan)

        # targets (post-game; never used as features)
        row["result"] = gm.result
        row["total"] = gm.total
        row["home_win"] = (
            np.nan if pd.isna(gm.result) else (1.0 if gm.result > 0 else (0.0 if gm.result < 0 else 0.5))
        )
        rows.append(row)

        # ---- state update AFTER features are read ----
        for team, opp_side, qb in (
            (gm.home_team, "away", gm.home_qb_name),
            (gm.away_team, "home", gm.away_qb_name),
        ):
            if isinstance(qb, str):
                last_qb[team] = qb
            s_row = tg_map.get((gm.game_id, team))
            if s_row is None:
                continue  # no PBP for this game (pre-coverage era or unplayed)
            st = state.setdefault(team, {"_n": 0, "_n_season": 0, "_season": gm.season})
            st["_season"] = gm.season
            # opponent adjustment uses the opponent's PREGAME state already
            # captured in `row`, so it cannot see this game's outcome
            vals: dict[str, float] = {}
            for s in EWMA_STATS:
                if s in ("adj_off_epa", "adj_def_epa"):
                    continue
                v = getattr(s_row, s, None)
                if v is not None:
                    vals[s] = v
            opp_def = row.get(f"{opp_side}_def_epa")
            opp_off = row.get(f"{opp_side}_off_epa")
            if vals.get("off_epa") is not None and not np.isnan(vals["off_epa"]):
                shift = 0.0
                if opp_def is not None and not pd.isna(opp_def) and league.get("def_epa") is not None:
                    shift = opp_def - league["def_epa"]
                vals["adj_off_epa"] = vals["off_epa"] - shift
            if vals.get("def_epa") is not None and not np.isnan(vals["def_epa"]):
                shift = 0.0
                if opp_off is not None and not pd.isna(opp_off) and league.get("off_epa") is not None:
                    shift = opp_off - league["off_epa"]
                vals["adj_def_epa"] = vals["def_epa"] - shift
            for s in EWMA_STATS:
                x = vals.get(s)
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    continue
                cur = st.get(s)
                st[s] = x if cur is None else alpha * x + (1 - alpha) * cur
                lg = league.get(s)
                league[s] = x if lg is None else 0.002 * x + 0.998 * lg
            st["_n"] += 1
            st["_n_season"] += 1

    out = pd.DataFrame(rows)
    # convenience differentials (home minus away)
    for s in EWMA_STATS:
        out[f"d_{s}"] = out[f"home_{s}"] - out[f"away_{s}"]
    log.info("features built: %d games, %d columns", len(out), out.shape[1])
    return out


def feature_columns(mode: str = "pure") -> list[str]:
    """Model input columns. 'pure' has no market info; 'market' adds the lines."""
    cols = [f"d_{s}" for s in EWMA_STATS]
    cols += [
        "rest_diff", "div_game", "playoff", "neutral", "dome", "surface_grass",
        "temp", "wind", "tz_travel", "week",
        "home_qb_change", "away_qb_change", "home_n_season", "away_n_season",
    ]
    if mode == "market":
        cols += ["spread_line", "total_line"]
    return cols
