"""Dynamic QB strength model.

Rating = shrunk, recency-weighted EPA per dropback, updated walk-forward:
    state read for each game's LISTED STARTERS (games.csv) before the game,
    then updated with every passer's actual production in that game.

Shrinkage: rating = (n_eff * ewma + K * prior) / (n_eff + K)
    n_eff  = decayed career dropbacks (recent games count more)
    K      = prior strength in dropbacks (~200): small samples stay near prior
    prior  = trailing league mean minus a debut penalty (rookies/backups start
             below average, matching historical first-start performance).

A QB rating is points-scale-free (EPA/dropback, league mean ~0.05); the
feature layer converts the home-away gap to points via dropback volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEBUT_PENALTY = 0.08          # EPA/dropback below league mean for a first start
PRIOR_STRENGTH = 200.0        # dropbacks of prior weight
GAME_DECAY = 0.97             # per-game decay of accumulated evidence
DROPBACKS_PER_GAME = 34.0     # for converting rating gaps to points
POINTS_PER_EPA = 1.0          # EPA is already in points


def qb_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, passer_id): dropbacks and EPA/dropback."""
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["epa"].notna() & pbp["passer_player_id"].notna()]
    g = db.groupby(["game_id", "passer_player_id"])
    out = pd.DataFrame({
        "dropbacks": g["epa"].size(),
        "epa_db": g["epa"].mean(),
        "name": g["passer_player_name"].first(),
    }).reset_index().rename(columns={"passer_player_id": "qb_id"})
    return out


def run_qb_model(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-game pregame QB ratings for the listed starters.

    Returns columns: game_id, home_qb_rating, away_qb_rating, home_qb_n_eff,
    away_qb_n_eff, d_qb_rating (home-away, EPA/dropback), d_qb_points.
    """
    qg = qb_game_stats(pbp)
    qg_by_game: dict[str, list] = {}
    for r in qg.itertuples(index=False):
        qg_by_game.setdefault(r.game_id, []).append(r)

    state: dict[str, dict] = {}      # qb_id -> {"ewma": x, "n_eff": n}
    last_starter: dict[str, str] = {}  # team -> last listed starter id
    league_mean = 0.03               # updated as evidence accumulates
    rows = []

    def rating_of(qb_id) -> tuple[float, float]:
        st = state.get(qb_id)
        prior = league_mean - DEBUT_PENALTY
        if st is None or st["n_eff"] <= 0:
            return prior, 0.0
        shrunk = (st["n_eff"] * st["ewma"] + PRIOR_STRENGTH * prior) / (st["n_eff"] + PRIOR_STRENGTH)
        return shrunk, st["n_eff"]

    g = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    for gm in g.itertuples(index=False):
        h_id = getattr(gm, "home_qb_id", None)
        a_id = getattr(gm, "away_qb_id", None)
        # upcoming games may not list a starter yet: assume QB continuity
        if not isinstance(h_id, str):
            h_id = last_starter.get(gm.home_team)
        if not isinstance(a_id, str):
            a_id = last_starter.get(gm.away_team)
        h_r, h_n = rating_of(h_id)
        a_r, a_n = rating_of(a_id)
        d = h_r - a_r
        rows.append({
            "game_id": gm.game_id,
            "home_qb_rating": h_r, "away_qb_rating": a_r,
            "home_qb_n_eff": h_n, "away_qb_n_eff": a_n,
            "d_qb_rating": d,
            "d_qb_points": d * DROPBACKS_PER_GAME * POINTS_PER_EPA,
        })
        if isinstance(getattr(gm, "home_qb_id", None), str):
            last_starter[gm.home_team] = gm.home_qb_id
        if isinstance(getattr(gm, "away_qb_id", None), str):
            last_starter[gm.away_team] = gm.away_qb_id
        # update with this game's actual passing (all passers, not just starters)
        for r in qg_by_game.get(gm.game_id, []):
            st = state.setdefault(r.qb_id, {"ewma": 0.0, "n_eff": 0.0})
            n_new = st["n_eff"] * GAME_DECAY + r.dropbacks
            st["ewma"] = (
                (st["ewma"] * st["n_eff"] * GAME_DECAY + r.epa_db * r.dropbacks) / n_new
                if n_new > 0 else 0.0
            )
            st["n_eff"] = n_new
            league_mean = 0.999 * league_mean + 0.001 * r.epa_db
    return pd.DataFrame(rows)
