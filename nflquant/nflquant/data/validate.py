"""Data validation: fail loudly on malformed inputs instead of training on garbage."""
from __future__ import annotations

import pandas as pd

VALID_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
}

GAMES_REQUIRED = [
    "game_id", "season", "game_type", "week", "gameday", "away_team",
    "home_team", "away_score", "home_score", "result", "spread_line",
    "total_line", "away_rest", "home_rest", "div_game", "roof",
    "away_qb_name", "home_qb_name",
]


class DataValidationError(ValueError):
    pass


def validate_games(g: pd.DataFrame) -> None:
    missing = [c for c in GAMES_REQUIRED if c not in g.columns]
    if missing:
        raise DataValidationError(f"games.csv missing columns: {missing}")
    bad = set(g["home_team"]) | set(g["away_team"])
    bad -= VALID_TEAMS
    if bad:
        raise DataValidationError(f"unknown team codes after remap: {sorted(bad)}")
    if g["game_id"].duplicated().any():
        dupes = g.loc[g["game_id"].duplicated(), "game_id"].head().tolist()
        raise DataValidationError(f"duplicate game_ids: {dupes}")
    played = g[g["result"].notna()]
    if len(played) and not (
        (played["result"] == played["home_score"] - played["away_score"]).all()
    ):
        raise DataValidationError("result != home_score - away_score for some games")
    if len(played) and (
        (played["home_score"] < 0).any() or (played["home_score"] > 80).any()
    ):
        raise DataValidationError("home_score outside [0, 80]")
    # Spreads: home-favored games have positive spread_line in nflverse convention
    sp = g["spread_line"].dropna()
    if len(sp) and (sp.abs() > 30).any():
        raise DataValidationError("spread_line magnitude > 30")


def validate_pbp(df: pd.DataFrame, season: int) -> None:
    need = {"game_id", "epa", "posteam", "defteam", "season", "week"}
    missing = need - set(df.columns)
    if missing:
        raise DataValidationError(f"pbp {season} missing columns: {missing}")
    if not len(df):
        raise DataValidationError(f"pbp {season} is empty")
    seasons = set(df["season"].dropna().unique())
    if seasons != {season}:
        raise DataValidationError(f"pbp file for {season} contains seasons {seasons}")
    epa = df["epa"].dropna()
    if len(epa) and (epa.abs() > 15).any():
        raise DataValidationError(f"pbp {season} has |EPA| > 15 (corrupt rows?)")
