"""Data ingestion: nflverse games spine + play-by-play parquets, with local caching.

Sources
-------
games.csv (nflverse/nfldata): one row per game 1999+, includes final scores,
    closing spread/total/moneylines, rest days, roof/surface/temp/wind,
    starting QBs and head coaches. This is the games spine.
play_by_play_{season}.parquet (nflverse-data releases): full PBP with EPA/WP.

All downloads cache to data_cache/ and are only re-fetched with force=True,
except the current in-progress season's games file which refreshes when older
than `max_age_hours`.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from nflquant.config import ca_bundle, cache_dir
from nflquant.data.validate import validate_games, validate_pbp
from nflquant.logging_utils import get_logger

log = get_logger(__name__)

# Franchise moves: normalize to current abbreviations so a franchise is one
# continuous entity for ratings.
TEAM_REMAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Columns actually consumed downstream; keeps PBP cache small.
PBP_COLS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "play_type", "epa", "success", "yards_gained",
    "pass", "rush", "down", "ydstogo", "yardline_100", "qtr",
    "wp", "wpa", "passer_player_id", "passer_player_name", "qb_dropback",
    "sack", "interception", "fumble_lost", "touchdown", "pass_touchdown",
    "rush_touchdown", "field_goal_result", "field_goal_attempt", "punt_attempt",
    "drive", "fixed_drive", "fixed_drive_result", "posteam_score", "defteam_score",
    "air_yards", "yards_after_catch", "cpoe", "series_success",
]


def current_season(today=None) -> int:
    """NFL season year for a date (a season spans the calendar-year boundary)."""
    d = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    return int(d.year if d.month >= 3 else d.year - 1)


def needs_refresh(cache: Path, season: int, max_age_hours: float) -> bool:
    """Completed seasons are immutable and cache forever; the in-progress one
    gains a week of data every Sunday, so it must re-download once it ages out.
    Without this the current season's files freeze at whatever existed the first
    time they were fetched, and every downstream feature silently stops updating.
    """
    if not cache.exists():
        return True
    if season < current_season():
        return False
    return (time.time() - cache.stat().st_mtime) > max_age_hours * 3600


def _download(url: str, dest: Path, verify) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300, verify=verify) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    tmp.rename(dest)


def load_games(cfg: dict, force: bool = False, max_age_hours: float = 24.0) -> pd.DataFrame:
    """Games spine with normalized team codes and derived columns."""
    cache = cache_dir(cfg) / "games.csv"
    local = Path(cfg["data"].get("games_local") or "")
    stale = cache.exists() and (time.time() - cache.stat().st_mtime) > max_age_hours * 3600
    if force or not cache.exists() or stale:
        if local.exists() and not force:
            shutil.copy(local, cache)
            log.info("games.csv copied from local clone %s", local)
        else:
            log.info("downloading games.csv ...")
            _download(cfg["data"]["games_url"], cache, ca_bundle(cfg))
    g = pd.read_csv(cache)
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(TEAM_REMAP)
    g["gameday"] = pd.to_datetime(g["gameday"])
    # result = home_score - away_score (nflverse convention); recompute defensively
    played = g["home_score"].notna() & g["away_score"].notna()
    g.loc[played, "result"] = g.loc[played, "home_score"] - g.loc[played, "away_score"]
    g.loc[played, "total"] = g.loc[played, "home_score"] + g.loc[played, "away_score"]
    g["playoff"] = (g["game_type"] != "REG").astype(int)
    g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    validate_games(g)
    return g


def load_pbp_season(cfg: dict, season: int, force: bool = False,
                    max_age_hours: float = 6.0) -> pd.DataFrame:
    """One season of play-by-play, reduced to the columns we consume."""
    cache = cache_dir(cfg) / f"pbp_{season}.parquet"
    if force or needs_refresh(cache, season, max_age_hours):
        url = cfg["data"]["pbp_url_tpl"].format(season=season)
        log.info("downloading pbp %s ...", season)
        raw = cache_dir(cfg) / f"pbp_{season}_raw.parquet"
        _download(url, raw, ca_bundle(cfg))
        df = pd.read_parquet(raw)
        keep = [c for c in PBP_COLS if c in df.columns]
        df[keep].to_parquet(cache, index=False)
        raw.unlink()
    df = pd.read_parquet(cache)
    for col in ("home_team", "away_team", "posteam", "defteam"):
        if col in df.columns:
            df[col] = df[col].replace(TEAM_REMAP)
    validate_pbp(df, season)
    return df


def load_pbp(cfg: dict, seasons: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    if seasons is None:
        lo, hi = cfg["data"]["pbp_seasons"]
        seasons = list(range(lo, hi + 1))
    frames = [load_pbp_season(cfg, s, force=force) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def load_injuries_season(cfg: dict, season: int, force: bool = False) -> pd.DataFrame | None:
    """Weekly injury reports; returns None when the release lacks this season."""
    cache = cache_dir(cfg) / f"injuries_{season}.parquet"
    if force or needs_refresh(cache, season, 6.0):
        url = cfg["data"]["injuries_url_tpl"].format(season=season)
        try:
            _download(url, cache, ca_bundle(cfg))
        except requests.HTTPError as e:
            # a failed refresh must not throw away a cache we already hold
            log.warning("injuries %s unavailable: %s", season, e)
            if not cache.exists():
                return None
    return pd.read_parquet(cache)
