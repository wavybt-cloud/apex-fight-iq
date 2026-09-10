"""Player prop models: anytime TD, QB passing yards, RB rushing yards.

Architecture
------------
1. Walk weekly player stats chronologically, keeping per-player EWMA state
   (yards, volume, TD rate) plus per-team offensive TD pace. State is read
   BEFORE each week is absorbed, so historical evaluation is leak-free.
2. Anytime TD: expected team offensive TDs come from the game engine's
   projected points and drive mix; a player's share of those TDs is his
   shrunk trailing share (rush+rec TDs; passing TDs do NOT count, a QB's
   rushing TDs do). P(anytime) = 1 - exp(-E[team TDs] x share)  (Poisson
   thinning).
3. Yardage props: player trailing mean, adjusted for opponent unit strength
   (pass/rush defensive EPA vs league) and game environment (projected team
   points vs the team's trailing average). Distributions: Normal for QB
   passing yards, Gamma for rushing yards (right-skewed, floored at zero).
   Spread parameters are fit empirically from 2023-2025 weekly residuals.
4. Current-team mapping uses the 2026 roster file, so offseason moves are
   respected. Rookies with no NFL games get no prop (no fabricated priors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from nflquant.config import cache_dir, ca_bundle
from nflquant.logging_utils import get_logger

log = get_logger(__name__)

PS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.parquet"
TEAM_REMAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

EWMA_HL_GAMES = 10          # player form halflife (games)
SEASON_CARRY = 0.85         # cross-season discount on accumulated evidence
TD_SHARE_PRIOR_K = 8.0      # games of prior weight for TD share shrinkage
# empirical prior TD share by position group (fit from 2023-25 in build script)
TD_SHARE_PRIOR = {"QB": 0.05, "RB": 0.16, "WR": 0.11, "TE": 0.09}
QB_SHARE_CAP = 0.18         # even mobile QBs rarely exceed this
QB_TD_MIN_CARRIES = 2.5     # pocket QBs get no anytime-TD prop


def load_player_weeks(cfg, seasons) -> pd.DataFrame:
    frames = []
    for s in seasons:
        cache = cache_dir(cfg) / f"ps_{s}.parquet"
        if not cache.exists():
            import requests
            r = requests.get(PS_URL.format(season=s), timeout=300, verify=ca_bundle(cfg))
            if r.status_code != 200:
                log.warning("player stats %s unavailable", s)
                continue
            cache.write_bytes(r.content)
        df = pd.read_parquet(cache)
        frames.append(df)
    ps = pd.concat(frames, ignore_index=True)
    ps["team"] = ps["team"].replace(TEAM_REMAP)
    ps["opponent_team"] = ps["opponent_team"].replace(TEAM_REMAP)
    for c in ["passing_yards", "rushing_yards", "receiving_yards", "attempts",
              "carries", "targets", "rushing_tds", "receiving_tds", "passing_tds"]:
        ps[c] = ps[c].fillna(0.0)
    ps["sk_tds"] = ps["rushing_tds"] + ps["receiving_tds"]   # anytime-TD relevant
    return ps.sort_values(["season", "week"]).reset_index(drop=True)


def load_roster(cfg, season: int) -> pd.DataFrame:
    cache = cache_dir(cfg) / f"roster_{season}.parquet"
    if not cache.exists():
        import requests
        r = requests.get(ROSTER_URL.format(season=season), timeout=300, verify=ca_bundle(cfg))
        r.raise_for_status()
        cache.write_bytes(r.content)
    r = pd.read_parquet(cache)
    r["team"] = r["team"].replace(TEAM_REMAP)
    return r[r.status == "ACT"][["gsis_id", "team", "position", "full_name",
                                 "depth_chart_position"]].dropna(subset=["gsis_id"])


class PlayerState:
    """Walk-forward EWMA state over player weeks + team TD pace."""

    def __init__(self, hl=EWMA_HL_GAMES):
        self.alpha = 1 - 0.5 ** (1 / hl)
        self.p: dict[str, dict] = {}       # player_id -> stats
        self.team_td: dict[str, float] = {}  # team -> off TDs/game EWMA
        self.team_pts: dict[str, float] = {}

    def absorb_week(self, wk: pd.DataFrame):
        """wk: all player rows of one (season, week)."""
        a = self.alpha
        # team totals for share denominators
        tt = wk.groupby("team").agg(td=("sk_tds", "sum")).reset_index()
        for r in tt.itertuples(index=False):
            cur = self.team_td.get(r.team)
            self.team_td[r.team] = r.td if cur is None else a * r.td + (1 - a) * cur
        for r in wk.itertuples(index=False):
            st = self.p.setdefault(r.player_id, {
                "n": 0.0, "name": r.player_display_name, "pos": r.position,
                "pass_y": None, "att": None, "rush_y": None, "car": None,
                "rec_y": None, "tgt": None, "td_pg": None, "team": r.team,
            })
            st["n"] += 1
            st["name"], st["pos"], st["team"] = r.player_display_name, r.position, r.team
            for key, val in [("pass_y", r.passing_yards), ("att", r.attempts),
                             ("rush_y", r.rushing_yards), ("car", r.carries),
                             ("rec_y", r.receiving_yards), ("tgt", r.targets),
                             ("td_pg", r.sk_tds)]:
                cur = st[key]
                st[key] = val if cur is None else a * val + (1 - a) * cur

    def season_break(self):
        for st in self.p.values():
            st["n"] *= SEASON_CARRY

    def td_share(self, pid: str) -> float | None:
        st = self.p.get(pid)
        if not st or st["td_pg"] is None:
            return None
        team_td = self.team_td.get(st["team"]) or 2.3
        raw = st["td_pg"] / max(team_td, 0.5)
        pos = st["pos"] if st["pos"] in TD_SHARE_PRIOR else (
            "RB" if st["pos"] in ("FB", "HB") else "WR")
        prior = TD_SHARE_PRIOR.get(pos, 0.08)
        n = min(st["n"], 40)
        cap = QB_SHARE_CAP if pos == "QB" else 0.55
        return float(np.clip((n * raw + TD_SHARE_PRIOR_K * prior) / (n + TD_SHARE_PRIOR_K), 0.005, cap))


def build_state(ps: pd.DataFrame, through=None) -> PlayerState:
    """Absorb all weeks (optionally only those strictly before `through` =
    (season, week)) into a fresh state."""
    st = PlayerState()
    last_season = None
    for (season, week), wk in ps.groupby(["season", "week"], sort=True):
        if through is not None and (season, week) >= through:
            break
        if last_season is not None and season != last_season:
            st.season_break()
        last_season = season
        st.absorb_week(wk)
    return st


# ---------- distributions ----------

def prob_to_american(p: float) -> str:
    p = min(max(p, 0.01), 0.99)
    return f"-{round(100*p/(1-p))}" if p >= 0.5 else f"+{round(100*(1-p)/p)}"


def anytime_td_prob(exp_team_tds: float, share: float) -> float:
    return float(1.0 - np.exp(-exp_team_tds * share))


def exp_offensive_tds(exp_pts: float, td_rate: float, fg_rate: float) -> float:
    """Invert the drive-mix scaling used by the simulator."""
    td0, fg0 = max(td_rate, 0.03), max(fg_rate, 0.03)
    return float(np.clip(exp_pts * td0 / (7.0 * td0 + 3.0 * fg0), 0.5, 6.5))


def pass_yards_dist(mean: float, sd_a: float, sd_b: float):
    sd = sd_a + sd_b * mean
    return norm(loc=mean, scale=max(sd, 25.0))


def rush_yards_dist(mean: float, sd_a: float, sd_b: float):
    sd = max(sd_a + sd_b * mean, 12.0)
    mean = max(mean, 5.0)
    k = (mean / sd) ** 2
    theta = sd ** 2 / mean
    return gamma_dist(a=k, scale=theta)


def fit_yard_spreads(ps: pd.DataFrame) -> dict:
    """Empirical sd-vs-mean relationships for weekly passing/rushing yards.

    For players with enough games: regress |weekly - trailing mean| to get a
    linear sd model. Returns {pass: (a, b), rush: (a, b)}."""
    out = {}
    for kind, ycol, vol_col, min_vol, min_trail in [
            ("pass", "passing_yards", "attempts", 15, 120),
            ("rush", "rushing_yards", "carries", 8, 25)]:
        rows = []
        for pid, grp in ps[ps[vol_col] >= min_vol].groupby("player_id"):
            if len(grp) < 8:
                continue
            y = grp.sort_values(["season", "week"])[ycol].values
            trail = pd.Series(y).ewm(halflife=EWMA_HL_GAMES).mean().shift(1).values
            m = ~np.isnan(trail) & (trail >= min_trail)  # established form only
            rows += list(zip(trail[m], np.abs(y[m] - trail[m])))
        arr = np.array(rows)
        # |err| of a normal = sd*sqrt(2/pi); invert to sd units
        X = np.column_stack([np.ones(len(arr)), arr[:, 0]])
        coef, *_ = np.linalg.lstsq(X, arr[:, 1] / np.sqrt(2 / np.pi), rcond=None)
        out[kind] = (float(coef[0]), float(coef[1]))
    return out


# ---------- opponent / environment adjustments ----------

def unit_adj(opp_def_epa: float | None, league_mean: float, beta: float = 1.6,
             lo: float = 0.86, hi: float = 1.14) -> float:
    """Yardage multiplier vs an opponent unit. def EPA below league mean =
    stingy defense = fewer yards."""
    if opp_def_epa is None or not np.isfinite(opp_def_epa):
        return 1.0
    return float(np.clip(1.0 + beta * (opp_def_epa - league_mean), lo, hi))
