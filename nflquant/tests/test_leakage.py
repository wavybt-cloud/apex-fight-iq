"""Automated leakage tests.

The core guarantee: a game's feature row is identical whether or not any
LATER game exists in the input data. If future information leaked into
features, truncating the future would change them.
"""
import numpy as np
import pandas as pd
import pytest

from nflquant.config import load_config
from nflquant.data.ingest import load_games, load_pbp
from nflquant.features.build import build_features, feature_columns


@pytest.fixture(scope="module")
def small_world():
    cfg = load_config()
    games = load_games(cfg)
    games = games[(games.season >= 2016) & (games.season <= 2018)].reset_index(drop=True)
    pbp = load_pbp(cfg, seasons=[2016, 2017, 2018])
    return games, pbp


def test_truncation_invariance(small_world):
    games, pbp = small_world
    feats_full = build_features(games, pbp)

    probe = feats_full[(feats_full.season == 2018) & (feats_full.week == 10)].iloc[0]
    cutoff = probe.gameday

    g_trunc = games[games.gameday <= cutoff].reset_index(drop=True)
    played_ids = set(g_trunc[g_trunc.gameday < cutoff].game_id)
    pbp_trunc = pbp[pbp.game_id.isin(played_ids)].reset_index(drop=True)

    feats_trunc = build_features(g_trunc, pbp_trunc)
    row_full = feats_full[feats_full.game_id == probe.game_id].iloc[0]
    row_trunc = feats_trunc[feats_trunc.game_id == probe.game_id].iloc[0]

    for col in feature_columns("market"):
        a, b = row_full[col], row_trunc[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, abs=1e-12), f"feature {col} changed when future removed"


def test_targets_not_in_features():
    cols = set(feature_columns("market"))
    assert {"result", "total", "home_win", "home_score", "away_score"}.isdisjoint(cols)


def test_first_game_has_no_stats(small_world):
    games, pbp = small_world
    feats = build_features(games, pbp)
    # very first chronological game: both teams must have empty EPA state
    first = feats.iloc[0]
    assert np.isnan(first["home_off_epa"]) and np.isnan(first["away_off_epa"])
    assert first["home_n_games"] == 0 and first["away_n_games"] == 0


def test_state_excludes_current_game(small_world):
    """A team's feature before game N must equal its EWMA after game N-1 only."""
    games, pbp = small_world
    feats = build_features(games, pbp)
    # find a team's 2nd game of 2016; its off_epa feature must equal exactly
    # the single-game value from its 1st game (EWMA of one sample = the sample)
    from nflquant.features.team_game import team_game_stats
    tg = team_game_stats(pbp)
    team = "NE"
    tf = feats[((feats.home_team == team) | (feats.away_team == team))].head(2)
    g1, g2 = tf.iloc[0], tf.iloc[1]
    side2 = "home" if g2.home_team == team else "away"
    v1 = tg[(tg.game_id == g1.game_id) & (tg.team == team)]["off_epa"].iloc[0]
    assert g2[f"{side2}_off_epa"] == pytest.approx(v1, abs=1e-12)
