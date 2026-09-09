import numpy as np

from nflquant.simulation.possession import simulate_game, summarize_sims


def _sim(exp_h=24.0, exp_a=21.0, n=20000, seed=42, **kw):
    return simulate_game(exp_h, exp_a, n_sims=n, seed=seed, **kw)


def test_means_match_inputs():
    s = summarize_sims(_sim())
    assert abs(s["mean_home"] - 24.0) < 1.2
    assert abs(s["mean_away"] - 21.0) < 1.2
    assert abs(s["mean_margin"] - 3.0) < 1.2


def test_favorite_wins_more():
    s = summarize_sims(_sim())
    assert 0.55 < s["p_home"] < 0.66  # ~3 pt favorite


def test_even_game_symmetric():
    s = summarize_sims(_sim(23.0, 23.0, seed=7))
    assert abs(s["p_home"] - 0.5) < 0.02


def test_distribution_shape():
    s = summarize_sims(_sim())
    assert 11.0 < s["sd_margin"] < 14.5
    assert 0.40 < s["p_one_score"] < 0.60
    assert 0.02 < s["p_ot"] < 0.09
    assert s["key_mass_3"] > 0.08          # key numbers present
    assert s["key_mass_7"] > 0.06


def test_cover_and_total_probs():
    s = summarize_sims(_sim(), spread_line=3.0, total_line=45.0)
    assert 0.35 < s["p_home_cover"] < 0.65
    assert 0.35 < s["p_over"] < 0.65


def test_reproducible():
    a = summarize_sims(_sim(seed=123))
    b = summarize_sims(_sim(seed=123))
    assert a["p_home"] == b["p_home"]
    assert a["mean_total"] == b["mean_total"]


def test_scores_are_football_numbers():
    sims = _sim()
    assert sims["home"].min() >= 0
    # 1-point games are nearly impossible in real football scoring
    margins = np.abs(sims["home"] - sims["away"])
    assert (margins == 1).mean() < 0.04
