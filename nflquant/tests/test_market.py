import numpy as np

from nflquant.market.lines import (
    american_to_prob, novig_home_prob, spread_to_home_prob, prob_to_american, cover_prob,
)


def test_american_to_prob():
    assert abs(american_to_prob(-110) - 110 / 210) < 1e-9
    assert abs(american_to_prob(150) - 100 / 250) < 1e-9


def test_novig_symmetric():
    p = novig_home_prob(-110, -110)
    assert abs(p - 0.5) < 1e-9


def test_novig_favorite():
    p = novig_home_prob(142, -170)  # home favored
    assert 0.55 < p < 0.65


def test_spread_prob_monotone():
    ps = spread_to_home_prob(np.array([-7.0, 0.0, 7.0]))
    assert ps[0] < 0.5 < ps[2]
    assert abs(ps[1] - 0.5) < 1e-9


def test_prob_american_roundtrip():
    for p in (0.3, 0.5, 0.65, 0.8):
        odds = prob_to_american(p)
        assert abs(american_to_prob(odds) - p) < 1e-9


def test_cover_prob():
    # model agrees with market -> 50%
    assert abs(cover_prob(-3.0, -3.0) - 0.5) < 1e-9
    # model likes home more than market -> home covers > 50%
    assert cover_prob(1.0, -3.0) > 0.5
