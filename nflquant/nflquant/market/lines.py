"""Betting-market conversions.

Conventions (nflverse games.csv):
    spread_line = expected HOME margin (positive => home favored).
    result      = home_score - away_score.
    Home covers when result > spread_line.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def american_to_prob(odds):
    """Vigged implied probability from American odds. Vectorized."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))
    return np.where(np.isfinite(odds) & (odds != 0), p, np.nan)


def american_to_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    d = np.where(odds < 0, 100.0 / -odds + 1.0, odds / 100.0 + 1.0)
    return np.where(np.isfinite(odds) & (odds != 0), d, np.nan)


def novig_home_prob(away_ml, home_ml):
    """De-vigged (proportional) home win probability from a moneyline pair."""
    pa, ph = american_to_prob(away_ml), american_to_prob(home_ml)
    s = pa + ph
    return np.where(np.isfinite(s) & (s > 0), ph / s, np.nan)


def spread_to_home_prob(spread_line, sigma: float = 13.45):
    """Home win probability implied by the spread under a normal margin model.

    P(home wins) = P(margin > 0) where margin ~ N(spread_line, sigma).
    A half-point continuity note: ties are possible only pre-OT; we treat the
    margin as continuous, which is fine at this level and validated in backtests.
    """
    return norm.cdf(np.asarray(spread_line, dtype=float) / sigma)


def prob_to_american(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fav = -100.0 * p / (1.0 - p)
        dog = 100.0 * (1.0 - p) / p
    return np.where(p >= 0.5, fav, dog)


def cover_prob(pred_margin, spread_line, sigma: float = 13.45):
    """P(home covers): P(margin > spread_line) with margin ~ N(pred_margin, sigma)."""
    return 1.0 - norm.cdf(
        (np.asarray(spread_line, dtype=float) - np.asarray(pred_margin, dtype=float)) / sigma
    )
