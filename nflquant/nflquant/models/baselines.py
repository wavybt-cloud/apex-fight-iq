"""Baseline models and the shared model interface.

Every model implements:
    fit(train: DataFrame) -> self
    predict(df: DataFrame) -> DataFrame with columns [p_home, margin] (+ total when supported)

Mandatory baselines: home-team, market-implied, Elo (see elo.py), plus simple
regression models. Anything fancier must beat these out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nflquant.market.lines import novig_home_prob, spread_to_home_prob


class HomeBaseline:
    """Predicts the trailing home win rate for every game."""
    name = "home"

    def fit(self, train: pd.DataFrame):
        played = train[train.home_win.notna() & (train.neutral == 0)]
        self.p_ = float((played.home_win > 0.5).mean())
        self.margin_ = float(played.result.mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        p = np.where(df["neutral"] == 1, 0.5, self.p_)
        m = np.where(df["neutral"] == 1, 0.0, self.margin_)
        return pd.DataFrame({"p_home": p, "margin": m}, index=df.index)


class MarketBaseline:
    """De-vigged moneyline probability; falls back to the spread when ML missing."""
    name = "market"

    def __init__(self, sigma: float = 13.2):
        self.sigma = sigma

    def fit(self, train: pd.DataFrame):
        resid = (train.result - train.spread_line).dropna()
        if len(resid) > 500:
            self.sigma = float(resid.std())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        p_ml = novig_home_prob(df["away_moneyline"].values, df["home_moneyline"].values)
        p_sp = spread_to_home_prob(df["spread_line"].values, sigma=self.sigma)
        p = np.where(np.isfinite(p_ml), p_ml, p_sp)
        return pd.DataFrame(
            {"p_home": p, "margin": df["spread_line"].values,
             "total": df["total_line"].values},
            index=df.index,
        )


def _season_weights(t: pd.DataFrame, halflife: float | None) -> np.ndarray | None:
    """Exponential decay by season age: recent football counts more."""
    if not halflife:
        return None
    age = t["season"].max() - t["season"]
    return np.power(0.5, age / halflife).values


class LogisticModel:
    """L2 logistic regression on the feature set (pure or market mode)."""
    def __init__(self, cols: list[str], name: str = "logit", C: float = 0.3,
                 season_halflife: float | None = None):
        self.cols, self.name, self.C = cols, name, C
        self.season_halflife = season_halflife

    def fit(self, train: pd.DataFrame):
        t = train[train.home_win.isin([0.0, 1.0])]
        w = _season_weights(t, self.season_halflife)
        self.pipe_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=2000)),
        ]).fit(t[self.cols], t.home_win.astype(int), lr__sample_weight=w)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.pipe_.predict_proba(df[self.cols])[:, 1]
        return pd.DataFrame({"p_home": p, "margin": np.nan}, index=df.index)


class RidgeMarginModel:
    """Ridge regression on margin; win prob via Normal(margin, resid sigma).
    Also fits a total model on the same features plus pace/efficiency levels."""
    def __init__(self, cols: list[str], name: str = "ridge", alpha: float = 10.0,
                 season_halflife: float | None = None):
        self.cols, self.name, self.alpha = cols, name, alpha
        self.season_halflife = season_halflife

    def fit(self, train: pd.DataFrame):
        t = train[train.result.notna()]
        w = _season_weights(t, self.season_halflife)
        self.pipe_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("rg", Ridge(alpha=self.alpha)),
        ]).fit(t[self.cols], t.result, rg__sample_weight=w)
        resid = t.result - self.pipe_.predict(t[self.cols])
        self.sigma_ = float(max(resid.std(), 9.0))
        tt = t[t.total.notna()]
        wt = _season_weights(tt, self.season_halflife)
        self.total_pipe_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("rg", Ridge(alpha=self.alpha)),
        ]).fit(tt[self.cols], tt.total, rg__sample_weight=wt)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        m = self.pipe_.predict(df[self.cols])
        p = 1.0 - norm.cdf(-m / self.sigma_)
        t = self.total_pipe_.predict(df[self.cols])
        return pd.DataFrame({"p_home": p, "margin": m, "total": t}, index=df.index)
