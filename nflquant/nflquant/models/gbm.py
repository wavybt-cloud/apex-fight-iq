"""Gradient-boosted models (LightGBM): win probability, margin, total.

Early stopping uses a CHRONOLOGICAL tail of the training data as the eval
set — never a random shuffle — so the stopping decision mimics deployment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import norm

DEFAULT_PARAMS = dict(
    num_leaves=31, learning_rate=0.03, n_estimators=600,
    min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=7, verbose=-1,
)


def _chrono_split(t: pd.DataFrame, frac: float = 0.85):
    t = t.sort_values("gameday")
    n = int(len(t) * frac)
    return t.iloc[:n], t.iloc[n:]


class GBMModel:
    """Joint wrapper: classifier for win, regressors for margin and total."""

    def __init__(self, cols: list[str], name: str = "gbm", params: dict | None = None,
                 early_stopping_rounds: int = 60):
        self.cols, self.name = cols, name
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.esr = early_stopping_rounds

    def _fit_one(self, model, tr, ev, target):
        model.fit(
            tr[self.cols], tr[target],
            eval_X=ev[self.cols], eval_y=ev[target],
            callbacks=[early_stopping(self.esr, verbose=False), log_evaluation(0)],
        )
        return model

    def fit(self, train: pd.DataFrame):
        t = train[train.result.notna()].copy()
        t["win"] = (t.result > 0).astype(int)
        tr, ev = _chrono_split(t)
        self.clf_ = self._fit_one(LGBMClassifier(**self.params), tr, ev, "win")
        self.margin_ = self._fit_one(LGBMRegressor(**self.params), tr, ev, "result")
        tt_tr, tt_ev = tr[tr.total.notna()], ev[ev.total.notna()]
        self.total_ = self._fit_one(LGBMRegressor(**self.params), tt_tr, tt_ev, "total")
        resid = t.result - self.margin_.predict(t[self.cols])
        self.sigma_ = float(max(resid.std(), 9.0))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        p_clf = self.clf_.predict_proba(df[self.cols])[:, 1]
        m = self.margin_.predict(df[self.cols])
        p_margin = 1.0 - norm.cdf(-m / self.sigma_)
        # average the two probability views; they disagree mostly in the tails
        p = 0.5 * p_clf + 0.5 * p_margin
        return pd.DataFrame(
            {"p_home": p, "margin": m, "total": self.total_.predict(df[self.cols])},
            index=df.index,
        )

    def importances(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.cols,
            "win": self.clf_.feature_importances_,
            "margin": self.margin_.feature_importances_,
        }).sort_values("margin", ascending=False)
