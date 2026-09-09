"""Ensembling over out-of-sample member predictions.

Weights are NEVER hand-picked: the probability stacker is a logistic
regression on member logits fit on earlier out-of-sample predictions, and
margin/total weights come from non-negative least squares on the same. Both
are then applied unchanged to later seasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression


def _logit(p, eps=1e-5):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class ProbStacker:
    """p_ens = sigmoid(w . logit(p_members) + b), fit on OOS predictions."""

    def __init__(self, members: list[str]):
        self.members = members

    def fit(self, oos: dict[str, pd.DataFrame]):
        X, y = self._design(oos)
        self.lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        return self

    def _design(self, oos: dict[str, pd.DataFrame]):
        base = None
        cols = {}
        for m in self.members:
            df = oos[m][["game_id", "p_home", "home_win"]].dropna(subset=["p_home"])
            df = df[df.home_win.isin([0.0, 1.0])]
            cols[m] = df.set_index("game_id")["p_home"]
            base = df.set_index("game_id")["home_win"] if base is None else base
        X = pd.DataFrame(cols).dropna()
        y = base.loc[X.index].astype(int)
        return _logit(X.values), y.values

    def predict(self, member_probs: pd.DataFrame) -> np.ndarray:
        """member_probs: columns = member names, aligned rows."""
        X = _logit(member_probs[self.members].values)
        return self.lr_.predict_proba(X)[:, 1]

    def weights(self) -> dict:
        return dict(zip(self.members, self.lr_.coef_[0].round(3))) | {
            "intercept": round(float(self.lr_.intercept_[0]), 3)
        }


class MarginBlender:
    """Non-negative least squares blend of member margin (or total) predictions."""

    def __init__(self, members: list[str], target: str = "result"):
        self.members = members
        self.target = target

    def fit(self, oos: dict[str, pd.DataFrame], pred_col: str = "margin",
            target_col: str = "result"):
        cols = {
            m: oos[m].dropna(subset=[pred_col]).set_index("game_id")[pred_col]
            for m in self.members
        }
        X = pd.DataFrame(cols).dropna()
        first = self.members[0]
        y = oos[first].set_index("game_id").loc[X.index, target_col]
        keep = y.notna()
        X, y = X[keep], y[keep]
        A = np.column_stack([X.values, np.ones(len(X))])
        w, _ = nnls(A, y.values)
        self.w_, self.b_ = w[:-1], w[-1]
        # normalize toward a convex combination when weights collapse
        s = self.w_.sum()
        if s > 0:
            resid = y.values - (X.values @ self.w_ + self.b_)
            self.sigma_ = float(np.std(resid))
        else:  # degenerate: fall back to equal weights
            self.w_ = np.ones(len(self.members)) / len(self.members)
            self.b_ = 0.0
            self.sigma_ = 13.2
        return self

    def predict(self, member_preds: pd.DataFrame) -> np.ndarray:
        return member_preds[self.members].values @ self.w_ + self.b_

    def weights(self) -> dict:
        return dict(zip(self.members, np.round(self.w_, 3))) | {"intercept": round(self.b_, 2)}
