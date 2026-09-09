"""Probability calibration: Platt, isotonic, and beta calibration.

Calibrators are fit ONLY on validation-window out-of-sample predictions and
applied unchanged afterwards. `pick_calibrator` selects by validation Brier.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


def _logit(p, eps=1e-5):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattCalibrator:
    name = "platt"

    def fit(self, p, y):
        s = _logit(p)

        def nll(ab):
            a, b = ab
            q = 1 / (1 + np.exp(-(a * s + b)))
            q = np.clip(q, 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
        self.a_, self.b_ = r.x
        return self

    def transform(self, p):
        return 1 / (1 + np.exp(-(self.a_ * _logit(p) + self.b_)))


class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        self.iso_.fit(p, y)
        return self

    def transform(self, p):
        return self.iso_.predict(p)


class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic on [ln p, ln(1-p)]."""
    name = "beta"

    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
        X1, X2 = np.log(p), -np.log(1 - p)

        def nll(abc):
            a, b, c = abc
            q = 1 / (1 + np.exp(-(a * X1 + b * X2 + c)))
            q = np.clip(q, 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        r = minimize(nll, x0=[1.0, 1.0, 0.0], method="Nelder-Mead")
        self.abc_ = r.x
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
        a, b, c = self.abc_
        return 1 / (1 + np.exp(-(a * np.log(p) + b * (-np.log(1 - p)) + c)))


class IdentityCalibrator:
    name = "none"

    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p, dtype=float)


def pick_calibrator(p_val, y_val, min_isotonic_n: int = 3000):
    """Fit calibrators on validation data; return (best, report) by Brier.

    Isotonic regression is only eligible with a large sample: its step
    function is the most flexible option and reliably wins IN-SAMPLE on
    small validation sets by fitting noise. Parametric calibrators (Platt,
    beta) are the safe default at NFL sample sizes.
    """
    m = np.isfinite(p_val) & np.isfinite(y_val)
    p_val, y_val = np.asarray(p_val)[m], np.asarray(y_val)[m]
    candidates = [IdentityCalibrator(), PlattCalibrator(), BetaCalibrator()]
    if len(p_val) >= min_isotonic_n:
        candidates.append(IsotonicCalibrator())
    report, best, best_brier = {}, None, np.inf
    for cal in candidates:
        cal.fit(p_val, y_val)
        b = float(np.mean((cal.transform(p_val) - y_val) ** 2))
        report[cal.name] = round(b, 5)
        if b < best_brier:
            best, best_brier = cal, b
    return best, report
