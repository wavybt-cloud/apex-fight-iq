"""Props-layer Platt calibration, fit on the leak-free 2025 walk-forward."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def fit_platt(p_pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p_pred, float), 1e-4, 1 - 1e-4)
    s = np.log(p / (1 - p))
    y = np.asarray(y, float)

    def nll(ab):
        a, b = ab
        q = np.clip(1 / (1 + np.exp(-(a * s + b))), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def apply_platt(p: float, a: float, b: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    s = np.log(p / (1 - p))
    return float(1 / (1 + np.exp(-(a * s + b))))
