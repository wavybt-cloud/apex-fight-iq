"""Evaluation metrics: probability quality, margin accuracy, market comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] - y[m]) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    m = np.isfinite(p) & np.isfinite(y)
    pc = np.clip(p[m], eps, 1 - eps)
    return float(-np.mean(y[m] * np.log(pc) + (1 - y[m]) * np.log(1 - pc)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] > 0.5) == (y[m] > 0.5)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins=None) -> pd.DataFrame:
    """Fold to P(favorite wins) so bins cover 0.5-1.0 with decent counts."""
    if bins is None:
        bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    pf = np.where(p >= 0.5, p, 1 - p)
    yf = np.where(p >= 0.5, y, 1 - y)
    idx = np.digitize(pf, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.2f}-{bins[b+1]:.2f}",
            "n": int(sel.sum()),
            "pred": float(pf[sel].mean()),
            "actual": float(yf[sel].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins=None) -> float:
    t = calibration_table(y, p, bins)
    if not len(t):
        return np.nan
    w = t.n / t.n.sum()
    return float(np.sum(w * np.abs(t.pred - t.actual)))


def margin_errors(result: np.ndarray, pred_margin: np.ndarray) -> dict:
    m = np.isfinite(result) & np.isfinite(pred_margin)
    err = result[m] - pred_margin[m]
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def ats_record(result: np.ndarray, pred_margin: np.ndarray, spread_line: np.ndarray) -> dict:
    """Against-the-spread accuracy when the model disagrees with the market.

    Pick home when pred_margin > spread_line. Pushes excluded.
    Break-even at -110 is 52.38%.
    """
    m = np.isfinite(result) & np.isfinite(pred_margin) & np.isfinite(spread_line)
    r, pm, sl = result[m], pred_margin[m], spread_line[m]
    push = r == sl
    pick_home = pm > sl
    win = np.where(pick_home, r > sl, r < sl)
    win = win[~push]
    return {"ats_n": int(len(win)), "ats_win_pct": float(np.mean(win)) if len(win) else np.nan}


def summarize(df: pd.DataFrame, p_col: str = "p_home", margin_col: str = "margin") -> dict:
    """One-line metric summary for a prediction frame carrying targets."""
    y = df["home_win"].values.astype(float)
    decided = np.isin(y, [0.0, 1.0])
    p = df[p_col].values.astype(float)
    out = {
        "n": int(decided.sum()),
        "brier": brier(y[decided], p[decided]),
        "logloss": log_loss(y[decided], p[decided]),
        "acc": accuracy(y[decided], p[decided]),
        "ece": expected_calibration_error(y[decided], p[decided]),
    }
    if margin_col in df and df[margin_col].notna().any():
        out |= margin_errors(df["result"].values, df[margin_col].values)
        out |= ats_record(df["result"].values, df[margin_col].values, df["spread_line"].values)
    return out
