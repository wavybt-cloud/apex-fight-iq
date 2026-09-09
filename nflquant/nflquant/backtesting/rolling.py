"""Rolling-origin backtesting.

For each evaluation season Y, models are trained on all seasons strictly
before Y and predict every game of season Y. Standing at the start of Y,
nothing later exists. Sequential models (Elo) are inherently walk-forward
and just get their per-game outputs joined in.

Season groups (from config):
    burn-in   : first PBP seasons, never evaluated
    development : rolled seasons before validation, for model iteration
    validation  : ensemble weights + calibration fitted here
    test        : untouched until final evaluation
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from nflquant.config import PACKAGE_ROOT
from nflquant.evaluation.metrics import summarize
from nflquant.logging_utils import get_logger

log = get_logger(__name__)


def rolling_predictions(
    feats: pd.DataFrame,
    model_factory,
    seasons: list[int],
    min_train_seasons: int = 3,
) -> pd.DataFrame:
    """Concatenated out-of-sample predictions for the given seasons.

    model_factory() -> fresh model instance per season (prevents state carryover).
    """
    outs = []
    for season in seasons:
        train = feats[(feats.season < season) & feats.result.notna()]
        if train.season.nunique() < min_train_seasons:
            continue
        test = feats[feats.season == season]
        if not len(test):
            continue
        model = model_factory()
        model.fit(train)
        pred = model.predict(test)
        pred = pred.assign(
            game_id=test.game_id.values, season=season,
            home_win=test.home_win.values, result=test.result.values,
            total_actual=test.total.values, spread_line=test.spread_line.values,
            total_line=test.total_line.values,
        )
        outs.append(pred)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def evaluate_models(
    feats: pd.DataFrame,
    factories: dict[str, callable],
    seasons: list[int],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run every model over the seasons; return metric table + raw predictions."""
    metrics, preds = [], {}
    for name, fac in factories.items():
        t0 = time.time()
        p = rolling_predictions(feats, fac, seasons)
        preds[name] = p
        row = {"model": name} | summarize(p)
        row["fit_s"] = round(time.time() - t0, 1)
        metrics.append(row)
        log.info("%-12s brier=%.4f logloss=%.4f acc=%.3f", name, row["brier"], row["logloss"], row["acc"])
    return pd.DataFrame(metrics), preds


def record_run(tag: str, cfg: dict, metrics: pd.DataFrame, extra: dict | None = None) -> Path:
    """Append run record (config hash, git commit, metrics) to runs/ registry."""
    runs = PACKAGE_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    rec = {
        "tag": tag,
        "when": pd.Timestamp.now().isoformat(),
        "config_hash": cfg.get("_meta", {}).get("config_hash"),
        "git_commit": cfg.get("_meta", {}).get("git_commit"),
        "metrics": metrics.to_dict("records"),
        "extra": extra or {},
    }
    path = runs / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_{tag}.json"
    path.write_text(json.dumps(rec, indent=2, default=str))
    return path
