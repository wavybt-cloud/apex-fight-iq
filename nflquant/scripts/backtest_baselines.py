"""Phase 3/4: baseline model backtest over the development window.

Development seasons only — validation (ensemble/calibration) and test seasons
stay untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nflquant.backtesting.rolling import evaluate_models, record_run
from nflquant.config import cache_dir, load_config
from nflquant.data.ingest import load_games
from nflquant.features.build import feature_columns
from nflquant.models.baselines import HomeBaseline, LogisticModel, MarketBaseline, RidgeMarginModel
from nflquant.models.elo import run_elo


class EloAsModel:
    """Expose precomputed sequential Elo through the model interface."""
    name = "elo"

    def fit(self, train):
        return self

    def predict(self, df):
        return pd.DataFrame(
            {"p_home": df["elo_home_prob"].values, "margin": df["elo_margin"].values},
            index=df.index,
        )


def load_feats_with_elo(cfg) -> pd.DataFrame:
    feats = pd.read_parquet(cache_dir(cfg) / "features.parquet")
    games = load_games(cfg)
    elo = run_elo(
        games,
        qb_change_flags=feats[["game_id", "home_qb_change", "away_qb_change"]],
        **{k: v for k, v in cfg["elo"].items()},
    )
    return feats.merge(elo, on="game_id", how="left")


def main():
    cfg = load_config()
    feats = load_feats_with_elo(cfg)

    dev_seasons = list(range(2012, cfg["seasons"]["validation"][0]))
    pure_cols = feature_columns("pure") + ["elo_diff_eff"]
    market_cols = feature_columns("market") + ["elo_diff_eff"]

    factories = {
        "home": lambda: HomeBaseline(),
        "market": lambda: MarketBaseline(),
        "elo": lambda: EloAsModel(),
        "logit_pure": lambda: LogisticModel(pure_cols, "logit_pure"),
        "ridge_pure": lambda: RidgeMarginModel(pure_cols, "ridge_pure"),
        "logit_mkt": lambda: LogisticModel(market_cols, "logit_mkt"),
        "ridge_mkt": lambda: RidgeMarginModel(market_cols, "ridge_mkt"),
    }
    metrics, preds = evaluate_models(feats, factories, dev_seasons)
    print("\n=== development window", dev_seasons[0], "-", dev_seasons[-1], "===")
    print(metrics.round(4).to_string(index=False))
    path = record_run("baselines_dev", cfg, metrics)
    print("run recorded:", path)


if __name__ == "__main__":
    main()
