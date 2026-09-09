"""Phase 5-6 backtest: QB model + opponent-adjusted features + GBM, dev window."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nflquant.backtesting.rolling import evaluate_models, record_run
from nflquant.config import cache_dir, load_config
from nflquant.data.ingest import load_games, load_pbp
from nflquant.features.build import feature_columns
from nflquant.models.baselines import LogisticModel, MarketBaseline, RidgeMarginModel
from nflquant.models.gbm import GBMModel
from nflquant.models.elo import run_elo
from nflquant.models.qb import run_qb_model
from nflquant.injuries.model import injury_features

QB_COLS = ["d_qb_points", "home_qb_n_eff", "away_qb_n_eff"]
INJ_COLS = ["d_inj", "home_inj", "away_inj"]
EXTRA_COLS = QB_COLS + INJ_COLS


def load_enriched(cfg) -> pd.DataFrame:
    """Features + Elo + QB model columns; cached."""
    cache = cache_dir(cfg) / "features_enriched.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    feats = pd.read_parquet(cache_dir(cfg) / "features.parquet")
    games = load_games(cfg)
    elo = run_elo(
        games,
        qb_change_flags=feats[["game_id", "home_qb_change", "away_qb_change"]],
        **{k: v for k, v in cfg["elo"].items()},
    )
    feats = feats.merge(elo, on="game_id", how="left")
    pbp = load_pbp(cfg)
    qb = run_qb_model(games, pbp)
    feats = feats.merge(qb, on="game_id", how="left")
    inj = injury_features(cfg, games)
    feats = feats.merge(inj, on="game_id", how="left")
    feats.to_parquet(cache, index=False)
    return feats


def main():
    cfg = load_config()
    feats = load_enriched(cfg)
    dev_seasons = list(range(2012, cfg["seasons"]["validation"][0]))

    pure = feature_columns("pure") + ["elo_diff_eff"] + EXTRA_COLS
    market = feature_columns("market") + ["elo_diff_eff"] + EXTRA_COLS

    factories = {
        "market": lambda: MarketBaseline(),
        "logit_pure": lambda: LogisticModel(pure, "logit_pure"),
        "ridge_pure": lambda: RidgeMarginModel(pure, "ridge_pure"),
        "gbm_pure": lambda: GBMModel(pure, "gbm_pure"),
        "logit_mkt": lambda: LogisticModel(market, "logit_mkt"),
        "ridge_mkt": lambda: RidgeMarginModel(market, "ridge_mkt"),
        "gbm_mkt": lambda: GBMModel(market, "gbm_mkt"),
    }
    metrics, preds = evaluate_models(feats, factories, dev_seasons)
    print("\n=== dev window with QB + opponent adjustment + GBM ===")
    print(metrics.round(4).to_string(index=False))
    record_run("v2_dev", cfg, metrics)

    # persist dev OOS predictions for ensemble weight fitting later
    for name, p in preds.items():
        p.to_parquet(cache_dir(cfg) / f"oos_dev_{name}.parquet", index=False)


if __name__ == "__main__":
    main()
