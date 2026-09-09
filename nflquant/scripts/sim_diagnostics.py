"""Phase 8: simulate the validation window and check realism vs actual NFL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from nflquant.config import cache_dir, load_config
from nflquant.data.ingest import load_games
from nflquant.simulation.diagnostics import realism_report, simulate_window
from backtest_v2 import load_enriched


def build_sim_inputs(cfg, seasons: list[int]) -> pd.DataFrame:
    """Join ensemble margin/total projections with drive-mix features."""
    bundle = joblib.load(cache_dir(cfg) / "ensemble_bundle.joblib")
    feats = load_enriched(cfg)
    games = load_games(cfg)[["game_id", "overtime"]]

    tag = "val"
    oos = {m: pd.read_parquet(cache_dir(cfg) / f"oos_{tag}_{m}.parquet")
           for m in set(bundle["margin_members_pure"] + bundle["total_members"])}
    vm = pd.DataFrame({m: oos[m].set_index("game_id")["margin"] for m in bundle["margin_members_pure"]}).dropna()
    vt = pd.DataFrame({m: oos[m].set_index("game_id")["total"] for m in bundle["total_members"]}).dropna()
    margin_pred = pd.Series(bundle["blend_pure"].predict(vm), index=vm.index, name="margin_pred")
    total_pred = pd.Series(bundle["blend_total"].predict(vt), index=vt.index, name="total_pred")

    f = feats.set_index("game_id")
    df = pd.concat([margin_pred, total_pred], axis=1).dropna()
    df = df.join(f[["season", "playoff", "result", "total", "spread_line", "total_line",
                    "home_td_rate", "home_fg_rate", "home_to_rate",
                    "away_td_rate", "away_fg_rate", "away_to_rate"]])
    df = df[df.season.isin(seasons) & df.result.notna()]
    df = df.rename(columns={
        "total": "total_actual",
        "home_td_rate": "h_td_rate", "home_fg_rate": "h_fg_rate", "home_to_rate": "h_to_rate",
        "away_td_rate": "a_td_rate", "away_fg_rate": "a_fg_rate", "away_to_rate": "a_to_rate",
    }).reset_index()
    for c in ["h_td_rate", "a_td_rate"]:
        df[c] = df[c].fillna(0.24)
    for c in ["h_fg_rate", "a_fg_rate"]:
        df[c] = df[c].fillna(0.15)
    for c in ["h_to_rate", "a_to_rate"]:
        df[c] = df[c].fillna(0.11)
    return df.merge(games, on="game_id", how="left")


def main():
    cfg = load_config()
    val = list(range(cfg["seasons"]["validation"][0], cfg["seasons"]["validation"][1] + 1))
    df = build_sim_inputs(cfg, val)
    print(f"simulating {len(df)} validation games x 4000 sims ...")
    combos = [tuple(float(x) for x in c.split(",")) for c in sys.argv[1:]] or [(2.4, 0.72, 0.30)]
    for sd, shrink, tie_p in combos:
        stats = simulate_window(df, n_sims=4000, param_sd_pts=sd,
                                drive_var_shrink=shrink, endgame_tie_prob=tie_p)
        rep = realism_report(stats, df)
        print(f"\n=== param_sd={sd} shrink={shrink} endgame_tie={tie_p} ===")
        print(rep.round(4).to_string(index=False))
        # probability quality of simulated win probs
        from nflquant.evaluation.metrics import brier
        y = (df.result > 0).astype(float).values
        print("sim p_home brier:", round(brier(y, stats.p_home.values), 4))


if __name__ == "__main__":
    main()
