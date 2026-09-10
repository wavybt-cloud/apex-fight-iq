"""Build and cache the full game feature matrix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflquant.config import cache_dir, load_config
from nflquant.data.ingest import load_games, load_pbp
from nflquant.features.build import build_features


def main(force: bool = False):
    cfg = load_config()
    out = cache_dir(cfg) / "features.parquet"
    if out.exists() and not force:
        print(f"features cache exists: {out} (use --force to rebuild)")
        return
    games = load_games(cfg)
    pbp = load_pbp(cfg)
    feats = build_features(
        games, pbp,
        ewma_halflife=cfg["features"]["ewma_halflife_games"],
        halflife_def=cfg["features"].get("halflife_def"),
        halflife_to=cfg["features"].get("halflife_to"),
        gt_band=tuple(cfg["features"]["garbage_time_wp"]),
    )
    feats.to_parquet(out, index=False)
    print(f"wrote {out}: {len(feats)} games x {feats.shape[1]} cols")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
