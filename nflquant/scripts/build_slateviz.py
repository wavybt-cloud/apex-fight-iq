"""Build the data files behind the published slate artifact.

Writes two JS files the artifact page loads verbatim:
  slateviz.json  -> `const SLATE={...}`   game sims, distributions, edge board
  propsdata.js   -> `const PROPS={...};const PROP_META={...}`

This lives in the repo on purpose. It was originally an ad-hoc scratchpad
script and was lost when the container recycled, which left the artifact
stuck on a stale week with no way to regenerate it.

Usage: python3 scripts/build_slateviz.py [--sims 100000] [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nflquant.config import PACKAGE_ROOT, load_config
from nflquant.market.lines import american_to_decimal
from nflquant.simulation.possession import simulate_game, summarize_sims

BINS = list(range(0, 60, 3))       # score axis for the joint heatmap
MARGIN_CLIP = 35                   # histogram is clipped, tails folded into the ends


def _latest_week_dir() -> Path:
    paths = sorted((PACKAGE_ROOT / "reports_out").glob("*_week*/predictions.json"))
    if not paths:
        raise SystemExit("no predictions.json found - run predict_week.py first")
    return paths[-1].parent


def _edge_board(g: dict, s: dict) -> dict:
    """Best single candidate for the game, priced the way the desk prices it."""
    cands = []
    if g.get("spread_line") is not None and s.get("p_home_cover") is not None:
        for p, lab in [(s["p_home_cover"], f"{g['home']} {-g['spread_line']:+.1f}"),
                       (1 - s["p_home_cover"], f"{g['away']} {g['spread_line']:+.1f}")]:
            cands.append({"lab": lab, "p": p, "ev": p * (100 / 110) - (1 - p)})
    if g.get("total_line") is not None and s.get("p_over") is not None:
        for p, lab in [(s["p_over"], f"Over {g['total_line']}"),
                       (1 - s["p_over"], f"Under {g['total_line']}")]:
            cands.append({"lab": lab, "p": p, "ev": p * (100 / 110) - (1 - p)})
    for price, p, team in [(g.get("home_ml"), g["p_home"], g["home"]),
                           (g.get("away_ml"), 1 - g["p_home"], g["away"])]:
        if price is None:
            continue
        d = float(american_to_decimal(price))
        cands.append({"lab": f"{team} ML {int(price):+d}", "p": p,
                      "ev": p * (d - 1) - (1 - p)})
    best = max(cands, key=lambda c: c["ev"])
    return {"lab": best["lab"], "p": round(best["p"], 4), "ev": round(best["ev"], 4)}


def _hist(values: np.ndarray, lo: int, hi: int) -> dict:
    """Percent of sims at each integer value, ends absorbing the tails."""
    v = np.clip(values, lo, hi).astype(int)
    counts = np.bincount(v - lo, minlength=hi - lo + 1)
    pct = 100.0 * counts / len(v)
    return {str(lo + i): round(float(p), 3) for i, p in enumerate(pct) if p > 0}


def _heat(away: np.ndarray, home: np.ndarray) -> list:
    """Joint score density on the BINS grid, as percentages."""
    h, _, _ = np.histogram2d(away, home, bins=[BINS + [200], BINS + [200]])
    return [[round(float(x), 3) for x in row] for row in (100.0 * h / h.sum())]


def build(sims_per_game: int, out_dir: Path) -> None:
    cfg = load_config()
    week_dir = _latest_week_dir()
    preds = json.load(open(week_dir / "predictions.json"))
    sim_cfg = cfg["simulation"]

    games = []
    for i, g in enumerate(sorted(preds, key=lambda x: (x["gameday"], x["away"]))):
        exp_h = (g["total"] + g["margin"]) / 2
        exp_a = (g["total"] - g["margin"]) / 2
        sims = simulate_game(
            exp_h, exp_a, n_sims=sims_per_game,
            drives_mean=sim_cfg["drives_mean"], drives_sd=sim_cfg["drives_sd"],
            param_sd_pts=sim_cfg["param_sd_pts"],
            drive_var_shrink=sim_cfg["drive_var_shrink"],
            endgame_tie_prob=sim_cfg["endgame_tie_prob"],
            walkoff_prob=sim_cfg["walkoff_prob"],
            consolation_prob=sim_cfg["consolation_prob"],
            seed=sim_cfg["seed"] + 1000 + i,
        )
        s = summarize_sims(sims, g.get("spread_line"), g.get("total_line"))
        hm, aw = sims["home"], sims["away"]
        games.append({
            "id": g["game_id"], "away": g["away"], "home": g["home"], "date": g["gameday"],
            "p_home": round(g["p_home"], 4), "conf": g["confidence"][0],
            "spread": g.get("spread_line"), "total_line": g.get("total_line"),
            "ml_a": g.get("away_ml"), "ml_h": g.get("home_ml"),
            "margin": round(g["margin"], 2), "total": round(g["total"], 2),
            "mean_h": round(float(hm.mean()), 1), "mean_a": round(float(aw.mean()), 1),
            "cover_h": round(s["p_home_cover"], 4) if s.get("p_home_cover") is not None else None,
            "over": round(s["p_over"], 4) if s.get("p_over") is not None else None,
            "ot": round(s["p_ot"], 4), "one": round(s["p_one_score"], 4),
            "blow": round(s["p_blowout_17"], 4),
            "h_pct": {str(q): int(np.percentile(hm, q)) for q in (10, 25, 50, 75, 90)},
            "a_pct": {str(q): int(np.percentile(aw, q)) for q in (10, 25, 50, 75, 90)},
            "best": _edge_board(g, s),
            "mh": _hist(hm - aw, -MARGIN_CLIP, MARGIN_CLIP),
            "th": _hist(hm + aw, 15, 80),
            "heat": _heat(aw, hm),
        })
        print(f"  {g['away']}@{g['home']}: {sims_per_game} sims")

    slate = {
        "n_per_game": sims_per_game,
        "season": int(week_dir.name.split("_")[0]),
        "week": int(preds[0]["week"]),
        "generated": __import__("pandas").Timestamp.now().isoformat()[:16],
        "bins": BINS,
        "finals": _finals(),
        "games": games,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "slateviz.json").write_text("const SLATE=" + json.dumps(slate))
    print(f"wrote {out_dir/'slateviz.json'}: {len(games)} games")

    props_path = week_dir / "props.json"
    if props_path.exists():
        props = json.load(open(props_path))
        # props.json stores a list of games each holding a `teams` list; the page
        # wants {game_id: {away: {...}, home: {...}}}, matched by team code rather
        # than list position so a reordering upstream cannot silently swap sides.
        by_game = {}
        for g in props["games"]:
            sides = {}
            for t in g["teams"]:
                sides["home" if t["team"] == g["home"] else "away"] = t
            if set(sides) != {"home", "away"}:
                raise SystemExit(f"props.json: cannot match sides for {g['id']}")
            by_game[g["id"]] = sides
        meta = {"spreads": props.get("spreads", {}), "platt": props.get("platt"),
                "generated": props.get("generated", "")}
        (out_dir / "propsdata.js").write_text(
            "const PROPS=" + json.dumps(by_game) + ";const PROP_META=" + json.dumps(meta) + ";")
        print(f"wrote {out_dir/'propsdata.js'}: {len(by_game)} games")
    else:
        print("no props.json for this week yet - skipping propsdata.js")


def _finals() -> list:
    """Graded-week ribbon shown above the slate: the running live record.

    Kept in viz_finals.json rather than derived, because grading a week needs
    the settled prices actually available at bet time, which the reports do not
    store. Update it when a week closes; an absent file just hides the ribbon.
    """
    p = PACKAGE_ROOT / "viz_finals.json"
    return json.load(open(p)) if p.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=100000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else PACKAGE_ROOT / "viz_out"
    build(a.sims, out)
