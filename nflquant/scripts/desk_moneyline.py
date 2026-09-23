"""Moneyline Desk: refresh data, re-run the game engine + sims, log the edge
board. Designed to be fired on a schedule; append-only log gives the desk a
history of how every edge moved between runs.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nflquant.config import PACKAGE_ROOT
from nflquant.market.lines import american_to_decimal, novig_home_prob

DESK_DIR = PACKAGE_ROOT / "desks"
DESK_DIR.mkdir(exist_ok=True)


def refresh_and_predict() -> Path:
    subprocess.run(["git", "-C", "/home/user/nflverse/nfldata", "pull", "-q"],
                   capture_output=True, timeout=300)
    # force a fresh games.csv copy
    cache = PACKAGE_ROOT / "data_cache" / "games.csv"
    if cache.exists():
        cache.unlink()
    r = subprocess.run([sys.executable, str(PACKAGE_ROOT / "scripts" / "predict_week.py"),
                        "--sims", "50000"], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("predict_week failed: " + r.stderr[-800:])
    outs = sorted((PACKAGE_ROOT / "reports_out").glob("*_week*/predictions.json"))
    return outs[-1]


def edge_board(pred_path: Path) -> list[dict]:
    preds = json.load(open(pred_path))
    rows = []
    for g in preds:
        s = g["sim"]
        cands = []
        if g.get("spread_line") is not None and s.get("p_home_cover") is not None:
            for side, p, lab in [("home", s["p_home_cover"], f"{g['home']} {-g['spread_line']:+.1f}"),
                                 ("away", 1 - s["p_home_cover"], f"{g['away']} {g['spread_line']:+.1f}")]:
                cands.append({"type": "spread", "label": lab, "p": p,
                              "ev": p * (100 / 110) - (1 - p)})
        if g.get("total_line") is not None and s.get("p_over") is not None:
            for p, lab in [(s["p_over"], f"Over {g['total_line']}"),
                           (1 - s["p_over"], f"Under {g['total_line']}")]:
                cands.append({"type": "total", "label": lab, "p": p,
                              "ev": p * (100 / 110) - (1 - p)})
        rows.append({
            "game": f"{g['away']}@{g['home']}", "date": g["gameday"],
            "p_home": g["p_home"], "margin": g["margin"], "total": g["total"],
            "confidence": g["confidence"][0],
            "members": g["members"],
            "cands": sorted(cands, key=lambda c: -c["ev"])[:2],
        })
    # ML prices ride along with the prediction (written by predict_week from the
    # freshly pulled games spine). The enriched feature parquet is a training
    # cache that is not refreshed per slate - pricing off it served ten-day-old
    # moneylines and manufactured edges that did not exist.
    for g, row in zip(preds, rows):
        if g.get("home_ml") is not None and g.get("away_ml") is not None:
            am, hm = float(g["away_ml"]), float(g["home_ml"])
            for side, p, price, team in [("home", g["p_home"], hm, g["home"]),
                                         ("away", 1 - g["p_home"], am, g["away"])]:
                d = float(american_to_decimal(price))
                row["cands"].append({"type": "ml", "label": f"{team} ML {int(price):+d}",
                                     "p": p, "price": price, "ev": p * (d - 1) - (1 - p)})
            row["cands"] = sorted(row["cands"], key=lambda c: -c["ev"])[:3]
    priced = sum(1 for g in preds if g.get("home_ml") is not None)
    if priced == 0:
        print("WARNING: no moneylines in predictions.json - the ML board is "
              "spread/total only this shift", file=sys.stderr)
    return rows


def main():
    pred_path = refresh_and_predict()
    board = edge_board(pred_path)
    snap = {"ts": pd.Timestamp.now().isoformat()[:19], "desk": "moneyline",
            "week_dir": pred_path.parent.name, "board": board}
    with open(DESK_DIR / "ml_log.jsonl", "a") as f:
        f.write(json.dumps(snap) + "\n")
    top = sorted((c | {"game": r["game"], "conf": r["confidence"]}
                  for r in board for c in r["cands"]), key=lambda c: -c["ev"])[:6]
    print(f"[ML desk {snap['ts']}] {pred_path.parent.name}: {len(board)} games")
    for c in top:
        print(f"  {c['label']:<18} {c['game']:<10} p={c['p']:.3f} EV={c['ev']:+.1%} ({c['conf']})")


if __name__ == "__main__":
    main()
