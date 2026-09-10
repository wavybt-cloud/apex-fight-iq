"""Props Desk: re-run the player-prop models (anytime TD, QB/RB yards) with
fresh data and log the board. Fired on a schedule, offset from the ML desk.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nflquant.config import PACKAGE_ROOT

DESK_DIR = PACKAGE_ROOT / "desks"
DESK_DIR.mkdir(exist_ok=True)


def main():
    r = subprocess.run([sys.executable, str(PACKAGE_ROOT / "scripts" / "props_week.py")],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("props_week failed: " + r.stderr[-800:])
    outs = sorted((PACKAGE_ROOT / "reports_out").glob("*_week*/props.json"))
    props = json.load(open(outs[-1]))
    snap = {"ts": pd.Timestamp.now().isoformat()[:19], "desk": "props",
            "week_dir": outs[-1].parent.name,
            "platt": props.get("platt"),
            "games": props["games"]}
    with open(DESK_DIR / "props_log.jsonl", "a") as f:
        f.write(json.dumps(snap) + "\n")
    n_td = sum(len(t["td"]) for g in props["games"] for t in g["teams"])
    print(f"[Props desk {snap['ts']}] {outs[-1].parent.name}: "
          f"{len(props['games'])} games, {n_td} TD props")
    # top TD board
    rows = [(r_["p"], r_["name"], r_["odds"], t["team"])
            for g in props["games"] for t in g["teams"] for r_ in t["td"]]
    for p, name, odds, team in sorted(rows, reverse=True)[:6]:
        print(f"  TD {name:<22} {team:<4} {p:.0%}  {odds}")


if __name__ == "__main__":
    main()
