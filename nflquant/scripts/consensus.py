"""Gameday Consensus: on days with games, take the latest snapshot from BOTH
desks and apply written approval rules to produce the plays-of-the-day report
with stakes. No qualifying play => an explicit PASS report, never a forced pick.

Approval rules (the whole point is that they are written down):
  GAME PLAYS
    A1. game kicks off today (engine's local calendar)
    A2. EV >= +10% at the snapshot price
    A3. calibrated probability in [0.35, 0.80] (outside = model artifact risk)
    A4. member disagreement <= 25 percentage points
    A5. totals plays additionally need EV >= +15% (totals are the engine's
        weakest market) and are capped at half stake
  PROP PLAYS (conditional - the desk cannot see your book's price)
    P1. anytime TD probability >= 25% after calibration
    P2. published as a TRIGGER: bet only if the book pays >= fair + 20 cents
  STAKES
    quarter-Kelly at snapshot price, bankroll from NFLQ_BANKROLL (default
    $1000), cap $50; x0.5 when the game confidence is VERY LOW; totals x0.5.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nflquant.config import PACKAGE_ROOT
from nflquant.market.lines import american_to_decimal

DESK_DIR = PACKAGE_ROOT / "desks"
BANKROLL = float(os.environ.get("NFLQ_BANKROLL", 1000))
KELLY_FRAC, MAX_BET = 0.25, 50.0


def last_snapshot(name):
    p = DESK_DIR / name
    if not p.exists():
        return None
    lines = p.read_text().strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def kelly(p, dec, mult=1.0):
    b = dec - 1
    f = (b * p - (1 - p)) / b
    return round(min(max(f, 0) * KELLY_FRAC * BANKROLL * mult, MAX_BET * mult), 0)


def trigger_price(p):
    """fair American + ~20 cents."""
    fair = 100 * (1 - p) / p if p < 0.5 else -100 * p / (1 - p)
    return f"+{round(fair + 25)}" if p < 0.5 else f"{round(fair + 25):+d}"


def main():
    today = pd.Timestamp.now().date().isoformat()
    ml = last_snapshot("ml_log.jsonl")
    pr = last_snapshot("props_log.jsonl")
    if not ml or not pr:
        print("consensus: desks have not both reported yet"); return

    todays = [r for r in ml["board"] if r["date"] == today]
    lines = [f"# Gameday consensus — {today}",
             f"desks: ML @ {ml['ts']} · props @ {pr['ts']} · bankroll ${BANKROLL:.0f}", ""]
    if not todays:
        lines.append("No games today. Desks keep running; nothing to approve.")
    approved = []
    for r in todays:
        dis = max(r["members"].values()) - min(r["members"].values())
        vlow = r["confidence"] == "VERY LOW"
        for c in r["cands"]:
            ev, p = c["ev"], c["p"]
            if ev < 0.10 or not (0.35 <= p <= 0.80) or dis > 0.25:
                continue
            if c["type"] == "total" and ev < 0.15:
                continue
            dec = float(american_to_decimal(c.get("price", -110)))
            mult = (0.5 if vlow else 1.0) * (0.5 if c["type"] == "total" else 1.0)
            stake = kelly(p, dec, mult)
            if stake < 5:
                continue
            approved.append((ev, r, c, stake, dis))
    approved.sort(reverse=True, key=lambda x: x[0])

    if todays:
        if approved:
            lines.append("## Approved game plays")
            for ev, r, c, stake, dis in approved:
                lines.append(f"- **{c['label']}** ({r['game']}) — model {c['p']:.1%}, "
                             f"EV {ev:+.1%}, stake **${stake:.0f}**"
                             f" · confidence {r['confidence']}, member spread {dis:.0%}")
        else:
            lines.append("## Approved game plays\n- **PASS** — nothing met the rules today. "
                         "Passing is a position.")
        # conditional props for today's games
        todays_games = {r["game"] for r in todays}
        lines.append("\n## Conditional prop plays (bet ONLY at trigger or better)")
        any_prop = False
        for g in pr["games"]:
            key = f"{g['away']}@{g['home']}"
            if key not in todays_games:
                continue
            for t in g["teams"]:
                for row in t["td"]:
                    if row["p"] >= 0.25:
                        any_prop = True
                        lines.append(f"- {row['name']} ({t['team']}) anytime TD — fair "
                                     f"{row['odds']} ({row['p']:.0%}); trigger "
                                     f"**{trigger_price(row['p'])} or longer**")
        if not any_prop:
            lines.append("- none met the probability floor")
        lines.append("\n*Rules A1–A5/P1–P2 in scripts/consensus.py. Fair prices are "
                     "no-vig; the trigger gap is the entire edge. The engine's "
                     "backtest says game edges vs closing lines are ~breakeven — "
                     "stakes are sized so variance can't hurt you while CLV data "
                     "accumulates.*")
    report = "\n".join(lines)
    out = PACKAGE_ROOT / "reports_out" / f"consensus_{today}.md"
    out.write_text(report)
    print(report)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
