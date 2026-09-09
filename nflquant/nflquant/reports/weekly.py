"""Weekly dashboard: games ranked along every axis that matters."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fmt(g):
    return f"{g['away']} @ {g['home']}"


def write_dashboard(games: list[dict], out_dir: Path, season: int, week: int) -> str:
    L = [f"# NFL week {week}, {season} - model dashboard", ""]
    if not games:
        return ""

    def rank(title, key, fmt, reverse=True, top=3):
        L.append(f"## {title}")
        for g in sorted(games, key=key, reverse=reverse)[:top]:
            L.append(f"- {_fmt(g)}: {fmt(g)}")
        L.append("")

    rank("Most confident winner", lambda g: max(g["p_home"], 1 - g["p_home"]),
         lambda g: f"{g['home'] if g['p_home'] >= .5 else g['away']} {max(g['p_home'], 1-g['p_home']):.0%}")
    rank("Largest projected margin", lambda g: abs(g["margin"]),
         lambda g: f"{g['margin']:+.1f} (home)")
    disagreement = lambda g: max(g["members"].values()) - min(g["members"].values())
    rank("Largest model disagreement", disagreement,
         lambda g: f"{disagreement(g):.0%} spread across members")
    rank("Lowest uncertainty", lambda g: disagreement(g),
         lambda g: f"{disagreement(g):.0%} member spread, confidence {g['confidence'][0]}",
         reverse=False)
    rank("Most likely one-score game", lambda g: g["sim"]["p_one_score"],
         lambda g: f"{g['sim']['p_one_score']:.0%}")
    rank("Highest projected total", lambda g: g["total"], lambda g: f"{g['total']:.1f}")
    rank("Lowest projected total", lambda g: g["total"], lambda g: f"{g['total']:.1f}",
         reverse=False)
    rank("Biggest upset chance", lambda g: g["sim"]["p_upset"],
         lambda g: f"{g['sim']['p_upset']:.0%}")

    with_lines = [g for g in games if g.get("spread_line") is not None]
    if with_lines:
        gap = lambda g: abs(g["margin"] - g["spread_line"])
        L.append("## Largest model-vs-market gaps (information, not a bet slip)")
        for g in sorted(with_lines, key=gap, reverse=True)[:5]:
            side = g["home"] if g["margin"] > g["spread_line"] else g["away"]
            L.append(f"- {_fmt(g)}: model {g['margin']:+.1f} vs line {g['spread_line']:+.1f} "
                     f"({gap(g):.1f} pts toward {side}); cover "
                     f"{g['sim'].get('p_home_cover', float('nan')):.0%} home")
        L.append("")
        L.append("*Backtest note: 2023-2025, edges vs closing lines returned negative"
                 " ROI at every threshold. Treat gaps as research signals.*")
        L.append("")

    text = "\n".join(L)
    (out_dir / "dashboard.md").write_text(text)
    return text
