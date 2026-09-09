"""Per-game professional report: projection, distribution, drivers, uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# human labels for the strongest feature drivers
DRIVER_LABELS = {
    "d_adj_off_epa": "opponent-adjusted offensive EPA",
    "d_adj_def_epa": "opponent-adjusted defensive EPA",
    "d_fg_rate": "drive FG rate",
    "d_n_drives": "pace (drives/game)",
    "d_off_plays": "offensive volume",
    "d_off_explosive": "explosive plays",
    "d_def_explosive": "explosive plays allowed",
    "d_def_success": "defensive success rate",
    "d_off_epa": "offensive EPA/play",
    "d_def_epa": "defensive EPA allowed",
    "d_off_pass_epa": "passing efficiency",
    "d_off_rush_epa": "rushing efficiency",
    "d_off_success": "offensive success rate",
    "d_off_explosive": "explosive-play rate",
    "d_def_pass_epa": "pass defense",
    "d_def_rush_epa": "run defense",
    "d_off_third_conv": "third-down conversion",
    "d_off_sack_rate": "sack avoidance / protection",
    "d_def_sack_rate": "pass rush",
    "d_off_cpoe": "completion % over expected",
    "d_td_rate": "drive TD rate",
    "d_to_rate": "drive turnover rate",
    "d_off_turnovers": "giveaways",
    "elo_diff_eff": "Elo rating gap (incl. HFA, rest, QB)",
    "d_qb_points": "QB rating gap",
    "rest_diff": "rest advantage",
    "div_game": "divisional familiarity",
    "wind": "wind",
    "temp": "temperature",
    "tz_travel": "time-zone travel",
    "home_qb_change": "home QB change",
    "away_qb_change": "away QB change",
}


# columns that are constant across a slate (or pure context) - true for every
# game that week, so meaningless as a per-game "driver"
DRIVER_EXCLUDE = {"week", "playoff", "neutral", "surface_grass", "dome",
                  "home_n_season", "away_n_season", "temp"}


def key_drivers(row: pd.Series, ridge_model, cols: list[str], top: int = 5):
    """Standardized ridge contributions: coefficient x scaled feature value.

    Linear attribution is exact for the ridge member (its contribution to the
    ensemble margin) - no approximation needed, unlike SHAP on the GBM.
    """
    pipe = ridge_model.pipe_
    imp, sc, rg = pipe.named_steps["imp"], pipe.named_steps["sc"], pipe.named_steps["rg"]
    x = imp.transform(pd.DataFrame([row[cols]]))
    xs = sc.transform(x)[0]
    contrib = xs * rg.coef_
    order = np.argsort(-np.abs(contrib))
    out = []
    for i in order:
        if cols[i] in DRIVER_EXCLUDE:
            continue
        label = DRIVER_LABELS.get(cols[i], cols[i])
        out.append((label, float(contrib[i])))
        if len(out) >= top:
            break
    return out


def confidence_grade(member_probs: dict[str, float], week: int,
                     qb_uncertain: bool, sd_margin: float) -> tuple[str, list[str]]:
    """Grade prediction confidence; VERY HIGH is intentionally rare."""
    probs = np.array(list(member_probs.values()))
    disagreement = probs.max() - probs.min()
    notes = []
    score = 2  # MODERATE
    if disagreement > 0.10:
        score -= 1; notes.append(f"models disagree by {disagreement:.0%}")
    elif disagreement < 0.045:
        score += 1
    if week <= 4:
        score -= 1; notes.append("early season: ratings lean on priors")
    if qb_uncertain:
        score -= 1; notes.append("QB situation unsettled")
    if sd_margin > 13.8:
        notes.append("wide simulated outcome range")
    edge = abs(np.mean(probs) - 0.5)
    if edge > 0.28 and disagreement < 0.06:
        score += 1
    grades = ["VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH"]
    return grades[int(np.clip(score, 0, 4))], notes


def format_game_report(g: dict) -> str:
    """g: dict with all computed fields (see predict_week.py)."""
    away, home = TEAM_NAMES.get(g["away"], g["away"]), TEAM_NAMES.get(g["home"], g["home"])
    s = g["sim"]
    fav = home if s["mean_margin"] >= 0 else away
    p_fav = g["p_home"] if fav == home else 1 - g["p_home"]
    L = []
    L.append("=" * 64)
    L.append(f"GAME  {away} @ {home}   ({g['gameday']}, week {g['week']})")
    L.append("=" * 64)
    L.append("")
    L.append("MODEL PROJECTION")
    L.append(f"  {home}: {s['mean_home']:.1f}    {away}: {s['mean_away']:.1f}"
             f"    (margin {s['mean_margin']:+.1f}, total {s['mean_total']:.1f})")
    L.append("")
    L.append("WIN PROBABILITY (calibrated ensemble)")
    L.append(f"  {home}: {g['p_home']:.1%}    {away}: {1-g['p_home']:.1%}")
    if g.get("p_home_mkt") is not None:
        L.append(f"  market-aware mode: {home} {g['p_home_mkt']:.1%}")
    L.append("")
    L.append(f"SIMULATION ({s['n_sims']:,} possession-level sims)")
    L.append(f"  median score:   {home} {s['median_home']:.0f} - {away} {s['median_away']:.0f}")
    hp, ap = s["home_pctiles"], s["away_pctiles"]
    L.append(f"  80% range:      {home} {hp[10]:.0f}-{hp[90]:.0f}, {away} {ap[10]:.0f}-{ap[90]:.0f}")
    L.append(f"  one-score game: {s['p_one_score']:.0%}    overtime: {s['p_ot']:.1%}"
             f"    blowout(17+): {s['p_blowout_17']:.0%}")
    L.append(f"  upset ({away if fav==home else home} wins): {s['p_upset']:.0%}")
    if "p_home_cover" in s:
        line = g.get("spread_line")
        L.append(f"  vs spread ({home} {-line:+.1f}): cover {s['p_home_cover']:.1%}")
    if "p_over" in s:
        L.append(f"  vs total ({g.get('total_line'):.1f}): over {s['p_over']:.1%}")
    L.append("")
    L.append("KEY FACTORS (points toward " + home + ")")
    for label, val in g["drivers"]:
        L.append(f"  {val:+.2f}  {label}")
    L.append("")
    L.append("MODEL AGREEMENT (home win prob)")
    for m, p in g["members"].items():
        L.append(f"  {m:12s} {p:.1%}")
    L.append(f"  {'ensemble':12s} {g['p_home']:.1%}")
    L.append("")
    grade, notes = g["confidence"]
    L.append(f"CONFIDENCE: {grade}" + (f"   ({'; '.join(notes)})" if notes else ""))
    L.append("")
    return "\n".join(L)
