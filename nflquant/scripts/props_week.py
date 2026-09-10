"""Generate player props for the upcoming week: anytime TD, QB passing yards,
RB rushing yards, with fair odds — plus a walk-forward calibration check of
the TD and yardage machinery over the 2025 season before anything is shipped.

Usage: python3 scripts/props_week.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from nflquant.config import cache_dir, load_config, PACKAGE_ROOT
from nflquant.props.model import (PlayerState, anytime_td_prob, build_state,
                                  exp_offensive_tds, fit_yard_spreads,
                                  load_player_weeks, load_roster,
                                  pass_yards_dist, prob_to_american,
                                  rush_yards_dist, unit_adj)

LEAGUE_DEF_PASS = None  # filled from features
LEAGUE_DEF_RUSH = None


def team_trailing_pts(frow, side):
    td = frow.get(f"{side}_td_rate")
    fg = frow.get(f"{side}_fg_rate")
    nd = frow.get(f"{side}_n_drives")
    if any(v is None or not np.isfinite(v) for v in (td, fg, nd)):
        return 22.5
    return float(np.clip(nd * (6.95 * td + 3 * fg), 14, 33))


def eligible(s, pos):
    """Mirror the live anytime-TD filter: skill positions with real usage;
    pocket QBs excluded."""
    if pos not in ("QB", "RB", "WR", "TE", "FB") or not isinstance(s["name"], str):
        return False
    usage = (s["car"] or 0) + (s["tgt"] or 0) + ((s["att"] or 0) * 0.15)
    if usage < 2.5:
        return False
    if pos == "QB" and (s["car"] or 0) < 2.5:
        return False
    return True


def calibration_check(ps, spreads):
    """Walk 2025 weekly: predict anytime-TD prob and QB/RB yards pregame,
    grade against what happened. Returns fitted Platt (a, b) for TD probs."""
    ps25 = ps[(ps.season == 2025)]
    weeks = sorted(ps25.week.unique())
    td_pred, td_act = [], []
    qb_err, qb_cov, rb_err, rb_cov = [], [], [], []
    for wk in weeks:
        st = build_state(ps, through=(2025, wk))
        cur = ps25[ps25.week == wk]
        for r in cur.itertuples(index=False):
            s = st.p.get(r.player_id)
            if not s or s["n"] < 4 or not eligible(s, r.position):
                continue
            share = st.td_share(r.player_id)
            team_td = st.team_td.get(r.team)
            if share and team_td:
                p = anytime_td_prob(team_td, share)
                td_pred.append(p)
                td_act.append(1.0 if r.sk_tds > 0 else 0.0)
            if r.attempts >= 15 and s["att"] and s["att"] >= 15:
                d = pass_yards_dist(s["pass_y"], *spreads["pass"])
                qb_err.append(abs(r.passing_yards - s["pass_y"]))
                lo, hi = d.ppf(0.1), d.ppf(0.9)
                qb_cov.append(1.0 if lo <= r.passing_yards <= hi else 0.0)
            if r.carries >= 8 and s["car"] and s["car"] >= 8:
                d = rush_yards_dist(s["rush_y"], *spreads["rush"])
                rb_err.append(abs(r.rushing_yards - s["rush_y"]))
                lo, hi = d.ppf(0.1), d.ppf(0.9)
                rb_cov.append(1.0 if lo <= r.rushing_yards <= hi else 0.0)
    td_pred, td_act = np.array(td_pred), np.array(td_act)
    from nflquant.props.calibrate import apply_platt, fit_platt
    a, b = fit_platt(td_pred, td_act)
    td_cal = np.array([apply_platt(p, a, b) for p in td_pred])
    print(f"\n=== 2025 walk-forward calibration ({len(td_pred)} eligible player-games) ===")
    print(f"TD Platt fit: a={a:.3f} b={b:.3f}")
    print("anytime TD reliability (raw pred -> calibrated -> actual):")
    for lo, hi in [(0, .15), (.15, .3), (.3, .45), (.45, .6), (.6, 1.0)]:
        m = (td_pred >= lo) & (td_pred < hi)
        if m.sum() > 20:
            print(f"  {lo:.2f}-{hi:.2f}: raw {td_pred[m].mean():.3f}  cal {td_cal[m].mean():.3f}  actual {td_act[m].mean():.3f}  (n={m.sum()})")
    print(f"QB pass yards: MAE {np.mean(qb_err):.1f}  | 80% interval coverage {np.mean(qb_cov):.1%} (n={len(qb_err)})")
    print(f"RB rush yards: MAE {np.mean(rb_err):.1f}  | 80% interval coverage {np.mean(rb_cov):.1%} (n={len(rb_err)})")
    return a, b


def main():
    cfg = load_config()
    seasons = [2023, 2024, 2025, 2026]
    ps = load_player_weeks(cfg, seasons)
    ps = ps[ps.season_type.isin(["REG", "POST"])]
    print("player-weeks:", len(ps), "| seasons:", sorted(ps.season.unique()))

    spreads = fit_yard_spreads(ps)
    print("yard sd fits: pass sd = %.1f + %.3f*mean | rush sd = %.1f + %.3f*mean"
          % (*spreads["pass"], *spreads["rush"]))

    cal_a, cal_b = calibration_check(ps, spreads)
    from nflquant.props.calibrate import apply_platt

    # ---------- live props ----------
    roster = load_roster(cfg, 2026)
    st = build_state(ps)   # everything through 2025 (and any 2026 weeks present)
    preds = json.load(open(PACKAGE_ROOT / "reports_out/2026_week01/predictions.json"))
    feats = pd.read_parquet(cache_dir(cfg) / "features_enriched.parquet").set_index("game_id")
    lg_dp = feats[feats.season == 2025]["home_def_pass_epa"].mean()
    lg_dr = feats[feats.season == 2025]["home_def_rush_epa"].mean()

    roster_by_team = {t: g for t, g in roster.groupby("team")}
    out_games = []
    for g in sorted(preds, key=lambda x: (x["gameday"], x["away"])):
        frow = feats.loc[g["game_id"]]
        teams = []
        for side, opp_side, team in (("home", "away", g["home"]), ("away", "home", g["away"])):
            exp_pts = (g["total"] + g["margin"]) / 2 if side == "home" else (g["total"] - g["margin"]) / 2
            e_td = exp_offensive_tds(exp_pts, frow.get(f"{side}_td_rate") or .24,
                                     frow.get(f"{side}_fg_rate") or .15)
            env = float(np.clip(exp_pts / team_trailing_pts(frow, side), 0.8, 1.2))
            adj_p = unit_adj(frow.get(f"{opp_side}_def_pass_epa"), lg_dp)
            adj_r = unit_adj(frow.get(f"{opp_side}_def_rush_epa"), lg_dr)
            script = 1.03 if (side == "home") == (g["margin"] > 2) and abs(g["margin"]) > 2 else \
                     (0.97 if abs(g["margin"]) > 2 else 1.0)

            ros = roster_by_team.get(team, pd.DataFrame())
            cands = []
            for r in ros.itertuples(index=False):
                s = st.p.get(r.gsis_id)
                if not s or s["n"] < 3:
                    continue
                cands.append((r.gsis_id, s, r.position))
            # anytime TD: top 6 eligible skill players (calibrated probability)
            td_rows = []
            for pid, s, pos in cands:
                if not eligible(s, pos):
                    continue
                share = st.td_share(pid)
                if not share:
                    continue
                p = apply_platt(anytime_td_prob(e_td * env, share), cal_a, cal_b)
                td_rows.append({"name": s["name"], "pos": pos, "p": round(p, 4),
                                "odds": prob_to_american(p)})
            td_rows = sorted(td_rows, key=lambda x: -x["p"])[:6]

            # QB passing yards: the LISTED starter gets priority; a listed
            # starter with no meaningful NFL sample (2026 rookie, career
            # backup) is reported honestly as no-line rather than silently
            # replaced by a teammate's number
            qb_name = frow.get(f"{side}_qb_name")
            qb_row = None
            qbs = [(pid, s) for pid, s, pos in cands if pos == "QB"]
            listed = None
            if isinstance(qb_name, str):
                for pid, s in qbs:
                    if isinstance(s["name"], str) and s["name"].split()[-1] == qb_name.split()[-1]:
                        listed = (pid, s); break
            pick, tag = None, ""
            if listed and (listed[1]["att"] or 0) >= 10 and listed[1]["n"] >= 3:
                pick, tag = listed, "listed starter"
            elif isinstance(qb_name, str):
                qb_row = {"name": qb_name, "line": None,
                          "note": "listed starter, insufficient NFL sample"}
            else:
                elig = [q for q in qbs if (q[1]["att"] or 0) >= 10 and q[1]["n"] >= 3]
                if elig:
                    pick, tag = max(elig, key=lambda x: x[1]["att"]), "projected (no starter listed)"
            if pick:
                mean = pick[1]["pass_y"] * adj_p * (env ** 0.6)
                d = pass_yards_dist(mean, *spreads["pass"])
                line = round(float(d.ppf(0.5)) * 2) / 2
                qb_row = {"name": pick[1]["name"], "mean": round(mean, 1), "line": line,
                          "sd": round(float(d.std()), 1),
                          "dist": "normal", "n": round(pick[1]["n"], 1), "note": tag}

            # RB rushing yards: top 2 by trailing carries
            rbs = sorted([(pid, s) for pid, s, pos in cands
                          if pos in ("RB", "FB") and (s["car"] or 0) >= 5],
                         key=lambda x: -(x[1]["car"] or 0))[:2]
            rb_rows = []
            for pid, s in rbs:
                mean = s["rush_y"] * adj_r * (env ** 0.5) * script
                d = rush_yards_dist(mean, *spreads["rush"])
                line = round(float(d.ppf(0.5)) * 2) / 2
                rb_rows.append({"name": s["name"], "mean": round(mean, 1), "line": line,
                                "k": round(float(d.kwds["a"]), 3), "theta": round(float(d.kwds["scale"]), 3),
                                "dist": "gamma", "car": round(s["car"], 1), "n": round(s["n"], 1)})

            teams.append({"team": team, "exp_pts": round(exp_pts, 1),
                          "exp_off_tds": round(e_td * env, 2),
                          "td": td_rows, "qb": qb_row, "rb": rb_rows})
        out_games.append({"id": g["game_id"], "away": g["away"], "home": g["home"],
                          "date": g["gameday"], "teams": teams})

    out = {"generated": pd.Timestamp.now().isoformat()[:16],
           "spreads": spreads, "platt": [round(cal_a, 4), round(cal_b, 4)],
           "games": out_games,
           "note": "Fair lines/odds, no vig. TD = rush+rec TDs (QB rushing counts). "
                   "Rookies without NFL games get no prop. Week-1 form = 2025 EWMA."}
    path = PACKAGE_ROOT / "reports_out/2026_week01/props.json"
    path.write_text(json.dumps(out, indent=1))
    print("\nwrote", path)
    # quick eyeball of two games
    for gg in out_games:
        if gg["away"] in ("MIA", "BAL"):
            for t in gg["teams"]:
                print(f"\n{t['team']} (exp {t['exp_pts']} pts, {t['exp_off_tds']} off TDs)")
                for r in t["td"][:4]:
                    print(f"  TD  {r['name']:<22} {r['p']:.0%}  {r['odds']}")
                if t["qb"]:
                    print(f"  QB  {t['qb']['name']:<22} {t['qb']['line']} pass yds (±{t['qb']['sd']})")
                for r in t["rb"]:
                    print(f"  RB  {r['name']:<22} {r['line']} rush yds")


if __name__ == "__main__":
    main()
