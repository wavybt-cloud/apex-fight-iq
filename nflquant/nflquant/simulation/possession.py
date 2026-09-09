"""Possession-level Monte Carlo game simulator.

Each simulated game is built from drives, not from a normal draw, so the
final-score distribution shows real NFL structure: mass on margins of 3 and
7, discrete totals, realistic overtime and one-score rates.

Per simulation:
  1. Parameter uncertainty: each team's strength is perturbed by a
     N(0, param_sd) points-per-game shock (shared across its drives -> fat
     tails and realistic correlation between a team's drive outcomes).
  2. Pace: both teams draw a common drive count (pace is a game property,
     correlated between opponents), split into halves.
  3. Drive outcomes: categorical over TD / FG / empty (punt, downs, missed
     FG) / turnover / defensive TD against, with probabilities scaled so the
     expected points per drive match the model's projected team points.
  4. Game script: at halftime, teams trailing by 9+ raise aggression
     (more TDs and turnovers, extra possession); big leads slow the game.
  5. Scoring increments: TDs are 7 (XP), occasionally 6 (miss) or 8 (2pt);
     FGs are 3. Ties go to a short overtime model with a small tie
     probability in the regular season.
"""
from __future__ import annotations

import numpy as np

TD_POINTS_XP = 7
DEF_TD_PER_TURNOVER = 0.18   # share of giveaways returned/converted directly to opp TD
SAFETY_RATE = 0.0015


def _drive_points(n_td: np.ndarray, rng) -> np.ndarray:
    """Points from n_td touchdowns each with XP/2pt/miss mix (vectorized)."""
    total = np.zeros_like(n_td, dtype=np.int64)
    max_td = int(n_td.max()) if len(n_td) else 0
    for i in range(max_td):
        active = n_td > i
        u = rng.random(active.sum())
        pts = np.where(u < 0.90, 7, np.where(u < 0.955, 8, 6))
        total[active] += pts
    return total


def simulate_game(
    exp_home_pts: float,
    exp_away_pts: float,
    home_drive_mix: tuple[float, float] = (0.24, 0.15),   # (td_rate, fg_rate) tendency
    away_drive_mix: tuple[float, float] = (0.24, 0.15),
    home_to_rate: float = 0.11,
    away_to_rate: float = 0.11,
    n_sims: int = 50_000,
    drives_mean: float = 11.2,
    drives_sd: float = 1.1,
    param_sd_pts: float = 2.6,
    drive_var_shrink: float = 0.72,
    endgame_tie_prob: float = 0.35,
    walkoff_prob: float = 0.50,
    consolation_prob: float = 0.24,
    playoff: bool = False,
    seed: int | None = None,
) -> dict:
    """Simulate one game; returns arrays of home/away scores plus summary()."""
    rng = np.random.default_rng(seed)

    # ---- pace: common game-level drive count + small per-team wiggle ----
    game_drives = rng.normal(drives_mean, drives_sd, n_sims)
    drives_h = np.clip(np.round(game_drives + rng.normal(0, 0.4, n_sims)), 8, 16).astype(int)
    drives_a = np.clip(np.round(game_drives + rng.normal(0, 0.4, n_sims)), 8, 16).astype(int)

    # ---- parameter uncertainty: per-sim team strength shocks (points) ----
    shock_h = rng.normal(0, param_sd_pts, n_sims)
    shock_a = rng.normal(0, param_sd_pts, n_sims)

    def team_scores(exp_pts, mix, opp_to_rate, drives, shock):
        td0, fg0 = max(mix[0], 0.03), max(mix[1], 0.03)
        # a team's projected points include what its DEFENSE scores off the
        # opponent's giveaways plus rare safeties; deduct those before scaling
        # drive probabilities, or totals inflate by ~3 points
        extra = opp_to_rate * DEF_TD_PER_TURNOVER * drives * 7.0 + 2.0 * SAFETY_RATE * 11
        target_ppd = np.clip((exp_pts + shock - extra) / drives, 0.5, 4.2)
        base_ppd = 7.0 * td0 + 3.0 * fg0
        f = target_ppd / base_ppd
        p_td = np.clip(td0 * f, 0.02, 0.62)
        p_fg = np.clip(fg0 * f, 0.02, 0.35)
        # halves for game-script adjustment
        d1 = np.maximum(drives // 2, 3)
        d2 = drives - d1
        return p_td, p_fg, d1, d2

    p_td_h, p_fg_h, d1_h, d2_h = team_scores(exp_home_pts, home_drive_mix, away_to_rate, drives_h, shock_h)
    p_td_a, p_fg_a, d1_a, d2_a = team_scores(exp_away_pts, away_drive_mix, home_to_rate, drives_a, shock_a)

    def _shrink_count(raw: np.ndarray, mean: np.ndarray, gamma: float) -> np.ndarray:
        """Shrink integer counts toward their mean with stochastic rounding.

        Independent-drive sampling over-disperses scores relative to real NFL
        games (field position, clock and strategy create negative serial
        correlation between drives). Shrinking count deviations by gamma
        reproduces the observed margin/total variance while keeping counts
        integer and unbiased.
        """
        adj = mean + gamma * (raw - mean)
        lo = np.floor(adj)
        return (lo + (rng.random(len(adj)) < (adj - lo))).astype(np.int64)

    def play_half(p_td, p_fg, drives_half, to_rate, max_d=9):
        """Vectorized: for each sim, count TDs and FGs over its active drives."""
        n = len(p_td)
        tds = np.zeros(n, dtype=np.int64)
        fgs = np.zeros(n, dtype=np.int64)
        opp_tds = np.zeros(n, dtype=np.int64)
        for i in range(max_d):
            active = drives_half > i
            if not active.any():
                break
            u = rng.random(n)
            tds += (active & (u < p_td)).astype(np.int64)
            fgs += (active & (u >= p_td) & (u < p_td + p_fg)).astype(np.int64)
            u2 = rng.random(n)
            opp_tds += (active & (u2 < to_rate * DEF_TD_PER_TURNOVER)).astype(np.int64)
        tds = _shrink_count(tds, drives_half * p_td, drive_var_shrink)
        fgs = _shrink_count(fgs, drives_half * p_fg, drive_var_shrink)
        return tds, fgs, opp_tds

    # ---- first half ----
    td1_h, fg1_h, dtd1_a = play_half(p_td_h, p_fg_h, d1_h, home_to_rate)
    td1_a, fg1_a, dtd1_h = play_half(p_td_a, p_fg_a, d1_a, away_to_rate)
    pts1_h = _drive_points(td1_h + dtd1_h, rng) + 3 * fg1_h
    pts1_a = _drive_points(td1_a + dtd1_a, rng) + 3 * fg1_a

    # ---- game script for the second half ----
    lead_h = pts1_h - pts1_a
    trail_big_h = lead_h <= -9
    trail_big_a = lead_h >= 9
    # trailing team: more aggression -> TDs up, FGs partially converted to TD tries
    p_td_h2 = np.where(trail_big_h, np.clip(p_td_h * 1.10, 0, 0.65), p_td_h)
    p_fg_h2 = np.where(trail_big_h, p_fg_h * 0.85, p_fg_h)
    p_td_a2 = np.where(trail_big_a, np.clip(p_td_a * 1.10, 0, 0.65), p_td_a)
    p_fg_a2 = np.where(trail_big_a, p_fg_a * 0.85, p_fg_a)
    # leading team slows the game: occasionally one fewer possession each
    slow = (np.abs(lead_h) >= 9) & (rng.random(len(lead_h)) < 0.35)
    d2_h = np.maximum(d2_h - slow.astype(int), 2)
    d2_a = np.maximum(d2_a - slow.astype(int), 2)

    td2_h, fg2_h, dtd2_a = play_half(p_td_h2, p_fg_h2, d2_h, home_to_rate)
    td2_a, fg2_a, dtd2_h = play_half(p_td_a2, p_fg_a2, d2_a, away_to_rate)
    pts2_h = _drive_points(td2_h + dtd2_h, rng) + 3 * fg2_h
    pts2_a = _drive_points(td2_a + dtd2_a, rng) + 3 * fg2_a

    # rare safeties
    home = pts1_h + pts2_h + 2 * (rng.random(n_sims) < SAFETY_RATE * 11)
    away = pts1_a + pts2_a + 2 * (rng.random(n_sims) < SAFETY_RATE * 11)

    # ---- late-game convergence ----
    # Real one-score endings cluster: a team down 1-3 in the final minutes
    # plays for the tying/winning field goal. Independent drives can't produce
    # that; nudge a share of 1-3 point finishes to a tie (which then resolves
    # by walk-off logic through the OT block, putting mass on margin 3).
    close = np.abs(home - away).astype(int)
    nudge = (close >= 1) & (close <= 3) & (rng.random(n_sims) < endgame_tie_prob)
    if nudge.any():
        h_lead = home > away
        # trailing team adds the equalizing points
        add = np.abs(home - away)
        home = np.where(nudge & ~h_lead, home + add, home)
        away = np.where(nudge & h_lead, away + add, away)

    # ---- late consolation scores ----
    # Trailing teams facing prevent defense add a late score that doesn't
    # threaten the result far more often than independent drives produce
    # (the "backdoor cover"). Compress a share of 9-13 point finishes into
    # one-score games without changing the winner.
    marg_now = home - away
    absm = np.abs(marg_now)
    consol = (absm >= 9) & (absm <= 13) & (rng.random(n_sims) < consolation_prob)
    if consol.any():
        cut = np.where(absm == 9, 3, 7)
        home = np.where(consol & (marg_now < 0), home + cut, home)
        away = np.where(consol & (marg_now > 0), away + cut, away)

    # ---- regulation walk-off ----
    # A tied game late usually gets decided before overtime: one more drive
    # ends in a field goal as the clock dies. Resolve that share of ties in
    # regulation (mass lands on margin exactly 3 - the biggest key number).
    strength_gap = (exp_home_pts + shock_h) - (exp_away_pts + shock_a)
    p_home_ot = 1.0 / (1.0 + np.exp(-strength_gap / 9.0))
    tied_now = home == away
    walkoff = tied_now & (rng.random(n_sims) < walkoff_prob)
    if walkoff.any():
        home_kicks = rng.random(n_sims) < p_home_ot
        home = np.where(walkoff & home_kicks, home + 3, home)
        away = np.where(walkoff & ~home_kicks, away + 3, away)

    # ---- overtime ----
    reg_tied = home == away
    went_ot = reg_tied.copy()
    if reg_tied.any():
        n_t = int(reg_tied.sum())
        u = rng.random(n_t)
        ties = np.zeros(n_t, dtype=bool) if playoff else (rng.random(n_t) < 0.055)
        home_wins_ot = u < p_home_ot[reg_tied]
        ot_pts = np.where(rng.random(n_t) < 0.55, 3, 7)
        h_add = np.where(~ties & home_wins_ot, ot_pts, 0)
        a_add = np.where(~ties & ~home_wins_ot, ot_pts, 0)
        home = home.astype(np.int64); away = away.astype(np.int64)
        home[reg_tied] += h_add
        away[reg_tied] += a_add

    return {"home": home.astype(int), "away": away.astype(int), "ot": went_ot,
            "n_sims": n_sims}


def summarize_sims(sims: dict, spread_line: float | None = None,
                   total_line: float | None = None) -> dict:
    """All requested distribution statistics from one simulate_game() output."""
    h, a, ot = sims["home"], sims["away"], sims["ot"]
    margin = h - a
    total = h + a
    out = {
        "n_sims": sims["n_sims"],
        "p_home": float((margin > 0).mean() + 0.5 * (margin == 0).mean()),
        "p_away": float((margin < 0).mean() + 0.5 * (margin == 0).mean()),
        "p_tie": float((margin == 0).mean()),
        "mean_home": float(h.mean()), "mean_away": float(a.mean()),
        "median_home": float(np.median(h)), "median_away": float(np.median(a)),
        "mean_margin": float(margin.mean()), "sd_margin": float(margin.std()),
        "mean_total": float(total.mean()), "sd_total": float(total.std()),
        "p_ot": float(ot.mean()),
        "p_one_score": float((np.abs(margin) <= 8).mean()),
        "p_blowout_17": float((np.abs(margin) >= 17).mean()),
        "margin_pctiles": {q: float(np.percentile(margin, q)) for q in (10, 25, 50, 75, 90)},
        "home_pctiles": {q: float(np.percentile(h, q)) for q in (10, 25, 50, 75, 90)},
        "away_pctiles": {q: float(np.percentile(a, q)) for q in (10, 25, 50, 75, 90)},
        "key_mass_3": float((np.abs(margin) == 3).mean()),
        "key_mass_7": float((np.abs(margin) == 7).mean()),
    }
    if spread_line is not None and np.isfinite(spread_line):
        push = margin == spread_line
        out["p_home_cover"] = float((margin > spread_line).mean() + 0.5 * push.mean())
        out["p_spread_push"] = float(push.mean())
    if total_line is not None and np.isfinite(total_line):
        push = total == total_line
        out["p_over"] = float((total > total_line).mean() + 0.5 * push.mean())
        out["p_total_push"] = float(push.mean())
    # upset probability: chance the pregame underdog (by mean margin) wins
    fav_home = out["mean_margin"] >= 0
    out["p_upset"] = out["p_away"] if fav_home else out["p_home"]
    return out
