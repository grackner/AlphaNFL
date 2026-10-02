import numpy as np
import pandas as pd
"""
Draft simulation for PPR
- Snake draft
- 14 teams
- Starters: QB, 2 RB, 2 WR, TE, 1 FLEX
- 7 bench spots
"""

# DRAFT CONFIG
SEED = 42
N_TEAMS = 14
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX = 1
BENCH = 7
ROUNDS = sum(STARTERS.values()) + FLEX + BENCH      # 14 rounds = 196 picks
POSITIONS = list(STARTERS)
ROSTER_CAP = {"QB": 2, "RB": 8, "WR": 8, "TE": 3}   # max per position
# 0-indexed depth of "replacement level" per position (league-wide starters + flex share)
# TODO: Calibrate replacement level
REPL_RANK = {"QB": 14, "RB": 36, "WR": 40, "TE": 14} # Relates to value of the player similar to WAR in baseball


def season_totals(weekly_df):
    """
    Calculate season totals for a player based on given dataset
    """
    g = weekly_df.sort_values("week").groupby("player_id")
    out = g.agg(
        name=("player_display_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        games=("week", "nunique"),
        ppr_pts=("fantasy_points_ppr", "sum"),
    ).reset_index()
    out["ppg"] = out["ppr_pts"] / out["games"]
    return out

def build_board(totals, shrink_games=4, season_games=16):
    """
    Build draft board based on past player's season totals
    """
    repl_ppg = {}
    for p in POSITIONS:
        s = totals[(totals.position == p) & (totals.games >= 8)].ppg.sort_values(ascending=False)
        repl_ppg[p] = s.iloc[min(REPL_RANK[p], len(s) - 1)]
    w = totals.games / (totals.games + shrink_games)
    totals["proj_ppg"] = w * totals.ppg + (1 - w) * totals.position.map(repl_ppg)
    totals["proj_pts"] = totals.proj_ppg * season_games
    base = {
        p: totals[totals.position == p].proj_pts.sort_values(ascending=False).iloc[REPL_RANK[p]]
        for p in POSITIONS
    }
    totals["vorp"] = totals.proj_pts - totals.position.map(base)
    totals = totals.sort_values("vorp", ascending=False).reset_index(drop=True)
    totals["rank"] = totals.index + 1
    return totals


def simulate_draft(board, seed=SEED, n_teams=N_TEAMS, rounds=ROUNDS, pool_size=60):
    """
    Run one simulation of the draft given a draft board, number of teams, rounds
    """
    rng = np.random.default_rng(seed)
    cfg = {
        t: {"noise": rng.uniform(8, 25),
            "bias": {p: rng.normal(0, 12) for p in POSITIONS}}
        for t in range(n_teams)
    }
    counts = {t: dict.fromkeys(POSITIONS, 0) for t in range(n_teams)}
    avail = board.copy()
    log = []
    pick = 0
    for rnd in range(1, rounds + 1):
        order = range(n_teams) if rnd % 2 == 1 else range(n_teams - 1, -1, -1)
        for slot, t in enumerate(order, 1):
            pick += 1
            picks_left = rounds - rnd + 1
            need = {p: max(0, STARTERS[p] - counts[t][p]) for p in POSITIONS}
            allowed = [p for p in POSITIONS if counts[t][p] < ROSTER_CAP[p]]
            if picks_left <= sum(need.values()):          # must fill required starters
                allowed = [p for p in allowed if need[p] > 0]
            cand = avail[avail.position.isin(allowed)].head(pool_size)
            score = (cand.vorp.values
                     + cand.position.map(cfg[t]["bias"]).values
                     + rng.normal(0, cfg[t]["noise"], len(cand)))
            row = cand.iloc[int(np.argmax(score))]
            counts[t][row.position] += 1
            avail = avail.drop(index=row.name)
            # Update draft log
            log.append({"pick": pick, "round": rnd, "slot": slot, "team": t + 1,
                        "player_id": row.player_id, "name": row["name"],
                        "position": row.position, "board_rank": row["rank"],
                        "proj_pts": row.proj_pts, "vorp": row.vorp})
    draft_log = pd.DataFrame(log)
    rosters = {t: g for t, g in draft_log.groupby("team")}
    return draft_log, rosters

def build_waiver(board, drafted_ids, tot_next):
    """
    Build waiver wire with remaining players not drafted from the board
    """
    ids = tot_next[["player_id", "name", "position", "team"]]
    w = ids[~ids.player_id.isin(drafted_ids)].merge(
        board[["player_id", "games", "ppg", "proj_pts", "vorp", "rank"]],
        on="player_id", how="left",
    ).rename(columns={"games": "games_prev", "ppg": "ppg_prev"})
    w["no_prior_data"] = w.proj_pts.isna()               # rookies etc.
    base = (board.proj_pts - board.vorp).groupby(board.position).first()
    w["proj_pts"] = w.proj_pts.fillna(w.position.map(base))   # replacement-level guess
    w["vorp"] = w.vorp.fillna(0.0)
    return w.sort_values("vorp", ascending=False).reset_index(drop=True)


def lineup_strength(draft_log):
    """
    Sanity check to ensure that optimal line-up was drafted per team
    """
    rows = []
    for t, g in draft_log.groupby("team"):
        g = g.sort_values("proj_pts", ascending=False)
        used, total = set(), 0.0
        for p, n in STARTERS.items():
            top = g[g.position == p].head(n)
            used |= set(top.index); total += top.proj_pts.sum()
        flex = g[g.position.isin(["RB", "WR", "TE"]) & ~g.index.isin(used)].head(FLEX)
        total += flex.proj_pts.sum()
        rows.append({"team": t, "proj_lineup_pts": round(total, 1)})
    return pd.DataFrame(rows).sort_values("proj_lineup_pts", ascending=False)
