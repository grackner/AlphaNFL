import numpy as np
import pandas as pd
from constants import POS_IDX, SLOT_NAMES, N_STARTERS, ROSTER

# Helper functions
def rosters_from_draft_log(draft_log, ids, n_teams=14):
    arr = np.zeros((n_teams, ROSTER), int)
    for t, g in draft_log.groupby("team"):
        arr[t - 1] = ids.get_indexer(g.player_id)
    assert (arr >= 0).all() and arr.shape == (n_teams, ROSTER)
    return arr

def round_robin(n):
    teams, rounds = list(range(n)), []
    for _ in range(n - 1):
        rounds.append([(teams[i], teams[n - 1 - i]) for i in range(n // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds

# Evaluation Helpers
def evaluate(env, policy, n_episodes=20, seed0=10_000):
    out = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs, env.action_masks()))
        out.append(info)
    df = pd.DataFrame(out)
    print(f"avg wins {df.wins.mean():.2f} | avg rank {df['rank'].mean():.2f} | avg PF {df.points_for.mean():.0f}")
    return df

autopilot = lambda obs, mask: 0                                   # (use with auto_lineup=True)
def random_policy(rng=np.random.default_rng(0)):
    return lambda obs, mask: int(rng.choice(np.flatnonzero(mask)))

