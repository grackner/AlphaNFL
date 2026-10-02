import numpy as np
POS = ["QB", "RB", "WR", "TE"]
POS_IDX = {p: i for i, p in enumerate(POS)}
SLOT_NAMES = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
N_STARTERS, N_BENCH = 7, 7
ROSTER = N_STARTERS + N_BENCH                     # 14, matches draft ROUNDS
REPL_RANK = {"QB": 14, "RB": 36, "WR": 40, "TE": 14}
NF = 11   

# ELIG[slot, pos] -> may a player of this position occupy this roster slot?
ELIG = np.zeros((ROSTER, 4), bool)
for s, nm in enumerate(SLOT_NAMES):
    if nm == "FLEX":
        ELIG[s, 1:] = True
    else:
        ELIG[s, POS_IDX[nm]] = True
ELIG[N_STARTERS:] = True