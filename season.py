import pandas as pd
from constants import POS, POS_IDX, REPL_RANK, NF
import numpy as np

class SeasonData:
    def __init__(self, board, weekly, shrink_k=4):
        weekly = weekly[weekly.position.isin(POS) & (weekly.week <= 18)]
        ids = pd.Index(pd.concat([board.player_id, weekly.player_id]).unique())
        self.ids, self.P, self.k = ids, len(ids), shrink_k

        pos_map = (pd.concat([board[["player_id", "position"]],
                              weekly[["player_id", "position"]]])
                   .drop_duplicates("player_id").set_index("player_id").position)
        self.pos = ids.map(pos_map).map(POS_IDX).to_numpy().astype(int)

        names = pd.concat([board[["player_id", "name"]],
                           weekly[["player_id", "player_display_name"]]
                           .rename(columns={"player_display_name": "name"})]
                          ).drop_duplicates("player_id").set_index("player_id").name
        self.name = ids.map(names).to_numpy()

        # prior ppg (from 2023); players with no prior get replacement level
        repl = []
        for p in POS:
            s = board[board.position == p].proj_ppg.sort_values(ascending=False).to_numpy()
            repl.append(s[min(REPL_RANK[p], len(s) - 1)])
        self.repl_ppg = np.array(repl)
        pr = board.set_index("player_id").proj_ppg.reindex(ids).to_numpy()
        self.prior = np.where(np.isnan(pr), self.repl_ppg[self.pos], pr)

        # weekly points / played matrices, columns = week 0..18
        self.pts = np.zeros((self.P, 19))
        self.played = np.zeros((self.P, 19), bool)
        r = ids.get_indexer(weekly.player_id)
        wk = weekly.week.to_numpy().astype(int)
        self.pts[r, wk] = weekly.fantasy_points_ppr.to_numpy()
        self.played[r, wk] = True
        self.cumpts = self.pts.cumsum(1)
        self.cumplayed = self.played.cumsum(1)

        # bye weeks: a team with no rows in a week is on bye (public schedule info)
        maxw = int(weekly.week.max())
        presence = weekly.groupby("team").week.apply(set)
        ptm = weekly.sort_values("week").groupby("player_id").team.last()
        self.on_bye = np.zeros((self.P, 19), bool)
        for i, pid in enumerate(ids):
            t = ptm.get(pid)
            if t is not None:
                for w in range(1, maxw + 1):
                    if w not in presence[t]:
                        self.on_bye[i, w] = True
        self._views = {}

    def view(self, w):
        """Everything knowable at the start of week w (uses weeks < w only)."""
        if w in self._views:
            return self._views[w]
        P, k = self.P, self.k
        est = (self.prior * k + self.cumpts[:, w - 1]) / (k + self.cumplayed[:, w - 1])
        if w > 1:
            sl = slice(max(1, w - 3), w)
            pl = self.played[:, sl]
            avail = pl.mean(1)
            last3 = self.pts[:, sl].sum(1) / np.maximum(1, pl.sum(1))
            last1 = self.pts[:, w - 1]
            gp = self.cumplayed[:, w - 1] / (w - 1)
        else:
            avail, last3, last1, gp = np.ones(P), np.zeros(P), np.zeros(P), np.ones(P)
        bye = self.on_bye[:, w]
        val = est * (0.4 + 0.6 * avail)             # season-level worth (trades, waivers)
        start = np.where(bye, -1.0, val)            # lineup-setting score
        F = np.zeros((P + 1, NF), np.float32)       # last row = padding for index -1
        F[np.arange(P), self.pos] = 1
        F[:P, 4], F[:P, 5] = est / 20, self.prior / 20
        F[:P, 6], F[:P, 7] = last1 / 30, last3 / 30
        F[:P, 8], F[:P, 9], F[:P, 10] = gp, avail, bye
        v = type("View", (), dict(est=est, val=val, start=start, F=F, bye=bye))
        self._views[w] = v
        return v