import numpy as np
from fantasy_rl import rosters_from_draft_log, round_robin
from constants import N_STARTERS, N_BENCH, ROSTER, NF, ELIG
from gymnasium import spaces
import gymnasium as gym


class FantasyEnv(gym.Env):
    """
    Creates the environment for the fantasy agent to train in taking in SeasonData,
    draft_logs, weeks in the season to train on, max_actions, etc. 
    Action ids:
    - 0 = end week
    - swap(starter s, bench b)
    - add(candidate k, drop slot r)
    - trade(market m, give slot a). Invalid actions are masked (use MaskablePPO).
    """
    metadata = {"render_modes": []}

    def __init__(self, data, draft_logs, n_weeks=14, max_actions=6, max_adds=2,
                 max_trades=1, cand_per_pos=5, n_market=40, auto_lineup=False,
                 margin_coef=0.01, final_bonus=1.0, bot_waiver_prob=0.5,
                 trade_margin=0.5, agent_team=None):
        super().__init__()
        self.data, self.n_weeks, self.n = data, n_weeks, 14
        self.scen = [rosters_from_draft_log(d, data.ids, self.n) for d in draft_logs]
        self.max_actions, self.max_adds, self.max_trades = max_actions, max_adds, max_trades
        self.cpp, self.KW, self.KM = cand_per_pos, 4 * cand_per_pos, n_market
        self.auto_lineup, self.margin_coef, self.final_bonus = auto_lineup, margin_coef, final_bonus
        self.bot_waiver_prob, self.trade_margin, self.fixed_me = bot_waiver_prob, trade_margin, agent_team

        self.o_add = 1 + N_STARTERS * N_BENCH
        self.o_trade = self.o_add + self.KW * ROSTER
        self.n_actions = self.o_trade + self.KM * ROSTER
        obs_dim = ROSTER * NF + self.KW * NF + self.KM * (NF + 1) + 8
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

    # ---------- helpers ----------
    def _lineup_val(self, ids, score):
        """
        (For bots)
        Calculates best line-up value using a set of player_ids and the score list.
        Used for trades and add/drops
        """
        s, p = score[ids], self.data.pos[ids] # Split into player's points & position ids
        tot, used = 0.0, np.zeros(len(ids), bool)
        for pos, cnt in ((0, 1), (1, 2), (2, 2), (3, 1)):
            idx = np.where(p == pos)[0]
            idx = idx[np.argsort(-s[idx])][:cnt]
            tot += s[idx].sum(); used[idx] = True
        rest = np.where(~used & (p > 0))[0]
        return tot + (s[rest].max() if len(rest) else 0.0)

    def _arrange(self, ids):
        """
        Arrange line-up
        Best lineup first (QB,RB,RB,WR,WR,TE,FLEX), bench sorted by score.
        """
        ids = np.asarray(ids)
        s, pos = self.D.start[ids], self.data.pos[ids]
        order = np.argsort(-s, kind="stable")
        used = np.zeros(len(ids), bool)
        spec = [({0}, 0), ({1}, 1), ({1}, 2), ({2}, 3), ({2}, 4), ({3}, 5), ({1, 2, 3}, 6)]
        slots = [None] * 7
        for allowed, si in spec:
            for i in order:
                if not used[i] and pos[i] in allowed:
                    slots[si] = i; used[i] = True; break
        for si in range(7):                       # fallback if a position is missing
            if slots[si] is None:
                i = next(i for i in order if not used[i]); slots[si] = i; used[i] = True
        bench = [i for i in order if not used[i]]
        return ids[slots + bench]

    def _refresh_lists(self):
        val, owner, pos = self.D.val, self.owner, self.data.pos
        fa = np.where(owner == -1)[0]
        cand = []
        for p in range(4):
            f = fa[pos[fa] == p]
            f = f[np.argsort(-val[f])][: self.cpp]
            cand += list(f) + [-1] * (self.cpp - len(f))
        self.cand = np.array(cand)
        oth = np.where((owner >= 0) & (owner != self.me))[0]
        oth = oth[np.argsort(-val[oth])][: self.KM]
        self.market = np.full(self.KM, -1); self.market[: len(oth)] = oth
        self._mask = None

    def _accept(self, o, give, get):
        """
        Accept trade 
        """
        key = (o, give, get)
        if key not in self._acc:
            r = self.roster[o]
            new = r.copy(); new[np.where(r == get)[0][0]] = give
            gain = self._lineup_val(new, self.D.val) - self._lineup_val(r, self.D.val)
            self._acc[key] = gain >= self.trade_margin
        return self._acc[key]

    # ---------- bots ----------
    def _bot_waiver(self, t):
        r = self.roster[t]
        val, pos = self.D.val, self.data.pos
        worst = N_STARTERS + int(np.argmin(val[r[N_STARTERS:]]))
        base = self._lineup_val(r, val)
        fa = np.where(self.owner == -1)[0]
        best_gain, best = 0.3, None
        for p in range(4):
            f = fa[pos[fa] == p]
            if len(f) == 0:
                continue
            c = f[np.argmax(val[f])]
            new = r.copy(); new[worst] = c
            g = self._lineup_val(new, val) - base
            if g > best_gain:
                best_gain, best = g, c
        if best is not None:
            self.owner[r[worst]] = -1; self.owner[best] = t; r[worst] = best

    def _bot_move(self, t):
        """
        Function for a bot to decide to move based on if roster value is
        less than the potential roster based on the waiver wire. 
        """
        self.roster[t] = self._arrange(self.roster[t])
        if self.np_random.random() < self.bot_waiver_prob:
            self._bot_waiver(t)
        self.roster[t] = self._arrange(self.roster[t])

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.roster = self.scen[rng.integers(len(self.scen))].copy()
        self.me = int(self.fixed_me if self.fixed_me is not None else rng.integers(self.n))
        self.owner = np.full(self.data.P, -1)
        for t in range(self.n):
            self.owner[self.roster[t]] = t
        perm = rng.permutation(self.n)
        rr = round_robin(self.n)
        self.pairs = {w: [(perm[a], perm[b]) for a, b in rr[(w - 1) % len(rr)]]
                      for w in range(1, self.n_weeks + 1)}
        self.week, self.wins, self.pf = 1, np.zeros(self.n), np.zeros(self.n)
        self.history, self._acc, self._mask = [], {}, None
        self._start_week()
        return self._obs(), {}

    def _start_week(self):
        self.D = self.data.view(self.week)
        self.actions_left, self.adds_left, self.trades_left = (
            self.max_actions, self.max_adds, self.max_trades)
        self._acc, self._mask = {}, None
        self.week_actions = []
        for t in range(self.n):
            if t != self.me:
                self._bot_move(t)
        if self.auto_lineup or self.week == 1:
            self.roster[self.me] = self._arrange(self.roster[self.me])
        self._refresh_lists()

    def action_masks(self):
        """
        Mask actions based on what is valid
        """
        if self._mask is not None:
            return self._mask
        m = np.zeros(self.n_actions, bool)
        m[0] = True
        pos, r = self.data.pos, self.roster[self.me]
        if self.actions_left > 0 and not self.auto_lineup:
            m[1:self.o_add] = ELIG[:N_STARTERS][:, pos[r[N_STARTERS:]]].ravel()
        if self.actions_left > 0 and self.adds_left > 0:
            cv = self.cand >= 0
            em = ELIG[:, pos[np.where(cv, self.cand, 0)]].T & cv[:, None]
            m[self.o_add:self.o_trade] = em.ravel()
        if self.actions_left > 0 and self.trades_left > 0:
            for mi, pid in enumerate(self.market):
                if pid < 0:
                    continue
                o = self.owner[pid]
                for a in range(ROSTER):
                    if ELIG[a, pos[pid]] and self._accept(o, r[a], pid):
                        m[self.o_trade + mi * ROSTER + a] = True
        self._mask = m
        return m

    def describe(self, a):
        nm, r = self.data.name, self.roster[self.me]
        if a == 0:
            return "end week"
        if a < self.o_add:
            s, b = divmod(a - 1, N_BENCH)
            return f"start {nm[r[N_STARTERS + b]]} over {nm[r[s]]}"
        if a < self.o_trade:
            k, slot = divmod(a - self.o_add, ROSTER)
            return f"add {nm[self.cand[k]]}, drop {nm[r[slot]]}"
        mi, slot = divmod(a - self.o_trade, ROSTER)
        return f"trade {nm[r[slot]]} for {nm[self.market[mi]]}"

    def _apply(self, a):
        r = self.roster[self.me]
        if a < self.o_add:
            s, b = divmod(a - 1, N_BENCH)
            r[s], r[N_STARTERS + b] = r[N_STARTERS + b], r[s]
        elif a < self.o_trade:
            k, slot = divmod(a - self.o_add, ROSTER)
            pid, drop = self.cand[k], r[slot]
            self.owner[drop], self.owner[pid], r[slot] = -1, self.me, pid
            self.adds_left -= 1
            self._refresh_lists()
        else:
            mi, slot = divmod(a - self.o_trade, ROSTER)
            get, give = self.market[mi], r[slot]
            o = self.owner[get]
            self.roster[o][np.where(self.roster[o] == get)[0][0]] = give
            r[slot] = get
            self.owner[give], self.owner[get] = o, self.me
            self.roster[o] = self._arrange(self.roster[o])
            self.trades_left -= 1
            self._acc = {}
            self._refresh_lists()
        self._mask = None

    def _play_week(self):
        """
        Play week by arranging line-up
        """
        w, d = self.week, self.data
        sl = np.arange(N_STARTERS)
        sc = np.array([(d.pts[self.roster[t][:N_STARTERS], w]
                        * ELIG[sl, d.pos[self.roster[t][:N_STARTERS]]]).sum()
                       for t in range(self.n)])
        opp = None
        for a, b in self.pairs[w]:
            if sc[a] > sc[b]: self.wins[a] += 1
            elif sc[b] > sc[a]: self.wins[b] += 1
            else: self.wins[a] += .5; self.wins[b] += .5
            if self.me in (a, b):
                opp = b if a == self.me else a
        self.pf += sc
        diff = sc[self.me] - sc[opp]
        self.history.append(dict(week=w, my_pts=round(sc[self.me], 1),
                                 opp_pts=round(sc[opp], 1), actions=self.week_actions))
        return float(np.sign(diff) + self.margin_coef * diff)

    def _rank(self):
        key = list(zip(self.wins, self.pf))
        return 1 + sum(k > key[self.me] for k in key)

    def step(self, action):
        """
        Function for agent choosing one action
        """
        a = int(action)
        reward, terminated, info = 0.0, False, {}
        if not self.action_masks()[a]:                 # shouldn't happen with masking
            # Penalize the agent for an invalid action
            reward -= 0.05
            a = 0
        end_week = a == 0
        if not end_week:
            self.week_actions.append(self.describe(a))
            self._apply(a)
            self.actions_left -= 1
            end_week = self.actions_left <= 0
        if end_week:
            reward += self._play_week()
            self.week += 1
            if self.week > self.n_weeks:
                terminated = True
                rank = self._rank()
                reward += self.final_bonus * (1 - 2 * (rank - 1) / (self.n - 1))
                info = dict(wins=self.wins[self.me], rank=rank, points_for=self.pf[self.me])
            else:
                self._start_week()
        return self._obs(), reward, terminated, False, info

    def _obs(self):
        """
        Builds what the agent sees before making a decision
        862 values in a vector- player's values, waiver values, etc.
        """
        w, D, me = self.week, self.D, self.me
        F, r = D.F, self.roster[me]
        denom = max(1, w - 1)
        owner_win = np.where(self.market >= 0, self.wins[self.owner[self.market]] / denom, 0.0)
        mk = np.concatenate([F[self.market], owner_win[:, None].astype(np.float32)], 1)

        def proj(t):
            rr = self.roster[t][:N_STARTERS]
            return (D.est[rr] * ~D.bye[rr] * ELIG[np.arange(N_STARTERS), self.data.pos[rr]]).sum()
        wk = min(w, self.n_weeks)                      # final obs is built after the last week
        opp = next(b if a == me else a for a, b in self.pairs[wk] if me in (a, b))
        glob = [w / self.n_weeks, self.wins[me] / self.n_weeks, (w - 1 - self.wins[me]) / self.n_weeks,
                (self._rank() - 1) / (self.n - 1), np.clip((proj(me) - proj(opp)) / 30, -3, 3),
                self.adds_left / max(1, self.max_adds), self.trades_left / max(1, self.max_trades),
                self.actions_left / self.max_actions]
        return np.concatenate([F[r].ravel(), F[self.cand].ravel(), mk.ravel(),
                               np.array(glob, np.float32)]).astype(np.float32)

