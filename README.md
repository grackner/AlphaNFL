# AlphaNFL
Building a fantasy football reinforcement learning agent

<img width="447" height="447" alt="tjwatt" src="https://github.com/user-attachments/assets/f9368dd5-9c66-4da6-8d5b-ae1841899a58" />


## League Rules:
- 14 team, PPR format
- Snake draft
- Positions: QB, RB, WR, TE (K and Defense will be later enhancements)

**RL Environment:**
- One agent that owns a team, plays against 13 bots
- Runs about 150 steps/second
- The agent is only learning from stats in earlier weeks, plus the 2023 prior and the public bye weeks. 
- Bots are also setting their line-ups each week, constrained by same rules as the agent

**Actions:**
- Actions for a player are start/sit, add/drop, trade, end week
- Agent is limited to 5 actions per week (add/drop is considered 2 actions)
- Add/drop would only consider top-5 players available for a position
- Trade is only accepted by a bot if it improves bot's projected line-up by given margin (hyperparameter to the model)
- Agent is limited to 2 add/drops, 1 trade

**Rewards:**
- Win = +1, loss = -1, small term for point margin
- At the end of the season there is a bonus based on final rank.

**Data:**
- Board is built from 2023 NFL season data
- Trained/Evaluated on 2024 NFL season data


## Agent Learning/Gymnasium
*For one season*
1. Random team is picked for the agent based on `draft_logs`
2. Bots set their line-ups, agent's line-up is built with no add/drops but vector is constructed with potential moves.
  The observation is a vector of 862 dimensions describing:
  - the agent's 14 players (plus 11 features per player- see below)
  - the 20 free-agent candidates (11 features)
  - the 40 players on other teams the agent could trade for (11 features + owner's win percentage to calculate trade value)
  - a few global values: week number, record, rank, projected score against this week's opponent, and how many adds, trades and actions are left
3. Once a week starts, the agent will make 1-6 actions (where reward=0) not seeing reward until the end of the week
4. Repeat every week until season ends and bonus points are calculated

### Player Features
The 11 features per player (these come from `data.view(week)`)
- 4 values for the position, as a one-hot (a single 1 marks which of QB/RB/WR/TE)
- `est`: projected points per game, blending the 2023 prior with 2024 results so far
- `prior`: the 2023-based projection alone, so the agent can tell the prior from the in-season update
- `last`1: points last week
- `last3`: average points over the last 3 weeks he played
- `gp`: fraction of games played so far this season
- `avail`: fraction of the last 3 weeks he played, which flags injuries
- `bye`: 1 if he's on bye this week

### Global values
- Week number / total weeks
- Wins / total weeks
- Losses / total weeks
- Current rank, scaled to 0-1
- Projected points for your lineup minus this week's opponent's, divided by 30 (clipped to ±3)
- Adds left, as a fraction of the allowance
- Trades left, as a fraction
- Actions left this week, as a fraction

### MLP Networks (Policy & Value)
- Utilizes `MaskablePPO` from `sb3_contrib` library
- Default MLPs have two hidden layers of size 64

<img width="1536" height="1024" alt="policymlp" src="https://github.com/user-attachments/assets/b67cb97d-1e7a-4124-8958-c2f88894c6d5" />

