# AlphaNFL
Building a fantasy football reinforcement learning agent

## League Rules:
- 14 team, PPR format
- Snake draft

**RL Environment:**
- One agent that owns a team, plays against 13 bots
- Runs about 150 steps/second
- The agent is only learning from stats in earlier weeks, plus the 2023 prior and the public bye weeks. 
- Bots are also setting their line-ups each week, constrained by same rules as the agent

**Actions:**
- Actions for a player are start/sit, add/drop, trade
- Agent is limited to 5 actions per week (add/drop is considered 2 actions)
- Add/drop would only consider top-5 players available for a position
- Trade is only accepted by a bot if it improves bot's projected line-up by given margin
- Agent is limited to 2 add/drops, 1 trade

Rewards:
- Win = +1, loss = -1, small term for point margin
- At the end of the season there is a bonus based on final rank.

Data:
- Trained on 2023 data
- Evaluated on 2024 season data


## Agent Learning/Gymnasium
*For one season*
1. Random team is picked for the agent based on `draft_logs`
2. Bots set their line-ups, agent's line-up is built with no add/drops but vector is constructed with potential moves
  The observation is a vector of 862 dimensions describing:
  - the agent's 14 players (position, projected points, recent form, bye status, and so on)
  - the 20 free-agent candidates
  - the 40 players on other teams the agent could trade for
  - a few global values: week number, record, rank, projected score against this week's opponent, and how many adds, trades and actions are left
3. Once a week starts, the agent will make 1-6 actions (where reward=0) not seeing reward until the end of the week
4. Repeat every week until season ends and bonus points are calculated
