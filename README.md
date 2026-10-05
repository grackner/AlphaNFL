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
1. Random team is picked for the agent based on `draft_logs`
