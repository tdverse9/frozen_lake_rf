# Frozen Lake Reinforcement Learning

A Q-Learning implementation to solve the classic **Frozen Lake** environment from OpenAI Gymnasium. This project explores reinforcement learning fundamentals including Q-table learning, curriculum learning, and hyperparameter tuning.

---

## Environment

**Frozen Lake** is a grid-world where the agent must navigate from the **Start (S)** to the **Goal (G)** without falling into **Holes (H)**.

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- `S` — Start position
- `F` — Frozen (safe) tile
- `H` — Hole (episode ends, reward = 0)
- `G` — Goal (reward = 1)

Two variants are explored:
- **Slippery** — Agent may slip in unintended directions (stochastic)
- **Non-Slippery** — Agent moves deterministically


## Project Structure

```
frozen_lake_combined.ipynb → Combined training notebook

frozen_non_slippery.ipynb → Deterministic environment

README.md → Project documentation


```
## Algorithm: Q-Learning

Q-Learning was chosen for three reasons:
1. **Small discrete state space** — FrozenLake has only 16 states and 4 actions, making a tabular Q-table (16×4) perfectly sufficient. Deep RL methods like DQN would be overkill.
2. **Deterministic environment** — with `is_slippery=False`, actions have guaranteed outcomes, so Q-Learning converges quickly and reliably without needing a model of the environment.
3. **Off-policy learning** — Q-Learning updates towards the greedy max action regardless of what the agent actually did, which means it can learn the optimal policy even while exploring randomly during early episodes.


**Q-value update rule:**

```
Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```
 Learning Rate - How fast the agent updates Q-values 
 Discount Factor -How much future rewards are valued 
 Exploration Rate - Probability of random action (epsilon-greedy) 

---

# Notebook 1 — frozen_non_slippery.ipynb
Overview-
Implements Q-Learning from scratch on the deterministic (non-slippery) version of FrozenLake. This serves as the foundation and baseline for the curriculum learning experiments.

##  #Notebook 2 — frozen_lake_combined.ipynb
Overview -
Extends the problem to a stochastic environment with a custom 20% slip probability. Under this setting, the agent only moves in the intended direction 80% of the time — slipping sideways (left or right of intended) with 10% each. A simple Q-Learner's performance drops significantly under this stochasticity. This notebook applies Curriculum Learning to recover performance within a budget of just 1000 total episodes.

---
### In future work, we aim to explore Proximal Policy Optimization (PPO)
Currently, we do not have deep practical experience with PPO, but we plan to study and implement it to replace the Q-table with a neural network–based policy
---
## References

- [OpenAI Gymnasium — Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- Watkins, C.J.C.H. (1989). *Learning from Delayed Rewards*
- Sutton & Barto — *Reinforcement Learning: An Introduction*
---
