# 🧊 Frozen Lake Reinforcement Learning

A Q-Learning implementation to solve the classic **Frozen Lake** environment from OpenAI Gymnasium. This project explores reinforcement learning fundamentals including Q-table learning, curriculum learning, and hyperparameter tuning.

---

## 📁 Project Structure

```
├── frozen_lake.py                  # Core Q-Learning agent and training logic
├── demo.py                         # Demo script to visualize trained agent
├── curriculum_learning.py          # Curriculum learning implementation
├── frozen_lake_slippy_final.ipynb  # Notebook: Stochastic (slippery) environment
├── frozen_non_slippery.ipynb       # Notebook: Deterministic (non-slippery) environment
├── q_table_frozenlake.npy          # Saved Q-table (trained weights)
├── curriculum_learning.png         # Curriculum learning reward curve
├── demo_comparison.png             # Comparison of agent performance
├── demo_policy_map.png             # Visualized policy map
├── demo_qtable_heatmap.png         # Q-table heatmap visualization
├── frozen_lake_complete.png        # Training results - complete run
├── frozen_lake_final.png           # Final training performance plot
├── hyperparameter_experiments.png  # Hyperparameter tuning results
└── reward_curve.png                # Training reward curve
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install gymnasium numpy matplotlib
```

### Train the Agent

```bash
python frozen_lake.py
```

### Run the Demo

```bash
python demo.py
```

### Curriculum Learning

```bash
python curriculum_learning.py
```

---

## Algorithm: Q-Learning

This project uses **Q-Learning**, a model-free, off-policy reinforcement learning algorithm. The agent learns a Q-table mapping state-action pairs to expected cumulative rewards.

**Q-value update rule:**

```
Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Learning Rate | α | How fast the agent updates Q-values |
| Discount Factor | γ | How much future rewards are valued |
| Exploration Rate | ε | Probability of random action (epsilon-greedy) |

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

---

## Results

| Experiment | Description |
|------------|-------------|
| `reward_curve.png` | Agent reward over training episodes |
| `frozen_lake_final.png` | Final training performance |
| `demo_policy_map.png` | Best action per state after training |
| `demo_qtable_heatmap.png` | Heatmap of learned Q-values |
| `hyperparameter_experiments.png` | Effect of varying α, γ, ε |
| `curriculum_learning.png` | Reward with curriculum training strategy |

---

## Curriculum Learning

The `curriculum_learning.py` script trains the agent progressively — starting on easier configurations before introducing the full stochastic environment. This approach speeds up convergence and improves final performance.

---

## Saved Model

The trained Q-table is saved as `q_table_frozenlake.npy` and can be loaded for inference:

```python
import numpy as np
q_table = np.load("q_table_frozenlake.npy")
```

---

## References

- [OpenAI Gymnasium — Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- Watkins, C.J.C.H. (1989). *Learning from Delayed Rewards*
- Sutton & Barto — *Reinforcement Learning: An Introduction*

---

## Algorithm Choice: Q-Learning

Q-Learning was chosen for three reasons:
1. **Small discrete state space** — FrozenLake has only 16 states and 4 actions, making a tabular Q-table (16×4) perfectly sufficient. Deep RL methods like DQN would be overkill.
2. **Deterministic environment** — with `is_slippery=False`, actions have guaranteed outcomes, so Q-Learning converges quickly and reliably without needing a model of the environment.
3. **Off-policy learning** — Q-Learning updates towards the greedy max action regardless of what the agent actually did, which means it can learn the optimal policy even while exploring randomly during early episodes.
