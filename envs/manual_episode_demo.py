"""Step 1 gate: a hand-solvable scripted episode.

Start at cell (row=0,col=0), goal at (row=4,col=4) (top-right), hazard at
(row=1,col=3). Repeatedly choosing "NE" walks the diagonal
(0,0)->(1,1)->(2,2)->(3,3)->(4,4)=goal, which never touches the hazard cell.
Prints (position, cell, reward, terminated) per decision step for manual
verification against the reward formula in nav_env.py.
"""

import numpy as np

from nav_env import GAMMA, GOAL_REWARD, STEP_PENALTY, NavEnv, default_easy_config
from partitions import BoxGridPartition

partition = BoxGridPartition(num_rows=5, num_cols=5)
config = default_easy_config(partition)
env = NavEnv(partition, config)

start_state = np.array([0.5, 0.5, 0.0, 0.0])
obs, info = env.reset(seed=0, options={"start_state": start_state})
goal_centroid = partition.cell_centroid(config.goal_cell)

print(f"goal cell = {config.goal_cell} {partition.cell_row_col(config.goal_cell)}, "
      f"centroid = {goal_centroid}")
print(f"hazard cells = {config.hazard_cells} "
      f"{[partition.cell_row_col(c) for c in config.hazard_cells]}")
print(f"start: pos={obs['state'][:2]}, cell={obs['cell']} {partition.cell_row_col(obs['cell'])}\n")

NE_ACTION = 1  # DIRECTION_NAMES = [E, NE, N, NW, W, SW, S, SE]
step_num = 0
while True:
    step_num += 1
    pos_before = env.state[:2].copy()
    phi_before = -np.linalg.norm(pos_before - goal_centroid)

    obs, reward, terminated, truncated, info = env.step(NE_ACTION)

    pos_after = obs["state"][:2]
    phi_after = 0.0 if terminated else -np.linalg.norm(pos_after - goal_centroid)
    shaping = GAMMA * phi_after - phi_before
    hand_reward = STEP_PENALTY + shaping + (GOAL_REWARD if info["outcome"] == "goal" else 0.0)

    print(f"step {step_num}: pos={pos_after}, cell={obs['cell']} {partition.cell_row_col(obs['cell'])}, "
          f"substeps={info['substeps']}, reward={reward:.6f} (hand={hand_reward:.6f}), "
          f"terminated={terminated}, truncated={truncated}, outcome={info['outcome']}")

    assert abs(reward - hand_reward) < 1e-9, "printed reward doesn't match the formula -- gate FAILED"

    if terminated or truncated:
        break

assert info["outcome"] == "goal", f"expected the diagonal path to reach the goal, got {info['outcome']}"
print(f"\nGate: reached goal in {step_num} decision steps via repeated NE, "
      f"every printed reward reproduced by hand. PASS.")
