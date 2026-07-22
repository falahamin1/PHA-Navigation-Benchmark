"""Step 6 Gate 0-CL: closed-loop reachability oracle.

Gate 0 (the open-loop scripted-path check) found that even after the Step
2c-fix (strict facet-sharing solvability), a residual gap remains: multi-hop
velocity-coupling drift makes an OPEN-LOOP action sequence land in a
valid-but-unintended facet-neighbor, compounding with path length. But a real
policy is never open-loop -- it observes its actual cell every decision step
and replans. This module answers the question that actually determines
sweep-readiness: can a greedy REPLANNING controller reach the goal despite
drift?

`closed_loop_oracle` also doubles as the Step 6 optimality-gap baseline: its
step count on a reached-goal trial is the "oracle steps" a trained policy's
own step count gets compared against, replacing the abstract BFS path length
(which never accounted for drift/replanning at all).

Design notes:
- Planning uses STRICT facet-sharing adjacency (`pool.solve_path`, the fixed
  Step 2c-fix predicate) at EVERY step, recomputed from the object's actual
  current cell (`obs["cell"]`, populated by NavEnv's own permissive-adjacency
  mode-switching -- unchanged, this module doesn't touch NavEnv). This is the
  precise closed-loop analogue of Gate 0's open-loop script: same planning
  predicate, but replanned from real state instead of committed up front.
- No training, no config/dynamics/reward changes: this is a hand-rolled
  greedy oracle, not a trained policy, and touches no other passed layer.
- `safe_start_bounds`/`instance_to_config`/`direction_for_hop` are the same
  tier-agnostic helpers used in the Gate 0 sweep (centroid-based, verified via
  `locate()` -- no dependency on cell_bounds/cell_row_col, so this works
  identically on box/mixed/Voronoi partitions).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamics"))

import numpy as np

from integrator import DIRECTIONS, DIRECTION_NAMES  # noqa: E402
from nav_env import HORIZON, NavEnv, NavInstanceConfig  # noqa: E402
from pool import solve_path  # noqa: E402


def safe_start_bounds(partition, cell_idx, initial_half_width=0.15, min_half_width=0.005):
    """Tier-agnostic interior sampling box: shrink a box around the cell
    centroid until all 4 corners empirically locate() to the same cell.
    Corners are clipped just inside the domain so probing near a domain edge
    never calls locate() on an out-of-domain point."""
    xmin, xmax, ymin, ymax = partition.domain
    eps = 1e-6
    cx, cy = partition.cell_centroid(cell_idx)
    hw = initial_half_width
    while hw >= min_half_width:
        raw_corners = [(cx - hw, cy - hw), (cx - hw, cy + hw), (cx + hw, cy - hw), (cx + hw, cy + hw)]
        corners = [(min(max(x, xmin + eps), xmax - eps), min(max(y, ymin + eps), ymax - eps))
                   for x, y in raw_corners]
        if all(partition.locate(np.array(c)) == cell_idx for c in corners):
            return ((cx - hw, cx + hw), (cy - hw, cy + hw))
        hw *= 0.5
    return ((cx - min_half_width, cx + min_half_width), (cy - min_half_width, cy + min_half_width))


def instance_to_config(partition, instance):
    start_bounds = safe_start_bounds(partition, instance.start_cell)
    return NavInstanceConfig(
        goal_cell=instance.goal_cell,
        hazard_cells=frozenset(instance.hazard_cells),
        start_bounds=start_bounds,
        v0_bounds=((-0.2, 0.2), (-0.2, 0.2)),
    )


def direction_for_hop(partition, cell_a, cell_b):
    """Nearest of the 8 compass directions to the actual centroid->centroid
    displacement vector -- generalizes cleanly to any convex partition."""
    ca = partition.cell_centroid(cell_a)
    cb = partition.cell_centroid(cell_b)
    delta = cb - ca
    delta_unit = delta / np.linalg.norm(delta)
    best_name, best_dot = None, -1e9
    for name in DIRECTION_NAMES:
        dot = float(np.dot(DIRECTIONS[name], delta_unit))
        if dot > best_dot:
            best_dot, best_name = dot, name
    return best_name


def closed_loop_oracle(partition, instance, seed, horizon=None):
    """Greedy replanning controller. At every decision step: observe the
    object's actual current cell, plan a strict-facet-adjacency BFS path
    (hazards removed) from there to the goal, take the first hop's direction,
    step the real NavEnv one decision epoch, repeat.

    Returns (reached_goal: bool, steps: int, trajectory: List[int], outcome: str).
    outcome is one of "goal", "hazard", "stall_timeout", "no_strict_path"
    (the last only if drift ever lands the object in a cell strict-BFS can't
    route out of -- distinct from an ordinary timeout, reported separately).
    """
    if horizon is None:
        horizon = HORIZON
    cfg = instance_to_config(partition, instance)
    env = NavEnv(partition, cfg, horizon=horizon)
    obs, _ = env.reset(seed=seed)
    trajectory = [obs["cell"]]
    steps = 0
    while True:
        current_cell = obs["cell"]
        path = solve_path(partition, current_cell, instance.goal_cell, instance.hazard_cells)
        if path is None or len(path) < 2:
            return False, steps, trajectory, "no_strict_path"
        next_cell = path[1]
        action_id = DIRECTION_NAMES.index(direction_for_hop(partition, current_cell, next_cell))
        obs, reward, terminated, truncated, info = env.step(action_id)
        steps += 1
        trajectory.append(obs["cell"])
        if terminated:
            outcome = info.get("outcome")
            return (outcome == "goal"), steps, trajectory, outcome
        if truncated:
            return False, steps, trajectory, "stall_timeout"
