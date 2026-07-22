"""Step 1: NavEnv -- gymnasium-style env over a pluggable Partition.

One env.step() call == one decision epoch: the agent picks a desired
direction, the physics integrates in fixed dt substeps until the position
crosses into a different cell (or terminates), and that crossing is the next
decision point. Horizon T counts decision steps, not physics substeps.

Two design decisions made here that aren't pinned down by DESIGN.md/SPEC.md
and are worth flagging for review rather than assuming silently:

1. Domain-boundary handling: SPEC.md doesn't say what happens if the agent
   drives itself past the outer [0,5]x[0,5] boundary (e.g. picking "E" from
   the rightmost column). Implemented as a wall: position is clamped to the
   domain and the outward velocity component is zeroed, non-terminal. An
   alternative would be to treat leaving the domain as a hazard-like failure;
   that felt too harsh for what's otherwise just picking a boundary-adjacent
   action once.
2. Potential-based shaping at terminal transitions: Phi(terminal) = 0 (goal
   or hazard), per Ng et al.'s requirement for the invariance guarantee to
   hold on absorbing states. Non-terminal Phi(s) = -||pos - goal_centroid||.
3. Decision-epoch timeout: a decision (agent picks a direction, physics runs
   until the next cell entry) can point straight into a wall from a boundary
   cell -- with wall-clamping there is then no new cell to ever cross into,
   so the epoch would hang forever. Discovered via the no-skip stress test in
   test_nav_env.py, not anticipated by DESIGN.md/SPEC.md. If no mode switch
   happens within DECISION_TIMEOUT_SUBSTEPS, the epoch is forcibly ended
   (non-terminal, cell unchanged, reward computed normally, decision counter
   still consumed) and control returns to the agent. That threshold is set
   far above any legitimately slow crossing (see DECISION_TIMEOUT_SUBSTEPS's
   own comment for the current from-rest worst case) so hitting it reliably
   means "genuinely stuck," not "still crossing." A separate, much higher
   MAX_SUBSTEPS_HARD_CAP remains as a pure safety net for an actual bug (e.g.
   runaway state).

STEP A -- dt recalibration (Step 6 Gate 0-CL fix; ROADMAP.md): the closed-loop
reachability oracle's real-dynamics stepping hit NavEnv's own no-skipped-cells
assertion on HARD -- a single Euler substep at the original dt=0.05 crossed an
entire cell without ever registering the intermediate transition. Root cause:
dt was tuned (Step 1) against EASY's unit-width(1.0) cells; measuring the
ACTUAL cell-scale distribution across all 3 tiers' generated pools found the
true global minimum is on MEDIUM (min linear width 0.1779, thinner than
HARD's own minimum 0.3827 -- MEDIUM's corner-cut template can produce thin
slivers), meaning even non-crashing MEDIUM trials could plausibly have been
suffering the SAME silent skip risk without having tripped the assertion yet.

Fix is UNIFORM-GLOBAL, not per-tier: one dt for all three tiers, sized to the
smallest cell in the ENTIRE benchmark pool (not just the tier that happened to
crash first). Chosen over per-tier because (a) the true worst case wasn't on
the tier the crash surfaced on, which is exactly the kind of thing a per-tier
value derived from "the tier that crashed" would get wrong; (b) it's the only
choice robust to a future pool regeneration drawing a different worst-case
seed; (c) EASY/MEDIUM simply get gratuitously finer (harmless, can't create a
skip) integration, and there is exactly one dt to justify in the write-up, not
three. drift_coupling (the physics) is UNCHANGED -- this only refines the
integration's accuracy at the existing physics, verified by
test_dt_recalibration.py's Step-0 convergence re-check (identical analytic
steady states at the new dt, just reached more precisely).

dt = 0.008 (from a target of >=8 Euler substeps to cross the smallest cell at
steady-state speed -- the same margin the original DRIFT_COUPLING=3.0 tuning
targeted for EASY's unit cells, applied here to the new global-minimum cell
width instead: 0.1779 / (8 * 2.7273 max steady-state speed) = 0.008152,
rounded down slightly for extra margin). See test_dt_recalibration.py for the
full cell-scale measurement and the resulting zero-skips confirmation.

DECISION_TIMEOUT_SUBSTEPS and MAX_SUBSTEPS_HARD_CAP are scaled up alongside dt
(see their own comments below) -- NOT a semantic change to the decision-epoch
model (still one reward/horizon-tick per env.step() call, wall-timeout still
non-terminal), but a NECESSARY companion adjustment: a legitimate from-rest
crossing of the largest cell in the pool (diameter 2.9010, HARD) now takes
~232 substeps at the new dt (was ~38 at the old dt) -- both constants are
rescaled to preserve the SAME safety-margin ratio over that figure the
original values held over the old dt's worst case, not left at their old
absolute numbers (which would still technically avoid a crash today, but with
far less margin than originally intended).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import FrozenSet, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamics"))
from integrator import A_V_C2E2, DIRECTION_NAMES, DIRECTIONS, DRIFT_COUPLING, euler_step  # noqa: E402

from partitions import BoxGridPartition, Partition  # noqa: E402

STEP_PENALTY = -0.01
GOAL_REWARD = 10.0
HAZARD_REWARD = -10.0
GAMMA = 0.99
HORIZON = 200

# STEP A (Step 6 Gate 0-CL fix): was 0.05, tuned only against EASY's
# unit-width cells. Recalibrated to the empirical global-minimum cell width
# across all 3 tiers' pools (0.1779, on MEDIUM) with an 8-substep-at-steady-
# state safety margin -- see this module's docstring and
# test_dt_recalibration.py. Uniform across all tiers/encoders by design (see
# docstring for the uniform-global-vs-per-tier justification).
DT = 0.008

# If no mode switch happens within this many substeps, the decision epoch is
# genuinely stuck (e.g. a wall-blocked direction), not just slow -- forcibly
# end it and return control to the agent. See design decision #3 above.
# STEP A: was 500 (worst legitimate from-rest crossing ~26-29 substeps at the
# old dt=0.05, ~17x margin). At the new dt=0.008 the same largest-cell
# crossing (diameter 2.9010, HARD) takes ~232 substeps from rest -- rescaled
# to 3000 to preserve a comparable (~13x) margin, not left at the old absolute
# value (see test_dt_recalibration.py for the from-rest measurement).
DECISION_TIMEOUT_SUBSTEPS = 3000

# True safety net: only fires on an actual bug (e.g. runaway state from a
# badly mistuned drift_coupling/dt), never in normal operation.
# STEP A: was 5000 (10x the old DECISION_TIMEOUT_SUBSTEPS); rescaled to 10x
# the new value, same ratio.
MAX_SUBSTEPS_HARD_CAP = 30000


@dataclass
class NavInstanceConfig:
    goal_cell: int
    hazard_cells: FrozenSet[int]
    start_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.5, 1.5), (0.5, 1.5))
    v0_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.2, 0.2), (-0.2, 0.2))


def default_easy_config(partition: BoxGridPartition) -> NavInstanceConfig:
    """SPEC.md 2.2/2.3 EASY defaults: goal = top-right cell, one interior hazard
    off the start->goal diagonal so the two don't accidentally collide."""
    goal_cell = partition.cell_index(partition.num_rows - 1, partition.num_cols - 1)
    hazard_cell = partition.cell_index(1, 3)
    return NavInstanceConfig(goal_cell=goal_cell, hazard_cells=frozenset({hazard_cell}))


class NavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        partition: Partition,
        instance_config: NavInstanceConfig,
        dt: float = DT,
        horizon: int = HORIZON,
        gamma: float = GAMMA,
        drift_coupling: float = DRIFT_COUPLING,
        A_v: np.ndarray = A_V_C2E2,
    ):
        super().__init__()
        self.partition = partition
        self.config = instance_config
        self.dt = dt
        self.horizon = horizon
        self.gamma = gamma
        self.k = drift_coupling
        self.A_v = A_v

        self.action_space = spaces.Discrete(len(DIRECTION_NAMES))
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "cell": spaces.Discrete(partition.num_cells),
        })

        self._goal_centroid = partition.cell_centroid(instance_config.goal_cell)
        self.state = None
        self.cell = None
        self.t = 0

    def _potential(self, pos: np.ndarray) -> float:
        return -np.linalg.norm(pos - self._goal_centroid)

    def _clamp_to_domain(self, pos: np.ndarray, vel: np.ndarray):
        xmin, xmax, ymin, ymax = self.partition.domain
        x, y = pos
        vx, vy = vel
        if x < xmin:
            x, vx = xmin, max(vx, 0.0)
        elif x > xmax:
            x, vx = xmax, min(vx, 0.0)
        if y < ymin:
            y, vy = ymin, max(vy, 0.0)
        elif y > ymax:
            y, vy = ymax, min(vy, 0.0)
        return np.array([x, y]), np.array([vx, vy])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "start_state" in options:
            self.state = np.array(options["start_state"], dtype=np.float64)
        else:
            (x_lo, x_hi), (y_lo, y_hi) = self.config.start_bounds
            (vx_lo, vx_hi), (vy_lo, vy_hi) = self.config.v0_bounds
            x = self.np_random.uniform(x_lo, x_hi)
            y = self.np_random.uniform(y_lo, y_hi)
            vx = self.np_random.uniform(vx_lo, vx_hi)
            vy = self.np_random.uniform(vy_lo, vy_hi)
            self.state = np.array([x, y, vx, vy])

        self.cell = self.partition.locate(self.state[:2])
        self.t = 0
        return self._obs(), {"cell": self.cell}

    def _obs(self):
        return {
            "state": self.state.copy(),
            "cell": self.cell,
            "goal_cell": self.config.goal_cell,
            "hazard_cells": self.config.hazard_cells,
            "is_goal": self.cell == self.config.goal_cell,
            "is_hazard": self.cell in self.config.hazard_cells,
        }

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("call reset() before step()")

        u_dir = DIRECTIONS[DIRECTION_NAMES[action]]
        start_pos = self.state[:2].copy()
        phi_s = self._potential(start_pos)

        terminated = False
        outcome = None
        substeps = 0

        while True:
            substeps += 1
            if substeps > MAX_SUBSTEPS_HARD_CAP:
                raise RuntimeError(
                    f"decision step exceeded the {MAX_SUBSTEPS_HARD_CAP}-substep hard cap -- this is "
                    "well past DECISION_TIMEOUT_SUBSTEPS too, so this is a real bug (e.g. runaway "
                    "state), not a wall-stuck action"
                )

            self.state = euler_step(self.state, u_dir, dt=self.dt, A_v=self.A_v, k=self.k)
            pos, vel = self._clamp_to_domain(self.state[:2], self.state[2:4])
            self.state = np.concatenate([pos, vel])

            new_cell = self.partition.locate(self.state[:2])

            if new_cell != self.cell:
                assert self.partition.neighbors_adjacent(self.cell, new_cell), (
                    f"skipped-cell violation: {self.cell} -> {new_cell} not adjacent "
                    f"(substep displacement too large for this cell size; retune drift_coupling/dt)"
                )
                self.cell = new_cell
                if new_cell == self.config.goal_cell:
                    terminated, outcome = True, "goal"
                elif new_cell in self.config.hazard_cells:
                    terminated, outcome = True, "hazard"
                break  # new decision point (or terminal) either way

            if substeps >= DECISION_TIMEOUT_SUBSTEPS:
                outcome = "timeout"  # genuinely stuck (e.g. wall-blocked direction); see decision #3
                break

        self.t += 1
        truncated = (not terminated) and (self.t >= self.horizon)

        phi_next = 0.0 if terminated else self._potential(self.state[:2])
        shaping = self.gamma * phi_next - phi_s
        reward = STEP_PENALTY + shaping
        if outcome == "goal":
            reward += GOAL_REWARD
        elif outcome == "hazard":
            reward += HAZARD_REWARD

        info = {"outcome": outcome, "substeps": substeps, "decision_step": self.t}
        return self._obs(), reward, terminated, truncated, info
