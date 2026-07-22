"""Step 5a: real PPO+GAE training loop + actor-critic heads, verified on
H-Rep DeepSet (the simplest faithful encoder) before any multi-encoder
harness exists (Step 5b) or full multi-tier sweep (Step 6).

NOT a modification of ppo_smoke.py: `EncoderActorCritic`/`train_overfit`
there remain exactly as built in Step 4 and are still imported by
test_deepset_encoders.py, test_baseline_encoders.py, and
test_relational_deepset.py for their overfit-one-instance smoke tests --
touching that file would risk those passed gates. This module is a parallel,
more rigorous promotion of the same pattern: masked action space, GAE with
correct episode-boundary handling, advantage normalization, and loss-scale
instrumentation.

MASKED ACTION SPACE -- design decision, not a NavEnv change: NavEnv (Step 1,
frozen) does not itself expose a feasible-action-mask field. Rather than
add one (which would require flagging and reopening a passed gate),
`compute_action_mask` derives the mask externally from information NavEnv
already exposes (`obs["state"]`, `partition.domain`): a direction is
infeasible iff the agent is already at/near a domain edge and that
direction points further outward -- exactly the wall-facing case Step 1's
`DECISION_TIMEOUT_SUBSTEPS` design decision exists to handle (picking such a
direction times out rather than crossing to a new cell). This is a
geometric fact about (position, domain), not an environment-behavior
change, and needs no NavEnv modification.

GAE + episode boundaries: the standard `mask = 0 if done else 1` term in the
backward recursion (see `compute_gae`) zeroes the bootstrap across a
terminal transition -- verified by an explicit hand-computable 2-episode
test in test_ppo_loop.py, not just trusted by inspection.

STEP 5A-FIX (frozen as of this fix -- see ROADMAP.md): the initial 5a loop's
loss-scale diagnostic showed value_loss dominating policy_loss by
1000x-47000x (raw returns reach ~19-28 here: the +10 goal reward plus
shaping), which could dominate the policy gradient through the shared
encoder. Two passes, both kept, neither a new paper-specified hyperparameter:
  1. Per-batch RETURN normalization for the value-loss computation only (the
     critic's raw output stays un-normalized for GAE bootstrapping elsewhere
     via `value.detach()` -- this is local to the loss term). Distinct from,
     and in addition to, the pre-existing ADVANTAGE normalization used in the
     policy loss -- the two do not overwrite or get conflated with each other.
  2. PPO2-style value-loss CLIPPING using the SAME `clip_eps` already used for
     the policy ratio (no new hyperparameter): the value prediction can move
     at most `clip_eps` from its rollout-time estimate per update; loss is the
     max of the clipped/unclipped squared error.
Both confirmed correct via: value_loss now stable ~0.5 (was climbing to
40-60+), gradient-norm ratio into the shared encoder ~8.9x (target 1x-10x,
the raw loss-VALUE ratio is a misleading proxy here since policy_loss is
intrinsically near-zero-mean under normalized advantages), and all 8
correctness tests in test_ppo_loop.py re-passing unchanged.

The EASY-curve plateau this fix was originally hoped to also resolve did NOT
fully clear -- but a follow-up greedy (argmax) per-instance evaluation
proved it is NOT a training-loop bug: with entropy decayed near zero
(converged, not unstable), a 6-clean-instance/100-iteration run solves 4/6
instances at a clean, deterministic 100% and fails the other 2 at a clean,
deterministic 0% (one always walks into an adjacent hazard, one always times
out without reaching goal or hazard) -- ruling out environment stochasticity
and pointing instead to a genuine generalization/capacity characteristic of
this small instance set and training budget, not a scale-mismatch bug. Out
of scope for this fix (by explicit user decision); noted in ROADMAP.md as a
known characteristic, not silently hidden.

This training configuration (both passes above) is FROZEN as the shared
harness Step 5b builds on across all six encoders -- do not tune it further
per-encoder.
"""

import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamics"))
from integrator import DIRECTIONS, DIRECTION_NAMES  # noqa: E402

from nav_env import NavEnv  # noqa: E402

AGENT_STATE_DIM = 4
NUM_ACTIONS = 8


def compute_action_mask(state, domain, tol: float = 1e-6) -> torch.Tensor:
    """(8,) bool mask, True = feasible. See module docstring."""
    x1, x2 = float(state[0]), float(state[1])
    xmin, xmax, ymin, ymax = domain
    mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
    for i, name in enumerate(DIRECTION_NAMES):
        u = DIRECTIONS[name]
        if x1 <= xmin + tol and u[0] < 0:
            mask[i] = False
        if x1 >= xmax - tol and u[0] > 0:
            mask[i] = False
        if x2 <= ymin + tol and u[1] < 0:
            mask[i] = False
        if x2 >= ymax - tol and u[1] > 0:
            mask[i] = False
    return mask


class MaskedEncoderActorCritic(nn.Module):
    """Shared encoder embedding -> separate actor (masked logits) and critic
    (scalar value) heads."""

    def __init__(self, encoder: nn.Module, embedding_dim: int = 128, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.encoder = encoder
        self.actor = nn.Linear(embedding_dim, num_actions)
        self.critic = nn.Linear(embedding_dim, 1)

    def forward(self, partition, instance, agent_state, current_cell=None, mask: Optional[torch.Tensor] = None):
        emb = self.encoder(partition, instance, agent_state, current_cell=current_cell)
        logits = self.actor(emb)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        value = self.critic(emb).squeeze(-1)
        return logits, value


def compute_gae(
    rewards: List[float], values: List[torch.Tensor], dones: List[bool], last_value: torch.Tensor,
    gamma: float = 0.99, lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE-lambda. `dones[t]=True` zeroes the bootstrap term so the
    backward recursion never carries value/advantage across an episode
    boundary -- verified explicitly in test_ppo_loop.py's 2-episode hand-check,
    not just asserted here."""
    T = len(rewards)
    values_ext = values + [last_value]
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values_ext[t + 1].item() * mask - values_ext[t].item()
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t = advantages_t + torch.stack(values)
    return advantages_t, returns_t


# PHA/paper-matching PPO hyperparameters (SPEC.md 2.6), rollout length from
# the existing harness pattern (Rush-hour-git run_*.py's steps_per_rollout,
# scaled down -- this single-encoder verification doesn't need 4096).
DEFAULT_HP = dict(
    lr=1e-4,
    clip_eps=0.2,
    gamma=0.99,
    lam=0.95,
    entropy_coef=0.05,
    entropy_coef_start=None,  # if both start/end set, linearly schedule entropy_coef over n_iterations
    entropy_coef_end=None,
    critic_coef=0.5,
    ppo_epochs=4,
    steps_per_iter=256,
    horizon=40,
)


def _entropy_coef_for_iteration(hp: dict, iteration: int, n_iterations: int) -> float:
    if hp.get("entropy_coef_start") is not None and hp.get("entropy_coef_end") is not None:
        frac = iteration / max(1, n_iterations - 1)
        return hp["entropy_coef_start"] + frac * (hp["entropy_coef_end"] - hp["entropy_coef_start"])
    return hp["entropy_coef"]


def train_ppo(
    encoder_factory,
    partition,
    instances: List,
    n_iterations: int = 60,
    hp: Optional[dict] = None,
    seed: int = 0,
    solve_rate_target: float = 0.9,
    solve_rate_window: int = 30,
    verbose: bool = True,
    return_diagnostics: bool = False,
):
    """Real PPO+GAE training loop over a pool of instances on one partition
    (Step 5a scope: single encoder, single partition/tier -- 5b adds the
    multi-encoder/multi-instance fairness harness).
    """
    hp = {**DEFAULT_HP, **(hp or {})}
    torch.manual_seed(seed)

    model = MaskedEncoderActorCritic(encoder_factory())
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    import numpy as np
    rng = np.random.default_rng(seed)

    def new_env():
        inst = instances[int(rng.integers(0, len(instances)))]
        cfg_env = NavEnv(partition, _instance_to_config(partition, inst), horizon=hp["horizon"])
        return cfg_env, inst

    env, current_instance = new_env()
    obs, _ = env.reset(seed=seed)

    episode_outcomes: List[str] = []
    solve_rate_history = []
    diagnostics = {"policy_loss": [], "value_loss": [], "entropy": [], "adv_mean": [], "adv_std": [],
                   "returns_mean": [], "returns_std": [], "entropy_coef_used": []}

    for iteration in range(n_iterations):
        entropy_coef = _entropy_coef_for_iteration(hp, iteration, n_iterations)
        buf_state, buf_cell, buf_mask, buf_action, buf_logp, buf_val = [], [], [], [], [], []
        buf_reward, buf_done, buf_instance = [], [], []

        for _ in range(hp["steps_per_iter"]):
            agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
            mask = compute_action_mask(obs["state"], partition.domain)
            logits, value = model(partition, current_instance, agent_state, current_cell=obs["cell"], mask=mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            buf_state.append(agent_state)
            buf_cell.append(obs["cell"])
            buf_mask.append(mask)
            buf_instance.append(current_instance)
            buf_action.append(action)
            buf_logp.append(dist.log_prob(action).detach())
            buf_val.append(value.detach())

            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            buf_reward.append(reward)
            buf_done.append(terminated or truncated)

            if terminated or truncated:
                episode_outcomes.append(info.get("outcome") or "truncated")
                env, current_instance = new_env()
                obs, _ = env.reset(seed=seed + len(episode_outcomes))

        with torch.no_grad():
            agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
            _, last_value = model(partition, current_instance, agent_state, current_cell=obs["cell"])

        advantages, returns = compute_gae(buf_reward, buf_val, buf_done, last_value.detach(),
                                           gamma=hp["gamma"], lam=hp["lam"])
        adv_mean_raw, adv_std_raw = advantages.mean().item(), advantages.std().item()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_logp = torch.stack(buf_logp)
        actions = torch.stack(buf_action)
        masks = torch.stack(buf_mask)

        # Step 5a-fix: RETURN normalization, distinct from and in addition to
        # the ADVANTAGE normalization above (not a replacement for it). Raw
        # returns reach ~19-28 here (the +10 goal reward plus shaping), so the
        # unnormalized squared-error value loss was 1000x-47000x the policy
        # loss, swamping the policy gradient through the shared encoder. Per-
        # batch, computed once per iteration (same pattern as advantages, no
        # persistent/running state -- least invasive first, per the brief).
        # Only the LOSS is rescaled: the critic's raw prediction (`value`,
        # used un-normalized for GAE bootstrapping elsewhere via
        # `value.detach()`) is untouched: normalizing it here is local to this
        # loss term and does not change what the critic outputs or how GAE
        # uses it.
        returns_mean, returns_std = returns.mean(), returns.std() + 1e-8

        # Step 5a-fix, second pass: value-loss clipping (per the user's own
        # pre-approved contingency -- normalization alone reduced the raw
        # value_loss magnitude and brought the gradient-norm ratio into the
        # shared encoder to ~8.9x, but the EASY curve still plateaued
        # ~50-70% rather than converging, so this layers on top rather than
        # replacing normalization). Standard PPO2-style clipped value loss:
        # the new value prediction can move at most `clip_eps` (the SAME
        # clip_eps already used for the policy ratio -- no new
        # hyperparameter) away from the ROLLOUT-TIME value estimate per
        # update; the loss is the max of the clipped/unclipped squared
        # error, discouraging large single-update swings in the value
        # function the same way the policy ratio is discouraged from large
        # swings. Computed in the same normalized space as the return
        # normalization above (old values normalized by the same
        # returns_mean/returns_std, so the two fixes compose rather than
        # conflict).
        old_values_norm = (torch.stack(buf_val) - returns_mean) / returns_std

        last_policy_loss = last_value_loss = last_entropy = 0.0
        for _ in range(hp["ppo_epochs"]):
            new_logits, new_values = [], []
            for t in range(hp["steps_per_iter"]):
                logits, value = model(partition, buf_instance[t], buf_state[t], current_cell=buf_cell[t],
                                       mask=masks[t])
                new_logits.append(logits)
                new_values.append(value)
            new_logits = torch.stack(new_logits)
            new_values = torch.stack(new_values)

            dist = torch.distributions.Categorical(logits=new_logits)
            new_logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - hp["clip_eps"], 1 + hp["clip_eps"]) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            returns_norm = (returns - returns_mean) / returns_std
            new_values_norm = (new_values - returns_mean) / returns_std
            value_clipped_norm = old_values_norm + torch.clamp(
                new_values_norm - old_values_norm, -hp["clip_eps"], hp["clip_eps"]
            )
            value_loss_unclipped = (new_values_norm - returns_norm).pow(2)
            value_loss_clipped = (value_clipped_norm - returns_norm).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            loss = policy_loss + hp["critic_coef"] * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_policy_loss, last_value_loss, last_entropy = policy_loss.item(), value_loss.item(), entropy.item()

        diagnostics["policy_loss"].append(last_policy_loss)
        diagnostics["value_loss"].append(last_value_loss)
        diagnostics["entropy"].append(last_entropy)
        diagnostics["adv_mean"].append(adv_mean_raw)
        diagnostics["adv_std"].append(adv_std_raw)
        diagnostics["returns_mean"].append(returns_mean.item())
        diagnostics["returns_std"].append(returns_std.item())
        diagnostics["entropy_coef_used"].append(entropy_coef)

        recent = episode_outcomes[-solve_rate_window:]
        solve_rate = (sum(1 for o in recent if o == "goal") / len(recent)) if recent else 0.0
        solve_rate_history.append(solve_rate)
        if verbose:
            print(f"  iter {iteration:3d}: episodes={len(episode_outcomes):4d}, solve_rate={solve_rate:.2%}, "
                  f"policy_loss={last_policy_loss:+.4f}, value_loss={last_value_loss:.4f}, entropy={last_entropy:.4f}")
        if solve_rate >= solve_rate_target and len(recent) >= solve_rate_window:
            if verbose:
                print(f"  reached solve_rate_target={solve_rate_target:.0%} at iteration {iteration}, stopping early")
            break

    if return_diagnostics:
        return model, solve_rate_history, diagnostics
    return model, solve_rate_history


def make_clean_easy_instances(n: int, seed: int = 0):
    """A small set of EASY instances whose shortest path uses only
    edge-adjacent (Manhattan) hops, never a diagonal corner-only hop -- the
    same trap Step 4a's overfit smoke test hit (a corner-hop instance is
    solvable but genuinely hard to execute, not a clean learning signal).
    Deterministic given `seed`."""
    import numpy as np

    from partitions import BoxGridPartition
    from pool import Instance, solve_path

    partition = BoxGridPartition()
    rng = np.random.default_rng(seed)
    instances = []
    attempts = 0
    while len(instances) < n and attempts < n * 200:
        attempts += 1
        start = int(rng.integers(0, partition.num_cells))
        goal = int(rng.integers(0, partition.num_cells))
        if start == goal:
            continue
        n_hazards = int(rng.integers(1, 3))
        candidates = [c for c in range(partition.num_cells) if c not in (start, goal)]
        rng.shuffle(candidates)
        hazards = tuple(candidates[:n_hazards])

        path = solve_path(partition, start, goal, hazards)
        if path is None:
            continue
        clean = all(
            (abs(partition.cell_row_col(path[i])[0] - partition.cell_row_col(path[i + 1])[0])
             + abs(partition.cell_row_col(path[i])[1] - partition.cell_row_col(path[i + 1])[1])) == 1
            for i in range(len(path) - 1)
        )
        if not clean:
            continue
        instances.append(Instance(tier="easy", partition_seed=0, start_cell=start, goal_cell=goal,
                                   hazard_cells=hazards, initial_velocity_sign=(1, 1)))
    if len(instances) < n:
        raise RuntimeError(f"only found {len(instances)}/{n} clean instances within the attempt budget")
    return partition, instances


def _instance_to_config(partition, instance):
    from nav_env import NavInstanceConfig
    x_lo, x_hi, y_lo, y_hi = partition.cell_bounds(instance.start_cell)
    margin = 0.1 * (x_hi - x_lo)
    start_bounds = ((x_lo + margin, x_hi - margin), (y_lo + margin, y_hi - margin))
    return NavInstanceConfig(
        goal_cell=instance.goal_cell,
        hazard_cells=frozenset(instance.hazard_cells),
        start_bounds=start_bounds,
        v0_bounds=((-0.2, 0.2), (-0.2, 0.2)),
    )
