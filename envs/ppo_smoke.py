"""Step 4a overfit-one-instance smoke test harness.

Not the Step 4b training loop -- a minimal PPO+GAE loop (adapted from this
repo's existing Rush-hour-git/PPOBuffer.py pattern, not copied verbatim)
whose only job is to prove an encoder is wired correctly by memorizing a
single fixed EASY instance to near-100% solve rate. Any encoder that can't
do this has a wiring bug, not a learning-difficulty problem.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nav_env import NavEnv, NavInstanceConfig
from partitions import BoxGridPartition
from pool import Instance


def make_easy_single_instance(seed: int = 0):
    """One fixed, solvable EASY instance, deliberately hand-picked rather
    than taken from generate_pool's first random draw: start=(0,0),
    goal=(0,2) -- a 2-hop, purely EDGE-adjacent path (both hops share a full
    facet, not a corner) -- with the hazard placed at the far corner (4,4),
    clear of that path. This matters: an early pool seed checked here
    (partition_seed=0's first instance, start=21/goal=15) turned out to have
    a shortest path that only exists via a *diagonal corner-only* hop
    (`neighbors_adjacent` permits it, matching pool.py's solvability
    predicate, but physically executing it requires threading exactly
    through a single point, with a hazard cell immediately adjacent to the
    overshoot direction) -- a genuinely hard control problem, not a wiring
    bug, and a bad choice for a smoke test whose entire point is to isolate
    "broken" from "hard". This instance is chosen to be unambiguously easy to
    execute, so a plateau below ~100% here is a real signal, not task noise.
    """
    partition = BoxGridPartition()
    start_cell = partition.cell_index(0, 0)
    goal_cell = partition.cell_index(0, 2)
    hazard_cells = (partition.cell_index(4, 4),)
    instance = Instance(
        tier="easy", partition_seed=0, start_cell=start_cell, goal_cell=goal_cell,
        hazard_cells=hazard_cells, initial_velocity_sign=(1, 1),
    )

    x_lo, x_hi, y_lo, y_hi = partition.cell_bounds(start_cell)
    margin = 0.1 * (x_hi - x_lo)
    start_bounds = ((x_lo + margin, x_hi - margin), (y_lo + margin, y_hi - margin))

    config = NavInstanceConfig(
        goal_cell=goal_cell,
        hazard_cells=frozenset(hazard_cells),
        start_bounds=start_bounds,
        v0_bounds=((-0.2, 0.2), (-0.2, 0.2)),
    )
    return partition, config, instance


# A bad/random early policy can wander for the full default horizon (200
# decision steps) before truncating, which starves a short smoke-test
# rollout (steps_per_iter ~128) of completed episodes. NavEnv's horizon is
# already a normal constructor parameter (not a frozen-layer change) --
# using a much shorter one here is a smoke-test-only convenience so bad
# early episodes fail fast and the policy gets more, shorter episodes of
# signal per iteration.
SMOKE_HORIZON = 30


class EncoderActorCritic(nn.Module):
    """Wraps any Step 3b/4a encoder (matching `forward(partition, instance,
    agent_state, current_cell=None) -> (embedding_dim,)`) with actor/critic
    heads. Not the Step 4b architecture -- a minimal wrapper for this smoke
    test only."""

    def __init__(self, encoder: nn.Module, embedding_dim: int = 128, num_actions: int = 8):
        super().__init__()
        self.encoder = encoder
        self.actor = nn.Linear(embedding_dim, num_actions)
        self.critic = nn.Linear(embedding_dim, 1)

    def forward(self, partition, instance, agent_state, current_cell=None):
        emb = self.encoder(partition, instance, agent_state, current_cell=current_cell)
        return self.actor(emb), self.critic(emb).squeeze(-1)


def train_overfit(
    encoder: nn.Module,
    n_iterations: int = 50,
    steps_per_iter: int = 128,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.02,
    critic_coef: float = 0.5,
    seed: int = 0,
    solve_rate_target: float = 0.95,
    solve_rate_window: int = 20,
    verbose: bool = True,
):
    """Minimal PPO+GAE loop over a single fixed EASY instance. Returns
    (model, solve_rate_history) -- solve_rate_history[i] is the fraction of
    the last `solve_rate_window` episodes (as of iteration i) that reached
    the goal."""
    torch.manual_seed(seed)
    partition, config, instance = make_easy_single_instance(seed=0)
    env = NavEnv(partition, config, horizon=SMOKE_HORIZON)
    model = EncoderActorCritic(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    episode_outcomes = []  # rolling list of "goal"/"hazard"/"timeout-truncated"
    solve_rate_history = []

    obs, _ = env.reset(seed=seed)

    for iteration in range(n_iterations):
        buf_logp, buf_val, buf_reward, buf_done, buf_action = [], [], [], [], []
        buf_state, buf_cell = [], []

        for _ in range(steps_per_iter):
            agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
            current_cell = obs["cell"]
            logits, value = model(partition, instance, agent_state, current_cell=current_cell)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            buf_state.append(agent_state)
            buf_cell.append(current_cell)
            buf_action.append(action)
            buf_logp.append(dist.log_prob(action).detach())
            buf_val.append(value.detach())

            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            buf_reward.append(reward)
            buf_done.append(terminated or truncated)

            if terminated or truncated:
                episode_outcomes.append(info.get("outcome") or "truncated")
                obs, _ = env.reset(seed=seed + len(episode_outcomes))

        # Bootstrap value for the final state.
        with torch.no_grad():
            agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
            _, last_value = model(partition, instance, agent_state, current_cell=obs["cell"])
        values = buf_val + [last_value.detach()]

        # GAE-lambda advantages.
        advantages = [0.0] * steps_per_iter
        gae = 0.0
        for t in reversed(range(steps_per_iter)):
            mask = 0.0 if buf_done[t] else 1.0
            delta = buf_reward[t] + gamma * values[t + 1].item() * mask - values[t].item()
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.stack(buf_val)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_logp = torch.stack(buf_logp)
        actions = torch.stack(buf_action)

        for _ in range(ppo_epochs):
            new_logits, new_values = [], []
            for t in range(steps_per_iter):
                logits, value = model(partition, instance, buf_state[t], current_cell=buf_cell[t])
                new_logits.append(logits)
                new_values.append(value)
            new_logits = torch.stack(new_logits)
            new_values = torch.stack(new_values)

            dist = torch.distributions.Categorical(logits=new_logits)
            new_logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            loss = policy_loss + critic_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        recent = episode_outcomes[-solve_rate_window:]
        solve_rate = (sum(1 for o in recent if o == "goal") / len(recent)) if recent else 0.0
        solve_rate_history.append(solve_rate)
        if verbose:
            print(f"  iter {iteration:3d}: episodes so far={len(episode_outcomes):4d}, "
                  f"recent solve_rate={solve_rate:.2%}")
        if solve_rate >= solve_rate_target and len(recent) >= solve_rate_window:
            if verbose:
                print(f"  reached solve_rate_target={solve_rate_target:.0%} at iteration {iteration}, stopping early")
            break

    return model, solve_rate_history
