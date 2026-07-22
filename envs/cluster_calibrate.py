"""Step 5b-cal, Parts 2+3: cluster budget calibration + entropy re-check.

Meant to run on Alpine via the Slurm scripts in jobs/ (see jobs/README.md),
not locally -- local runs showed ~7-10s/iteration for H-Rep/EASY, so a
several-thousand-iteration ceiling with periodic dual-mode eval is an
hours-to-a-day job, exactly what the tangram-git/Rush-hour-git jobs already
budget 12-24h wall-time for.

One parameterized script serves both Part 2 (budget calibration,
entropy_coef=0.05 default) and Part 3 (entropy re-check, entropy_coef=0.01)
-- same training loop, same dual-mode+entropy logging, just a different
--entropy-coef flag, so the two runs are directly comparable apples-to-
apples (identical seed/config/checkpoint schedule).

Checkpointing/resume: periodically saves (model, optimizer, iteration,
running RNG-relevant counters) to --checkpoint-dir. If that directory
already has a checkpoint, resumes from it -- same contract as the existing
tangram-git jobs ("if the job times out or crashes, just resubmit this same
script"). Determinism note: episode/reset-seed assignment is a pure
function of (seed, episode_index) (see fairness_harness._episode_assignment),
so resuming mid-run and continuing the episode counter reproduces the
identical instance stream a single uninterrupted run would have seen --
resuming does not perturb Invariant 1.

Convergence criteria to look for in the reported curves (Step 5b-cal):
  1. entropy has meaningfully decayed from ~1.8 (policy became decisive)
  2. greedy held-out solve rate has plateaued (more training doesn't help)
  3. greedy and stochastic solve rates have converged toward each other
     (gap closing = decisive, not diffuse-but-lucky)
All three together, not any one alone, mark the calibration point.
"""
import argparse
import json
import os

import numpy as np
import torch

from fairness_harness import (
    ENCODER_REGISTRY,
    FROZEN_CONFIG,
    RESULTS_DIR,
    _episode_assignment,
    _evaluate,
    get_final_pool,
)
from nav_env import NavEnv
from pool import build_partition
from closed_loop_oracle import instance_to_config
from ppo_train import MaskedEncoderActorCritic, compute_action_mask, compute_gae

TIER = "easy"
ENCODER = "hrep"

# Checkpoint schedule: frequent early (behavior changes fastest early in
# training), sparser later. Intermediate checkpoints use a moderate eval
# sample (cheap enough to run often); the LAST N_FULL_EVAL_CHECKPOINTS
# checkpoints use the full 203-instance test set at the full reset count,
# to pin down the exact convergence point precisely once it's in view.
DEFAULT_CHECKPOINTS = (
    list(range(100, 1000, 100)) + list(range(1000, 3000, 250)) + list(range(3000, 5001, 500))
)
N_FULL_EVAL_CHECKPOINTS = 4
INTERMEDIATE_EVAL_SAMPLE_SIZE = 100
INTERMEDIATE_EVAL_RESETS = 20
FULL_EVAL_RESETS = 30


def _checkpoint_paths(checkpoint_dir):
    return (os.path.join(checkpoint_dir, "state.pt"), os.path.join(checkpoint_dir, "curve.json"))


def run(entropy_coef, max_iterations, checkpoints, seed, checkpoint_dir, tier=TIER, encoder=ENCODER,
        verbose=True):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_path, curve_path = _checkpoint_paths(checkpoint_dir)

    cfg = {**FROZEN_CONFIG, **{"entropy_coef": entropy_coef}}
    train_instances, test_instances = get_final_pool(tier)
    partition_cache = {}

    def get_partition(pseed):
        if pseed not in partition_cache:
            partition_cache[pseed] = build_partition(tier, pseed)
        return partition_cache[pseed]

    torch.manual_seed(seed)
    model = MaskedEncoderActorCritic(ENCODER_REGISTRY[encoder]())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    action_gen = torch.Generator().manual_seed(seed)

    start_iteration = 0
    episode_outcomes = []
    curve = {"iteration": [], "entropy": [], "greedy_solve": [], "stochastic_solve": [],
             "greedy_dist_100": [], "greedy_dist_0": [], "n_eval_instances": [], "elapsed_s": []}

    if os.path.exists(state_path):
        print(f"Resuming from checkpoint at {state_path}", flush=True)
        ckpt = torch.load(state_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        action_gen.set_state(ckpt["action_gen_state"])
        start_iteration = ckpt["iteration"]
        episode_outcomes = ckpt["episode_outcomes"]
        with open(curve_path) as f:
            curve = json.load(f)
        print(f"Resumed at iteration {start_iteration}", flush=True)

    episode_count = start_iteration  # placeholder, corrected below if resuming mid-episode-count
    if os.path.exists(state_path):
        episode_count = ckpt["episode_count"]

    def new_env():
        nonlocal episode_count
        idx, reset_seed = _episode_assignment(seed, episode_count, len(train_instances))
        inst = train_instances[idx]
        partition = get_partition(inst.partition_seed)
        env_cfg = instance_to_config(partition, inst)
        env = NavEnv(partition, env_cfg, horizon=cfg["horizon"])
        obs, _ = env.reset(seed=reset_seed)
        episode_count += 1
        return env, inst, partition, obs

    env, current_instance, current_partition, obs = new_env()

    import time
    t_start = time.time()

    for iteration in range(start_iteration, max_iterations):
        buf_state, buf_cell, buf_mask, buf_action, buf_logp, buf_val = [], [], [], [], [], []
        buf_reward, buf_done, buf_instance, buf_partition = [], [], [], []

        for _ in range(cfg["steps_per_iter"]):
            agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
            mask = compute_action_mask(obs["state"], current_partition.domain)
            logits, value = model(current_partition, current_instance, agent_state,
                                   current_cell=obs["cell"], mask=mask)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1, generator=action_gen).squeeze(-1)
            log_prob = torch.distributions.Categorical(logits=logits).log_prob(action)

            buf_state.append(agent_state)
            buf_cell.append(obs["cell"])
            buf_mask.append(mask)
            buf_instance.append(current_instance)
            buf_partition.append(current_partition)
            buf_action.append(action)
            buf_logp.append(log_prob.detach())
            buf_val.append(value.detach())

            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            buf_reward.append(reward)
            buf_done.append(terminated or truncated)

            if terminated or truncated:
                episode_outcomes.append(info.get("outcome") or "truncated")
                env, current_instance, current_partition, obs = new_env()

        with torch.no_grad():
            agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
            _, last_value = model(current_partition, current_instance, agent_state, current_cell=obs["cell"])

        advantages, returns = compute_gae(buf_reward, buf_val, buf_done, last_value.detach(),
                                           gamma=cfg["gamma"], lam=cfg["lam"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_logp = torch.stack(buf_logp)
        actions = torch.stack(buf_action)
        masks = torch.stack(buf_mask)
        returns_mean, returns_std = returns.mean(), returns.std() + 1e-8
        old_values_norm = (torch.stack(buf_val) - returns_mean) / returns_std

        last_entropy = 0.0
        for _epoch in range(cfg["ppo_epochs"]):
            new_logits, new_values = [], []
            for t in range(cfg["steps_per_iter"]):
                logits, value = model(buf_partition[t], buf_instance[t], buf_state[t],
                                       current_cell=buf_cell[t], mask=masks[t])
                new_logits.append(logits)
                new_values.append(value)
            new_logits = torch.stack(new_logits)
            new_values = torch.stack(new_values)

            dist = torch.distributions.Categorical(logits=new_logits)
            new_logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            returns_norm = (returns - returns_mean) / returns_std
            new_values_norm = (new_values - returns_mean) / returns_std
            value_clipped_norm = old_values_norm + torch.clamp(
                new_values_norm - old_values_norm, -cfg["clip_eps"], cfg["clip_eps"]
            )
            value_loss_unclipped = (new_values_norm - returns_norm).pow(2)
            value_loss_clipped = (value_clipped_norm - returns_norm).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            loss = policy_loss + cfg["critic_coef"] * value_loss - cfg["entropy_coef"] * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_entropy = entropy.item()

        if (iteration + 1) in checkpoints:
            is_final_stretch = checkpoints.index(iteration + 1) >= len(checkpoints) - N_FULL_EVAL_CHECKPOINTS
            if is_final_stretch:
                eval_instances, n_resets = test_instances, FULL_EVAL_RESETS
            else:
                rng = np.random.default_rng(999)
                idx = rng.choice(len(test_instances), size=min(INTERMEDIATE_EVAL_SAMPLE_SIZE, len(test_instances)),
                                  replace=False)
                eval_instances, n_resets = [test_instances[i] for i in idx], INTERMEDIATE_EVAL_RESETS

            greedy = _evaluate(model, tier, eval_instances, partition_cache, n_resets=n_resets,
                                horizon=cfg["horizon"], mode="greedy")
            stochastic_rng = np.random.default_rng(seed + iteration)
            stochastic = _evaluate(model, tier, eval_instances, partition_cache, n_resets=n_resets,
                                    horizon=cfg["horizon"], mode="stochastic", stochastic_rng=stochastic_rng)
            greedy_rates = [v["solve_rate"] for v in greedy.values()]
            greedy_solve = float(np.mean(greedy_rates))
            stochastic_solve = float(np.mean([v["solve_rate"] for v in stochastic.values()]))
            dist_100 = sum(1 for r in greedy_rates if r == 1.0)
            dist_0 = sum(1 for r in greedy_rates if r == 0.0)
            elapsed = time.time() - t_start

            curve["iteration"].append(iteration + 1)
            curve["entropy"].append(last_entropy)
            curve["greedy_solve"].append(greedy_solve)
            curve["stochastic_solve"].append(stochastic_solve)
            curve["greedy_dist_100"].append(dist_100)
            curve["greedy_dist_0"].append(dist_0)
            curve["n_eval_instances"].append(len(eval_instances))
            curve["elapsed_s"].append(elapsed)

            if verbose:
                print(f"  [entropy_coef={entropy_coef}] iter={iteration + 1:5d} "
                      f"({'FULL' if is_final_stretch else 'sample'} n={len(eval_instances)}x{n_resets}): "
                      f"entropy={last_entropy:.4f}, greedy={greedy_solve:.2%}, stochastic={stochastic_solve:.2%}, "
                      f"gap={stochastic_solve - greedy_solve:+.2%}, elapsed={elapsed:.0f}s", flush=True)

            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "action_gen_state": action_gen.get_state(), "iteration": iteration + 1,
                "episode_outcomes": episode_outcomes, "episode_count": episode_count,
            }, state_path)
            with open(curve_path, "w") as f:
                json.dump(curve, f, indent=2)

    return model, curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or os.path.join(
        RESULTS_DIR, f"cluster_calibrate_entropy{args.entropy_coef}_seed{args.seed}"
    )
    checkpoints = [c for c in DEFAULT_CHECKPOINTS if c <= args.max_iterations]
    if checkpoints[-1] != args.max_iterations:
        checkpoints.append(args.max_iterations)

    print(f"=== cluster_calibrate: entropy_coef={args.entropy_coef}, max_iterations={args.max_iterations}, "
          f"seed={args.seed}, checkpoint_dir={checkpoint_dir} ===", flush=True)
    model, curve = run(args.entropy_coef, args.max_iterations, checkpoints, args.seed, checkpoint_dir)

    out_path = args.output or os.path.join(RESULTS_DIR, f"cluster_calibrate_entropy{args.entropy_coef}"
                                            f"_seed{args.seed}_curve.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"entropy_coef": args.entropy_coef, "max_iterations": args.max_iterations,
                   "seed": args.seed, "curve": curve}, f, indent=2)
    print(f"\nFinal curve written to {out_path}", flush=True)
    print("DONE")
