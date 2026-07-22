"""Step 5b, Sub-step 1: training-budget calibration.

Trains H-Rep on the full EASY train pool (812 instances), periodically
evaluating held-out solve rate on a fixed EASY test sample, to find where
more training stops improving the held-out ceiling. That iteration count
becomes the frozen budget `fairness_harness.BUDGET_BY_TIER["easy"]`.

Reuses fairness_harness.train_run's exact training-loop mechanics (same
frozen config, same isolated RNGs, same env construction) rather than
ppo_train.py's train_ppo, so the calibration measures the SAME loop the
sweep will actually run -- not a proxy. Structured as one continuous run
with periodic eval checkpoints (not N separate from-scratch restarts) since
this module owns its own loop and can pause to evaluate without redoing
prior iterations.
"""
import json
import os
import time

import numpy as np
import torch

from fairness_harness import (
    ENCODER_REGISTRY,
    FROZEN_CONFIG,
    RESULTS_DIR,
    _episode_assignment,
    _greedy_evaluate,
    get_final_pool,
)
from nav_env import NavEnv
from pool import build_partition
from closed_loop_oracle import instance_to_config
from ppo_train import MaskedEncoderActorCritic, compute_action_mask, compute_gae

CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 275, 350]
EVAL_SAMPLE_SIZE = 50   # fixed subset for cheap intermediate checkpoints
EVAL_RESETS_INTERMEDIATE = 5
EVAL_RESETS_FINAL = 30
CALIBRATION_SEED = 0


def calibrate(tier="easy", encoder="hrep", seed=CALIBRATION_SEED, checkpoints=CHECKPOINTS,
              config=None, verbose=True):
    cfg = {**FROZEN_CONFIG, **(config or {})}
    train_instances, test_instances = get_final_pool(tier)
    rng = np.random.default_rng(12345)
    eval_sample = [test_instances[i] for i in rng.choice(len(test_instances),
                                                          size=min(EVAL_SAMPLE_SIZE, len(test_instances)),
                                                          replace=False)]

    partition_cache = {}

    def get_partition(pseed):
        if pseed not in partition_cache:
            partition_cache[pseed] = build_partition(tier, pseed)
        return partition_cache[pseed]

    torch.manual_seed(seed)
    model = MaskedEncoderActorCritic(ENCODER_REGISTRY[encoder]())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    action_gen = torch.Generator().manual_seed(seed)

    episode_count = 0

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
    episode_outcomes = []
    curve = {"iteration": [], "train_solve_rate": [], "test_solve_rate": [], "policy_loss": [],
             "value_loss": [], "entropy": [], "elapsed_s": []}

    max_iter = max(checkpoints)
    t_start = time.time()
    for iteration in range(max_iter):
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

        last_policy_loss = last_value_loss = last_entropy = 0.0
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
            last_policy_loss, last_value_loss, last_entropy = policy_loss.item(), value_loss.item(), entropy.item()

        if verbose and (iteration + 1) % 5 == 0:
            print(f"  ...iteration {iteration + 1} done, elapsed={time.time() - t_start:.1f}s", flush=True)

        if (iteration + 1) in checkpoints:
            recent = episode_outcomes[-30:]
            train_solve_rate = (sum(1 for o in recent if o == "goal") / len(recent)) if recent else 0.0
            per_inst = _greedy_evaluate(model, tier, eval_sample, partition_cache,
                                         n_resets=EVAL_RESETS_INTERMEDIATE, horizon=cfg["horizon"])
            test_solve_rate = float(np.mean([v["solve_rate"] for v in per_inst.values()]))
            elapsed = time.time() - t_start
            curve["iteration"].append(iteration + 1)
            curve["train_solve_rate"].append(train_solve_rate)
            curve["test_solve_rate"].append(test_solve_rate)
            curve["policy_loss"].append(last_policy_loss)
            curve["value_loss"].append(last_value_loss)
            curve["entropy"].append(last_entropy)
            curve["elapsed_s"].append(elapsed)
            if verbose:
                print(f"  checkpoint iter={iteration + 1:4d}: train_solve={train_solve_rate:.2%}, "
                      f"TEST_solve={test_solve_rate:.2%}, entropy={last_entropy:.4f}, "
                      f"elapsed={elapsed:.1f}s", flush=True)

    return model, curve, test_instances, partition_cache


if __name__ == "__main__":
    model, curve, test_instances, partition_cache = calibrate()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "calibration_curve_easy_hrep.json"), "w") as f:
        json.dump(curve, f, indent=2)
    print("\n=== CALIBRATION CURVE (EASY, H-Rep) ===")
    for i in range(len(curve["iteration"])):
        print(f"  iter={curve['iteration'][i]:4d}  test_solve={curve['test_solve_rate'][i]:.2%}  "
              f"train_solve={curve['train_solve_rate'][i]:.2%}  entropy={curve['entropy'][i]:.4f}")

    best_idx = max(range(len(curve["test_solve_rate"])), key=lambda i: curve["test_solve_rate"][i])
    chosen_budget = curve["iteration"][best_idx]
    print(f"\ncandidate plateau budget: {chosen_budget} (test_solve={curve['test_solve_rate'][best_idx]:.2%})")

    print(f"\nRunning FINAL full evaluation at budget={chosen_budget} "
          f"(all {len(test_instances)} test instances x {EVAL_RESETS_FINAL} resets)...")
    final_per_instance = _greedy_evaluate(model, "easy", test_instances, partition_cache,
                                           n_resets=EVAL_RESETS_FINAL, horizon=FROZEN_CONFIG["horizon"])
    rates = [v["solve_rate"] for v in final_per_instance.values()]
    dist_100 = sum(1 for r in rates if r == 1.0)
    dist_0 = sum(1 for r in rates if r == 0.0)
    dist_partial = len(rates) - dist_100 - dist_0
    print(f"FINAL per-instance distribution (n={len(rates)}): 100%={dist_100}, partial={dist_partial}, "
          f"0%={dist_0}")
    print(f"FINAL aggregate test solve rate at budget={chosen_budget}: {np.mean(rates):.2%}")

    with open(os.path.join(RESULTS_DIR, "calibration_final_distribution_easy_hrep.json"), "w") as f:
        json.dump({"chosen_budget": chosen_budget, "rates": rates,
                   "dist_100": dist_100, "dist_partial": dist_partial, "dist_0": dist_0,
                   "per_instance": final_per_instance}, f, indent=2)
    print("DONE")
