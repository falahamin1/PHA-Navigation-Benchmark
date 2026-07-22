"""Step 5b-diag: optimization-vs-generalization fork diagnostic.

Context: H-Rep at budget=200 on the full clean EASY pool (812 train/203
test) solves 14.42% held-out, with entropy pinned at 1.6-1.9 (near max
ln(8)=2.08) for all 350 iterations of the calibration run. Hypothesis: the
entropy bonus (0.05 * ~1.8 = 0.09) dominates the policy-gradient term
(~0.006, per Step 5a), pinning the policy near-uniform once instance
diversity (812, vs the 6-instance smoke test) dilutes the per-instance
advantage signal below the entropy pull.

This script:
1. Retrains H-Rep to n_iterations=200 (same seed=0, same frozen config as
   the calibration run it reproduces) to get a live model + a captured
   final-iteration buffer.
2. Evaluates greedy solve rate on a 203-instance TRAIN sample (same size/
   reset protocol as the held-out TEST eval, for a direct, apples-to-apples
   train-vs-test comparison) -- the decisive fork.
3. Evaluates a RANDOM-policy baseline (uniform over feasible/masked
   actions, no model at all) on the full 203 TEST instances -- converts
   "underperforming" into "not learning" if scores land near 14.42%.
4. Computes the gradient-norm ratio of the entropy term vs the policy term
   into the shared encoder, using the captured buffer -- the Step 5a-fix
   method (separate backward() calls, sum abs-grad over encoder params),
   not the raw loss-magnitude ratio (which misled Step 5a).
5. Reports the learning-curve slope at iteration 350 from the existing
   calibration curve (calibration_curve_easy_hrep.json) -- no retraining
   needed for this part.

Part 2 (entropy mini-sweep at 0.01, 0.005) is a SEPARATE function,
`run_entropy_variant`, invoked from __main__ only if Part 1's fork lands on
"optimization failure" (train solve rate also low, not 80%+).
"""
import copy
import json
import os

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

SEED = 0
BUDGET = 200
TIER = "easy"
ENCODER = "hrep"
TRAIN_SAMPLE_SIZE = 203  # matches the test set size, for a direct fork comparison
N_RESETS_FORK = 30       # match the "trustworthy" protocol the final test eval used
KNOWN_TEST_SOLVE_RATE = 0.1442  # already measured (full 203x30) -- reused, not recomputed


def train_to_checkpoint(entropy_coef, n_iterations=BUDGET, seed=SEED, tier=TIER, encoder=ENCODER, verbose=True):
    """Reproduces fairness_harness's exact training-loop mechanics up to
    n_iterations, returning (model, last_iteration_buffer, train_solve_history).
    last_iteration_buffer is what the gradient-ratio diagnostic needs."""
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
    train_solve_history = []
    entropy_history = []
    last_buffer = None

    for iteration in range(n_iterations):
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

        recent = episode_outcomes[-30:]
        train_solve = (sum(1 for o in recent if o == "goal") / len(recent)) if recent else 0.0
        train_solve_history.append(train_solve)
        entropy_history.append(last_entropy)
        if verbose and (iteration + 1) % 25 == 0:
            print(f"  [entropy={entropy_coef}] iter {iteration + 1}: train_solve={train_solve:.2%}, "
                  f"entropy={last_entropy:.4f}", flush=True)

        if iteration == n_iterations - 1:
            last_buffer = dict(buf_state=buf_state, buf_cell=buf_cell, buf_mask=masks, buf_action=actions,
                               buf_instance=buf_instance, buf_partition=buf_partition, buf_logp=old_logp,
                               advantages=advantages, returns=returns, returns_mean=returns_mean,
                               returns_std=returns_std, old_values_norm=old_values_norm, cfg=cfg)

    return model, last_buffer, train_solve_history, partition_cache, train_instances, test_instances, entropy_history


def gradient_norm_ratio(model, buffer):
    """Entropy-term vs policy-term gradient-norm ratio into the shared
    encoder -- Step 5a-fix's method (separate backward() calls, sum
    abs-grad over encoder params), not raw loss magnitudes."""
    cfg = buffer["cfg"]
    new_logits, new_values = [], []
    for t in range(cfg["steps_per_iter"]):
        logits, value = model(buffer["buf_partition"][t], buffer["buf_instance"][t], buffer["buf_state"][t],
                               current_cell=buffer["buf_cell"][t], mask=buffer["buf_mask"][t])
        new_logits.append(logits)
        new_values.append(value)
    new_logits = torch.stack(new_logits)
    dist = torch.distributions.Categorical(logits=new_logits)
    new_logp = dist.log_prob(buffer["buf_action"])
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_logp - buffer["buf_logp"])
    advantages = buffer["advantages"]
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy_term = -cfg["entropy_coef"] * entropy

    model.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_grad_norm = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)

    model.zero_grad()
    entropy_term.backward()
    entropy_grad_norm = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)

    model.zero_grad()
    return entropy_grad_norm, policy_grad_norm, entropy_grad_norm / max(policy_grad_norm, 1e-12)


def random_policy_evaluate(tier, test_instances, n_resets=30, horizon=40, seed_base=0):
    """Uniform-over-feasible-actions baseline -- no model at all."""
    partition_cache = {}
    rng = np.random.default_rng(seed_base)
    results = {}
    for inst in test_instances:
        if inst.partition_seed not in partition_cache:
            partition_cache[inst.partition_seed] = build_partition(tier, inst.partition_seed)
        partition = partition_cache[inst.partition_seed]
        cfg = instance_to_config(partition, inst)
        outcomes = []
        for reset_seed in range(n_resets):
            env = NavEnv(partition, cfg, horizon=horizon)
            obs, _ = env.reset(seed=reset_seed)
            outcome = None
            for _ in range(horizon):
                mask = compute_action_mask(obs["state"], partition.domain)
                feasible = np.nonzero(mask.numpy())[0]
                action = int(rng.choice(feasible))
                obs, _reward, terminated, truncated, info = env.step(action)
                if terminated:
                    outcome = info.get("outcome")
                    break
                if truncated:
                    outcome = "truncated"
                    break
            else:
                outcome = "truncated"
            outcomes.append(outcome)
        key = f"{inst.start_cell}|{inst.goal_cell}|{','.join(map(str, inst.hazard_cells))}|{inst.partition_seed}"
        results[key] = {"outcomes": outcomes, "solve_rate": sum(1 for o in outcomes if o == "goal") / n_resets}
    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Part 1: optimization-vs-generalization fork ===", flush=True)
    print(f"Retraining H-Rep to iteration={BUDGET}, entropy_coef=0.05 (reproducing the calibration checkpoint)...",
          flush=True)
    model, buffer, train_solve_history, partition_cache, train_instances, test_instances, entropy_history = (
        train_to_checkpoint(entropy_coef=0.05, n_iterations=BUDGET)
    )

    print("\nEvaluating greedy TRAIN-set solve rate (203-instance sample, 30 resets)...", flush=True)
    rng = np.random.default_rng(777)
    train_sample = [train_instances[i] for i in rng.choice(len(train_instances), size=TRAIN_SAMPLE_SIZE,
                                                            replace=False)]
    train_eval = _greedy_evaluate(model, TIER, train_sample, partition_cache, n_resets=N_RESETS_FORK, horizon=40)
    train_solve_rate = float(np.mean([v["solve_rate"] for v in train_eval.values()]))
    train_dist_100 = sum(1 for v in train_eval.values() if v["solve_rate"] == 1.0)
    train_dist_0 = sum(1 for v in train_eval.values() if v["solve_rate"] == 0.0)
    print(f"TRAIN solve rate (greedy, {TRAIN_SAMPLE_SIZE} instances x{N_RESETS_FORK}): {train_solve_rate:.2%} "
          f"(100%={train_dist_100}, 0%={train_dist_0}, partial={TRAIN_SAMPLE_SIZE - train_dist_100 - train_dist_0})",
          flush=True)
    print(f"KNOWN test solve rate (already measured, full 203x30): {KNOWN_TEST_SOLVE_RATE:.2%}", flush=True)

    print("\nEvaluating RANDOM-policy baseline (full 203 test instances x30 resets)...", flush=True)
    random_eval = random_policy_evaluate(TIER, test_instances, n_resets=30)
    random_solve_rate = float(np.mean([v["solve_rate"] for v in random_eval.values()]))
    print(f"RANDOM baseline solve rate: {random_solve_rate:.2%}", flush=True)

    print("\nComputing gradient-norm ratio (entropy term vs policy term)...", flush=True)
    entropy_grad, policy_grad, ratio = gradient_norm_ratio(model, buffer)
    print(f"entropy_grad_norm={entropy_grad:.6f}, policy_grad_norm={policy_grad:.6f}, ratio={ratio:.2f}x",
          flush=True)

    with open(os.path.join(RESULTS_DIR, "calibration_curve_easy_hrep.json")) as f:
        curve = json.load(f)
    last3_iter = curve["iteration"][-3:]
    last3_solve = curve["test_solve_rate"][-3:]
    slope = (last3_solve[-1] - last3_solve[0]) / (last3_iter[-1] - last3_iter[0])
    print(f"\ncurve slope over last 3 checkpoints {list(zip(last3_iter, last3_solve))}: {slope:.6f} per iteration "
          f"({'still climbing' if slope > 0.0005 else 'flat/declining'})", flush=True)

    fork = "GENERALIZATION_FAILURE" if (train_solve_rate >= 0.6 and KNOWN_TEST_SOLVE_RATE < 0.3) else \
        "OPTIMIZATION_FAILURE"
    print(f"\n=== FORK RESULT: {fork} ===", flush=True)
    print(f"train={train_solve_rate:.2%}, test={KNOWN_TEST_SOLVE_RATE:.2%}, random={random_solve_rate:.2%}, "
          f"entropy/policy grad ratio={ratio:.2f}x, curve slope={slope:.6f}", flush=True)

    part1_result = {
        "train_solve_rate": train_solve_rate, "test_solve_rate": KNOWN_TEST_SOLVE_RATE,
        "random_solve_rate": random_solve_rate, "entropy_grad_norm": entropy_grad, "policy_grad_norm": policy_grad,
        "grad_ratio": ratio, "curve_slope_last3": slope, "fork": fork,
    }
    with open(os.path.join(RESULTS_DIR, "diag_part1.json"), "w") as f:
        json.dump(part1_result, f, indent=2)
    print("\nDONE PART 1")
