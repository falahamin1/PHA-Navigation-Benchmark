"""B3: convergence calibration. Resumes an existing sweep_results/<tag>
checkpoint (Stage A/B's validated, frozen 3000-iteration runs) and continues
training to a much longer ceiling (15,000 iterations by default), using a
SINGLE evaluation protocol -- the full held-out test set at n_eval_resets
(30) -- at EVERY checkpoint, instead of sweep_cluster.py's cheap-sample-
then-full-set switch. That switch is exactly the "protocol seam" A1 found
and B0e showed changes a headline ranking on MEDIUM; Stage D is meant to use
a seamless protocol, so this is where that starts.

Also logs, at every checkpoint, the argmax-invariance measure B0a
introduced as a one-shot diagnostic (argmax_invariance.py) -- how much of a
fixed, deliberately-dissimilar observation batch collapses onto the same
argmax action. Lets the "is this policy actually still input-sensitive"
question be read off a curve instead of inferred from solve rate alone.

Reads sweep_results/<tag>_checkpoint.pt (and, for provenance only,
sweep_results/<tag>.json) but never writes there -- that's Stage A/B's
validated layer. Writes sweep_results_b3/<tag>_b3_checkpoint.pt,
_b3_progress.json, and (on reaching n_iterations) _b3.json.

Resume chain: first invocation loads the ORIGINAL sweep checkpoint (fixed
at iteration 3000); every subsequent invocation (e.g. after a 72h wall-time
cutoff) loads this script's OWN checkpoint instead, continuing further.
Idempotent once _b3.json exists, matching sweep_cluster.py's convention.

Same frozen hyperparameters as the original sweep (FROZEN_CONFIG +
entropy_coef=0.01) -- this is a training-BUDGET question, not a
hyperparameter-retune, so nothing else about the run changes.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from fairness_harness import (
    ENCODER_REGISTRY,
    FROZEN_CONFIG,
    _episode_assignment,
    _evaluate,
    _is_sim_violation,
    get_final_pool,
    verify_pool_fingerprint,
)
from launch_reduced_sweep import index_to_run
from nav_env import NavEnv
from pool import build_partition
from closed_loop_oracle import instance_to_config
from ppo_train import MaskedEncoderActorCritic, compute_action_mask, compute_gae
from argmax_invariance import build_dissimilar_batch, argmax_dominant_fraction

ORIGINAL_RESULTS_DIR = "sweep_results"
OUTPUT_DIR_DEFAULT = "sweep_results_b3"

SWEEP_ENTROPY_COEF = 0.01  # matches sweep_cluster.py -- frozen, not re-tuned here
DEFAULT_N_ITERATIONS = 15000
DEFAULT_EVAL_EVERY = 1000
DEFAULT_CHECKPOINT_EVERY = 1000

SIM_VIOLATION_REWARD = -0.01 + -10.0  # matches sweep_cluster.py verbatim


def _run_tag(tier, encoder, seed):
    return f"{tier}_{encoder}_seed{seed}"


def _count_sim_violations(per_instance_eval):
    return sum(o == "sim_violation" for v in per_instance_eval.values() for o in v["outcomes"])


def run(tier, encoder, seed, output_dir, n_iterations=DEFAULT_N_ITERATIONS,
        eval_every=DEFAULT_EVAL_EVERY, checkpoint_every=DEFAULT_CHECKPOINT_EVERY, verbose=True,
        config_overrides=None, argmax_batch_n=24):
    """config_overrides exists for tests only (shrink steps_per_iter/
    n_eval_resets/horizon so a smoke run finishes in seconds) -- real B3
    tasks never pass it, so entropy_coef/n_eval_resets(=30, full protocol)
    stay fixed for every real run."""
    if encoder not in ENCODER_REGISTRY:
        raise ValueError(f"unknown encoder {encoder!r}, must be one of {list(ENCODER_REGISTRY)}")

    os.makedirs(output_dir, exist_ok=True)
    tag = _run_tag(tier, encoder, seed)
    own_ckpt_path = os.path.join(output_dir, f"{tag}_b3_checkpoint.pt")
    progress_path = os.path.join(output_dir, f"{tag}_b3_progress.json")
    final_path = os.path.join(output_dir, f"{tag}_b3.json")
    original_ckpt_path = os.path.join(ORIGINAL_RESULTS_DIR, f"{tag}_checkpoint.pt")
    original_result_path = os.path.join(ORIGINAL_RESULTS_DIR, f"{tag}.json")

    if os.path.exists(final_path):
        print(f"[done] {tag} already complete at {n_iterations} -- {final_path} exists, nothing to do", flush=True)
        with open(final_path) as f:
            return json.load(f)

    train_instances, test_instances = get_final_pool(tier)
    pool_provenance = verify_pool_fingerprint(tier, train_instances, test_instances)
    print(f"[pool] tier={tier} n_train={pool_provenance['n_train']} n_test={pool_provenance['n_test']} "
          f"train_fp={pool_provenance['train_fingerprint']} test_fp={pool_provenance['test_fingerprint']}",
          flush=True)

    if not os.path.exists(original_result_path):
        raise RuntimeError(
            f"{original_result_path} does not exist -- B3 resumes an already-completed Stage A/B sweep run, "
            f"it does not start one from scratch. Run the original sweep for {tag} first."
        )

    cfg = {**FROZEN_CONFIG, "entropy_coef": SWEEP_ENTROPY_COEF, "n_iterations": n_iterations,
           **(config_overrides or {})}
    partition_cache = {}

    def get_partition(pseed):
        if pseed not in partition_cache:
            partition_cache[pseed] = build_partition(tier, pseed)
        return partition_cache[pseed]

    torch.manual_seed(seed)
    model = MaskedEncoderActorCritic(ENCODER_REGISTRY[encoder]())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    action_gen = torch.Generator().manual_seed(seed)

    curve = {"iteration": [], "entropy": [], "greedy_solve": [], "stochastic_solve": [],
             "argmax_dominant_fraction": [], "argmax_n_distinct": [],
             "n_eval_instances": [], "elapsed_s": [], "train_sim_violations": [], "eval_sim_violations": [],
             "pool_provenance": pool_provenance}

    if os.path.exists(own_ckpt_path):
        ckpt_path_used = own_ckpt_path
    elif os.path.exists(original_ckpt_path):
        ckpt_path_used = original_ckpt_path
    else:
        raise RuntimeError(f"neither {own_ckpt_path} nor {original_ckpt_path} exists -- cannot resume {tag}")

    ckpt = torch.load(ckpt_path_used, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    action_gen.set_state(ckpt["action_gen_state"])
    start_iteration = ckpt["iteration"]
    episode_count = ckpt["episode_count"]
    train_sim_violations = ckpt.get("sim_violation_count", 0)
    print(f"[resume] {tag} loaded {'own B3' if ckpt_path_used == own_ckpt_path else 'ORIGINAL sweep'} "
          f"checkpoint ({ckpt_path_used}) at iteration={start_iteration}, episode_count={episode_count}, "
          f"train_sim_violations={train_sim_violations}", flush=True)

    if start_iteration >= n_iterations:
        raise RuntimeError(
            f"{tag}: loaded checkpoint is already at iteration={start_iteration} >= n_iterations={n_iterations} "
            "-- nothing to extend. Raise n_iterations or check you're pointed at the right checkpoint."
        )

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            curve = json.load(f)
        prior_provenance = curve.get("pool_provenance")
        if prior_provenance is not None and prior_provenance != pool_provenance:
            raise RuntimeError(
                f"POOL MISMATCH on resume for {tag}: this run's pool ({pool_provenance}) differs from the "
                f"pool the existing B3 progress file was trained against ({prior_provenance}). Do not continue."
            )
        curve["pool_provenance"] = pool_provenance

    argmax_batch = build_dissimilar_batch(tier, test_instances, n_obs=argmax_batch_n)

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
    t_start = time.time()

    print(f"[start] tag={tag} start_iteration={start_iteration} n_iterations={n_iterations} "
          f"eval_every={eval_every} (FULL protocol every checkpoint, no seam) "
          f"entropy_coef={cfg['entropy_coef']}", flush=True)

    for iteration in range(start_iteration, n_iterations):
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

            try:
                obs, reward, terminated, truncated, info = env.step(int(action.item()))
            except AssertionError as e:
                if not _is_sim_violation(e):
                    raise
                train_sim_violations += 1
                reward, terminated, truncated = SIM_VIOLATION_REWARD, True, False
            buf_reward.append(reward)
            buf_done.append(terminated or truncated)

            if terminated or truncated:
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

        it1 = iteration + 1
        is_checkpoint = (it1 % eval_every == 0) or (it1 == n_iterations)
        if is_checkpoint:
            # ALWAYS full protocol -- the entire point of B3's eval redesign
            # (A1 + B0e: the sample/full switch changes a headline ranking).
            greedy_eval = _evaluate(model, tier, test_instances, partition_cache,
                                     n_resets=cfg["n_eval_resets"], horizon=cfg["horizon"], mode="greedy")
            stochastic_eval = _evaluate(model, tier, test_instances, partition_cache, n_resets=cfg["n_eval_resets"],
                                         horizon=cfg["horizon"], mode="stochastic",
                                         stochastic_rng=np.random.default_rng((seed, it1)))
            greedy_rate = float(np.mean([v["solve_rate"] for v in greedy_eval.values()]))
            stochastic_rate = float(np.mean([v["solve_rate"] for v in stochastic_eval.values()]))
            eval_violations = _count_sim_violations(greedy_eval) + _count_sim_violations(stochastic_eval)

            dominant_frac, n_distinct, _counts = argmax_dominant_fraction(model, argmax_batch)

            elapsed = time.time() - t_start

            curve["iteration"].append(it1)
            curve["entropy"].append(last_entropy)
            curve["greedy_solve"].append(greedy_rate)
            curve["stochastic_solve"].append(stochastic_rate)
            curve["argmax_dominant_fraction"].append(dominant_frac)
            curve["argmax_n_distinct"].append(n_distinct)
            curve["n_eval_instances"].append(len(test_instances))
            curve["elapsed_s"].append(elapsed)
            curve["train_sim_violations"].append(train_sim_violations)
            curve["eval_sim_violations"].append(eval_violations)
            with open(progress_path, "w") as f:
                json.dump(curve, f, indent=2)

            if verbose:
                print(f"[checkpoint] iter={it1:6d} (FULL n={len(test_instances)}x{cfg['n_eval_resets']}) "
                      f"entropy={last_entropy:.4f} greedy={greedy_rate:.2%} stochastic={stochastic_rate:.2%} "
                      f"gap={stochastic_rate - greedy_rate:+.2%} argmax_dominant={dominant_frac:.3f} "
                      f"(n_distinct={n_distinct}) sim_violations(train_cum={train_sim_violations}, "
                      f"eval_this_ckpt={eval_violations}) elapsed={elapsed:.0f}s", flush=True)

            last_full_greedy, last_full_stochastic = greedy_eval, stochastic_eval

        if (it1 % checkpoint_every == 0) or (it1 == n_iterations):
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "action_gen_state": action_gen.get_state(), "iteration": it1, "episode_count": episode_count,
                "sim_violation_count": train_sim_violations,
            }, own_ckpt_path)
            print(f"[checkpoint-saved] iter={it1}", flush=True)

    if "last_full_greedy" not in locals():
        # Resumed exactly at a prior n_iterations with the loop body never
        # executing (mirrors sweep_cluster.py's same fallback comment).
        last_full_greedy = _evaluate(model, tier, test_instances, partition_cache,
                                      n_resets=cfg["n_eval_resets"], horizon=cfg["horizon"], mode="greedy")
        last_full_stochastic = _evaluate(model, tier, test_instances, partition_cache,
                                          n_resets=cfg["n_eval_resets"], horizon=cfg["horizon"], mode="stochastic",
                                          stochastic_rng=np.random.default_rng((seed, n_iterations)))

    aggregate_solve_rate_greedy = float(np.mean([v["solve_rate"] for v in last_full_greedy.values()]))
    aggregate_solve_rate_stochastic = float(np.mean([v["solve_rate"] for v in last_full_stochastic.values()]))

    with open(original_result_path) as f:
        original_result = json.load(f)

    result = {
        "tier": tier, "encoder": encoder, "seed": seed,
        "resumed_from": original_ckpt_path,
        "original_final_greedy": original_result["aggregate_solve_rate_greedy"],
        "original_final_stochastic": original_result["aggregate_solve_rate_stochastic"],
        "curve": curve,
        "aggregate_solve_rate_greedy": aggregate_solve_rate_greedy,
        "aggregate_solve_rate_stochastic": aggregate_solve_rate_stochastic,
        "config": cfg,
        "pool_provenance": pool_provenance,
        "n_train_instances": len(train_instances), "n_test_instances": len(test_instances),
        "train_sim_violations": train_sim_violations,
    }
    with open(final_path, "w") as f:
        json.dump(result, f)
    print(f"[done] tag={tag} wrote {final_path} (greedy={aggregate_solve_rate_greedy:.2%}, "
          f"stochastic={aggregate_solve_rate_stochastic:.2%})", flush=True)
    return result


# The 6 (tier, encoder) cells B3 actually targets, all at seed 0 -- chosen so
# the SAME training/eval episode stream (a pure function of (seed, episode_
# count), independent of encoder -- see fairness_harness._episode_assignment)
# underlies every comparison, keeping this the fairest possible read across
# encoders. See the B3 report for why HARD and EASY are both in scope (EASY
# showed the largest final-step curve movement of any tier in the original
# sweep, so it is not safe to assume HARD is the slow-to-converge one).
TARGET_RUNS = [
    ("hard", "hrep", 0), ("hard", "relational", 0), ("hard", "cnn", 0),
    ("easy", "hrep", 0), ("easy", "relational", 0), ("easy", "cnn", 0),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--array-index", type=int, default=None, help="index into TARGET_RUNS, [0,6)")
    p.add_argument("--tier", default=None)
    p.add_argument("--encoder", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-iterations", type=int, default=DEFAULT_N_ITERATIONS)
    p.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    p.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = p.parse_args()

    if args.array_index is not None:
        tier, encoder, seed = TARGET_RUNS[args.array_index]
        print(f"array_index={args.array_index} -> tier={tier}, encoder={encoder}, seed={seed}", flush=True)
    else:
        if args.tier is None or args.encoder is None or args.seed is None:
            p.error("either --array-index, or all of --tier/--encoder/--seed, must be given")
        tier, encoder, seed = args.tier, args.encoder, args.seed

    run(tier, encoder, seed, args.output_dir, n_iterations=args.n_iterations,
        eval_every=args.eval_every, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
