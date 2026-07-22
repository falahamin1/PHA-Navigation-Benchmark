"""Step 5b-cal, Parts 2/3: cluster-side budget calibration + entropy re-check.

CLI entry point for the Slurm jobs in jobs/nav-calibrate-*.slurm. Trains one
encoder on one tier at one entropy_coef to a large iteration ceiling,
logging entropy + greedy solve rate + stochastic solve rate at frequent
checkpoints (the three-curve convergence picture Step 5b-cal asks for), and
periodically checkpointing model/optimizer/RNG state to disk so a
timed-out/interrupted job can simply be resubmitted and resume -- same
convention as tangram-git/jobs/hrep.slurm ("resubmit this same script,
training resumes from the latest checkpoint").

Reuses fairness_harness's training-loop mechanics (same RNG design, same
frozen config apart from entropy_coef/n_iterations, same env construction)
-- this is NOT a different training loop, just this one wrapped with
resumability and a coarser/more frequent eval cadence than train_run's
"evaluate once at the end" default (appropriate for the 180-run sweep, but
not for extracting a convergence curve).

Usage (see jobs/nav-calibrate-*.slurm for the actual sbatch invocations):
    python3 calibrate_cluster.py --tier easy --encoder hrep --entropy-coef 0.05 \\
        --seed 0 --n-iterations 5000 --eval-every 50 --checkpoint-every 250 \\
        --n-eval-resets 30 --output-dir calibration_results

Resuming: just rerun the identical command -- if
  <output-dir>/<run_tag>_checkpoint.pt exists, training resumes from it
(model/optimizer/action-generator state + iteration/episode counters); the
curve file (<run_tag>_curve.json) is loaded and appended to, not overwritten.
Note: the exact env/in-flight-episode state at the moment of interruption is
NOT part of the checkpoint (a fresh episode is simply started on resume) --
irrelevant for a calibration curve, which is about long-run trend, not
bit-exact reproducibility across a crash boundary.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from fairness_harness import ENCODER_REGISTRY, FROZEN_CONFIG, _episode_assignment, _evaluate, get_final_pool, \
    pool_fingerprint
from nav_env import NavEnv
from pool import build_partition
from closed_loop_oracle import instance_to_config
from ppo_train import MaskedEncoderActorCritic, compute_action_mask, compute_gae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--encoder", default="hrep", choices=list(ENCODER_REGISTRY.keys()))
    p.add_argument("--entropy-coef", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iterations", type=int, default=5000, help="generous ceiling, not a target")
    p.add_argument("--eval-every", type=int, default=50, help="checkpoint eval frequency, in iterations")
    p.add_argument("--checkpoint-every", type=int, default=250, help="model/optimizer save frequency")
    p.add_argument("--n-eval-resets", type=int, default=30)
    p.add_argument("--output-dir", default="calibration_results")
    return p.parse_args()


def main():
    args = parse_args()
    run_tag = f"{args.tier}_{args.encoder}_entropy{args.entropy_coef}_seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{run_tag}_checkpoint.pt")
    curve_path = os.path.join(args.output_dir, f"{run_tag}_curve.json")

    cfg = {**FROZEN_CONFIG, "entropy_coef": args.entropy_coef}
    train_instances, test_instances = get_final_pool(args.tier)
    pool_provenance = {
        "tier": args.tier, "n_train": len(train_instances), "n_test": len(test_instances),
        "train_fingerprint": pool_fingerprint(train_instances), "test_fingerprint": pool_fingerprint(test_instances),
    }
    print(f"[pool] tier={args.tier} n_train={pool_provenance['n_train']} n_test={pool_provenance['n_test']} "
          f"train_fp={pool_provenance['train_fingerprint']} test_fp={pool_provenance['test_fingerprint']}",
          flush=True)
    partition_cache = {}

    def get_partition(pseed):
        if pseed not in partition_cache:
            partition_cache[pseed] = build_partition(args.tier, pseed)
        return partition_cache[pseed]

    torch.manual_seed(args.seed)
    model = MaskedEncoderActorCritic(ENCODER_REGISTRY[args.encoder]())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    action_gen = torch.Generator().manual_seed(args.seed)

    start_iteration = 0
    episode_count = 0
    curve = {"iteration": [], "entropy": [], "greedy_solve": [], "stochastic_solve": [], "elapsed_s": [],
             "pool_provenance": pool_provenance}

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        action_gen.set_state(ckpt["action_gen_state"])
        start_iteration = ckpt["iteration"]
        episode_count = ckpt["episode_count"]
        print(f"[resume] loaded checkpoint at iteration={start_iteration}, episode_count={episode_count}",
              flush=True)
    if os.path.exists(curve_path):
        with open(curve_path) as f:
            curve = json.load(f)
        prior_provenance = curve.get("pool_provenance")
        if prior_provenance is not None and prior_provenance != pool_provenance:
            raise RuntimeError(
                f"POOL MISMATCH on resume: this run's pool ({pool_provenance}) differs from the pool "
                f"the existing curve file was trained against ({prior_provenance}). Do not continue -- "
                "the calibration would be measuring two different instance sets. Investigate the pool "
                "cache before resubmitting (see jobs/README.md's pool-provenance note)."
            )
        curve["pool_provenance"] = pool_provenance

    def new_env():
        nonlocal episode_count
        idx, reset_seed = _episode_assignment(args.seed, episode_count, len(train_instances))
        inst = train_instances[idx]
        partition = get_partition(inst.partition_seed)
        env_cfg = instance_to_config(partition, inst)
        env = NavEnv(partition, env_cfg, horizon=cfg["horizon"])
        obs, _ = env.reset(seed=reset_seed)
        episode_count += 1
        return env, inst, partition, obs

    env, current_instance, current_partition, obs = new_env()
    t_start = time.time()

    print(f"[start] run_tag={run_tag} start_iteration={start_iteration} n_iterations={args.n_iterations}",
          flush=True)

    for iteration in range(start_iteration, args.n_iterations):
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
        if it1 % args.eval_every == 0 or it1 == args.n_iterations:
            greedy_eval = _evaluate(model, args.tier, test_instances, partition_cache,
                                     n_resets=args.n_eval_resets, horizon=cfg["horizon"], mode="greedy")
            stochastic_eval = _evaluate(model, args.tier, test_instances, partition_cache,
                                         n_resets=args.n_eval_resets, horizon=cfg["horizon"], mode="stochastic",
                                         stochastic_rng=np.random.default_rng((args.seed, it1)))
            greedy_rate = float(np.mean([v["solve_rate"] for v in greedy_eval.values()]))
            stochastic_rate = float(np.mean([v["solve_rate"] for v in stochastic_eval.values()]))
            elapsed = time.time() - t_start

            curve["iteration"].append(it1)
            curve["entropy"].append(last_entropy)
            curve["greedy_solve"].append(greedy_rate)
            curve["stochastic_solve"].append(stochastic_rate)
            curve["elapsed_s"].append(elapsed)
            with open(curve_path, "w") as f:
                json.dump(curve, f, indent=2)

            print(f"[checkpoint] iter={it1:5d} entropy={last_entropy:.4f} greedy={greedy_rate:.2%} "
                  f"stochastic={stochastic_rate:.2%} gap={stochastic_rate - greedy_rate:+.2%} "
                  f"elapsed={elapsed:.0f}s", flush=True)

        if it1 % args.checkpoint_every == 0 or it1 == args.n_iterations:
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "action_gen_state": action_gen.get_state(), "iteration": it1, "episode_count": episode_count,
            }, ckpt_path)
            print(f"[checkpoint-saved] iter={it1}", flush=True)

    print(f"[done] run_tag={run_tag}", flush=True)


if __name__ == "__main__":
    main()
