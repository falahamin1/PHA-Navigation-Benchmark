"""AAAI reduced sweep: CLI entry point for jobs/nav-sweep-*.slurm.

One Slurm array task calls this once, trains ONE (tier, encoder, seed) at
the frozen config (entropy_coef=0.01, 3000 iterations), and writes the
self-contained result file `sweep_results/<tier>_<encoder>_seed<seed>.json`
the statistics module consumes. This is fairness_harness.train_run's job in
spirit, but with two capabilities train_run doesn't have and this sweep
needs:

1. Checkpoint/resume, modeled on calibrate_cluster.py's proven contract
   ("if a job times out, resubmit the identical command; it resumes from
   the last checkpoint and continues the SAME episode/instance stream" --
   fairness_harness._episode_assignment is a pure function of
   (seed, episode_count), so resuming mid-run and continuing the episode
   counter does not perturb the fairness invariant). If the tagged result
   file already exists, the task is a no-op (idempotent under resubmission
   after real completion, not just after a timeout).
2. Entropy + both solve rates logged at EVERY checkpoint (not just once at
   the end, which is all train_run does) -- the three-curve convergence
   record the launch prompt asks for. Full per-instance dual-mode eval
   (the schema's per-instance outcomes) is expensive (state x resets x 2
   modes), so intermediate checkpoints eval a fixed sample of the test set
   at a lighter reset count for the curve only, matching calibrate_cluster.
   py's DEFAULT_CHECKPOINTS convention; only the LAST checkpoint (iteration
   == n_iterations) runs the full held-out set at FROZEN_CONFIG's
   n_eval_resets and is what per_instance_test_outcomes_* in the final
   result file records.

Pool fingerprint verification (fairness_harness.verify_pool_fingerprint)
runs before any training, every invocation (fresh or resumed) -- a job must
fail loudly rather than silently train on / resume onto a mismatched pool.

Usage:
    python3 sweep_cluster.py --tier easy --array-index 0 --output-dir sweep_results
    python3 sweep_cluster.py --tier easy --encoder hrep --seed 0   # equivalent, direct
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from fairness_harness import (
    ENCODER_REGISTRY,
    FROZEN_CONFIG,
    RESULTS_DIR,
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

# Frozen for this sweep -- see the launch prompt. Not CLI-overridable on
# purpose: this is the one config every one of the 90 runs must share.
SWEEP_ENTROPY_COEF = 0.01
SWEEP_BUDGET = 3000

# Checkpoint/eval schedule: 10 checkpoints over 3000 iterations. The last 2
# run the full held-out set at FROZEN_CONFIG's n_eval_resets (30); the
# earlier 8 sample a fixed subset at fewer resets, cheap enough to not
# dominate wall-clock the way calibration's every-100-iterations full dual
# eval did (see jobs/README.md's "eval frequency is the knob to turn" note).
DEFAULT_EVAL_EVERY = 300
DEFAULT_CHECKPOINT_EVERY = 300
N_FULL_EVAL_CHECKPOINTS = 2
INTERMEDIATE_EVAL_SAMPLE_SIZE = 100
INTERMEDIATE_EVAL_RESETS = 10

# See fairness_harness._is_sim_violation's comment for the full story: a rare
# but real skipped-cell physics edge case, much more frequent on MEDIUM/HARD
# than EASY, that the frozen env raises as a hard AssertionError. Harness-
# level fix (this sweep's decision, not a dt/drift_coupling change): treat
# the training step that triggered it as an ordinary terminal transition
# with a hazard-sized penalty, then start a fresh episode. SIM_VIOLATION_
# REWARD mirrors nav_env.py's STEP_PENALTY + HAZARD_REWARD (no shaping term
# -- the position at the moment of violation isn't a trustworthy point to
# potential-shape from).
SIM_VIOLATION_REWARD = -0.01 + -10.0


def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _count_sim_violations(per_instance_eval):
    return sum(o == "sim_violation" for v in per_instance_eval.values() for o in v["outcomes"])


def _run_tag(tier, encoder, seed):
    return f"{tier}_{encoder}_seed{seed}"


def _paths(output_dir, tier, encoder, seed):
    tag = _run_tag(tier, encoder, seed)
    return (
        os.path.join(output_dir, f"{tag}_checkpoint.pt"),
        os.path.join(output_dir, f"{tag}_progress.json"),
        os.path.join(output_dir, f"{tag}.json"),
    )


def run(tier, encoder, seed, output_dir, n_iterations=SWEEP_BUDGET,
        eval_every=DEFAULT_EVAL_EVERY, checkpoint_every=DEFAULT_CHECKPOINT_EVERY, verbose=True,
        config_overrides=None, eval_sample_size=INTERMEDIATE_EVAL_SAMPLE_SIZE,
        intermediate_eval_resets=INTERMEDIATE_EVAL_RESETS):
    """config_overrides/eval_sample_size/intermediate_eval_resets exist for
    tests only (shrink steps_per_iter/n_eval_resets/sample size so a smoke
    run finishes in seconds) -- real sweep tasks never pass them, so
    entropy_coef/n_iterations stay the only per-run knobs, matching the
    frozen-config requirement."""
    if encoder not in ENCODER_REGISTRY:
        raise ValueError(f"unknown encoder {encoder!r}, must be one of {list(ENCODER_REGISTRY)}")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path, progress_path, final_path = _paths(output_dir, tier, encoder, seed)

    train_instances, test_instances = get_final_pool(tier)
    pool_provenance = verify_pool_fingerprint(tier, train_instances, test_instances)
    print(f"[pool] tier={tier} n_train={pool_provenance['n_train']} n_test={pool_provenance['n_test']} "
          f"train_fp={pool_provenance['train_fingerprint']} test_fp={pool_provenance['test_fingerprint']}",
          flush=True)

    run_tag = _run_tag(tier, encoder, seed)
    if os.path.exists(final_path):
        print(f"[done] {run_tag} already complete -- {final_path} exists, nothing to do", flush=True)
        with open(final_path) as f:
            return json.load(f)

    cfg = {**FROZEN_CONFIG, "entropy_coef": SWEEP_ENTROPY_COEF, "n_iterations": n_iterations,
           **(config_overrides or {})}
    commit_hash = _git_commit_hash()
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
    episode_count = 0
    train_sim_violations = 0
    curve = {"iteration": [], "entropy": [], "greedy_solve": [], "stochastic_solve": [],
             "n_eval_instances": [], "elapsed_s": [], "train_sim_violations": [], "eval_sim_violations": [],
             "pool_provenance": pool_provenance}

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        action_gen.set_state(ckpt["action_gen_state"])
        start_iteration = ckpt["iteration"]
        episode_count = ckpt["episode_count"]
        train_sim_violations = ckpt.get("sim_violation_count", 0)
        print(f"[resume] {run_tag} loaded checkpoint at iteration={start_iteration}, "
              f"episode_count={episode_count}, train_sim_violations={train_sim_violations}", flush=True)
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            curve = json.load(f)
        prior_provenance = curve.get("pool_provenance")
        if prior_provenance is not None and prior_provenance != pool_provenance:
            raise RuntimeError(
                f"POOL MISMATCH on resume for {run_tag}: this run's pool ({pool_provenance}) differs "
                f"from the pool the existing progress file was trained against ({prior_provenance}). "
                "Do not continue -- investigate pool_cache/ provenance before resubmitting."
            )
        curve["pool_provenance"] = pool_provenance
        # Backward-compat: a progress file written before sim_violation
        # tracking existed won't have these keys.
        curve.setdefault("train_sim_violations", [])
        curve.setdefault("eval_sim_violations", [])

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

    print(f"[start] run_tag={run_tag} start_iteration={start_iteration} n_iterations={n_iterations} "
          f"entropy_coef={cfg['entropy_coef']} commit={commit_hash}", flush=True)

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
                # env's internal state is now inconsistent (position moved
                # past a cell boundary before the check fired) -- do not
                # reuse it. Treat this step as an ordinary hazard-like
                # terminal transition and let the normal new_env() call
                # below start a clean episode.
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
            remaining_checkpoints = (n_iterations - it1) // eval_every + 1
            is_final_stretch = remaining_checkpoints <= N_FULL_EVAL_CHECKPOINTS or it1 == n_iterations
            if is_final_stretch:
                eval_instances, n_resets = test_instances, cfg["n_eval_resets"]
            else:
                rng = np.random.default_rng(999)
                sample_n = min(eval_sample_size, len(test_instances))
                idx = rng.choice(len(test_instances), size=sample_n, replace=False)
                eval_instances, n_resets = [test_instances[i] for i in idx], intermediate_eval_resets

            greedy_eval = _evaluate(model, tier, eval_instances, partition_cache,
                                     n_resets=n_resets, horizon=cfg["horizon"], mode="greedy")
            stochastic_eval = _evaluate(model, tier, eval_instances, partition_cache, n_resets=n_resets,
                                         horizon=cfg["horizon"], mode="stochastic",
                                         stochastic_rng=np.random.default_rng((seed, it1)))
            greedy_rate = float(np.mean([v["solve_rate"] for v in greedy_eval.values()]))
            stochastic_rate = float(np.mean([v["solve_rate"] for v in stochastic_eval.values()]))
            eval_violations = _count_sim_violations(greedy_eval) + _count_sim_violations(stochastic_eval)
            elapsed = time.time() - t_start

            curve["iteration"].append(it1)
            curve["entropy"].append(last_entropy)
            curve["greedy_solve"].append(greedy_rate)
            curve["stochastic_solve"].append(stochastic_rate)
            curve["n_eval_instances"].append(len(eval_instances))
            curve["elapsed_s"].append(elapsed)
            curve["train_sim_violations"].append(train_sim_violations)
            curve["eval_sim_violations"].append(eval_violations)
            with open(progress_path, "w") as f:
                json.dump(curve, f, indent=2)

            if verbose:
                print(f"[checkpoint] iter={it1:5d} ({'FULL' if is_final_stretch else 'sample'} "
                      f"n={len(eval_instances)}x{n_resets}) entropy={last_entropy:.4f} "
                      f"greedy={greedy_rate:.2%} stochastic={stochastic_rate:.2%} "
                      f"gap={stochastic_rate - greedy_rate:+.2%} sim_violations(train_cum={train_sim_violations}, "
                      f"eval_this_ckpt={eval_violations}) elapsed={elapsed:.0f}s", flush=True)

            if is_final_stretch:
                last_full_greedy, last_full_stochastic = greedy_eval, stochastic_eval

        if (it1 % checkpoint_every == 0) or (it1 == n_iterations):
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "action_gen_state": action_gen.get_state(), "iteration": it1, "episode_count": episode_count,
                "sim_violation_count": train_sim_violations,
            }, ckpt_path)
            print(f"[checkpoint-saved] iter={it1}", flush=True)

    # Final full held-out eval. The loop above always makes the last
    # iteration a full-eval checkpoint (is_final_stretch forced True at
    # it1 == n_iterations), so the normal path already has
    # last_full_greedy/last_full_stochastic in scope. The one case it
    # doesn't: a checkpoint previously reached n_iterations but the process
    # was killed before this final block ran (range(start_iteration,
    # n_iterations) is then empty, so the loop body never executes on
    # resume) -- fall back to evaluating the resumed model directly.
    if "last_full_greedy" not in locals():
        last_full_greedy = _evaluate(model, tier, test_instances, partition_cache,
                                      n_resets=cfg["n_eval_resets"], horizon=cfg["horizon"], mode="greedy")
        last_full_stochastic = _evaluate(model, tier, test_instances, partition_cache,
                                          n_resets=cfg["n_eval_resets"], horizon=cfg["horizon"], mode="stochastic",
                                          stochastic_rng=np.random.default_rng((seed, n_iterations)))
    aggregate_solve_rate_greedy = float(np.mean([v["solve_rate"] for v in last_full_greedy.values()]))
    aggregate_solve_rate_stochastic = float(np.mean([v["solve_rate"] for v in last_full_stochastic.values()]))
    final_eval_sim_violations = _count_sim_violations(last_full_greedy) + _count_sim_violations(last_full_stochastic)

    result = {
        "tier": tier, "encoder": encoder, "seed": seed,
        "curve": curve,
        "per_instance_test_outcomes_greedy": last_full_greedy,
        "per_instance_test_outcomes_stochastic": last_full_stochastic,
        "aggregate_solve_rate_greedy": aggregate_solve_rate_greedy,
        "aggregate_solve_rate_stochastic": aggregate_solve_rate_stochastic,
        "aggregate_solve_rate": aggregate_solve_rate_greedy,
        "per_instance_test_outcomes": last_full_greedy,
        "config": {**cfg, "commit_hash": commit_hash},
        "pool_provenance": pool_provenance,
        "n_train_instances": len(train_instances), "n_test_instances": len(test_instances),
        # See fairness_harness._is_sim_violation: count of the harness-level
        # skipped-cell-physics catch, training-time (cumulative over the
        # whole run) and eval-time (from this final full evaluation only).
        # Non-zero here doesn't invalidate the run -- those episodes are
        # already counted as not-solved in the solve rates above -- but a
        # high count is a real signal worth surfacing to the statistics
        # module/paper, not silently dropped.
        "train_sim_violations": train_sim_violations,
        "final_eval_sim_violations": final_eval_sim_violations,
    }
    with open(final_path, "w") as f:
        json.dump(result, f)
    print(f"[done] run_tag={run_tag} wrote {final_path} "
          f"(greedy={aggregate_solve_rate_greedy:.2%}, stochastic={aggregate_solve_rate_stochastic:.2%}, "
          f"train_sim_violations={train_sim_violations}, final_eval_sim_violations={final_eval_sim_violations})",
          flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["easy", "medium", "hard"])
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--encoder", default=None, help="direct mode, alternative to --array-index")
    p.add_argument("--seed", type=int, default=None, help="direct mode, alternative to --array-index")
    p.add_argument("--n-iterations", type=int, default=SWEEP_BUDGET)
    p.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    p.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    p.add_argument("--output-dir", default=RESULTS_DIR)
    args = p.parse_args()

    if args.array_index is not None:
        tier, encoder, seed = index_to_run(args.tier, args.array_index)
        print(f"array_index={args.array_index} -> tier={tier}, encoder={encoder}, seed={seed}", flush=True)
    else:
        if args.encoder is None or args.seed is None or args.tier is None:
            p.error("either --array-index, or all of --tier/--encoder/--seed, must be given")
        tier, encoder, seed = args.tier, args.encoder, args.seed

    run(tier, encoder, seed, args.output_dir, n_iterations=args.n_iterations,
        eval_every=args.eval_every, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
