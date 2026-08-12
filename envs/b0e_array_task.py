"""B0e: protocol-matched chance baseline. One Slurm array task = one
(tier, encoder, seed) cell (same index_to_run bijection sweep_cluster.py
already uses -- array_index in [0, 30) within a --tier). For that cell,
computes BOTH:

  (1) an untrained baseline: fresh encoder weights, torch.manual_seed(seed)
      (the same seed value already used to train that cell -- reused here as
      the "5 random initializations" the task asks for, not a new seed
      enumeration).
  (2) a re-evaluation of that cell's already-trained final checkpoint

under the IDENTICAL interval protocol already in the codebase (sweep_
cluster.py's intermediate-checkpoint eval): the fixed 100-instance subsample
drawn by np.random.default_rng(999), 10 resets/instance, greedy AND
stochastic modes. This is the exact code path A1 already exercised
(fairness_harness._evaluate with the same sampling), reused verbatim -- no
new evaluation logic.

Horizon policy -- an EVALUATION protocol parameter (already an existing
argument to _evaluate/NavEnv, not a change to env dynamics): capped at 15 for
EASY/MEDIUM, left at the frozen 40 for HARD. Empirically checked locally
before submission (see B0e report): horizon=15 vs 40 gave IDENTICAL solve
rates on every easy/medium spot check (untrained and trained); on HARD, even
horizon=30 lost 1 point off the untrained baseline (0.125 -> 0.115) because
an undirected random walk occasionally reaches the goal only via a long,
inefficient path that a shorter cap clips -- trained/collapsed policies
solve quickly if they solve at all (HARD checkpoint solve rate was IDENTICAL
from horizon=40 down to horizon=20), but the untrained side has no such
floor, so HARD keeps the full budget. The SAME horizon is used for both the
baseline and the checkpoint within a tier -- capping only one side would
reintroduce exactly the kind of protocol mismatch B0e exists to eliminate.

Self-check embedded in every task (cheap: first 10 of the 100 instances x 2
resets, greedy mode): re-evaluates that slice at the uncapped horizon=40 and
logs whether it matches the capped-horizon rate on the same slice -- gives
per-task, per-encoder evidence for the cap assumption across all 90 cells,
not just the 2-3 local spot checks the cap was originally chosen from. Never
blocks the task; only annotates the output.

Idempotent (matches sweep_cluster.py's convention): if the output file
already exists, no-op.

Writes sweep_results_b0e/<tier>_<encoder>_seed<seed>_b0e.json -- a NEW
directory. sweep_results/*.json (the validated Stage A/B sweep output) is
read-only here, never opened for writing.
"""
import argparse
import json
import os

import numpy as np
import torch

from fairness_harness import ENCODER_REGISTRY, get_final_pool, verify_pool_fingerprint, _evaluate
from ppo_train import MaskedEncoderActorCritic
from launch_reduced_sweep import index_to_run

RESULTS_DIR = "sweep_results"
OUTPUT_DIR_DEFAULT = "sweep_results_b0e"

EVAL_SUBSAMPLE_SEED = 999   # same constant sweep_cluster.py uses
N_EVAL_INSTANCES = 100
N_RESETS = 10
HORIZON_BY_TIER = {"easy": 15, "medium": 15, "hard": 40}
SELF_CHECK_N_INSTANCES = 10
SELF_CHECK_N_RESETS = 2
UNCAPPED_HORIZON = 40


def _run_tag(tier, encoder, seed):
    return f"{tier}_{encoder}_seed{seed}"


def _rate(eval_result):
    return float(np.mean([v["solve_rate"] for v in eval_result.values()]))


def evaluate_both_modes(model, tier, instances, partition_cache, n_resets, horizon, stochastic_seed):
    greedy = _evaluate(model, tier, instances, partition_cache, n_resets=n_resets,
                        horizon=horizon, mode="greedy")
    stochastic = _evaluate(model, tier, instances, partition_cache, n_resets=n_resets,
                            horizon=horizon, mode="stochastic",
                            stochastic_rng=np.random.default_rng(stochastic_seed))
    return _rate(greedy), _rate(stochastic)


def self_check(model, tier, probe_instances, partition_cache, horizon_capped, n_resets=SELF_CHECK_N_RESETS):
    if horizon_capped == UNCAPPED_HORIZON:
        # HARD never caps -- capped == uncapped by construction, skip the
        # redundant second eval rather than silently reporting a trivial match.
        rate = _rate(_evaluate(model, tier, probe_instances, partition_cache,
                                n_resets=n_resets, horizon=UNCAPPED_HORIZON, mode="greedy"))
        return dict(capped_rate=rate, uncapped_rate=rate, matched=True, skipped_redundant_uncapped_run=True)
    capped = _evaluate(model, tier, probe_instances, partition_cache,
                        n_resets=n_resets, horizon=horizon_capped, mode="greedy")
    uncapped = _evaluate(model, tier, probe_instances, partition_cache,
                          n_resets=n_resets, horizon=UNCAPPED_HORIZON, mode="greedy")
    capped_rate, uncapped_rate = _rate(capped), _rate(uncapped)
    return dict(capped_rate=capped_rate, uncapped_rate=uncapped_rate, matched=(capped_rate == uncapped_rate))


def run(tier, encoder, seed, array_index, output_dir,
        n_eval_instances=N_EVAL_INSTANCES, n_resets=N_RESETS,
        self_check_n_instances=SELF_CHECK_N_INSTANCES, self_check_n_resets=SELF_CHECK_N_RESETS):
    """n_eval_instances/n_resets/self_check_* exist for tests only (shrink a
    real cluster-scale call to something that finishes in seconds) -- real
    array tasks never pass them, so the interval protocol's defining
    constants (100 instances, 10 resets, seed=999) stay fixed for every real
    run."""
    os.makedirs(output_dir, exist_ok=True)
    tag = _run_tag(tier, encoder, seed)
    out_path = os.path.join(output_dir, f"{tag}_b0e.json")
    if os.path.exists(out_path):
        print(f"[done] {tag} already complete -- {out_path} exists, nothing to do", flush=True)
        return

    train_instances, test_instances = get_final_pool(tier)
    pool_provenance = verify_pool_fingerprint(tier, train_instances, test_instances)
    print(f"[pool] tier={tier} n_train={pool_provenance['n_train']} n_test={pool_provenance['n_test']} "
          f"train_fp={pool_provenance['train_fingerprint']} test_fp={pool_provenance['test_fingerprint']}",
          flush=True)

    rng = np.random.default_rng(EVAL_SUBSAMPLE_SEED)
    sample_n = min(n_eval_instances, len(test_instances))
    idx = rng.choice(len(test_instances), size=sample_n, replace=False)
    eval_instances = [test_instances[i] for i in idx]
    probe_instances = eval_instances[:self_check_n_instances]

    horizon = HORIZON_BY_TIER[tier]
    partition_cache = {}

    # (1) Untrained baseline.
    torch.manual_seed(seed)
    baseline_model = MaskedEncoderActorCritic(ENCODER_REGISTRY[encoder]())
    baseline_model.eval()
    baseline_greedy, baseline_stochastic = evaluate_both_modes(
        baseline_model, tier, eval_instances, partition_cache,
        n_resets=n_resets, horizon=horizon, stochastic_seed=(90001, seed))
    baseline_self_check = self_check(baseline_model, tier, probe_instances, partition_cache, horizon,
                                      n_resets=self_check_n_resets)
    print(f"[baseline] {tag} greedy={baseline_greedy:.4f} stochastic={baseline_stochastic:.4f} "
          f"self_check={baseline_self_check}", flush=True)

    # (2) Trained-checkpoint re-eval, interval protocol.
    ckpt_path = os.path.join(RESULTS_DIR, f"{tag}_checkpoint.pt")
    result_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    checkpoint_model = MaskedEncoderActorCritic(ENCODER_REGISTRY[encoder]())
    ckpt = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model.load_state_dict(ckpt["model_state"])
    checkpoint_model.eval()
    checkpoint_greedy, checkpoint_stochastic = evaluate_both_modes(
        checkpoint_model, tier, eval_instances, partition_cache,
        n_resets=n_resets, horizon=horizon, stochastic_seed=(90002, seed))
    checkpoint_self_check = self_check(checkpoint_model, tier, probe_instances, partition_cache, horizon,
                                        n_resets=self_check_n_resets)
    print(f"[checkpoint] {tag} greedy={checkpoint_greedy:.4f} stochastic={checkpoint_stochastic:.4f} "
          f"self_check={checkpoint_self_check}", flush=True)

    with open(result_path) as f:
        original_result = json.load(f)

    out = dict(
        tier=tier, encoder=encoder, seed=seed, array_index=array_index,
        horizon_used=horizon, n_eval_instances=sample_n, n_resets=n_resets,
        eval_subsample_seed=EVAL_SUBSAMPLE_SEED, pool_provenance=pool_provenance,
        baseline=dict(greedy_rate=baseline_greedy, stochastic_rate=baseline_stochastic,
                      self_check=baseline_self_check),
        checkpoint=dict(greedy_rate=checkpoint_greedy, stochastic_rate=checkpoint_stochastic,
                        self_check=checkpoint_self_check,
                        reported_full_protocol_greedy=original_result["aggregate_solve_rate_greedy"],
                        reported_full_protocol_stochastic=original_result["aggregate_solve_rate_stochastic"]),
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", required=True, choices=["easy", "medium", "hard"])
    p.add_argument("--array-index", type=int, required=True)
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = p.parse_args()

    tier, encoder, seed = index_to_run(args.tier, args.array_index)
    print(f"array_index={args.array_index} -> tier={tier}, encoder={encoder}, seed={seed}", flush=True)
    run(tier, encoder, seed, args.array_index, args.output_dir)


if __name__ == "__main__":
    main()
