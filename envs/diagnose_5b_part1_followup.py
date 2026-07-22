"""Step 5b-audit, Part 1: confirm training (not the forward/eval path) is
what's destroying performance.

1. Untrained baseline: a freshly-initialized (0 training steps) H-Rep policy,
   evaluated greedily on the full EASY test protocol (203 instances x30
   resets). Expected ~31% (matching random) if the forward/masking/eval path
   is sane.
2. Stochastic vs greedy eval on the SAME trained (entropy=0.05, budget=200)
   checkpoint: if stochastic sampling scores near/above random while greedy
   sits at ~18%, the bug is in the eval path (argmax interacting badly with
   something), not the learned policy itself. If both are low, the policy
   genuinely learned bad preferences.
"""
import numpy as np
import torch

from diagnose_5b import BUDGET, train_to_checkpoint
from fairness_harness import ENCODER_REGISTRY, get_final_pool
from nav_env import NavEnv
from pool import build_partition
from closed_loop_oracle import instance_to_config
from ppo_train import MaskedEncoderActorCritic, compute_action_mask

SEED = 0
TIER = "easy"
N_RESETS = 30
HORIZON = 40


def _evaluate(model, tier, test_instances, partition_cache, n_resets, horizon, stochastic=False, seed_base=0):
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
                agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
                mask = compute_action_mask(obs["state"], partition.domain)
                with torch.no_grad():
                    logits, _value = model(partition, inst, agent_state, current_cell=obs["cell"], mask=mask)
                if stochastic:
                    probs = torch.softmax(logits, dim=-1).numpy()
                    action = int(rng.choice(len(probs), p=probs))
                else:
                    action = int(torch.argmax(logits).item())
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
    print("=== Untrained baseline (0 training steps, H-Rep, EASY) ===", flush=True)
    train_instances, test_instances = get_final_pool(TIER)
    torch.manual_seed(SEED)
    untrained_model = MaskedEncoderActorCritic(ENCODER_REGISTRY["hrep"]())
    partition_cache_untrained = {}
    untrained_eval = _evaluate(untrained_model, TIER, test_instances, partition_cache_untrained,
                                n_resets=N_RESETS, horizon=HORIZON, stochastic=False)
    untrained_rate = float(np.mean([v["solve_rate"] for v in untrained_eval.values()]))
    print(f"UNTRAINED greedy solve rate: {untrained_rate:.2%}", flush=True)

    print("\n=== Trained checkpoint (entropy=0.05, budget=200): stochastic vs greedy ===", flush=True)
    model, buffer, train_solve_history, partition_cache, train_instances2, test_instances2, entropy_history = (
        train_to_checkpoint(entropy_coef=0.05, n_iterations=BUDGET, verbose=True)
    )
    greedy_eval = _evaluate(model, TIER, test_instances2, partition_cache, n_resets=N_RESETS, horizon=HORIZON,
                             stochastic=False)
    greedy_rate = float(np.mean([v["solve_rate"] for v in greedy_eval.values()]))
    print(f"TRAINED greedy solve rate: {greedy_rate:.2%}", flush=True)

    stochastic_eval = _evaluate(model, TIER, test_instances2, partition_cache, n_resets=N_RESETS, horizon=HORIZON,
                                 stochastic=True, seed_base=1)
    stochastic_rate = float(np.mean([v["solve_rate"] for v in stochastic_eval.values()]))
    print(f"TRAINED stochastic solve rate: {stochastic_rate:.2%}", flush=True)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"random baseline (previously measured): 31.05%", flush=True)
    print(f"untrained (this run): {untrained_rate:.2%}", flush=True)
    print(f"trained greedy (this run): {greedy_rate:.2%}", flush=True)
    print(f"trained stochastic (this run): {stochastic_rate:.2%}", flush=True)
    print("DONE")
