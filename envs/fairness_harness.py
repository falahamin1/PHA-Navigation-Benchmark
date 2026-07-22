"""Step 5b: the fairness harness -- trains one encoder on one tier with one
seed under a single frozen config, for the eventual 6-encoder x 3-tier x
10-seed sweep on Alpine.

WHY THIS IS A NEW TRAINING LOOP, NOT A CALL TO ppo_train.train_ppo
-------------------------------------------------------------------
train_ppo (Step 5a, frozen -- not modified here) has a real gap against this
step's Invariant 1 ("same seed -> identical training-instance sequence
across different encoders"):

1. `torch.manual_seed(seed)` is called once, then `MaskedEncoderActorCritic
   (encoder_factory())` is constructed. Different encoder architectures
   consume different NUMBERS of draws from torch's global RNG during
   initialization (different layer counts/shapes) -- so immediately after
   construction, the global RNG's state already differs across encoders,
   even at the same seed. Any later torch-RNG-consuming call (action
   sampling via `dist.sample()`, which reads the global RNG) then diverges
   across encoders from that point on.
2. Instance selection (`new_env()`) draws from a `np.random.default_rng(seed)`
   lazily, once per episode boundary. The NUMBER of episodes completed within
   a fixed `steps_per_iter` budget depends on episode length, which depends
   on the POLICY's behavior -- i.e. on the encoder. Two encoders that solve
   at different rates will consume this RNG at different rates and see
   DIFFERENT instances from very early in training.

Neither of these is a bug in train_ppo -- Step 5a never needed cross-encoder
determinism, only within-run determinism (which it has, and which
test_ppo_loop.py's test_determinism already covers and remains valid). But
Step 5b's fairness invariant needs something train_ppo's design doesn't
provide, and Step 5b's brief is explicit that the PPO loop must not be
modified. So this module reuses every frozen PPO/GAE/masking ingredient
UNCHANGED (`MaskedEncoderActorCritic`, `compute_action_mask`, `compute_gae`
from ppo_train.py) inside a NEW outer loop that fixes exactly the two gaps
above:

1. Model init uses `torch.manual_seed(seed)` as before (architecture-
   dependent draw count is fine, expected, harmless) -- but ALL subsequent
   action sampling uses a SEPARATE, explicitly-passed `torch.Generator`
   seeded independently, so model-init's architecture-dependent RNG
   consumption can never perturb the action-sampling stream.
2. Instance/reset-seed assignment is a pure, stateless function of
   (seed, episode_index) -- `_episode_assignment` -- via a fresh
   `np.random.default_rng((seed, episode_index))` per episode, addressed by
   episode COUNT rather than drawn lazily from a running generator. Two
   encoders at the same seed request the SAME (instance, reset_seed) for
   their Nth episode regardless of how many total steps either took to get
   there. This is exactly "the identical training-instance sequence" --
   tested directly in test_fairness_harness.py's Invariant 1.

Env construction reuses closed_loop_oracle.py's `instance_to_config`
(tier-agnostic, already verified safe on all 3 tiers) rather than
ppo_train.py's EASY-only `_instance_to_config` (which calls `cell_bounds`,
absent on MEDIUM/HARD) -- this also makes the trained policy's start-region
convention match the one the closed-loop-oracle baseline already uses, so
the eventual optimality-gap comparison (agent steps vs oracle steps) is
apples-to-apples.

FROZEN CONFIG: one dict, `FROZEN_CONFIG`, is the single source of every
hyperparameter. `train_run` loads it (with tier as the only override point
for the calibrated budget, since MEDIUM/HARD budgets may legitimately differ
-- see BUDGET_BY_TIER) and never branches on encoder identity for anything
except architecture construction and each encoder's own already-fixed
CNN-pooling default (mean, set in baseline_encoders.py, not touched here).
"""
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamics"))
from integrator import DIRECTION_NAMES  # noqa: E402

from baseline_encoders import CNNEncoder, MLPEncoder  # noqa: E402
from closed_loop_oracle import instance_to_config  # noqa: E402
from deepset_encoders import HRepDeepSet, VRepDeepSet  # noqa: E402
from nav_env import NavEnv  # noqa: E402
from pool import Instance, build_partition, train_test_split  # noqa: E402
from ppo_train import (  # noqa: E402
    NUM_ACTIONS,
    MaskedEncoderActorCritic,
    compute_action_mask,
    compute_gae,
)
from reachability_filter import generate_reachable_pool  # noqa: E402
from region_gnn import RegionGraphGNN  # noqa: E402
from region_graph import build_region_graph  # noqa: E402
from relational_deepset import RelationalDeepSet  # noqa: E402

_HERE = os.path.dirname(__file__)
POOL_CACHE_DIR = os.path.join(_HERE, "pool_cache")
RESULTS_DIR = os.path.join(_HERE, "sweep_results")

POOL_TARGET_SIZES = {"easy": 800, "medium": 700, "hard": 700}
POOL_RNG_SEED = 42
POOL_SPLIT_SEED = 7

class _RegionGraphGNNAdapter(torch.nn.Module):
    """Bridges RegionGraphGNN's own interface -- forward(graph, agent_state),
    always called in region_gnn.py/test_region_gnn.py with a pre-built
    RegionGraph -- to the uniform forward(partition, instance, agent_state,
    current_cell=None) every other encoder (and MaskedEncoderActorCritic)
    expects. RegionGraphGNN was never previously run through the PPO loop
    (Step 5a used HRepDeepSet as the "simplest faithful encoder" stand-in
    throughout); this adapter is new glue code, not a modification of
    region_gnn.py.

    Caches the built RegionGraph per (partition, instance) -- rebuilding it
    (the O(cells^2) facet-sharing geometry in build_region_graph) on every
    single env step would be expensive and is unnecessary: only the
    dynamic contains_agent feature changes step to step, which
    graph.set_agent_cell() updates cheaply in place. Cache key is content-
    based (not id(partition)) for the same reason pool.py's strict-adjacency
    cache had to move off id() -- see pool.py's module docstring."""

    def __init__(self, **kwargs):
        super().__init__()
        self.gnn = RegionGraphGNN(**kwargs)
        self._graph_cache: Dict[tuple, object] = {}

    def _cache_key(self, partition, instance):
        return (type(partition).__name__, getattr(partition, "grid_seed", None), partition.num_cells, instance)

    def forward(self, partition, instance, agent_state, current_cell=None):
        key = self._cache_key(partition, instance)
        graph = self._graph_cache.get(key)
        if graph is None:
            graph = build_region_graph(partition, instance)
            self._graph_cache[key] = graph
        if current_cell is not None:
            graph.set_agent_cell(current_cell)
        return self.gnn(graph, agent_state)


ENCODER_REGISTRY = {
    "hrep": lambda: HRepDeepSet(),
    "vrep": lambda: VRepDeepSet(),
    "mlp": lambda: MLPEncoder(),
    "cnn": lambda: CNNEncoder(),  # pool_type="mean" default -- the fixed, carried-forward decision
    "relational": lambda: RelationalDeepSet(),
    "gnn": lambda: _RegionGraphGNNAdapter(hidden_dim=80, edge_type_mode="relational"),
}
ENCODER_NAMES = tuple(ENCODER_REGISTRY.keys())
TIERS = ("easy", "medium", "hard")
SEEDS = tuple(range(10))

# Navigation-specific frozen config (per-benchmark-uniform entropy, matching
# Tangram's convention of setting entropy per-benchmark, not globally).
# n_iterations (the calibrated training budget) is filled in per-tier via
# BUDGET_BY_TIER once Sub-step 1's calibration is run -- see that section's
# gate report for how these numbers were chosen.
FROZEN_CONFIG = dict(
    lr=1e-4,
    clip_eps=0.2,
    gamma=0.99,
    lam=0.95,
    entropy_coef=0.05,
    critic_coef=0.5,
    ppo_epochs=4,
    steps_per_iter=256,
    horizon=40,
    n_eval_resets=30,  # per-instance greedy resets for the held-out solve distribution
)

# Placeholder until Sub-step 1's calibration sets this; train_run requires an
# explicit override until then so a stale/undecided budget can never be used
# silently.
BUDGET_BY_TIER: Dict[str, Optional[int]] = {"easy": None, "medium": None, "hard": None}


# --- Pool caching (regenerating the Step B reachability-filtered pool from
# scratch costs 40-230s/tier -- caching once to disk makes the 180-run sweep
# tractable) ---------------------------------------------------------------

def _pool_cache_path(tier):
    return os.path.join(POOL_CACHE_DIR, f"{tier}_pool.pkl")


def get_final_pool(tier: str):
    """Returns (train_instances, test_instances), the Step B
    reachability-filtered, strict-solvable pool -- cached to disk after the
    first call."""
    path = _pool_cache_path(tier)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    os.makedirs(POOL_CACHE_DIR, exist_ok=True)
    kept, _removed, _excluded_unsafe, _stats, _raw_n = generate_reachable_pool(
        tier, POOL_TARGET_SIZES[tier], rng_seed=POOL_RNG_SEED
    )
    train, test = train_test_split(kept, test_fraction=0.2, rng_seed=POOL_SPLIT_SEED)
    with open(path, "wb") as f:
        pickle.dump((train, test), f)
    return train, test


def pool_fingerprint(instances) -> str:
    """Deterministic SHA-256 fingerprint (first 16 hex chars) over an
    ordered instance list. NOT Python's built-in hash() -- that's salted per
    process (PYTHONHASHSEED) for str fields, so it isn't stable across
    machines/runs, which defeats the entire point of a cross-machine
    provenance check. Order-sensitive on purpose: _episode_assignment
    indexes train_instances positionally, so a reordering (not just a
    different SET of instances) would itself be a real fairness bug this
    should catch, not something to normalize away by sorting first."""
    canonical = "|".join(
        f"{inst.tier},{inst.partition_seed},{inst.start_cell},{inst.goal_cell},"
        f"{'-'.join(map(str, inst.hazard_cells))},{inst.initial_velocity_sign[0]}:{inst.initial_velocity_sign[1]}"
        for inst in instances
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# --- Deterministic, encoder-independent episode assignment ----------------

def _episode_assignment(seed: int, episode_index: int, n_instances: int):
    """Pure function of (seed, episode_index, n_instances) -- identical
    across every encoder, since it never sees the encoder or the model.
    This IS "the identical training-instance sequence" Invariant 1 requires."""
    rng = np.random.default_rng((seed, episode_index))
    instance_idx = int(rng.integers(0, n_instances))
    reset_seed = int(rng.integers(0, 2**31 - 1))
    return instance_idx, reset_seed


def _direction_for_hop(partition, cell_a, cell_b):
    from integrator import DIRECTIONS
    ca, cb = partition.cell_centroid(cell_a), partition.cell_centroid(cell_b)
    delta = cb - ca
    delta_unit = delta / np.linalg.norm(delta)
    best_name, best_dot = None, -1e9
    for name in DIRECTION_NAMES:
        dot = float(np.dot(DIRECTIONS[name], delta_unit))
        if dot > best_dot:
            best_dot, best_name = dot, name
    return best_name


def _evaluate(model, tier, test_instances, partition_cache, n_resets=30, horizon=40, mode="greedy",
               stochastic_rng=None):
    """Per-instance evaluation on the held-out test set, in either mode:
    - "greedy": deterministic argmax -- measures whether the policy has
      become DECISIVE and correct. Standard reported protocol (matches
      Tangram).
    - "stochastic": samples from the policy's own distribution -- measures
      whether the policy carries a useful BIAS even before it has fully
      committed. The gap between the two modes is itself a convergence
      diagnostic (Step 5b-cal): large gap = still diffuse, gap closing =
      becoming decisive. See fairness_harness's module docstring and
      diagnose_5b_part1_followup.py, which is what surfaced the need for
      this (greedy alone was noise-dominated on a high-entropy policy and
      produced a misleading below-random number).

    Returns {instance_key: {"outcomes": [...], "steps": [...], "solve_rate": float}}.
    """
    if mode not in ("greedy", "stochastic"):
        raise ValueError(f"mode must be 'greedy' or 'stochastic', got {mode!r}")
    rng = stochastic_rng if stochastic_rng is not None else np.random.default_rng(0)
    results = {}
    for inst in test_instances:
        if inst.partition_seed not in partition_cache:
            partition_cache[inst.partition_seed] = build_partition(tier, inst.partition_seed)
        partition = partition_cache[inst.partition_seed]
        cfg = instance_to_config(partition, inst)
        outcomes, steps_list = [], []
        for reset_seed in range(n_resets):
            env = NavEnv(partition, cfg, horizon=horizon)
            obs, _ = env.reset(seed=reset_seed)
            steps = 0
            outcome = None
            for _ in range(horizon):
                agent_state = torch.as_tensor(obs["state"], dtype=torch.float32)
                mask = compute_action_mask(obs["state"], partition.domain)
                with torch.no_grad():
                    logits, _value = model(partition, inst, agent_state, current_cell=obs["cell"], mask=mask)
                if mode == "greedy":
                    action = int(torch.argmax(logits).item())
                else:
                    probs = torch.softmax(logits, dim=-1).numpy()
                    action = int(rng.choice(len(probs), p=probs))
                obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated:
                    outcome = info.get("outcome")
                    break
                if truncated:
                    outcome = "truncated"
                    break
            else:
                outcome = "truncated"
            outcomes.append(outcome)
            if outcome == "goal":
                steps_list.append(steps)
        key = f"{inst.start_cell}|{inst.goal_cell}|{','.join(map(str, inst.hazard_cells))}|{inst.partition_seed}"
        results[key] = {"outcomes": outcomes, "steps": steps_list,
                         "solve_rate": sum(1 for o in outcomes if o == "goal") / n_resets}
    return results


def _greedy_evaluate(model, tier, test_instances, partition_cache, n_resets=30, horizon=40):
    """Backward-compatible alias -- greedy-only evaluation. Prefer
    `_evaluate(..., mode="greedy")` / `_evaluate(..., mode="stochastic")`
    directly in new code, which is what train_run now uses for dual-mode
    logging (Step 5b-cal)."""
    return _evaluate(model, tier, test_instances, partition_cache, n_resets=n_resets, horizon=horizon, mode="greedy")


def train_run(
    tier: str, encoder: str, seed: int, n_iterations: Optional[int] = None,
    config: Optional[dict] = None, verbose: bool = False, results_dir: Optional[str] = None,
) -> dict:
    """The unit a Slurm array task calls. Trains `encoder` on `tier` with
    `seed` under FROZEN_CONFIG (n_iterations from BUDGET_BY_TIER unless
    explicitly overridden -- calibration runs use the override), evaluates
    greedily on the tier's held-out test set, and writes a self-contained
    result file tagged (tier, encoder, seed)."""
    if encoder not in ENCODER_REGISTRY:
        raise ValueError(f"unknown encoder {encoder!r}, must be one of {ENCODER_NAMES}")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    cfg = {**FROZEN_CONFIG, **(config or {})}
    if n_iterations is None:
        n_iterations = BUDGET_BY_TIER[tier]
    if n_iterations is None:
        raise ValueError(
            f"no training budget set for tier={tier!r} -- BUDGET_BY_TIER must be set by calibration, "
            "or pass n_iterations explicitly for a calibration run"
        )
    cfg["n_iterations"] = n_iterations

    train_instances, test_instances = get_final_pool(tier)
    partition_cache: Dict[int, object] = {}

    def get_partition(pseed):
        if pseed not in partition_cache:
            partition_cache[pseed] = build_partition(tier, pseed)
        return partition_cache[pseed]

    # --- Isolated RNGs: model init on torch's global RNG (architecture-
    # dependent draw count, fine/expected); ALL action sampling on a
    # separate, independently-seeded generator so it can never be perturbed
    # by model-init's per-architecture RNG consumption. ---
    torch.manual_seed(seed)
    model = MaskedEncoderActorCritic(ENCODER_REGISTRY[encoder]())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    action_gen = torch.Generator().manual_seed(seed)

    episode_count = 0
    instance_log: List[int] = []  # for the Invariant-1 test: the realized instance-index sequence

    def new_env():
        nonlocal episode_count
        idx, reset_seed = _episode_assignment(seed, episode_count, len(train_instances))
        instance_log.append(idx)
        inst = train_instances[idx]
        partition = get_partition(inst.partition_seed)
        env_cfg = instance_to_config(partition, inst)
        env = NavEnv(partition, env_cfg, horizon=cfg["horizon"])
        obs, _ = env.reset(seed=reset_seed)
        episode_count += 1
        return env, inst, partition, obs

    env, current_instance, current_partition, obs = new_env()
    episode_outcomes: List[str] = []
    learning_curve = {"iteration": [], "solve_rate": [], "policy_loss": [], "value_loss": [], "entropy": []}

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
        solve_rate = (sum(1 for o in recent if o == "goal") / len(recent)) if recent else 0.0
        learning_curve["iteration"].append(iteration)
        learning_curve["solve_rate"].append(solve_rate)
        learning_curve["policy_loss"].append(last_policy_loss)
        learning_curve["value_loss"].append(last_value_loss)
        learning_curve["entropy"].append(last_entropy)
        if verbose:
            print(f"  [{tier}/{encoder}/seed{seed}] iter {iteration:3d}: solve_rate={solve_rate:.2%}, "
                  f"policy_loss={last_policy_loss:+.4f}, value_loss={last_value_loss:.4f}, "
                  f"entropy={last_entropy:.4f}")

    # Step 5b-cal, Part 1: dual-mode logging. Greedy measures whether the
    # policy has become decisive and correct (the standard reported
    # protocol, matches Tangram); stochastic measures whether it carries a
    # useful bias even while still diffuse. The gap between them is a
    # convergence diagnostic in its own right -- see fairness_harness's
    # _evaluate docstring for why greedy alone can be noise-dominated and
    # misleading on a high-entropy policy.
    stochastic_rng = np.random.default_rng(seed)
    per_instance_greedy = _evaluate(model, tier, test_instances, partition_cache, n_resets=cfg["n_eval_resets"],
                                     horizon=cfg["horizon"], mode="greedy")
    per_instance_stochastic = _evaluate(model, tier, test_instances, partition_cache, n_resets=cfg["n_eval_resets"],
                                         horizon=cfg["horizon"], mode="stochastic", stochastic_rng=stochastic_rng)
    aggregate_solve_rate_greedy = float(np.mean([v["solve_rate"] for v in per_instance_greedy.values()]))
    aggregate_solve_rate_stochastic = float(np.mean([v["solve_rate"] for v in per_instance_stochastic.values()]))

    result = {
        "tier": tier, "encoder": encoder, "seed": seed,
        "learning_curve": learning_curve,
        "per_instance_test_outcomes_greedy": per_instance_greedy,
        "per_instance_test_outcomes_stochastic": per_instance_stochastic,
        "aggregate_solve_rate_greedy": aggregate_solve_rate_greedy,
        "aggregate_solve_rate_stochastic": aggregate_solve_rate_stochastic,
        # "aggregate_solve_rate"/"per_instance_test_outcomes" kept as aliases
        # to the greedy numbers -- greedy is the standard reported protocol
        # (matches Tangram) -- so any code still reading the pre-5b-cal
        # field names gets the same (greedy) numbers, not a KeyError.
        "aggregate_solve_rate": aggregate_solve_rate_greedy,
        "per_instance_test_outcomes": per_instance_greedy,
        "config": cfg,
        "n_train_instances": len(train_instances), "n_test_instances": len(test_instances),
    }

    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tier}_{encoder}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    result["_result_path"] = out_path
    result["_instance_log"] = instance_log  # not persisted -- Invariant-1 test hook only
    return result
