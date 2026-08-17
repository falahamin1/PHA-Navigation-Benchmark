"""B3: formalizes B0a's observation-dependence diagnostic into a reusable,
per-checkpoint metric a training loop can log alongside entropy/solve rate.

B0a's finding (Stage B, local ad hoc script): four collapsed checkpoints all
produced a LIVE, input-varying encoder embedding, but the actor head's
argmax barely moved across a batch of deliberately dissimilar observations
(different partitions, start cells, goal cells) -- two locked onto a single
action 24/24 times, one split 12/12 between two competing actions. That was
a one-shot post-hoc measurement. This module makes it a trackable curve:
build a FIXED dissimilar-observation batch once per (tier, seed), then call
`argmax_dominant_fraction` at every checkpoint through training on that same
fixed batch, so consecutive checkpoints are directly comparable.

Primary metric: dominant_fraction = (count of the single most-common argmax
action) / (batch size). 1.0 = fully locked (every observation picks the same
action); 1/8 = uniform across all 8 headings, the most diverse a batch this
size could look. Not a solve-rate or a p-value -- a direct, cheap read on
whether the policy's DECISION is input-sensitive, independent of whatever
the eval solve rate says.
"""
import numpy as np
import torch

from closed_loop_oracle import instance_to_config
from nav_env import NavEnv
from pool import build_partition

N_OBS_DEFAULT = 24
BATCH_RNG_SEED = 12345  # fixed, independent of training seed -- see module docstring


def build_dissimilar_batch(tier, test_instances, n_obs=N_OBS_DEFAULT, seed=BATCH_RNG_SEED):
    """N_OBS observations spanning distinct partitions, start/goal cells, and
    continuous agent states (via distinct reset seeds). Deterministic given
    (tier, seed) -- callable fresh at every checkpoint without needing to be
    saved/reloaded across a resumed run.
    """
    rng = np.random.default_rng(seed)
    n_obs = min(n_obs, len(test_instances))
    idx = rng.choice(len(test_instances), size=n_obs, replace=False)
    partition_cache = {}
    batch = []
    for i in idx:
        inst = test_instances[int(i)]
        if inst.partition_seed not in partition_cache:
            partition_cache[inst.partition_seed] = build_partition(tier, inst.partition_seed)
        partition = partition_cache[inst.partition_seed]
        cfg = instance_to_config(partition, inst)
        env = NavEnv(partition, cfg, horizon=40)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        batch.append(dict(partition=partition, inst=inst, state=obs["state"], cell=obs["cell"]))
    return batch


def argmax_dominant_fraction(model, batch):
    """Runs the batch through model.encoder + model.actor (the same split
    MaskedEncoderActorCritic.forward uses internally -- see ppo_train.py),
    no masking (matches B0a: pre-mask logits, so a policy can't look
    "decisive" merely because the mask killed 7 of 8 actions). Returns
    (dominant_fraction, n_distinct_actions, argmax_counts)."""
    model.eval()
    argmaxes = []
    with torch.no_grad():
        for b in batch:
            agent_state = torch.as_tensor(b["state"], dtype=torch.float32)
            emb = model.encoder(b["partition"], b["inst"], agent_state, current_cell=b["cell"])
            logits = model.actor(emb)
            argmaxes.append(int(torch.argmax(logits).item()))
    counts = np.bincount(argmaxes, minlength=8)
    dominant_fraction = float(counts.max() / len(batch))
    n_distinct = int((counts > 0).sum())
    return dominant_fraction, n_distinct, counts.tolist()
