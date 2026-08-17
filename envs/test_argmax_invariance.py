import torch
import torch.nn as nn

from argmax_invariance import argmax_dominant_fraction, build_dissimilar_batch
from fairness_harness import ENCODER_REGISTRY, get_final_pool
from ppo_train import MaskedEncoderActorCritic


def test_build_dissimilar_batch_is_deterministic_and_diverse():
    _train, test_instances = get_final_pool("medium")
    batch1 = build_dissimilar_batch("medium", test_instances, n_obs=16)
    batch2 = build_dissimilar_batch("medium", test_instances, n_obs=16)

    assert len(batch1) == 16
    # Deterministic: same seed -> same instances/states.
    for b1, b2 in zip(batch1, batch2):
        assert b1["inst"].start_cell == b2["inst"].start_cell
        assert b1["inst"].goal_cell == b2["inst"].goal_cell
        assert (b1["state"] == b2["state"]).all()

    n_distinct_partitions = len({b["inst"].partition_seed for b in batch1})
    n_distinct_starts = len({b["inst"].start_cell for b in batch1})
    assert n_distinct_partitions > 1
    assert n_distinct_starts > 1


def test_dominant_fraction_is_one_for_a_constant_actor():
    """A model whose actor layer is a constant function of the embedding
    (weight=0, only bias) must argmax to the same action on every input --
    dominant_fraction must be exactly 1.0, matching the collapsed-checkpoint
    signature B0a found (medium_hrep_seed0/medium_vrep_seed2: 24/24)."""
    _train, test_instances = get_final_pool("medium")
    batch = build_dissimilar_batch("medium", test_instances, n_obs=12)

    model = MaskedEncoderActorCritic(ENCODER_REGISTRY["hrep"]())
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
        model.actor.bias[3] = 10.0  # action index 3 dominates regardless of embedding

    frac, n_distinct, counts = argmax_dominant_fraction(model, batch)
    assert frac == 1.0
    assert n_distinct == 1
    assert counts[3] == len(batch)


def test_dominant_fraction_reflects_a_split_between_two_actions():
    """Mirrors the medium_relational_seed0 signature from B0a: not fully
    locked, but concentrated on exactly 2 of 8 actions. Two biases tied well
    above every other action's bias (0) means only those two can ever win
    argmax, regardless of embedding contribution -- forces a clean 2-way
    split to test against, without needing full lock-in."""
    _train, test_instances = get_final_pool("medium")
    batch = build_dissimilar_batch("medium", test_instances, n_obs=12)

    model = MaskedEncoderActorCritic(ENCODER_REGISTRY["hrep"]())
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
        model.actor.bias[0] = 5.0
        model.actor.bias[4] = 5.0

    frac, n_distinct, counts = argmax_dominant_fraction(model, batch)
    assert n_distinct <= 2
    assert counts[0] + counts[4] == len(batch)
    assert frac >= 0.5  # the more common of the two must be at least half the batch


def test_dominant_fraction_low_for_a_genuinely_input_sensitive_actor():
    """A healthy, trained checkpoint (medium_hrep_seed4 -- Stage A/B's
    designated healthy comparison run) should NOT show full lock-in."""
    _train, test_instances = get_final_pool("medium")
    batch = build_dissimilar_batch("medium", test_instances, n_obs=24)

    ckpt = torch.load("sweep_results/medium_hrep_seed4_checkpoint.pt", map_location="cpu")
    model = MaskedEncoderActorCritic(ENCODER_REGISTRY["hrep"]())
    model.load_state_dict(ckpt["model_state"])

    frac, n_distinct, counts = argmax_dominant_fraction(model, batch)
    assert frac < 1.0
    assert n_distinct >= 2


def test_counts_sum_to_batch_size():
    _train, test_instances = get_final_pool("easy")
    batch = build_dissimilar_batch("easy", test_instances, n_obs=10)
    model = MaskedEncoderActorCritic(ENCODER_REGISTRY["cnn"]())
    frac, n_distinct, counts = argmax_dominant_fraction(model, batch)
    assert sum(counts) == len(batch)
    assert len(counts) == 8
