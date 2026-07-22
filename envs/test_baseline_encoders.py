"""Step 4b tests: run directly with `python test_baseline_encoders.py`.

- test_shape: MEDIUM/HARD -> embedding of dim d for both encoders; MLP
  max-cell-count no-truncation-analog check (prints pool-wide max and bound).
- test_mlp_not_permutation_invariant: reordering cells DOES change the MLP's
  embedding -- confirms it's genuinely the point-wise strawman, not
  accidentally set-invariant.
- test_cnn_param_cap: CNN params <= 1.2x the ~53k budget; confirms global
  pooling is used (head's input dim is the channel count, not
  resolution^2 x channels -- the flatten-blowup this step is guarding against).
- test_cnn_spatial_sensitivity: moving the goal/hazard to a different cell
  changes the CNN embedding.
- test_agent_state_global_concat: both encoders; for CNN, isolates velocity
  (position legitimately enters the raster by design, so a position-only
  change is not a clean test of "global-only" -- velocity-only is).
- test_overfit_one_instance: both baselines reach 95%+ solve rate on the
  SAME clean, edge-adjacent EASY instance 4a used (not a pool-default
  instance that might hinge on a corner-hop-near-hazard).
- test_param_count_table: MLP, CNN, H-Rep, V-Rep, GNN-80-rel side by side.
- test_determinism: same input + weights -> identical embedding, both.
"""

import numpy as np
import torch

from baseline_encoders import CNN_RESOLUTION, CNNEncoder, MAX_CELLS, MLPEncoder
from deepset_encoders import HRepDeepSet, VRepDeepSet
from partitions import IrregularConvexPartition, MixedConvexPartition
from pool import Instance, build_partition, generate_pool_with_stats
from region_gnn import RegionGraphGNN
from test_region_graph import _ListPartition

torch.manual_seed(0)

BUDGET = 53000
CNN_CAP = 1.2 * BUDGET


def _instance(goal_cell=24, hazard_cells=(8,)):
    return Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=goal_cell,
                     hazard_cells=tuple(sorted(hazard_cells)), initial_velocity_sign=(1, 1))


def test_shape():
    agent_state = torch.tensor([2.5, 2.5, 0.1, -0.1])
    medium = MixedConvexPartition(grid_seed=0)
    hard = IrregularConvexPartition(grid_seed=0)
    inst_medium = _instance(goal_cell=24, hazard_cells=(8,))
    inst_hard = _instance(goal_cell=5, hazard_cells=(1, 2))

    for name, Encoder in [("MLP", MLPEncoder), ("CNN", CNNEncoder)]:
        model = Encoder()
        model.eval()
        with torch.no_grad():
            emb_medium = model(medium, inst_medium, agent_state)
            emb_hard = model(hard, inst_hard, agent_state)
        assert emb_medium.shape == (128,), f"{name} MEDIUM shape {emb_medium.shape} != (128,)"
        assert emb_hard.shape == (128,), f"{name} HARD shape {emb_hard.shape} != (128,)"
        print(f"  {name}: MEDIUM ({medium.num_cells} cells) -> (128,); HARD ({hard.num_cells} cells) -> (128,)")

    pool_sizes = {"easy": 800, "medium": 700, "hard": 700}
    overall_max = 0
    for tier in ("easy", "medium", "hard"):
        pool, _stats = generate_pool_with_stats(tier, pool_sizes[tier], rng_seed=42)
        seeds = {inst.partition_seed for inst in pool}
        tier_max = max(build_partition(tier, s).num_cells for s in seeds)
        overall_max = max(overall_max, tier_max)
    print(f"  MLP no-truncation check: pool-wide observed max num_cells = {overall_max}, MAX_CELLS = {MAX_CELLS}")
    assert overall_max <= MAX_CELLS, f"observed max cells {overall_max} > MAX_CELLS={MAX_CELLS} -- would truncate"
    print("test_shape: PASS")


def test_mlp_not_permutation_invariant():
    squares = [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],
        [[4.0, 0.0], [5.0, 0.0], [5.0, 1.0], [4.0, 1.0]],
        [[6.0, 0.0], [7.0, 0.0], [7.0, 1.0], [6.0, 1.0]],
    ]
    partition = _ListPartition(squares)
    instance = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=1, hazard_cells=(2,),
                         initial_velocity_sign=(1, 1))
    perm_map = [2, 0, 3, 1]
    permuted_squares, inverse = [None] * 4, [0] * 4
    for old_idx, new_idx in enumerate(perm_map):
        permuted_squares[new_idx] = squares[old_idx]
        inverse[old_idx] = new_idx
    permuted_partition = _ListPartition(permuted_squares)
    permuted_instance = Instance(tier="test", partition_seed=0, start_cell=inverse[0], goal_cell=inverse[1],
                                  hazard_cells=(inverse[2],), initial_velocity_sign=(1, 1))
    agent_state = torch.tensor([0.5, 0.5, 0.0, 0.0])

    model = MLPEncoder()
    model.eval()
    with torch.no_grad():
        emb = model(partition, instance, agent_state)
        emb_permuted = model(permuted_partition, permuted_instance, agent_state)
    residual = (emb - emb_permuted).abs().max().item()
    assert residual > 1e-3, (
        f"MLP embedding barely changed under cell reordering (residual={residual}) -- "
        "it may have accidentally acquired permutation invariance and stopped being the strawman"
    )
    print(f"test_mlp_not_permutation_invariant: PASS (residual = {residual:.4f}, confirms genuine non-invariance)")
    return residual


def test_cnn_param_cap():
    model = CNNEncoder()
    count = model.param_count()
    print(f"  CNN param count = {count:,} (cap = {CNN_CAP:,.0f}, {count/BUDGET:.2f}x the ~53k budget)")
    assert count <= CNN_CAP, f"CNN params {count} exceed the {CNN_CAP:.0f} cap"

    head_in_features = model.head[0].in_features
    spatial_flatten_size = model.conv2.out_channels * CNN_RESOLUTION * CNN_RESOLUTION
    assert head_in_features != spatial_flatten_size, (
        "head's input dim equals channels x resolution^2 -- looks like a flatten, not global pooling"
    )
    assert head_in_features == model.conv2.out_channels + 4, (
        f"expected head input = channel_count + 4 (global-pooled, not flattened), got {head_in_features}"
    )
    print(f"  confirmed global pooling: head input dim = {head_in_features} "
          f"(channels={model.conv2.out_channels} + agent_state=4), NOT {spatial_flatten_size} (the flatten size)")
    print("test_cnn_param_cap: PASS")
    return count


def test_cnn_spatial_sensitivity():
    partition = MixedConvexPartition(grid_seed=0)
    agent_state = torch.tensor([2.5, 2.5, 0.0, 0.0])
    inst_a = _instance(goal_cell=1, hazard_cells=(8,))
    inst_b = _instance(goal_cell=20, hazard_cells=(8,))  # different goal cell

    model = CNNEncoder()
    model.eval()
    with torch.no_grad():
        emb_a = model(partition, inst_a, agent_state)
        emb_b = model(partition, inst_b, agent_state)
    diff = (emb_a - emb_b).norm().item()
    assert diff > 1e-3, f"moving the goal cell barely changed the CNN embedding (diff={diff})"
    print(f"test_cnn_spatial_sensitivity: PASS (moving goal cell changed embedding by {diff:.4f})")


def test_agent_state_global_concat():
    partition = MixedConvexPartition(grid_seed=1)
    instance = _instance()

    # MLP: encode_region() takes no agent_state at all -- same structural
    # guarantee as the GNN/DeepSet encoders.
    mlp = MLPEncoder()
    mlp.eval()
    with torch.no_grad():
        pooled_a = mlp.encode_region(partition, instance)
        pooled_b = mlp.encode_region(partition, instance)
        assert torch.equal(pooled_a, pooled_b), "MLP: encode_region should be exactly deterministic"
        agent1 = torch.tensor([0.5, 0.5, 0.0, 0.0])
        agent2 = torch.tensor([4.5, 4.5, -0.2, 0.3])
        emb1 = mlp(partition, instance, agent1)
        emb2 = mlp(partition, instance, agent2)
    assert not torch.allclose(emb1, emb2), "MLP: changing agent state should change the embedding"
    print(f"  MLP: encode_region() takes no agent_state parameter; forward() changed by "
          f"{(emb1 - emb2).norm().item():.4f} when agent state changed")

    # CNN: position legitimately enters the raster by design (splatted
    # channel), so isolate velocity (position held fixed) to cleanly test
    # the global-only concat pathway without conflating it with the raster's
    # intentional position sensitivity (already covered by
    # test_cnn_spatial_sensitivity).
    cnn = CNNEncoder()
    cnn.eval()
    with torch.no_grad():
        agent1 = torch.tensor([2.0, 2.0, 0.0, 0.0])
        agent2 = torch.tensor([2.0, 2.0, -0.5, 0.6])
        emb1 = cnn(partition, instance, agent1)
        emb2 = cnn(partition, instance, agent2)
    assert not torch.allclose(emb1, emb2), "CNN: changing agent velocity (position fixed) should change the embedding"
    print(f"  CNN: forward() changed by {(emb1 - emb2).norm().item():.4f} when agent VELOCITY changed "
          f"(position held fixed, isolating the global-concat pathway)")
    print("test_agent_state_global_concat: PASS")


def test_determinism():
    partition = IrregularConvexPartition(grid_seed=2)
    instance = _instance(goal_cell=3, hazard_cells=(1,))
    agent_state = torch.tensor([1.0, 1.0, 0.1, 0.1])
    for name, Encoder in [("MLP", MLPEncoder), ("CNN", CNNEncoder)]:
        model = Encoder()
        model.eval()
        with torch.no_grad():
            emb1 = model(partition, instance, agent_state)
            emb2 = model(partition, instance, agent_state)
        assert torch.equal(emb1, emb2), f"{name}: identical input+weights should give a bit-identical embedding"
    print("test_determinism: PASS (both baselines bit-identical across repeated calls)")


def test_param_count_table():
    mlp, cnn = MLPEncoder(), CNNEncoder()
    h, v = HRepDeepSet(), VRepDeepSet()
    gnn = RegionGraphGNN(hidden_dim=80, edge_type_mode="relational")
    print(f"  MLP                       = {mlp.param_count():>7,}  ({mlp.param_count()/BUDGET:.2f}x)")
    print(f"  CNN                       = {cnn.param_count():>7,}  ({cnn.param_count()/BUDGET:.2f}x)")
    print(f"  H-Rep DeepSet             = {h.param_count():>7,}  ({h.param_count()/BUDGET:.2f}x)")
    print(f"  V-Rep DeepSet             = {v.param_count():>7,}  ({v.param_count()/BUDGET:.2f}x)")
    print(f"  GNN (hidden_dim=80, rel.) = {gnn.param_count():>7,}  ({gnn.param_count()/BUDGET:.2f}x)")
    print("test_param_count_table: PASS (printed for review)")


def test_overfit_one_instance():
    from ppo_smoke import train_overfit
    for name, Encoder in [("MLP", MLPEncoder), ("CNN", CNNEncoder)]:
        print(f"  training {name} on the same clean EASY instance 4a used...")
        _model, history = train_overfit(Encoder(), n_iterations=40, steps_per_iter=128, ppo_epochs=3, verbose=False)
        final_rate = history[-1] if history else 0.0
        assert final_rate >= 0.95, f"{name}: failed to reach 95% solve rate (final={final_rate:.0%}) -- wiring bug"
        print(f"  {name}: solve-rate curve = {[f'{r:.2f}' for r in history]}")
        print(f"  {name}: reached {final_rate:.0%} in {len(history)} iterations")
    print("test_overfit_one_instance: PASS")


if __name__ == "__main__":
    test_shape()
    test_mlp_not_permutation_invariant()
    test_cnn_param_cap()
    test_cnn_spatial_sensitivity()
    test_agent_state_global_concat()
    test_determinism()
    test_param_count_table()
    test_overfit_one_instance()
    print("Step 4b tests: ALL PASS")
