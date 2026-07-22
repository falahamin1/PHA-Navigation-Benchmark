"""Step 4c tests: run directly with `python test_relational_deepset.py`.

- test_shape_determinism_permutation: shape, determinism, within- and
  across-cell permutation invariance (the relational features, being
  aggregated summaries, must not break across-cell invariance).
- test_relational_features_influence_embedding: changing a neighbor's hazard
  status (same geometry) changes the embedding -- proves the augmentation is
  wired, not dead weight.
- test_identity_blind_boundary: THE central boundary test -- swapping WHICH
  neighbor carries a property (hazard vs not), while the aggregate multiset
  of neighbor types/geometry is unchanged, leaves a cell's relational
  features identical. Proves the features are identity-blind aggregates, not
  a preserved edge structure. Prints the per-cell feature dimension.
- test_no_multihop_propagation: THE sharpest test -- a concrete 3-cell chain
  A-B-C. Changing C's hazard status (2 hops from A, holding A-B connectivity
  fixed) must NOT change A's relational features, while it DOES change B's
  (B is C's direct neighbor). A GNN would propagate the 2-hop change; this
  control must not.
- test_no_truncation: inherits F_MAX=12 from 4a.
- test_param_count: printed alongside the (now six-encoder) table.
- test_overfit_one_instance: same clean EASY instance as 4a/4b.
"""

import numpy as np
import torch

from deepset_encoders import F_MAX, HRepDeepSet, VRepDeepSet
from partitions import IrregularConvexPartition, MixedConvexPartition
from pool import Instance, build_partition, generate_pool_with_stats
from region_gnn import RegionGraphGNN
from relational_deepset import NUM_RELATIONAL_FEATURES, RELATIONAL_FEATURE_NAMES, RelationalDeepSet, \
    compute_cell_relational_features
from test_region_graph import _ListPartition
from baseline_encoders import MLPEncoder, CNNEncoder

torch.manual_seed(0)


def _instance(goal_cell=24, hazard_cells=(8,)):
    return Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=goal_cell,
                     hazard_cells=tuple(sorted(hazard_cells)), initial_velocity_sign=(1, 1))


def test_shape_determinism_permutation():
    model = RelationalDeepSet()
    model.eval()
    agent_state = torch.tensor([2.5, 2.5, 0.1, -0.1])

    medium = MixedConvexPartition(grid_seed=0)
    hard = IrregularConvexPartition(grid_seed=0)
    with torch.no_grad():
        emb_medium = model(medium, _instance(goal_cell=24, hazard_cells=(8,)), agent_state)
        emb_hard = model(hard, _instance(goal_cell=5, hazard_cells=(1, 2)), agent_state)
    assert emb_medium.shape == (128,) and emb_hard.shape == (128,)
    print(f"  shape: MEDIUM ({medium.num_cells} cells) -> (128,); HARD ({hard.num_cells} cells) -> (128,)")

    with torch.no_grad():
        e1 = model(medium, _instance(), agent_state)
        e2 = model(medium, _instance(), agent_state)
    assert torch.equal(e1, e2), "determinism failed"
    print("  determinism: PASS")

    rng = np.random.default_rng(42)
    k = 5
    tensor = torch.zeros(F_MAX, 3)
    tensor[:k] = torch.tensor(rng.uniform(-2, 2, size=(k, 3)), dtype=torch.float32)
    mask = torch.zeros(F_MAX, dtype=torch.bool)
    mask[:k] = True
    perm = torch.tensor(rng.permutation(F_MAX))
    with torch.no_grad():
        pooled = model._masked_pool(model.phi(tensor), mask)
        pooled_perm = model._masked_pool(model.phi(tensor[perm]), mask[perm])
    within_residual = (pooled - pooled_perm).abs().max().item()
    assert within_residual < 1e-5, f"within-cell permutation residual too large: {within_residual}"
    print(f"  within-cell permutation residual = {within_residual:.3e}")

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
    agent_state2 = torch.tensor([0.5, 0.5, 0.0, 0.0])
    with torch.no_grad():
        emb = model(partition, instance, agent_state2)
        emb_perm = model(permuted_partition, permuted_instance, agent_state2)
    across_residual = (emb - emb_perm).abs().max().item()
    assert across_residual < 1e-4, f"across-cell permutation residual too large: {across_residual}"
    print(f"  across-cell permutation residual = {across_residual:.3e} "
          f"(relational features did not break across-cell invariance)")
    print("test_shape_determinism_permutation: PASS")
    return within_residual, across_residual


def test_relational_features_influence_embedding():
    partition = MixedConvexPartition(grid_seed=0)
    model = RelationalDeepSet()
    model.eval()
    agent_state = torch.tensor([2.0, 2.0, 0.0, 0.0])

    # Same goal, same geometry; hazard placed on a direct neighbor of cell 0
    # in one instance and somewhere unrelated in the other, so cell 0's
    # hazard_neighbor_count differs.
    neighbor_of_0 = next(i for i in range(1, partition.num_cells) if partition.neighbors_adjacent(0, i))
    far_cell = next(i for i in range(partition.num_cells) if not partition.neighbors_adjacent(0, i) and i != 0)

    inst_a = _instance(goal_cell=24, hazard_cells=(neighbor_of_0,))
    inst_b = _instance(goal_cell=24, hazard_cells=(far_cell,))

    feats_a = compute_cell_relational_features(partition, inst_a)
    feats_b = compute_cell_relational_features(partition, inst_b)
    assert feats_a[0, 1].item() != feats_b[0, 1].item(), "cell 0's hazard_neighbor_count should differ"

    with torch.no_grad():
        emb_a = model(partition, inst_a, agent_state)
        emb_b = model(partition, inst_b, agent_state)
    diff = (emb_a - emb_b).norm().item()
    assert diff > 1e-3, f"changing a neighbor's hazard status barely changed the embedding (diff={diff})"
    print(f"test_relational_features_influence_embedding: PASS (embedding changed by {diff:.4f} when "
          f"cell 0's neighbor-hazard-count changed from {feats_a[0,1].item():.0f} to {feats_b[0,1].item():.0f})")


def test_identity_blind_boundary():
    """Center cell A with two facet-adjacent neighbors L, R of equal size --
    swapping WHICH one is the hazard (aggregate multiset of neighbor types is
    the same either way: one hazard, one clear) must leave A's relational
    feature vector identical."""
    squares = [
        [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]],  # A (center), index 0
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # L, index 1
        [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],  # R, index 2
    ]
    partition = _ListPartition(squares)
    A, L, R = 0, 1, 2

    inst_L_hazard = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=-1, hazard_cells=(L,),
                              initial_velocity_sign=(1, 1))
    inst_R_hazard = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=-1, hazard_cells=(R,),
                              initial_velocity_sign=(1, 1))

    feats_L = compute_cell_relational_features(partition, inst_L_hazard)
    feats_R = compute_cell_relational_features(partition, inst_R_hazard)

    assert feats_L.shape[1] == NUM_RELATIONAL_FEATURES == 5, (
        f"expected a small fixed per-cell feature dimension, got {feats_L.shape[1]}"
    )
    residual = (feats_L[A] - feats_R[A]).abs().max().item()
    assert residual < 1e-9, (
        f"cell A's relational features changed ({residual}) when only WHICH neighbor was the hazard "
        "changed, not the aggregate neighbor-type multiset -- features are not identity-blind"
    )
    print(f"  per-cell relational feature names: {RELATIONAL_FEATURE_NAMES}")
    print(f"  per-cell relational feature dimension = {NUM_RELATIONAL_FEATURES} (fixed constant, "
          f"independent of degree/partition size)")
    print(f"  cell A features (L hazard) = {feats_L[A].tolist()}")
    print(f"  cell A features (R hazard) = {feats_R[A].tolist()}")
    print(f"  max residual when swapping which neighbor is the hazard = {residual:.3e}")
    print("test_identity_blind_boundary: PASS (features are identity-blind aggregates)")
    return residual


def test_no_multihop_propagation():
    """THE sharpest test. 3-cell chain A-B-C (A-B and B-C facet-adjacent,
    A-C NOT adjacent). Changing C's hazard status is a 2-hop change relative
    to A (A's direct neighborhood, i.e. just B, is unchanged). A GNN would
    propagate this in >=2 message-passing layers; this control must not."""
    squares = [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # A, index 0
        [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]],  # B, index 1
        [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],  # C, index 2
    ]
    partition = _ListPartition(squares)
    A, B, C = 0, 1, 2
    assert partition.cell_vertices  # sanity: using the minimal test partition

    from geometry import segment_overlap_length  # just to confirm A,C share no boundary
    vA, vC = partition.cell_vertices(A), partition.cell_vertices(C)
    max_overlap = max(
        segment_overlap_length(vA[i], vA[(i + 1) % len(vA)], vC[j], vC[(j + 1) % len(vC)])
        for i in range(len(vA)) for j in range(len(vC))
    )
    assert max_overlap < 1e-9, "test setup broken: A and C should not be adjacent at all"

    inst_c_clear = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=-1, hazard_cells=(),
                             initial_velocity_sign=(1, 1))
    inst_c_hazard = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=-1, hazard_cells=(C,),
                              initial_velocity_sign=(1, 1))

    feats_clear = compute_cell_relational_features(partition, inst_c_clear)
    feats_hazard = compute_cell_relational_features(partition, inst_c_hazard)

    a_residual = (feats_clear[A] - feats_hazard[A]).abs().max().item()
    b_residual = (feats_clear[B] - feats_hazard[B]).abs().max().item()

    assert a_residual < 1e-9, (
        f"cell A's (2 hops from C) relational features changed by {a_residual} when C became a hazard "
        "-- this control is propagating information beyond 1 hop, i.e. it has become a disguised GNN"
    )
    assert b_residual > 1e-9, (
        f"cell B's (1 hop from C, C's direct neighbor) features did NOT change ({b_residual}) when C "
        "became a hazard -- the feature computation itself looks broken, not just non-propagating"
    )
    print(f"  cell A (2 hops from C) residual = {a_residual:.3e}  (must be ~0 -- confirmed no multi-hop propagation)")
    print(f"  cell B (1 hop from C) residual  = {b_residual:.3e}  (must be >0 -- confirms the feature is live)")
    print("test_no_multihop_propagation: PASS (this is a 1-hop control, not a disguised GNN)")
    return a_residual, b_residual


def test_no_truncation():
    pool_sizes = {"easy": 800, "medium": 700, "hard": 700}
    overall_max = 0
    for tier in ("easy", "medium", "hard"):
        pool, _stats = generate_pool_with_stats(tier, pool_sizes[tier], rng_seed=42)
        seeds = {inst.partition_seed for inst in pool}
        tier_max = max(
            build_partition(tier, seed).cell_facet_count(i)
            for seed in seeds for i in range(build_partition(tier, seed).num_cells)
        )
        overall_max = max(overall_max, tier_max)
    print(f"test_no_truncation: PASS (pool-wide observed max facet count = {overall_max}, F_MAX = {F_MAX})")
    assert overall_max <= F_MAX


def test_param_count():
    rel = RelationalDeepSet()
    mlp, cnn = MLPEncoder(), CNNEncoder()
    h, v = HRepDeepSet(), VRepDeepSet()
    gnn = RegionGraphGNN(hidden_dim=80, edge_type_mode="relational")
    budget = 53000
    print(f"  MLP                             = {mlp.param_count():>7,}  ({mlp.param_count()/budget:.2f}x)")
    print(f"  CNN                             = {cnn.param_count():>7,}  ({cnn.param_count()/budget:.2f}x)")
    print(f"  H-Rep DeepSet                   = {h.param_count():>7,}  ({h.param_count()/budget:.2f}x)")
    print(f"  V-Rep DeepSet                   = {v.param_count():>7,}  ({v.param_count()/budget:.2f}x)")
    print(f"  GNN (hidden_dim=80, relational) = {gnn.param_count():>7,}  ({gnn.param_count()/budget:.2f}x)")
    print(f"  RelationalDeepSet               = {rel.param_count():>7,}  ({rel.param_count()/budget:.2f}x)")
    print("test_param_count: PASS")


def test_overfit_one_instance():
    from ppo_smoke import train_overfit
    _model, history = train_overfit(RelationalDeepSet(), n_iterations=40, steps_per_iter=128, ppo_epochs=3,
                                     verbose=False)
    final_rate = history[-1] if history else 0.0
    assert final_rate >= 0.95, f"RelationalDeepSet failed to reach 95% solve rate (final={final_rate:.0%})"
    print(f"  solve-rate curve = {[f'{r:.2f}' for r in history]}")
    print(f"test_overfit_one_instance: PASS (reached {final_rate:.0%} in {len(history)} iterations)")


if __name__ == "__main__":
    test_shape_determinism_permutation()
    test_relational_features_influence_embedding()
    test_identity_blind_boundary()
    test_no_multihop_propagation()
    test_no_truncation()
    test_param_count()
    test_overfit_one_instance()
    print("Step 4c tests: ALL PASS")
