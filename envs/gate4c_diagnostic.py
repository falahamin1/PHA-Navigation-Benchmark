"""Step 4c gate diagnostic: run directly with `python gate4c_diagnostic.py`.

One place: the relational feature list + dimension + justification, the
2-hop test result (the non-GNN proof), the across-cell permutation residual,
the six-encoder param table, and the overfit curve.
"""

import torch

from baseline_encoders import CNNEncoder, MLPEncoder
from deepset_encoders import HRepDeepSet, VRepDeepSet
from ppo_smoke import train_overfit
from region_gnn import RegionGraphGNN
from relational_deepset import NUM_RELATIONAL_FEATURES, RELATIONAL_FEATURE_NAMES, RelationalDeepSet, \
    compute_cell_relational_features
from test_region_graph import _ListPartition
from pool import Instance

torch.manual_seed(0)
BUDGET = 53000


def main():
    print("=" * 78)
    print("1) RELATIONAL FEATURES CHOSEN + JUSTIFICATION")
    print("=" * 78)
    print(f"  Features ({NUM_RELATIONAL_FEATURES}, fixed constant, from strict facet-sharing adjacency, "
          f"the same predicate the GNN uses):")
    for i, name in enumerate(RELATIONAL_FEATURE_NAMES):
        print(f"    {i}: {name}")
    print(
        "\n  Why this is fair augmentation, not a disguised graph:\n"
        "  - every feature is a SUM/COUNT over the neighbor set: permutation-invariant\n"
        "    and lossy -- a cell knows '2 neighbors are hazards' but not which two.\n"
        "  - fixed dimension (5), independent of degree/partition size/tier -- unlike a\n"
        "    GNN's per-node hidden state (refreshed from a SPECIFIC neighbor each message).\n"
        "  - strictly 1-hop: computed only from a cell's own inter-cell edges; a\n"
        "    neighbor's neighbor never enters the computation. Proven empirically below,\n"
        "    not just asserted by inspection."
    )

    print("=" * 78)
    print("2) THE 2-HOP TEST (the sharpest line between this and a GNN)")
    print("=" * 78)
    squares = [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # A, index 0
        [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]],  # B, index 1
        [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],  # C, index 2
    ]
    partition = _ListPartition(squares)
    A, B, C = 0, 1, 2
    inst_clear = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=-1, hazard_cells=(),
                           initial_velocity_sign=(1, 1))
    inst_hazard = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=-1, hazard_cells=(C,),
                            initial_velocity_sign=(1, 1))
    feats_clear = compute_cell_relational_features(partition, inst_clear)
    feats_hazard = compute_cell_relational_features(partition, inst_hazard)
    a_residual = (feats_clear[A] - feats_hazard[A]).abs().max().item()
    b_residual = (feats_clear[B] - feats_hazard[B]).abs().max().item()
    print(f"  3-cell chain A-B-C (A-B and B-C facet-adjacent, A-C NOT adjacent).")
    print(f"  C becomes a hazard (2 hops from A, 1 hop from B):")
    print(f"    cell A residual (2 hops away) = {a_residual:.3e}   {'PASS (no propagation)' if a_residual < 1e-9 else 'FAIL'}")
    print(f"    cell B residual (1 hop away)  = {b_residual:.3e}   {'PASS (feature is live)' if b_residual > 1e-9 else 'FAIL'}")

    print("=" * 78)
    print("3) ACROSS-CELL PERMUTATION RESIDUAL (relational features didn't break invariance)")
    print("=" * 78)
    squares2 = [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],
        [[4.0, 0.0], [5.0, 0.0], [5.0, 1.0], [4.0, 1.0]],
        [[6.0, 0.0], [7.0, 0.0], [7.0, 1.0], [6.0, 1.0]],
    ]
    partition2 = _ListPartition(squares2)
    instance2 = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=1, hazard_cells=(2,),
                          initial_velocity_sign=(1, 1))
    perm_map = [2, 0, 3, 1]
    permuted_squares, inverse = [None] * 4, [0] * 4
    for old_idx, new_idx in enumerate(perm_map):
        permuted_squares[new_idx] = squares2[old_idx]
        inverse[old_idx] = new_idx
    permuted_partition = _ListPartition(permuted_squares)
    permuted_instance = Instance(tier="test", partition_seed=0, start_cell=inverse[0], goal_cell=inverse[1],
                                  hazard_cells=(inverse[2],), initial_velocity_sign=(1, 1))
    model = RelationalDeepSet()
    model.eval()
    agent_state = torch.tensor([0.5, 0.5, 0.0, 0.0])
    with torch.no_grad():
        emb = model(partition2, instance2, agent_state)
        emb_perm = model(permuted_partition, permuted_instance, agent_state)
    residual = (emb - emb_perm).abs().max().item()
    print(f"  across-cell permutation residual = {residual:.3e}  (tolerance 1e-4)")

    print("=" * 78)
    print("4) SIX-ENCODER PARAM TABLE")
    print("=" * 78)
    mlp, cnn = MLPEncoder(), CNNEncoder()
    h, v = HRepDeepSet(), VRepDeepSet()
    gnn = RegionGraphGNN(hidden_dim=80, edge_type_mode="relational")
    rel = model
    for name, m in [("MLP", mlp), ("CNN", cnn), ("H-Rep DeepSet", h), ("V-Rep DeepSet", v),
                    ("GNN (hidden_dim=80, relational)", gnn), ("RelationalDeepSet", rel)]:
        c = m.param_count()
        print(f"  {name:34s} = {c:>7,}  ({c/BUDGET:.2f}x ~53k budget)")

    print("=" * 78)
    print("5) OVERFIT-ONE-INSTANCE SOLVE CURVE (clean, edge-adjacent EASY instance)")
    print("=" * 78)
    _model, history = train_overfit(RelationalDeepSet(), n_iterations=40, steps_per_iter=128, ppo_epochs=3,
                                     verbose=False)
    curve = " -> ".join(f"{r:.2f}" for r in history)
    print(f"  RelationalDeepSet ({len(history)} iterations to reach target): {curve}")
    print("=" * 78)


if __name__ == "__main__":
    main()
