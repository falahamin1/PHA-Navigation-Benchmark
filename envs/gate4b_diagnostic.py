"""Step 4b gate diagnostic: run directly with `python gate4b_diagnostic.py`.

One place: five-encoder param table, CNN global-pooling confirmation + param
cap, MLP-not-invariant residual, pool-wide max cell count vs MAX_CELLS, and
the two overfit-one-instance solve curves on the clean EASY instance.
"""

import torch

from baseline_encoders import CNN_RESOLUTION, CNNEncoder, MAX_CELLS, MLPEncoder
from deepset_encoders import HRepDeepSet, VRepDeepSet
from partitions import MixedConvexPartition
from pool import Instance, build_partition, generate_pool_with_stats
from ppo_smoke import train_overfit
from region_gnn import RegionGraphGNN
from test_region_graph import _ListPartition

torch.manual_seed(0)
BUDGET = 53000


def main():
    print("=" * 78)
    print("1) FIVE-ENCODER PARAM TABLE")
    print("=" * 78)
    mlp, cnn = MLPEncoder(), CNNEncoder()
    h, v = HRepDeepSet(), VRepDeepSet()
    gnn = RegionGraphGNN(hidden_dim=80, edge_type_mode="relational")
    for name, model in [("MLP", mlp), ("CNN", cnn), ("H-Rep DeepSet", h), ("V-Rep DeepSet", v),
                        ("GNN (hidden_dim=80, relational)", gnn)]:
        c = model.param_count()
        print(f"  {name:34s} = {c:>7,}  ({c/BUDGET:.2f}x ~53k budget)")

    print("=" * 78)
    print("2) CNN GLOBAL-POOLING CONFIRMATION + PARAM CAP")
    print("=" * 78)
    cap = 1.2 * BUDGET
    count = cnn.param_count()
    head_in = cnn.head[0].in_features
    flatten_size = cnn.conv2.out_channels * CNN_RESOLUTION * CNN_RESOLUTION
    print(f"  CNN params = {count:,}  <=  cap {cap:,.0f}  ({'OK' if count <= cap else 'FAIL'})")
    print(f"  head input dim = {head_in} (= channels {cnn.conv2.out_channels} + agent_state 4)")
    print(f"  flatten would have been = {flatten_size:,}  -- NOT what's used (confirms global pooling, not flatten)")
    print(f"  pool_type = {cnn.pool_type!r} (config flag; 'max' available -- see baseline_encoders.py note on "
          f"mean-pool diluting small localized raster changes ~28x more than max-pool would)")

    print("=" * 78)
    print("3) MLP-NOT-PERMUTATION-INVARIANT RESIDUAL")
    print("=" * 78)
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
    mlp.eval()
    with torch.no_grad():
        emb = mlp(partition, instance, agent_state)
        emb_perm = mlp(permuted_partition, permuted_instance, agent_state)
    residual = (emb - emb_perm).abs().max().item()
    print(f"  MLP embedding changed by {residual:.4f} under cell reordering (NOT invariant, as intended -- "
          f"a real DeepSet/GNN would be ~1e-7 here, see Step 3b/4a diagnostics)")

    print("=" * 78)
    print("4) MAX-CELL-COUNT NO-TRUNCATION CHECK (MLP's F_max analog)")
    print("=" * 78)
    pool_sizes = {"easy": 800, "medium": 700, "hard": 700}
    overall_max = 0
    for tier in ("easy", "medium", "hard"):
        pool, _stats = generate_pool_with_stats(tier, pool_sizes[tier], rng_seed=42)
        seeds = {inst.partition_seed for inst in pool}
        tier_max = max(build_partition(tier, s).num_cells for s in seeds)
        print(f"  {tier:6s}: {len(seeds):3d} distinct partition seeds, max num_cells = {tier_max}")
        overall_max = max(overall_max, tier_max)
    print(f"  OBSERVED POOL-WIDE MAX = {overall_max}   MAX_CELLS USED = {MAX_CELLS}   "
          f"{'OK' if MAX_CELLS >= overall_max else 'FAIL -- TRUNCATION RISK'}")

    print("=" * 78)
    print("5) OVERFIT-ONE-INSTANCE SOLVE CURVES (clean, edge-adjacent EASY instance)")
    print("=" * 78)
    for name, Encoder in [("MLP", MLPEncoder), ("CNN", CNNEncoder)]:
        _model, history = train_overfit(Encoder(), n_iterations=40, steps_per_iter=128, ppo_epochs=3, verbose=False)
        curve = " -> ".join(f"{r:.2f}" for r in history)
        print(f"  {name} ({len(history)} iterations to reach target): {curve}")
    print("=" * 78)


if __name__ == "__main__":
    main()
