"""Step 4a gate diagnostic: run directly with `python gate4a_diagnostic.py`.

One place: F_max vs observed pool-wide max (truncation check), H/V/GNN-
matched param counts, both permutation-invariance residuals, the padding-
mask test result, and the two overfit-one-instance solve curves.
"""

import numpy as np
import torch

from deepset_encoders import F_MAX, HRepDeepSet, VRepDeepSet, _hrep_cell_tensor, _vrep_cell_tensor
from partitions import MixedConvexPartition
from pool import Instance, build_partition, generate_pool_with_stats
from ppo_smoke import train_overfit
from region_gnn import RegionGraphGNN
from test_region_graph import _ListPartition

torch.manual_seed(0)


def main():
    print("=" * 78)
    print("1) NO-TRUNCATION CHECK: F_MAX vs. observed pool-wide max facet/vertex count")
    print("=" * 78)
    pool_sizes = {"easy": 800, "medium": 700, "hard": 700}
    overall_max = 0
    for tier in ("easy", "medium", "hard"):
        pool, _stats = generate_pool_with_stats(tier, pool_sizes[tier], rng_seed=42)
        seeds = {inst.partition_seed for inst in pool}
        tier_max = max(
            build_partition(tier, seed).cell_facet_count(i)
            for seed in seeds
            for i in range(build_partition(tier, seed).num_cells)
        )
        print(f"  {tier:6s}: {len(seeds):3d} distinct partition seeds, max facet count = {tier_max}")
        overall_max = max(overall_max, tier_max)
    print(f"  OBSERVED POOL-WIDE MAX = {overall_max}   F_MAX USED = {F_MAX}   "
          f"{'OK (F_MAX >= observed max)' if F_MAX >= overall_max else 'FAIL -- TRUNCATION RISK'}")

    print("=" * 78)
    print("2) PARAMETER COUNTS (H-Rep, V-Rep, GNN param-matched)")
    print("=" * 78)
    h, v = HRepDeepSet(), VRepDeepSet()
    gnn_matched = RegionGraphGNN(hidden_dim=80, edge_type_mode="relational")
    print(f"  H-Rep DeepSet             = {h.param_count():>7,}  ({h.param_count()/53000:.2f}x ~53k budget)")
    print(f"  V-Rep DeepSet             = {v.param_count():>7,}  ({v.param_count()/53000:.2f}x ~53k budget)")
    print(f"  GNN (hidden_dim=80, rel.) = {gnn_matched.param_count():>7,}  "
          f"({gnn_matched.param_count()/53000:.2f}x ~53k budget)")

    print("=" * 78)
    print("3) PERMUTATION INVARIANCE (within-cell, across-cell)")
    print("=" * 78)
    rng = np.random.default_rng(42)
    for name, Encoder, elem_dim in [("H-Rep", HRepDeepSet, 3), ("V-Rep", VRepDeepSet, 2)]:
        model = Encoder()
        model.eval()
        k = 5
        tensor = torch.zeros(F_MAX, elem_dim)
        tensor[:k] = torch.tensor(rng.uniform(-2, 2, size=(k, elem_dim)), dtype=torch.float32)
        mask = torch.zeros(F_MAX, dtype=torch.bool)
        mask[:k] = True
        perm = torch.tensor(rng.permutation(F_MAX))
        with torch.no_grad():
            pooled = model._masked_pool(model.phi(tensor), mask)
            pooled_perm = model._masked_pool(model.phi(tensor[perm]), mask[perm])
        residual_within = (pooled - pooled_perm).abs().max().item()
        print(f"  {name}: within-cell residual  = {residual_within:.3e}")

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
    for name, Encoder in [("H-Rep", HRepDeepSet), ("V-Rep", VRepDeepSet)]:
        model = Encoder()
        model.eval()
        with torch.no_grad():
            emb = model(partition, instance, agent_state)
            emb_perm = model(permuted_partition, permuted_instance, agent_state)
        residual_across = (emb - emb_perm).abs().max().item()
        print(f"  {name}: across-cell residual  = {residual_across:.3e}")

    print("=" * 78)
    print("4) PADDING-MASK CORRECTNESS (same cell, f_max=6 vs f_max=12)")
    print("=" * 78)
    medium = MixedConvexPartition(grid_seed=0)
    cell_idx = next(i for i in range(medium.num_cells) if medium.cell_facet_count(i) == 4)
    for name, Encoder, tensor_fn in [("H-Rep", HRepDeepSet, _hrep_cell_tensor), ("V-Rep", VRepDeepSet, _vrep_cell_tensor)]:
        model = Encoder()
        model.eval()
        t6, m6 = tensor_fn(medium, cell_idx, f_max=6)
        t12, m12 = tensor_fn(medium, cell_idx, f_max=F_MAX)
        with torch.no_grad():
            p6 = model._masked_pool(model.phi(t6), m6)
            p12 = model._masked_pool(model.phi(t12), m12)
        residual = (p6 - p12).abs().max().item()
        status = "OK (masking correct)" if residual < 1e-6 else "FAIL -- padding leaking into sum"
        print(f"  {name}: f_max=6 vs f_max=12 residual = {residual:.3e}   {status}")

    print("=" * 78)
    print("5) OVERFIT-ONE-INSTANCE SOLVE CURVES")
    print("=" * 78)
    for name, Encoder in [("H-Rep", HRepDeepSet), ("V-Rep", VRepDeepSet)]:
        _model, history = train_overfit(Encoder(), n_iterations=40, steps_per_iter=128, ppo_epochs=3, verbose=False)
        curve = " -> ".join(f"{r:.2f}" for r in history)
        print(f"  {name} ({len(history)} iterations to reach target): {curve}")
    print("=" * 78)


if __name__ == "__main__":
    main()
