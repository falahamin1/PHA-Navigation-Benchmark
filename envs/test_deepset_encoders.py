"""Step 4a tests: run directly with `python test_deepset_encoders.py`.

- test_shape: MEDIUM/HARD instances, both encoders -> embedding of dim d.
- test_permutation_invariance_within_cell: permuting a cell's (tensor, mask)
  pair -- the operational definition of "permuting the set of constraints/
  vertices" -- must not change phi+masked-pool output.
- test_permutation_invariance_across_cell: reordering the list of cells
  (goal/hazard indices remapped consistently) must not change the region
  embedding.
- test_no_truncation: the F_max carry-forward -- scans real generated pools
  (all tiers) and asserts every cell's facet/vertex count <= F_MAX.
- test_padding_mask_correctness: the same real cell padded to two different
  lengths must give the identical per-cell embedding.
- test_agent_state_global_concat: agent state can't reach encode_region (no
  parameter for it), but does change forward()'s output.
- test_overfit_one_instance: both encoders must reach ~100% solve rate
  memorizing one clean, easy-to-execute EASY instance.
- test_param_count: H, V, and GNN-matched param counts side by side.
- test_determinism: same input + weights -> identical embedding.
"""

import numpy as np
import torch

from deepset_encoders import F_MAX, HRepDeepSet, VRepDeepSet, _hrep_cell_tensor, _vrep_cell_tensor
from partitions import IrregularConvexPartition, MixedConvexPartition
from pool import Instance, build_partition, generate_pool_with_stats
from region_gnn import RegionGraphGNN
from test_region_graph import _ListPartition

torch.manual_seed(0)


def _instance(goal_cell=24, hazard_cells=(8,)):
    return Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=goal_cell,
                     hazard_cells=tuple(sorted(hazard_cells)), initial_velocity_sign=(1, 1))


def test_shape():
    agent_state = torch.tensor([2.5, 2.5, 0.1, -0.1])
    medium = MixedConvexPartition(grid_seed=0)
    hard = IrregularConvexPartition(grid_seed=0)
    inst_medium = _instance(goal_cell=24, hazard_cells=(8,))
    inst_hard = _instance(goal_cell=5, hazard_cells=(1, 2))

    for name, Encoder in [("H-Rep", HRepDeepSet), ("V-Rep", VRepDeepSet)]:
        model = Encoder()
        model.eval()
        with torch.no_grad():
            emb_medium = model(medium, inst_medium, agent_state)
            emb_hard = model(hard, inst_hard, agent_state)
        assert emb_medium.shape == (128,), f"{name} MEDIUM shape {emb_medium.shape} != (128,)"
        assert emb_hard.shape == (128,), f"{name} HARD shape {emb_hard.shape} != (128,)"
        print(f"  {name}: MEDIUM ({medium.num_cells} cells) -> (128,); HARD ({hard.num_cells} cells) -> (128,)")
    print("test_shape: PASS")


def test_permutation_invariance_within_cell():
    """Permuting a cell's (tensor, mask) pair together IS the operational
    definition of permuting the unordered set of constraints/vertices -- an
    arbitrary permutation of the underlying polygon's vertex LIST would
    change which edges get computed (invalid), so this is tested at the
    (tensor, mask) level, which is exactly the set the DeepSet pools over.
    """
    rng = np.random.default_rng(42)
    for name, Encoder, elem_dim in [("H-Rep", HRepDeepSet, 3), ("V-Rep", VRepDeepSet, 2)]:
        model = Encoder()
        model.eval()

        k = 5  # real elements
        tensor = torch.zeros(F_MAX, elem_dim)
        tensor[:k] = torch.tensor(rng.uniform(-2, 2, size=(k, elem_dim)), dtype=torch.float32)
        mask = torch.zeros(F_MAX, dtype=torch.bool)
        mask[:k] = True

        perm = torch.tensor(rng.permutation(F_MAX))
        tensor_perm = tensor[perm]
        mask_perm = mask[perm]

        with torch.no_grad():
            pooled = model._masked_pool(model.phi(tensor), mask)
            pooled_perm = model._masked_pool(model.phi(tensor_perm), mask_perm)

        residual = (pooled - pooled_perm).abs().max().item()
        assert residual < 1e-5, f"{name}: within-cell permutation changed pooled output, residual={residual}"
        print(f"  {name}: within-cell permutation residual = {residual:.3e}")
    print("test_permutation_invariance_within_cell: PASS")


def test_permutation_invariance_across_cell():
    squares = [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],
        [[4.0, 0.0], [5.0, 0.0], [5.0, 1.0], [4.0, 1.0]],
        [[6.0, 0.0], [7.0, 0.0], [7.0, 1.0], [6.0, 1.0]],
    ]
    partition = _ListPartition(squares)
    instance = Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=1, hazard_cells=(2,),
                         initial_velocity_sign=(1, 1))
    agent_state = torch.tensor([0.5, 0.5, 0.0, 0.0])

    perm = [2, 0, 3, 1]  # old_idx -> new position
    permuted_squares = [None] * len(squares)
    inverse = [0] * len(squares)
    for old_idx, new_idx in enumerate(perm):
        permuted_squares[new_idx] = squares[old_idx]
        inverse[old_idx] = new_idx
    permuted_partition = _ListPartition(permuted_squares)
    permuted_instance = Instance(
        tier="test", partition_seed=0, start_cell=inverse[0], goal_cell=inverse[1],
        hazard_cells=(inverse[2],), initial_velocity_sign=(1, 1),
    )

    for name, Encoder in [("H-Rep", HRepDeepSet), ("V-Rep", VRepDeepSet)]:
        model = Encoder()
        model.eval()
        with torch.no_grad():
            emb = model(partition, instance, agent_state)
            emb_permuted = model(permuted_partition, permuted_instance, agent_state)
        residual = (emb - emb_permuted).abs().max().item()
        assert residual < 1e-5, f"{name}: across-cell permutation changed embedding, residual={residual}"
        print(f"  {name}: across-cell permutation residual = {residual:.3e}")
    print("test_permutation_invariance_across_cell: PASS")


def test_no_truncation():
    pool_sizes = {"easy": 800, "medium": 700, "hard": 700}
    overall_max = 0
    for tier in ("easy", "medium", "hard"):
        pool, _stats = generate_pool_with_stats(tier, pool_sizes[tier], rng_seed=42)
        seeds = {inst.partition_seed for inst in pool}
        tier_max = 0
        for seed in seeds:
            partition = build_partition(tier, seed)
            for i in range(partition.num_cells):
                tier_max = max(tier_max, partition.cell_facet_count(i))
        print(f"  {tier}: {len(seeds)} distinct partition seeds, max facet count = {tier_max}")
        overall_max = max(overall_max, tier_max)
        assert tier_max <= F_MAX, f"{tier}: observed max facet count {tier_max} > F_MAX={F_MAX} -- would truncate"
    print(f"test_no_truncation: PASS (pool-wide observed max = {overall_max}, F_MAX = {F_MAX})")


def test_padding_mask_correctness():
    partition = MixedConvexPartition(grid_seed=0)
    # Find a cell with a moderate facet count so both a "tight" and a "loose"
    # padding size are >= its real facet count.
    cell_idx = next(i for i in range(partition.num_cells) if partition.cell_facet_count(i) == 4)

    for name, Encoder, tensor_fn in [
        ("H-Rep", HRepDeepSet, _hrep_cell_tensor),
        ("V-Rep", VRepDeepSet, _vrep_cell_tensor),
    ]:
        model = Encoder()
        model.eval()
        tensor_tight, mask_tight = tensor_fn(partition, cell_idx, f_max=6)
        tensor_loose, mask_loose = tensor_fn(partition, cell_idx, f_max=F_MAX)

        with torch.no_grad():
            pooled_tight = model._masked_pool(model.phi(tensor_tight), mask_tight)
            pooled_loose = model._masked_pool(model.phi(tensor_loose), mask_loose)

        residual = (pooled_tight - pooled_loose).abs().max().item()
        assert residual < 1e-6, (
            f"{name}: padding count changed the per-cell embedding (residual={residual}) -- "
            "padded slots are leaking into the sum (phi's bias term is likely not being masked out)"
        )
        print(f"  {name}: f_max=6 vs f_max={F_MAX} (same {int(mask_tight.sum())} real elements) "
              f"-> residual = {residual:.3e}")
    print("test_padding_mask_correctness: PASS")


def test_agent_state_global_concat():
    partition = MixedConvexPartition(grid_seed=1)
    instance = _instance()
    for name, Encoder in [("H-Rep", HRepDeepSet), ("V-Rep", VRepDeepSet)]:
        model = Encoder()
        model.eval()
        with torch.no_grad():
            pooled_a = model.encode_region(partition, instance)
            pooled_b = model.encode_region(partition, instance)
            assert torch.equal(pooled_a, pooled_b), f"{name}: encode_region should be exactly deterministic"

            agent1 = torch.tensor([0.5, 0.5, 0.0, 0.0])
            agent2 = torch.tensor([4.5, 4.5, -0.2, 0.3])
            emb1 = model(partition, instance, agent1)
            emb2 = model(partition, instance, agent2)
        assert not torch.allclose(emb1, emb2), f"{name}: changing agent state should change the embedding"
        print(f"  {name}: encode_region() takes no agent_state parameter; forward() changed by "
              f"{(emb1 - emb2).norm().item():.4f} when agent state changed")
    print("test_agent_state_global_concat: PASS")


def test_determinism():
    partition = IrregularConvexPartition(grid_seed=2)
    instance = _instance(goal_cell=3, hazard_cells=(1,))
    agent_state = torch.tensor([1.0, 1.0, 0.1, 0.1])
    for name, Encoder in [("H-Rep", HRepDeepSet), ("V-Rep", VRepDeepSet)]:
        model = Encoder()
        model.eval()
        with torch.no_grad():
            emb1 = model(partition, instance, agent_state)
            emb2 = model(partition, instance, agent_state)
        assert torch.equal(emb1, emb2), f"{name}: identical input+weights should give a bit-identical embedding"
    print("test_determinism: PASS (both encoders bit-identical across repeated calls)")


def test_param_count():
    h = HRepDeepSet()
    v = VRepDeepSet()
    gnn_matched = RegionGraphGNN(hidden_dim=80, edge_type_mode="relational")
    print(f"  H-Rep DeepSet params      = {h.param_count():>7,}")
    print(f"  V-Rep DeepSet params      = {v.param_count():>7,}")
    print(f"  GNN (hidden_dim=80, rel.) = {gnn_matched.param_count():>7,}")
    print("test_param_count: PASS (printed for review against the ~53k DeepSet budget)")


def test_overfit_one_instance():
    from ppo_smoke import train_overfit
    for name, Encoder in [("H-Rep", HRepDeepSet), ("V-Rep", VRepDeepSet)]:
        print(f"  training {name} on a single fixed EASY instance...")
        _model, history = train_overfit(Encoder(), n_iterations=40, steps_per_iter=128, ppo_epochs=3, verbose=False)
        final_rate = history[-1] if history else 0.0
        assert final_rate >= 0.95, f"{name}: failed to reach 95% solve rate on a single instance (final={final_rate:.0%}) -- wiring bug"
        print(f"  {name}: solve-rate curve = {[f'{r:.2f}' for r in history]}")
        print(f"  {name}: reached {final_rate:.0%} in {len(history)} iterations")
    print("test_overfit_one_instance: PASS")


if __name__ == "__main__":
    test_shape()
    test_permutation_invariance_within_cell()
    test_permutation_invariance_across_cell()
    test_no_truncation()
    test_padding_mask_correctness()
    test_agent_state_global_concat()
    test_determinism()
    test_param_count()
    test_overfit_one_instance()
    print("Step 4a tests: ALL PASS")
