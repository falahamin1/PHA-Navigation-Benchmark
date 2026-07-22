"""Step 3b tests: run directly with `python test_region_gnn.py`.

- test_shape: fixed embedding dim regardless of graph size (MEDIUM ~120
  nodes, HARD ~160 nodes).
- test_permutation_invariance: relabeling nodes (+ remapping edges
  consistently) must not change the embedding.
- test_inter_cell_ablation_matters / test_intra_cell_ablation_matters: the
  mini-ablations -- removing either edge type must measurably change the
  embedding. Note these run on an UNTRAINED (randomly initialized) model, so
  they show architectural sensitivity to each edge type, not learned
  reliance -- that's an empirical Step 4 training question, not this step's.
- test_agent_state_global_concat: encode_graph() has no agent_state
  parameter at all (structural guarantee, not just empirical), so agent
  state cannot influence message passing; forward() output still must
  change when agent state changes.
- test_determinism: same graph + agent state + weights -> identical output.
- test_gradient_flow: dummy backward pass gives non-None, non-zero grads on
  every message-passing weight (both w_intra and w_inter for the default
  relational mode) -- catches a dead subnetwork.
"""

import numpy as np
import torch

from partitions import IrregularConvexPartition, MixedConvexPartition
from pool import Instance
from region_graph import RegionGraph, build_region_graph
from region_gnn import RegionGraphGNN

torch.manual_seed(0)


def _instance(goal_cell=24, hazard_cells=(8,)):
    return Instance(tier="test", partition_seed=0, start_cell=0, goal_cell=goal_cell,
                     hazard_cells=tuple(sorted(hazard_cells)), initial_velocity_sign=(1, 1))


def _permute_region_graph(graph: RegionGraph, perm: np.ndarray) -> RegionGraph:
    """perm[old_idx] = new_idx. Relabels nodes and remaps edges consistently."""
    n = graph.num_nodes
    new_node_cell = np.empty(n, dtype=int)
    new_node_facet = np.empty(n, dtype=int)
    new_node_features = np.empty_like(graph.node_features)
    new_node_endpoints = np.empty_like(graph.node_endpoints)
    for old in range(n):
        new = perm[old]
        new_node_cell[new] = graph.node_cell[old]
        new_node_facet[new] = graph.node_facet[old]
        new_node_features[new] = graph.node_features[old]
        new_node_endpoints[new] = graph.node_endpoints[old]

    new_edges, new_edge_type = [], []
    for (i, j), et in zip(graph.edges, graph.edge_type):
        ni, nj = int(perm[i]), int(perm[j])
        new_edges.append((min(ni, nj), max(ni, nj)))
        new_edge_type.append(et)
    order = sorted(range(len(new_edges)), key=lambda k: new_edges[k])
    new_edges = np.array([new_edges[k] for k in order], dtype=int) if new_edges else np.zeros((0, 2), dtype=int)
    new_edge_type = [new_edge_type[k] for k in order]

    return RegionGraph(
        node_cell=new_node_cell, node_facet=new_node_facet, node_features=new_node_features,
        node_endpoints=new_node_endpoints, edges=new_edges, edge_type=new_edge_type,
    )


def test_shape():
    model = RegionGraphGNN(embedding_dim=128)
    model.eval()
    agent_state = torch.tensor([2.5, 2.5, 0.1, -0.1])

    medium_graph = build_region_graph(MixedConvexPartition(grid_seed=0), _instance())
    hard_graph = build_region_graph(IrregularConvexPartition(grid_seed=0), _instance(goal_cell=5, hazard_cells=(1, 2)))

    emb_medium = model(medium_graph, agent_state)
    emb_hard = model(hard_graph, agent_state)

    assert emb_medium.shape == (128,), f"MEDIUM embedding shape {emb_medium.shape} != (128,)"
    assert emb_hard.shape == (128,), f"HARD embedding shape {emb_hard.shape} != (128,)"
    print(f"test_shape: PASS (MEDIUM {medium_graph.num_nodes} nodes -> (128,); "
          f"HARD {hard_graph.num_nodes} nodes -> (128,))")


def test_permutation_invariance():
    model = RegionGraphGNN()
    model.eval()
    graph = build_region_graph(MixedConvexPartition(grid_seed=1), _instance())
    agent_state = torch.tensor([1.5, 3.0, 0.0, 0.2])

    rng = np.random.default_rng(7)
    perm = rng.permutation(graph.num_nodes)
    permuted_graph = _permute_region_graph(graph, perm)

    with torch.no_grad():
        emb_original = model(graph, agent_state)
        emb_permuted = model(permuted_graph, agent_state)

    residual = (emb_original - emb_permuted).abs().max().item()
    assert residual < 1e-4, f"permutation changed the embedding: max abs residual {residual}"
    print(f"test_permutation_invariance: PASS (max abs residual = {residual:.3e})")
    return residual


def test_inter_cell_ablation_matters():
    model = RegionGraphGNN()
    model.eval()
    graph = build_region_graph(MixedConvexPartition(grid_seed=2), _instance())
    agent_state = torch.tensor([2.0, 2.0, 0.0, 0.0])

    with torch.no_grad():
        emb_full = model(graph, agent_state)
        emb_no_inter = model(graph, agent_state, ablate_inter=True)

    diff = (emb_full - emb_no_inter).norm().item()
    rel = diff / (emb_full.norm().item() + 1e-12)
    assert diff > 1e-3, f"removing inter-cell edges barely changed the embedding (diff={diff}) -- architecturally inert"
    print(f"test_inter_cell_ablation_matters: PASS (L2 change = {diff:.4f}, relative = {rel:.2%})")
    return diff


def test_intra_cell_ablation_matters():
    model = RegionGraphGNN()
    model.eval()
    graph = build_region_graph(MixedConvexPartition(grid_seed=2), _instance())
    agent_state = torch.tensor([2.0, 2.0, 0.0, 0.0])

    with torch.no_grad():
        emb_full = model(graph, agent_state)
        emb_no_intra = model(graph, agent_state, ablate_intra=True)

    diff = (emb_full - emb_no_intra).norm().item()
    rel = diff / (emb_full.norm().item() + 1e-12)
    assert diff > 1e-3, f"removing intra-cell edges barely changed the embedding (diff={diff})"
    print(f"test_intra_cell_ablation_matters: PASS (L2 change = {diff:.4f}, relative = {rel:.2%})")
    return diff


def test_agent_state_global_concat():
    model = RegionGraphGNN()
    model.eval()
    graph = build_region_graph(MixedConvexPartition(grid_seed=3), _instance())

    with torch.no_grad():
        pooled_a = model.encode_graph(graph)
        pooled_b = model.encode_graph(graph)  # encode_graph has no agent_state param at all
        assert torch.equal(pooled_a, pooled_b), "encode_graph should be exactly deterministic (no agent_state input)"

        agent1 = torch.tensor([0.5, 0.5, 0.0, 0.0])
        agent2 = torch.tensor([4.5, 4.5, -0.2, 0.3])
        emb1 = model(graph, agent1)
        emb2 = model(graph, agent2)

    assert not torch.allclose(emb1, emb2), "changing agent state should change the final embedding"
    print(f"test_agent_state_global_concat: PASS (encode_graph() takes no agent_state parameter -- "
          f"structurally cannot leak into message passing; forward() output changed by "
          f"{(emb1 - emb2).norm().item():.4f} when agent state changed)")


def test_determinism():
    model = RegionGraphGNN()
    model.eval()
    graph = build_region_graph(IrregularConvexPartition(grid_seed=4), _instance(goal_cell=3, hazard_cells=(1,)))
    agent_state = torch.tensor([1.0, 1.0, 0.1, 0.1])

    with torch.no_grad():
        emb1 = model(graph, agent_state)
        emb2 = model(graph, agent_state)
    assert torch.equal(emb1, emb2), "identical graph+agent_state+weights should give a bit-identical embedding"
    print("test_determinism: PASS (bit-identical across repeated calls)")


def test_gradient_flow():
    model = RegionGraphGNN(edge_type_mode="relational")
    graph = build_region_graph(MixedConvexPartition(grid_seed=5), _instance())
    agent_state = torch.tensor([2.0, 2.0, 0.0, 0.0])

    embedding = model(graph, agent_state)
    embedding.sum().backward()

    checked = []
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name}: no gradient at all (disconnected from the loss)"
        grad_norm = param.grad.abs().sum().item()
        assert grad_norm > 1e-10, f"{name}: gradient is all-zero (dead subnetwork)"
        checked.append((name, grad_norm))

    for layer_idx, layer in enumerate(model.layers):
        assert layer.w_intra.weight.grad.abs().sum().item() > 1e-10, f"layer {layer_idx}: w_intra has dead gradient"
        assert layer.w_inter.weight.grad.abs().sum().item() > 1e-10, f"layer {layer_idx}: w_inter has dead gradient"

    print(f"test_gradient_flow: PASS ({len(checked)} parameter tensors all have non-None, non-zero gradients, "
          f"including w_intra/w_inter on all {len(model.layers)} layers)")


if __name__ == "__main__":
    test_shape()
    residual = test_permutation_invariance()
    inter_diff = test_inter_cell_ablation_matters()
    intra_diff = test_intra_cell_ablation_matters()
    test_agent_state_global_concat()
    test_determinism()
    test_gradient_flow()

    print("\n--- ablation magnitudes side by side ---")
    print(f"inter-cell ablation change: {inter_diff:.4f}")
    print(f"intra-cell ablation change: {intra_diff:.4f}")
    print(f"ratio inter/intra: {inter_diff / intra_diff:.3f}")
    print("Step 3b tests: ALL PASS")
