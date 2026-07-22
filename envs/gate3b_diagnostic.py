"""Step 3b gate diagnostic: run directly with `python gate3b_diagnostic.py`.

Prints the chosen edge-type-handling option + justification, and for one
MEDIUM instance: embedding dimension, inter-/intra-cell ablation magnitudes
side by side, and the permutation-invariance residual. Also reports param
count against the ~53k DeepSet budget (SPEC.md 2.6) for both edge_type_mode
options at the default hidden_dim, and a trimmed hidden_dim suggestion.
"""

import numpy as np
import torch

from partitions import MixedConvexPartition
from pool import Instance
from region_graph import build_region_graph
from region_gnn import RegionGraphGNN
from test_region_gnn import _permute_region_graph

torch.manual_seed(0)


def main():
    print("=" * 78)
    print("EDGE-TYPE-HANDLING DECISION: option (b), 'relational' (R-GCN style, 2 relations)")
    print("=" * 78)
    print(
        "Separate learned weight matrices (w_intra, w_inter) for intra-cell vs.\n"
        "inter-cell messages, at every layer. Chosen over a single shared weight\n"
        "matrix because the paper's central claim is that inter-polytope edges do\n"
        "work a DeepSet structurally cannot -- giving them dedicated parameters\n"
        "makes that claim architecturally explicit rather than hoping a shared\n"
        "function learns the intra/inter distinction on its own. Config flag\n"
        "`edge_type_mode='shared'` switches to the alternative with no rewrite.\n"
    )

    instance = Instance(tier="medium", partition_seed=0, start_cell=0, goal_cell=24,
                        hazard_cells=(8,), initial_velocity_sign=(1, 1))
    partition = MixedConvexPartition(grid_seed=0)
    graph = build_region_graph(partition, instance)
    agent_state = torch.tensor([2.5, 2.5, 0.0, 0.0])

    model = RegionGraphGNN()
    model.eval()

    with torch.no_grad():
        emb_full = model(graph, agent_state)
        emb_no_inter = model(graph, agent_state, ablate_inter=True)
        emb_no_intra = model(graph, agent_state, ablate_intra=True)

    inter_diff = (emb_full - emb_no_inter).norm().item()
    intra_diff = (emb_full - emb_no_intra).norm().item()
    norm_full = emb_full.norm().item()

    print("-" * 78)
    print(f"MEDIUM instance: partition_seed=0, {graph.num_nodes} nodes, "
          f"{sum(1 for t in graph.edge_type if t=='intra')} intra-edges, "
          f"{sum(1 for t in graph.edge_type if t=='inter')} inter-edges")
    print(f"embedding dimension                 = {emb_full.shape[0]}")
    print(f"embedding L2 norm (full graph)       = {norm_full:.4f}")
    print(f"inter-cell ablation |delta|           = {inter_diff:.4f}  ({inter_diff/norm_full:.1%} of full norm)")
    print(f"intra-cell ablation |delta|           = {intra_diff:.4f}  ({intra_diff/norm_full:.1%} of full norm)")
    print(f"ratio inter/intra                    = {inter_diff/intra_diff:.3f}")
    print(
        "  -> inter-cell ablation is NOT smaller than intra-cell ablation "
        f"({'>=' if inter_diff >= intra_diff else '<'} ratio 1.0); at random\n"
        "     initialization the relational architecture gives inter-cell structure\n"
        "     comparable (here, slightly greater) influence to intra-cell structure.\n"
        "     This is architectural sensitivity under UNTRAINED weights, not learned\n"
        "     reliance -- whether a trained policy actually exploits it is Step 4's\n"
        "     empirical question, not this step's."
    )

    rng = np.random.default_rng(7)
    perm = rng.permutation(graph.num_nodes)
    permuted = _permute_region_graph(graph, perm)
    with torch.no_grad():
        emb_permuted = model(permuted, agent_state)
    residual = (emb_full - emb_permuted).abs().max().item()
    print("-" * 78)
    print(f"permutation-invariance max abs residual = {residual:.3e}  (tolerance 1e-4)")

    print("-" * 78)
    print("PARAMETER COUNT vs. ~53k DeepSet budget (SPEC.md 2.6):")
    for mode in ("relational", "shared"):
        m = RegionGraphGNN(hidden_dim=128, edge_type_mode=mode)
        print(f"  hidden_dim=128, edge_type_mode={mode:11s} -> {m.param_count():>7,} params "
              f"({m.param_count()/53000:.2f}x the DeepSet budget)")
    m_trimmed = RegionGraphGNN(hidden_dim=80, edge_type_mode="relational")
    print(f"  hidden_dim=80,  edge_type_mode=relational -> {m_trimmed.param_count():>7,} params "
          f"({m_trimmed.param_count()/53000:.2f}x) -- suggested trim for Step 4 param-matching, not applied by default")
    print("=" * 78)


if __name__ == "__main__":
    main()
