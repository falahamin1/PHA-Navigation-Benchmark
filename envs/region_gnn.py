"""Step 3b: RegionGraphGNN -- message-passing GNN encoder mapping a Step 3a
RegionGraph (+ agent continuous state) to a fixed-dimensional embedding.

Not wired to any actor/critic head or training loop (Step 4). Hand-rolled
(dense adjacency + nn.Linear), no torch_geometric/DGL -- matches this repo's
existing GNN convention (Rush-hour-git/GraphNNRL.py's GCNLayer), and dense
(N,N) adjacency is cheap at this scale (N <= ~200 nodes for HARD).

FLAGGED DESIGN DECISION -- edge-type handling (`edge_type_mode`):

- "relational" (DEFAULT): separate learned weight matrices for intra-cell
  and inter-cell messages -- R-GCN style, 2 relations. Chosen because the
  paper's central claim is that inter-polytope edges do work a DeepSet
  structurally cannot; giving them their own weight matrix makes that claim
  architecturally explicit rather than hoping a single shared function
  happens to learn the intra/inter distinction from graph structure alone.
  The mini-ablation test (test_region_gnn.py) is the empirical check on
  whether this pays off -- see the printed ablation magnitudes.
- "shared": one weight matrix for all neighbor messages regardless of edge
  type (edge type isn't otherwise fed into the message). Simpler, fewer
  parameters, but risks the inter-cell signal being learned only weakly,
  diluted into the same channel as intra-cell structure.

Both paths live in RegionGNNLayer behind `edge_type_mode`, so switching is a
config change, not a rewrite. Cost of "relational": roughly 1.5x the
message-passing parameters of "shared" (two neighbor weight matrices instead
of one, per layer) -- see param_count() and the Step 3b gate diagnostic for
the actual number against the ~53k DeepSet budget (SPEC.md 2.6); this may
need hidden_dim trimmed for a fair Step 4 comparison, flagged there rather
than resolved here since param-matching is explicitly a Step 4 concern.

F_max note (carried forward from 2b/3a): this encoder does NOT pad to a
fixed per-cell facet count -- it consumes the region graph's actual node set
directly (that's precisely the structural advantage a graph-based encoder
has over a padded-tensor one). There is consequently no facet-count-bound
tensor anywhere in this module to size or assert against F_max.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_FEATURE_DIM = 6  # matches region_graph.FEATURE_NAMES
AGENT_STATE_DIM = 4   # (x1, x2, v1, v2)


def _dense_relation_adjacency(graph, num_nodes: int):
    """Two symmetric (N,N) 0/1 adjacency matrices, one per edge type."""
    adj_intra = torch.zeros(num_nodes, num_nodes)
    adj_inter = torch.zeros(num_nodes, num_nodes)
    for (i, j), et in zip(graph.edges, graph.edge_type):
        target = adj_intra if et == "intra" else adj_inter
        target[i, j] = 1.0
        target[j, i] = 1.0
    return adj_intra, adj_inter


def _symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
    """Standard GCN (Kipf & Welling) symmetric normalization D^-1/2 A D^-1/2,
    from this matrix's own degree. No self-loops folded in here -- the
    self-transformation is a separate learned term in RegionGNNLayer."""
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    return adj * deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)


class RegionGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_type_mode: str = "relational"):
        super().__init__()
        if edge_type_mode not in ("relational", "shared"):
            raise ValueError(f"edge_type_mode must be 'relational' or 'shared', got {edge_type_mode!r}")
        self.edge_type_mode = edge_type_mode
        self.w_self = nn.Linear(in_dim, out_dim, bias=True)
        if edge_type_mode == "relational":
            self.w_intra = nn.Linear(in_dim, out_dim, bias=False)
            self.w_inter = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.w_neighbor = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h, adj_intra_norm, adj_inter_norm, adj_combined_norm):
        out = self.w_self(h)
        if self.edge_type_mode == "relational":
            out = out + adj_intra_norm @ self.w_intra(h) + adj_inter_norm @ self.w_inter(h)
        else:
            out = out + adj_combined_norm @ self.w_neighbor(h)
        return F.relu(out)


class RegionGraphGNN(nn.Module):
    """2-layer (default) relational GCN over the Step 3a RegionGraph, mean-
    pooled (default) readout, agent (x,v) concatenated globally after
    pooling -- never enters message passing."""

    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 2,
        edge_type_mode: str = "relational",
        pool_type: str = "mean",
    ):
        super().__init__()
        if pool_type not in ("mean", "sum", "max"):
            raise ValueError(f"pool_type must be mean/sum/max, got {pool_type!r}")
        self.pool_type = pool_type
        self.edge_type_mode = edge_type_mode
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.layers = nn.ModuleList([
            RegionGNNLayer(hidden_dim, hidden_dim, edge_type_mode) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + AGENT_STATE_DIM, embedding_dim),
            nn.ReLU(),
        )

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "mean":
            return h.mean(dim=0)
        if self.pool_type == "sum":
            return h.sum(dim=0)
        return h.max(dim=0).values

    def encode_graph(self, graph, ablate_intra: bool = False, ablate_inter: bool = False) -> torch.Tensor:
        """Message passing + pooling only -- deliberately does NOT take
        agent_state, so it's structurally impossible for agent state to
        influence anything computed here (see test_agent_state_global_concat)."""
        num_nodes = graph.num_nodes
        adj_intra_raw, adj_inter_raw = _dense_relation_adjacency(graph, num_nodes)
        if ablate_intra:
            adj_intra_raw = torch.zeros_like(adj_intra_raw)
        if ablate_inter:
            adj_inter_raw = torch.zeros_like(adj_inter_raw)

        adj_intra_norm = _symmetric_normalize(adj_intra_raw)
        adj_inter_norm = _symmetric_normalize(adj_inter_raw)
        adj_combined_norm = _symmetric_normalize(adj_intra_raw + adj_inter_raw)

        h = self.input_proj(torch.as_tensor(graph.node_features, dtype=torch.float32))
        for layer in self.layers:
            h = layer(h, adj_intra_norm, adj_inter_norm, adj_combined_norm)
        return self._pool(h)

    def forward(self, graph, agent_state, ablate_intra: bool = False, ablate_inter: bool = False) -> torch.Tensor:
        pooled = self.encode_graph(graph, ablate_intra=ablate_intra, ablate_inter=ablate_inter)
        agent_state = torch.as_tensor(agent_state, dtype=torch.float32)
        combined = torch.cat([pooled, agent_state], dim=-1)
        return self.head(combined)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
