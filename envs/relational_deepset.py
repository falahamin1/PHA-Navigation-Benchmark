"""Step 4c: RelationalDeepSet -- the honest control, not another baseline.

The GNN's thesis is that inter-polytope message passing does work a set
encoder cannot. The obvious reviewer objection is "a DeepSet with a few
hand-designed adjacency features would do just as well" -- this encoder IS
that objection, built and run. If the GNN beats it, message passing is doing
something feature-augmentation can't; if it only ties, the honest finding is
"explicit relational features suffice."  Either result is publishable, but
they're different claims, so this control has to be genuinely strong and
genuinely NOT a disguised GNN.

Architecture: the Step 4a H-Rep DeepSet (per-cell constraint set -> phi ->
masked sum-pool -> per-cell embedding), with each cell's descriptor augmented
by a small fixed-size RELATIONAL feature vector before the region-level pool.

FLAGGED CENTRAL DESIGN DECISION -- what makes this genuine augmentation and
not a smuggled graph:

Chosen features per cell (NUM_RELATIONAL_FEATURES = 5), computed from the
Step 3a `build_region_graph`'s STRICT facet-sharing inter-cell edges (the
exact same predicate the GNN's message passing uses -- "relational" means
the same thing in both encoders, so the comparison is apples-to-apples):

  1. degree                -- number of distinct facet-adjacent neighbors
  2. hazard_neighbor_count -- how many of those neighbors are hazard cells
  3. goal_neighbor_count   -- how many of those neighbors are the goal cell
  4. shared_facet_count    -- total number of facet-sharing constraint-pairs
                               touching this cell (can exceed degree at a
                               T-junction, where one cell's edge is split
                               against a neighbor's several sub-edges)
  5. shared_facet_length   -- summed length of all shared boundary segments
                               with neighbors (aggregate geometric contact)

Why this is fair augmentation, not a disguised graph:
- Every feature is a SUM or COUNT over the neighbor set -- permutation-
  invariant and LOSSY. A cell knows "2 of my neighbors are hazards" but not
  WHICH TWO, and cannot recover per-neighbor identity from this vector.
- The feature dimension is a FIXED CONSTANT (5), independent of degree,
  partition size, or tier -- unlike a GNN's per-node hidden state (which
  scales with hidden_dim and is refreshed per-message from a SPECIFIC
  neighbor's own representation), or a full adjacency list (which would
  scale with degree).
- Strictly 1-hop: computed only from a cell's OWN inter-cell edges. A
  neighbor's neighbor never enters the computation at all -- there is no
  mechanism here for a 2-hop change to reach this cell's features (see
  test_relational_deepset.py's test_no_multihop_propagation, which verifies
  this empirically with a concrete 3-cell chain, not just by inspection).
- No per-neighbor embedding, no edge list, no iterative update -- one static
  vector, computed once from geometry+instance, never refined by message
  passing. There is nothing here a GNN layer could "unfold" into multi-hop
  reasoning; the aggregation already discarded per-neighbor identity.

What would have crossed the line (NOT done here): passing each neighbor's
raw embedding (preserves identity, enables the cell_mlp to learn to
distinguish neighbors -- that's message passing), any full edge list, or
computing this feature more than once per forward pass in a way that lets
information compound across hops.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from deepset_encoders import F_MAX, _hrep_cell_tensor
from geometry import segment_overlap_length
from region_graph import build_region_graph

AGENT_STATE_DIM = 4
NUM_CELL_FLAGS = 3
NUM_RELATIONAL_FEATURES = 5
RELATIONAL_FEATURE_NAMES = (
    "degree", "hazard_neighbor_count", "goal_neighbor_count",
    "shared_facet_count", "shared_facet_length",
)


def compute_cell_relational_features(partition, instance) -> torch.Tensor:
    """(num_cells, NUM_RELATIONAL_FEATURES) fixed-size, 1-hop, identity-blind
    summary of each cell's local adjacency. See module docstring."""
    graph = build_region_graph(partition, instance)
    num_cells = partition.num_cells
    hazard_set = set(instance.hazard_cells)
    goal_cell = instance.goal_cell

    neighbor_sets = [set() for _ in range(num_cells)]
    shared_facet_count = [0] * num_cells
    shared_facet_length = [0.0] * num_cells

    for (i, j), et in zip(graph.edges, graph.edge_type):
        if et != "inter":
            continue
        ci, cj = int(graph.node_cell[i]), int(graph.node_cell[j])
        neighbor_sets[ci].add(cj)
        neighbor_sets[cj].add(ci)
        a1, a2 = graph.node_endpoints[i]
        b1, b2 = graph.node_endpoints[j]
        length = float(segment_overlap_length(a1, a2, b1, b2))
        shared_facet_count[ci] += 1
        shared_facet_count[cj] += 1
        shared_facet_length[ci] += length
        shared_facet_length[cj] += length

    features = torch.zeros(num_cells, NUM_RELATIONAL_FEATURES)
    for c in range(num_cells):
        neighbors = neighbor_sets[c]
        degree = len(neighbors)
        hazard_neighbors = sum(1 for n in neighbors if n in hazard_set)
        goal_neighbors = sum(1 for n in neighbors if n == goal_cell)
        features[c, 0] = float(degree)
        features[c, 1] = float(hazard_neighbors)
        features[c, 2] = float(goal_neighbors)
        features[c, 3] = float(shared_facet_count[c])
        features[c, 4] = shared_facet_length[c]
    return features


class RelationalDeepSet(nn.Module):
    """Step 4a's H-Rep DeepSet + per-cell relational feature concat before
    the region pool. Same interface as every other Step 3/4 encoder."""

    def __init__(self, embedding_dim: int = 128, phi_hidden: int = 128, cell_hidden: int = 128,
                 pool_type: str = "sum", f_max: int = F_MAX):
        super().__init__()
        if pool_type not in ("sum", "mean", "max"):
            raise ValueError(f"pool_type must be sum/mean/max, got {pool_type!r}")
        self.pool_type = pool_type
        self.f_max = f_max

        self.phi = nn.Sequential(
            nn.Linear(3, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden), nn.ReLU(),
        )
        cell_input_dim = phi_hidden + NUM_CELL_FLAGS + NUM_RELATIONAL_FEATURES
        self.cell_mlp = nn.Sequential(nn.Linear(cell_input_dim, cell_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(cell_hidden + AGENT_STATE_DIM, embedding_dim), nn.ReLU())

        # compute_cell_relational_features(partition, instance) is fixed for
        # the lifetime of an instance (it depends on partition geometry and
        # instance.hazard_cells/goal_cell only -- never on current_cell or
        # continuous state), so rebuilding it from scratch on every forward
        # call is pure waste. Same cache-key shape as fairness_harness.py's
        # _RegionGraphGNNAdapter._cache_key -- content-based, not id()-based,
        # for the same reason pool.py's strict-adjacency cache had to move
        # off id() (see that module's docstring).
        self._relational_feature_cache: Dict[tuple, torch.Tensor] = {}

    def _cache_key(self, partition, instance):
        return (type(partition).__name__, getattr(partition, "grid_seed", None), partition.num_cells, instance)

    def _masked_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        if self.pool_type == "sum":
            return (x * mask_f).sum(dim=0)
        if self.pool_type == "mean":
            count = mask.to(x.dtype).sum().clamp(min=1.0)
            return (x * mask_f).sum(dim=0) / count
        neg_inf = torch.finfo(x.dtype).min
        x_masked = torch.where(mask.unsqueeze(-1), x, torch.full_like(x, neg_inf))
        return x_masked.max(dim=0).values

    def encode_region(self, partition, instance, current_cell: Optional[int] = None) -> torch.Tensor:
        hazard_set = set(instance.hazard_cells)
        key = self._cache_key(partition, instance)
        relational = self._relational_feature_cache.get(key)
        if relational is None:
            relational = compute_cell_relational_features(partition, instance)  # (num_cells, 5)
            self._relational_feature_cache[key] = relational

        cell_descriptors = []
        for cell_idx in range(partition.num_cells):
            elem_tensor, mask = _hrep_cell_tensor(partition, cell_idx, self.f_max)
            phi_out = self.phi(elem_tensor)
            pooled_cell = self._masked_pool(phi_out, mask)

            is_goal = float(cell_idx == instance.goal_cell)
            is_hazard = float(cell_idx in hazard_set)
            contains_agent = float(current_cell is not None and cell_idx == current_cell)
            flags = torch.tensor([is_goal, is_hazard, contains_agent], dtype=pooled_cell.dtype)
            cell_descriptors.append(torch.cat([pooled_cell, flags, relational[cell_idx]]))

        cell_descriptors = torch.stack(cell_descriptors, dim=0)
        cell_embeds = self.cell_mlp(cell_descriptors)

        if self.pool_type == "sum":
            return cell_embeds.sum(dim=0)
        if self.pool_type == "mean":
            return cell_embeds.mean(dim=0)
        return cell_embeds.max(dim=0).values

    def forward(self, partition, instance, agent_state, current_cell: Optional[int] = None) -> torch.Tensor:
        pooled = self.encode_region(partition, instance, current_cell=current_cell)
        agent_state = torch.as_tensor(agent_state, dtype=torch.float32)
        return self.head(torch.cat([pooled, agent_state], dim=-1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
