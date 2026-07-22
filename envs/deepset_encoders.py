"""Step 4a: H-Rep and V-Rep DeepSet encoders (paper Section 3.2), built to
the same interface as the Step 3b RegionGraphGNN -- consume (partition,
instance, agent_state) directly (not a RegionGraph; these encoders don't use
the graph at all, only per-cell element sets) and produce an embedding in
R^d, d=128 default, agent (x,v) concatenated globally after the region-level
pool, exactly like the GNN.

Two-level DeepSet, per instance (unbatched, matching RegionGraphGNN.forward's
one-graph-at-a-time convention):
  Level 1 (per cell): phi(element) for each element in the cell's unordered
    set (H-Rep: half-space constraints (nx,ny,offset); V-Rep: vertices
    (x,y)) -> masked sum-pool -> per-cell embedding. Cell-context flags
    (is_goal, is_hazard, contains_agent) concatenated onto the per-cell
    embedding before the region-level pool (SPEC.md 2.5), same convention as
    RegionGraph's per-node flags -- contains_agent is a settable/optional
    signal (via `current_cell=`), defaulting to "no cell flagged" just like
    RegionGraph.set_agent_cell defaults to all-False at construction.
  Level 2 (per region): a shared cell_mlp transforms each per-cell
    descriptor, then a second permutation-invariant pool (sum, matching the
    paper) over cells -> region embedding.
  Agent (x,v) concatenated to the region embedding -> head -> final
  embedding.

CARRY-FORWARD, F_max (due since Step 2b): the per-cell padding dimension.
Computed from the ACTUAL generated pool across all three tiers (not the
design doc's illustrative "{3,...,6}"), via:

    from pool import generate_pool_with_stats, build_partition
    # scan every distinct partition_seed present in an 800/700/700-instance
    # pool (same sizes as Step 2c) for max(cell_facet_count) over all cells

Result: EASY max=4, MEDIUM max=5, HARD max=10 (at partition_seed=1892010513,
cell 19) -- pool-wide max is 10, not the design doc's 6 and not any single
instance's 8. F_MAX is set to 12 (10 + a small margin against a slightly
larger outlier under a different generation rng_seed). Since a convex
polygon's vertex count equals its facet count, this same F_MAX bounds both
H-Rep (max constraints/cell) and V-Rep (max vertices/cell) padding.

Padding is masked out of the sum-pool (see `_masked_pool`) -- a padded slot
contributes exactly zero regardless of phi's bias term, not "phi(0)".
"""

from typing import Optional

import torch
import torch.nn as nn

from region_graph import _cell_edge_constraints, _get_cell_vertices

F_MAX = 12  # see module docstring for derivation
AGENT_STATE_DIM = 4  # (x1, x2, v1, v2)
NUM_CELL_FLAGS = 3   # (is_goal, is_hazard, contains_agent)


def _hrep_cell_tensor(partition, cell_idx: int, f_max: int = F_MAX):
    """(f_max, 3) constraint tensor [normal_x, normal_y, offset] + (f_max,) bool mask."""
    verts = _get_cell_vertices(partition, cell_idx)
    constraints = _cell_edge_constraints(verts)
    k = len(constraints)
    assert k <= f_max, f"cell {cell_idx} has {k} facets > F_MAX={f_max} -- would silently truncate"

    tensor = torch.zeros(f_max, 3)
    mask = torch.zeros(f_max, dtype=torch.bool)
    for i, c in enumerate(constraints):
        tensor[i, 0] = float(c["normal"][0])
        tensor[i, 1] = float(c["normal"][1])
        tensor[i, 2] = float(c["offset"])
        mask[i] = True
    return tensor, mask


def _vrep_cell_tensor(partition, cell_idx: int, f_max: int = F_MAX):
    """(f_max, 2) vertex tensor [x, y] + (f_max,) bool mask."""
    verts = _get_cell_vertices(partition, cell_idx)
    k = len(verts)
    assert k <= f_max, f"cell {cell_idx} has {k} vertices > F_MAX={f_max} -- would silently truncate"

    tensor = torch.zeros(f_max, 2)
    mask = torch.zeros(f_max, dtype=torch.bool)
    for i in range(k):
        tensor[i, 0] = float(verts[i][0])
        tensor[i, 1] = float(verts[i][1])
        mask[i] = True
    return tensor, mask


class _TwoLevelDeepSet(nn.Module):
    """Shared implementation for H-Rep and V-Rep -- they differ only in
    per-element dimension and the cell_tensor_fn used to build (tensor, mask)
    pairs, exactly mirroring how this repo's existing DeepSetEncoder
    (Rush-hour-git/DeepSetRL.py) is reused verbatim for both H-rep and V-rep
    by varying only input_dim."""

    def __init__(
        self,
        element_dim: int,
        cell_tensor_fn,
        embedding_dim: int = 128,
        phi_hidden: int = 64,
        cell_hidden: int = 96,
        pool_type: str = "sum",
        f_max: int = F_MAX,
    ):
        super().__init__()
        if pool_type not in ("sum", "mean", "max"):
            raise ValueError(f"pool_type must be sum/mean/max, got {pool_type!r}")
        self.element_dim = element_dim
        self.cell_tensor_fn = cell_tensor_fn
        self.pool_type = pool_type
        self.f_max = f_max
        self.embedding_dim = embedding_dim

        self.phi = nn.Sequential(
            nn.Linear(element_dim, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden), nn.ReLU(),
        )
        self.cell_mlp = nn.Sequential(nn.Linear(phi_hidden + NUM_CELL_FLAGS, cell_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(cell_hidden + AGENT_STATE_DIM, embedding_dim), nn.ReLU())

    def _masked_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (f_max, H), mask: (f_max,) bool. Masked-out slots contribute
        exactly zero (or -inf pre-max for max-pool), never phi's bias term."""
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
        """Message-passing-equivalent stage: per-cell pooled sets -> region
        pool. Deliberately takes no agent_state -- matches RegionGraphGNN's
        encode_graph(), so agent state cannot influence this stage."""
        hazard_set = set(instance.hazard_cells)
        cell_descriptors = []
        for cell_idx in range(partition.num_cells):
            elem_tensor, mask = self.cell_tensor_fn(partition, cell_idx, self.f_max)
            phi_out = self.phi(elem_tensor)
            pooled_cell = self._masked_pool(phi_out, mask)

            is_goal = float(cell_idx == instance.goal_cell)
            is_hazard = float(cell_idx in hazard_set)
            contains_agent = float(current_cell is not None and cell_idx == current_cell)
            flags = torch.tensor([is_goal, is_hazard, contains_agent], dtype=pooled_cell.dtype)
            cell_descriptors.append(torch.cat([pooled_cell, flags]))

        cell_descriptors = torch.stack(cell_descriptors, dim=0)  # (num_cells, phi_hidden+3)
        cell_embeds = self.cell_mlp(cell_descriptors)             # (num_cells, cell_hidden)

        if self.pool_type == "sum":
            return cell_embeds.sum(dim=0)
        if self.pool_type == "mean":
            return cell_embeds.mean(dim=0)
        return cell_embeds.max(dim=0).values

    def forward(self, partition, instance, agent_state, current_cell: Optional[int] = None) -> torch.Tensor:
        region_embedding = self.encode_region(partition, instance, current_cell=current_cell)
        agent_state = torch.as_tensor(agent_state, dtype=torch.float32)
        combined = torch.cat([region_embedding, agent_state], dim=-1)
        return self.head(combined)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HRepDeepSet(_TwoLevelDeepSet):
    # phi_hidden=cell_hidden=128 tuned to land at ~51k params, matching the
    # ~53k DeepSet budget (SPEC.md 2.6) for a fair Step 4 comparison.
    def __init__(self, embedding_dim: int = 128, phi_hidden: int = 128, cell_hidden: int = 128,
                 pool_type: str = "sum", f_max: int = F_MAX):
        super().__init__(
            element_dim=3, cell_tensor_fn=_hrep_cell_tensor, embedding_dim=embedding_dim,
            phi_hidden=phi_hidden, cell_hidden=cell_hidden, pool_type=pool_type, f_max=f_max,
        )


class VRepDeepSet(_TwoLevelDeepSet):
    def __init__(self, embedding_dim: int = 128, phi_hidden: int = 128, cell_hidden: int = 128,
                 pool_type: str = "sum", f_max: int = F_MAX):
        super().__init__(
            element_dim=2, cell_tensor_fn=_vrep_cell_tensor, embedding_dim=embedding_dim,
            phi_hidden=phi_hidden, cell_hidden=cell_hidden, pool_type=pool_type, f_max=f_max,
        )
