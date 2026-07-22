"""Step 4b: MLP (point-wise strawman) and CNN (spatial baseline) encoders,
built to the same interface as the Step 3b GNN and Step 4a DeepSets --
`forward(partition, instance, agent_state, current_cell=None) -> (embedding_dim,)`,
agent (x,v) concatenated globally, d=128 default.

MLP -- the deliberate strawman: every cell's features (centroid, its
constraints padded to F_max=12 -- the exact same F_MAX locked in Step 4a's
deepset_encoders.py, reused not redefined --, and context flags) are written
to a FIXED SLOT by cell index and flattened into one vector, zero-padded to
MAX_CELLS (see below). There is no pooling anywhere -- cell i always lives at
the same position in the vector by index accident, so this has no object/set
structure and is NOT permutation-invariant (test_baseline_encoders.py's
`test_mlp_not_permutation_invariant` confirms this is a positive, not a bug).

MAX_CELLS carry-forward (F_max's sibling, for cell COUNT rather than facet
count): computed the same way as Step 4a's F_MAX, from the actual generated
pool across all three tiers:

    from pool import generate_pool_with_stats, build_partition
    # scan every distinct partition_seed for max(partition.num_cells)

Result: EASY max=25, MEDIUM max=40, HARD max=30 -- MEDIUM, not HARD, has the
highest cell count (its template split can produce up to ~40 sub-cells from
25 base squares, whereas HARD's Voronoi generator always produces exactly
num_seed_points=30). MAX_CELLS is set to 42 (40 + a small margin, same spirit
as F_MAX's margin over its observed max of 10).

Budget note (flagged, not silent): the spec's "~128 wide" MLP trunk
guideline is for a differently-scaled input; here MAX_CELLS x per-cell-dim =
42 x 41 = 1722, so a literal 128-wide first layer alone would cost ~220k
params -- 4x the ~53k comparison budget every other encoder targets. Trunk
widths (28, 24) were chosen instead to land at ~52.7k, applying the same
budget-over-literal-width priority the brief explicitly states for the CNN.

CNN -- 5 raster channels over [0,5]x[0,5] at 40x40 (SPEC.md 2.5, adapted):
occupancy (constant 1 -- the domain is fully tiled by the partition, so this
channel carries no information for our tiers, kept for interface fidelity),
goal-cell mask, hazard-cell mask, agent-position (splatted, a small Gaussian
blob at the agent's actual (x,y) -- NOT masked out of this stage, unlike the
GNN/DeepSet's message-passing stage, because spatially relating the agent to
nearby geometry via convolution is the whole point of a spatial encoder; see
module note on the agent-state test), and a cell-id field (normalized
cell index per pixel) replacing the original vanilla-benchmark's "desired-
direction field" -- that field doesn't exist as an input in this RL
formulation (direction is the POLICY's output, not an observable static map;
SPEC.md 1.2), so the brief's explicit "desired-direction-OR-cell-id" wording
is read as inviting this substitution.

CNN param cap (carried forward, flagged since the original plan): two 3x3
conv layers -> GLOBAL pooling (mean over spatial dims) -> head, never
flatten -> FC. A flattened 40x40x64 into a 128-wide FC would be ~295k params
in that one layer alone, 6x over budget; global pooling keeps the FC input
at the channel count (64), not the spatial map size. Channel counts
(c1=64, c2=64) chosen to land at ~48.7k, well under the 1.2x-budget (~63.6k)
hard cap.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepset_encoders import F_MAX as CONSTRAINT_F_MAX
from geometry import polygon_centroid
from region_graph import _cell_edge_constraints, _get_cell_vertices

AGENT_STATE_DIM = 4
NUM_CELL_FLAGS = 3
MAX_CELLS = 42  # see module docstring for derivation (pool-wide observed max=40, MEDIUM tier)
CELL_FEATURE_DIM = 2 + CONSTRAINT_F_MAX * 3 + NUM_CELL_FLAGS  # centroid + padded constraints + flags = 41


def _mlp_cell_feature(partition, cell_idx: int, instance, current_cell: Optional[int]) -> torch.Tensor:
    verts = _get_cell_vertices(partition, cell_idx)
    constraints = _cell_edge_constraints(verts)
    k = len(constraints)
    assert k <= CONSTRAINT_F_MAX, f"cell {cell_idx} has {k} facets > F_MAX={CONSTRAINT_F_MAX} -- would truncate"

    # Computed locally from vertices (geometry.polygon_centroid) rather than
    # via partition.cell_centroid: identical value for every real Partition
    # tier (Box/MixedConvex/IrregularConvex all store/derive the true
    # geometric centroid too), but doesn't require cell_centroid to exist at
    # all -- keeps this encoder usable against minimal test-only partition
    # stand-ins (e.g. test_region_graph._ListPartition) that only implement
    # cell_vertices, matching how region_graph.py itself only depends on
    # duck-typed vertex access, not the full Partition ABC.
    centroid = torch.as_tensor(polygon_centroid(verts), dtype=torch.float32)
    padded = torch.zeros(CONSTRAINT_F_MAX * 3)
    for i, c in enumerate(constraints):
        padded[i * 3] = float(c["normal"][0])
        padded[i * 3 + 1] = float(c["normal"][1])
        padded[i * 3 + 2] = float(c["offset"])

    is_goal = float(cell_idx == instance.goal_cell)
    is_hazard = float(cell_idx in set(instance.hazard_cells))
    contains_agent = float(current_cell is not None and cell_idx == current_cell)
    flags = torch.tensor([is_goal, is_hazard, contains_agent], dtype=torch.float32)
    return torch.cat([centroid, padded, flags])


class MLPEncoder(nn.Module):
    """Point-wise strawman: fixed-slot flattened cell vector, plain MLP
    trunk, no pooling, no permutation invariance."""

    def __init__(self, embedding_dim: int = 128, hidden1: int = 28, hidden2: int = 24,
                 max_cells: int = MAX_CELLS):
        super().__init__()
        self.max_cells = max_cells
        input_dim = max_cells * CELL_FEATURE_DIM
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden2 + AGENT_STATE_DIM, embedding_dim), nn.ReLU())

    def encode_region(self, partition, instance, current_cell: Optional[int] = None) -> torch.Tensor:
        num_cells = partition.num_cells
        assert num_cells <= self.max_cells, (
            f"partition has {num_cells} cells > MAX_CELLS={self.max_cells} -- would truncate the fixed-slot vector"
        )
        slots = [torch.zeros(CELL_FEATURE_DIM) for _ in range(self.max_cells)]
        for cell_idx in range(num_cells):
            slots[cell_idx] = _mlp_cell_feature(partition, cell_idx, instance, current_cell)
        flat = torch.cat(slots)
        return self.trunk(flat)

    def forward(self, partition, instance, agent_state, current_cell: Optional[int] = None) -> torch.Tensor:
        pooled = self.encode_region(partition, instance, current_cell=current_cell)
        agent_state = torch.as_tensor(agent_state, dtype=torch.float32)
        return self.head(torch.cat([pooled, agent_state], dim=-1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- CNN --------------------------------------------------------------

CNN_RESOLUTION = 40
_cell_idx_grid_cache = {}


def _get_cell_idx_grid(partition, resolution: int = CNN_RESOLUTION) -> np.ndarray:
    """(R,R) int array, one cell index per pixel -- the only expensive part
    of rasterization (partition.locate per pixel), cached per (partition
    identity, resolution) since it never changes for a fixed partition."""
    key = (id(partition), resolution)
    if key not in _cell_idx_grid_cache:
        xmin, xmax, ymin, ymax = partition.domain
        xs = xmin + (np.arange(resolution) + 0.5) / resolution * (xmax - xmin)
        ys = ymin + (np.arange(resolution) + 0.5) / resolution * (ymax - ymin)
        grid = np.zeros((resolution, resolution), dtype=int)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                grid[i, j] = partition.locate(np.array([x, y]))
        _cell_idx_grid_cache[key] = grid
    return _cell_idx_grid_cache[key]


def _rasterize(partition, instance, agent_state, resolution: int = CNN_RESOLUTION) -> torch.Tensor:
    grid = _get_cell_idx_grid(partition, resolution)
    num_cells = partition.num_cells

    occupancy = np.ones((resolution, resolution), dtype=np.float32)
    goal_mask = (grid == instance.goal_cell).astype(np.float32)
    hazard_set = instance.hazard_cells
    hazard_mask = np.isin(grid, hazard_set).astype(np.float32) if hazard_set else np.zeros_like(occupancy)
    cell_id_field = ((grid + 1) / num_cells).astype(np.float32)

    xmin, xmax, ymin, ymax = partition.domain
    xs = xmin + (np.arange(resolution) + 0.5) / resolution * (xmax - xmin)
    ys = ymin + (np.arange(resolution) + 0.5) / resolution * (ymax - ymin)
    XX, YY = np.meshgrid(xs, ys)  # default 'xy': XX[i,j]=xs[j], YY[i,j]=ys[i] -- matches grid[i,j] (i=row/y, j=col/x)
    ax, ay = float(agent_state[0]), float(agent_state[1])
    sigma = 0.3
    agent_channel = np.exp(-((XX - ax) ** 2 + (YY - ay) ** 2) / (2 * sigma ** 2)).astype(np.float32)

    img = np.stack([occupancy, goal_mask, hazard_mask, agent_channel, cell_id_field], axis=0)
    return torch.from_numpy(img)


class CNNEncoder(nn.Module):
    # Global-pool choice matters more than it might look: on a random-init
    # sanity check, moving the goal cell to a different location changed the
    # pre-head representation by L2=0.011 under mean-pooling vs. L2=0.292
    # under max-pooling (~28x) -- mean-pooling dilutes a small localized mask
    # change (e.g. one cell's goal flag, ~4% of pixels) across the whole
    # spatial average, while max-pooling preserves the peak signal. Both are
    # valid "global pooling, not flatten" per the spec; exposed as a config
    # flag (mirroring edge_type_mode/pool_type on the GNN/DeepSet encoders)
    # rather than silently committing to one. Default kept at "mean" since
    # test_cnn_spatial_sensitivity still passes it, comfortably above
    # tolerance -- but if the CNN underperforms in later training, this
    # dilution is the first architectural knob worth revisiting.
    def __init__(self, embedding_dim: int = 128, c1: int = 64, c2: int = 64,
                 resolution: int = CNN_RESOLUTION, pool_type: str = "mean"):
        super().__init__()
        if pool_type not in ("mean", "max"):
            raise ValueError(f"pool_type must be mean/max, got {pool_type!r}")
        self.resolution = resolution
        self.pool_type = pool_type
        self.conv1 = nn.Conv2d(5, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.head = nn.Sequential(nn.Linear(c2 + AGENT_STATE_DIM, embedding_dim), nn.ReLU())

    def encode_image(self, partition, instance, agent_state) -> torch.Tensor:
        """Unlike the GNN/DeepSet's encode_graph()/encode_region(), this DOES
        depend on agent position -- the position is splatted into the raster
        by design (SPEC.md 2.5), so convolution can relate the agent to
        nearby geometry. Only agent VELOCITY is exclusively global-only; see
        test_agent_state_global_concat's velocity-only variant for the CNN.
        """
        img = _rasterize(partition, instance, agent_state, self.resolution).unsqueeze(0)  # (1,5,R,R)
        h = F.relu(self.conv1(img))
        h = F.relu(self.conv2(h))
        if self.pool_type == "mean":
            return h.mean(dim=(2, 3)).squeeze(0)  # global pool, NOT flatten
        return h.amax(dim=(2, 3)).squeeze(0)

    def forward(self, partition, instance, agent_state, current_cell: Optional[int] = None) -> torch.Tensor:
        pooled = self.encode_image(partition, instance, agent_state)
        agent_state_t = torch.as_tensor(agent_state, dtype=torch.float32)
        return self.head(torch.cat([pooled, agent_state_t], dim=-1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
