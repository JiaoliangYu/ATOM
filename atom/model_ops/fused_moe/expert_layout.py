# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Expert slot layout for FusedMoE parameters.

Where each expert's weights live inside the fused ``w13_weight`` / ``w2_weight``
buffers, and which of those slots a given rank owns.  Both the weight loader
and the batched staging path need this, and they must agree byte for byte, so
it lives in one place instead of being re-derived at each call site.

Deliberately pure ``torch`` — no AITER, no ``atom.config`` — so the model
loader's unit tests can exercise the real layout on a plain CPU runner.
"""

import torch

# Which dimension of a single expert slot a shard is sharded along, before the
# `is_transposed` flip.  `w1`/`w3` are the two halves of the gate_up projection
# (column-parallel, output dim); `w2` is the down projection (row-parallel).
_SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}


def determine_expert_map(
    ep_size: int, ep_rank: int, global_num_experts: int
) -> tuple[int, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size (int): The size of the expert parallel group
        global_num_experts (int): The total number of experts in the model.

    Returns:
        Tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts] = (
            torch.arange(0, local_num_experts, dtype=torch.int32)
        )
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = global_num_experts - ep_rank * local_num_experts

        expert_map[-local_num_experts:] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    return (local_num_experts, expert_map)


def count_local_base_experts(
    expert_map: torch.Tensor | None,
    global_num_experts: int,
    num_redundant_experts: int,
    local_num_experts: int,
    num_fused_shared_experts: int = 0,
) -> int:
    """Number of local slots holding a routed base expert.

    Excludes both EPLB redundant replicas — `fill_redundant` populates those
    after loading — and fused shared experts, which the checkpoint delivers
    separately from the routed ones and which therefore must not be counted
    when deciding whether a batch is complete.

    `determine_expert_map` assigns a rank a contiguous run of global ids
    starting at local index 0, and both the redundant replicas and the shared
    experts are appended after them, so the returned count is also the length
    of the local slot *prefix* holding routed base experts.
    """
    if expert_map is None:
        return local_num_experts - num_fused_shared_experts
    num_logical = global_num_experts - num_redundant_experts
    return int((expert_map[:num_logical] != -1).sum().item())


def physical_expert_id(
    expert_id: int,
    global_num_experts: int,
    num_redundant_experts: int,
    num_fused_shared_experts: int,
) -> int:
    """Translate a checkpoint-side expert id into a physical slot index.

    Checkpoints (and `FusedMoE.make_expert_params_mapping`) number a fused
    shared expert immediately after the *logical* routed experts, because that
    is all the HF config knows about.  `expert_map` is indexed by *physical*
    slot, and EPLB inserts `num_redundant_experts` replicas between the two —
    so without this translation a shared expert would be written over a
    redundant replica while its own slot kept its init value.

    A no-op unless EPLB is configured with redundant experts.
    """
    num_logical = global_num_experts - num_redundant_experts
    if num_fused_shared_experts and expert_id >= num_logical:
        return global_num_experts + (expert_id - num_logical)
    return expert_id


class SharedExpertDispatchLayout:
    """Translation between the EPLB physical id space and the dispatch id space.

    Two id spaces coexist once a shared expert is fused into an all2all-dispatched
    MoE:

    * **EPLB space** (width ``eplb_num_physical``) is what placement, migration and
      the load histogram speak.  It contains routed experts only -- base plus EPLB
      redundant replicas -- and is what ``determine_expert_map`` partitions.
    * **dispatch space** (width ``num_physical``) is what the all2all backend and
      the local weight buffers speak.  Every rank gets one extra slot per fused
      shared expert, appended after that rank's routed slots.

    The split exists because MoRI derives a token's destination rank from the raw
    expert id (``id // num_experts_per_rank``), so a shared expert replicated on
    every rank cannot share a single global id the way ``expert_map`` allows.
    Giving each rank its own shared id is the whole point of this layout.

    The per-rank slot order is ``[routed ...][shared ...]``, matching the buffer
    that ``FusedMoE.__init__`` already allocates (``local_num_experts`` is bumped
    by ``num_fused_shared_experts`` there).  Keeping routed as a *prefix* is load
    bearing: ``count_local_base_experts`` and ``is_batched_expert_slot`` both rely
    on it.

    ``num_shared == 0`` degrades every translation to the identity, so callers can
    build this unconditionally and stay branch-free.
    """

    __slots__ = ("eplb_num_physical", "ep_size", "num_shared")

    def __init__(
        self, eplb_num_physical: int, ep_size: int, num_shared: int = 0
    ) -> None:
        assert ep_size > 0, f"ep_size must be positive, got {ep_size}"
        assert eplb_num_physical % ep_size == 0, (
            "EPLB physical experts must divide evenly across ranks: "
            f"eplb_num_physical={eplb_num_physical}, ep_size={ep_size}"
        )
        assert num_shared >= 0, f"num_shared must be non-negative, got {num_shared}"
        self.eplb_num_physical = int(eplb_num_physical)
        self.ep_size = int(ep_size)
        self.num_shared = int(num_shared)

    @property
    def routed_slots_per_rank(self) -> int:
        """Routed physical slots each rank owns (EPLB space, per rank)."""
        return self.eplb_num_physical // self.ep_size

    @property
    def slots_per_rank(self) -> int:
        """Total local slots each rank owns (dispatch space, per rank).

        This is what the all2all backend must be told as its
        ``num_experts_per_rank`` -- NOT ``num_experts // world_size``, which
        would drop the shared slot and make the backend route to the wrong rank.
        """
        return self.routed_slots_per_rank + self.num_shared

    @property
    def num_physical(self) -> int:
        """Total dispatch-space slots across all ranks."""
        return self.slots_per_rank * self.ep_size

    def routed_to_dispatch(self, eplb_id: int) -> int:
        """EPLB physical id -> dispatch id.

        Each rank's block grows by ``num_shared``, so an id in rank ``r``'s block
        shifts right by ``num_shared * r``.  The local offset within the block is
        preserved, which is why the weight buffer needs no reordering.
        """
        return eplb_id + self.num_shared * (eplb_id // self.routed_slots_per_rank)

    def shared_dispatch_id(self, ep_rank: int, shared_index: int = 0) -> int:
        """Dispatch id of this rank's ``shared_index``-th shared expert.

        Constant per rank: the fixed-local choice that makes a token's shared
        column resolve to its own rank, costing no cross-GPU traffic.
        """
        assert 0 <= shared_index < self.num_shared, (
            f"shared_index={shared_index} out of range for "
            f"num_shared={self.num_shared}"
        )
        return ep_rank * self.slots_per_rank + self.routed_slots_per_rank + shared_index

    def local_slot_of_dispatch_id(self, dispatch_id: int) -> int:
        """Dispatch id -> index within its owning rank's local buffer."""
        return dispatch_id % self.slots_per_rank

    def owner_rank_of_dispatch_id(self, dispatch_id: int) -> int:
        """Dispatch id -> owning rank, the way the all2all backend computes it."""
        return dispatch_id // self.slots_per_rank

    def eplb_gather_index(
        self, device: torch.device | None = None
    ) -> torch.Tensor | None:
        """Index that folds a dispatch-space load histogram back to EPLB space.

        ``out[p] = histogram[routed_to_dispatch(p)]`` for every EPLB physical id
        ``p``.  Needed because the histogram is recorded in dispatch space (so the
        shared slots get their own bins instead of colliding with routed ids),
        while ``physical_load_to_logical_load`` and the placement policies expect
        EPLB-space widths.

        Returns ``None`` when no shared expert is fused -- the two spaces are then
        identical and the caller should skip the gather entirely.
        """
        if self.num_shared == 0:
            return None
        eplb_ids = torch.arange(self.eplb_num_physical, dtype=torch.long, device=device)
        return eplb_ids + self.num_shared * (eplb_ids // self.routed_slots_per_rank)

    def shared_slot_index(
        self, device: torch.device | None = None
    ) -> torch.Tensor | None:
        """Dispatch-space bins holding shared experts, one block per rank.

        The load histogram must zero these before it is consumed: a shared expert
        is hit by every token, so its per-rank load is a large constant.  Feeding
        that into ``_compute_balancedness`` would add the same offset to every
        GPU's total and drive mean/max toward 1, masking the routed imbalance the
        gate is supposed to detect.

        Returns ``None`` when no shared expert is fused.
        """
        if self.num_shared == 0:
            return None
        ranks = torch.arange(self.ep_size, dtype=torch.long, device=device)
        block = ranks * self.slots_per_rank + self.routed_slots_per_rank
        offsets = torch.arange(self.num_shared, dtype=torch.long, device=device)
        return (block[:, None] + offsets[None, :]).reshape(-1)


def expert_shard_dim(shard_id: str, is_transposed: bool = False) -> int:
    """Dimension of a single expert slot that `shard_id` is sharded along."""
    dim = _SHARD_ID_TO_SHARDED_DIM[shard_id]
    return int(not dim) if is_transposed else dim


def expert_shard_view(
    slot: torch.Tensor, shard_id: str, shard_dim: int
) -> torch.Tensor:
    """The part of one expert slot that `shard_id` owns.

    `w13` stacks the gate and up projections along `shard_dim`, so `w1` owns
    the first half and `w3` the second; `w2` owns the whole slot.
    """
    if shard_id == "w2":
        return slot
    assert shard_id in ("w1", "w3"), f"unexpected shard_id {shard_id!r}"
    half = slot.shape[shard_dim] // 2
    return slot.narrow(shard_dim, 0 if shard_id == "w1" else half, half)


def expert_region(
    tensor: torch.Tensor,
    local_expert_id: int,
    shard_id: str,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Sub-view of a param-shaped `tensor` owned by one (expert, shard) arrival.

    The region is the *full* half-slot, not the width a particular arrival
    copies: `_load_w13` / `_load_w2` narrow further to `loaded_weight`'s size
    when the parameter is padded (MXFP4 alignment), and a flush does not have
    `loaded_weight` to hand.  The difference is the padding tail, which is zero
    on both sides — staging buffers are zero-initialised and MXFP4 parameters
    are zeroed in `create_weights` — so copying the full half matches what a
    whole-parameter flush would have written.  Ownership is therefore tracked
    at (slot, shard) granularity, not per byte.
    """
    return expert_shard_view(
        tensor[local_expert_id], shard_id, expert_shard_dim(shard_id, is_transposed)
    )
