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
    """Translation between the routed physical id space and the dispatch id space.

    Routed space (width ``num_routed_physical`` == ``global_num_experts``) holds
    routed experts only, base plus EPLB redundant replicas; it is what
    ``expert_map`` indexes and, when EPLB is on, what placement and the load
    histogram speak. Dispatch space (width ``num_dispatch_slots``) is what the
    all2all backend and the local weight buffers speak, with one extra slot per
    rank for the fused shared expert. They have to differ because MoRI derives a
    token's destination rank from the raw expert id, so a shared expert
    replicated everywhere cannot share a single global id the way ``expert_map``
    allows.

    Per-rank slot order is ``[routed ...][shared ...]``; keeping routed as a
    prefix is load bearing for ``count_local_base_experts`` and
    ``is_batched_expert_slot``.
    """

    __slots__ = ("ep_size", "num_routed_physical", "num_shared")

    def __init__(
        self, num_routed_physical: int, ep_size: int, num_shared: int = 0
    ) -> None:
        assert ep_size > 0, f"ep_size must be positive, got {ep_size}"
        assert num_routed_physical % ep_size == 0, (
            "routed physical experts must divide evenly across ranks: "
            f"num_routed_physical={num_routed_physical}, ep_size={ep_size}"
        )
        assert num_shared >= 0, f"num_shared must be non-negative, got {num_shared}"
        self.num_routed_physical = int(num_routed_physical)
        self.ep_size = int(ep_size)
        self.num_shared = int(num_shared)

    @property
    def routed_slots_per_rank(self) -> int:
        """Routed physical slots each rank owns."""
        return self.num_routed_physical // self.ep_size

    @property
    def slots_per_rank(self) -> int:
        """Local slots per rank (dispatch space); the backend's num_experts_per_rank."""
        return self.routed_slots_per_rank + self.num_shared

    @property
    def num_dispatch_slots(self) -> int:
        """Total dispatch-space slots across all ranks."""
        return self.slots_per_rank * self.ep_size

    def routed_to_dispatch(self, physical_id: "int | torch.Tensor"):
        """Routed physical id -> dispatch id. Scalar or tensor.

        The offset within a rank's block is preserved, so the weight buffer
        needs no reordering. Callers must screen ids outside
        ``[0, num_routed_physical)`` themselves: floor division sends -1 to -2.
        The Triton kernel inlines this same arithmetic.
        """
        return physical_id + self.num_shared * (
            physical_id // self.routed_slots_per_rank
        )

    def shared_dispatch_id(self, ep_rank: int, shared_index: int = 0) -> int:
        """Dispatch id of this rank's shared expert: fixed-local, no xGMI traffic."""
        assert 0 <= shared_index < self.num_shared, (
            f"shared_index={shared_index} out of range for "
            f"num_shared={self.num_shared}"
        )
        return ep_rank * self.slots_per_rank + self.routed_slots_per_rank + shared_index

    def apply_to_topk(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        ep_rank: int,
        shared_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite a routed topk into dispatch space and append the shared column.

        Reference implementation for the fused Triton kernel; also the CPU path.

        Must run AFTER the EPLB logical->physical remap. Earlier, the shared
        column would land in EPLB space and enter the load histogram -- it is hit
        by every token, so it would drive balancedness toward 1 and mask the
        routed imbalance the rebalance gate exists to detect.

        Ids outside ``[0, num_routed_physical)`` pass through untouched: floor
        division would send the -1 sentinel to -2.
        """
        assert self.num_shared > 0, "apply_to_topk requires a fused shared expert"
        num_tokens = topk_ids.shape[0]

        in_range = (topk_ids >= 0) & (topk_ids < self.num_routed_physical)
        routed = torch.where(in_range, self.routed_to_dispatch(topk_ids), topk_ids)

        # Built on-device: `torch.tensor(list, device=...)` is a H2D copy, which
        # CUDA graph capture rejects unless pinned.
        shared_ids = (
            torch.arange(self.num_shared, dtype=topk_ids.dtype, device=topk_ids.device)
            + self.shared_dispatch_id(ep_rank, 0)
        ).expand(num_tokens, self.num_shared)
        shared_w = torch.full(
            (num_tokens, self.num_shared),
            shared_weight,
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        return (
            torch.cat((routed, shared_ids), dim=1),
            torch.cat((topk_weights, shared_w), dim=1),
        )


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
