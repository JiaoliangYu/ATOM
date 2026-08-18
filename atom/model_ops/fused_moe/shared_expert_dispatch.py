# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Fold a per-rank shared expert into an all2all dispatch.

Used when EPLB is off. With EPLB on the shared expert is instead one more
routed logical expert, so placement and the existing remap handle it and none
of this runs.
"""

import torch

from atom.model_ops.fused_moe.expert_layout import SharedExpertDispatchLayout

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on CPU-only builds
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _shared_dispatch_kernel(
        topk_ids_ptr,  # [ntok, topk]      routed physical ids
        topk_w_ptr,  # [ntok, topk]      routed weights
        out_ids_ptr,  # [ntok, out_width] dispatch-space ids
        out_w_ptr,  # [ntok, out_width] weights
        num_routed_physical,
        routed_per_rank,
        shared_base,
        shared_weight,
        topk,
        n_out,
        out_width,
        NUM_SHARED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Rewrite routed ids into dispatch space and fill the shared column.

        Iterates over output elements, so the shared column is written in place
        instead of concatenated.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out

        row = offs // out_width
        col = offs % out_width
        is_routed = col < topk
        rmask = mask & is_routed

        in_off = row * topk + col
        phys = tl.load(topk_ids_ptr + in_off, mask=rmask, other=-1).to(tl.int64)

        # SharedExpertDispatchLayout.routed_to_dispatch, inlined. Out-of-range
        # ids pass through: floor division would send -1 to -2.
        in_range = (phys >= 0) & (phys < num_routed_physical)
        shifted = phys + NUM_SHARED * (phys // routed_per_rank)
        disp = tl.where(in_range, shifted, phys)

        sid = shared_base + (col - topk)
        tl.store(out_ids_ptr + offs, tl.where(is_routed, disp, sid), mask=mask)

        w = tl.load(topk_w_ptr + in_off, mask=rmask, other=0.0)
        tl.store(out_w_ptr + offs, tl.where(is_routed, w, shared_weight), mask=mask)


def apply_shared_expert_dispatch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layout: SharedExpertDispatchLayout,
    ep_rank: int,
    shared_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen a routed topk into dispatch space. Returns ``(weights, ids)``.

    The torch equivalent is several elementwise kernels plus two ``torch.cat``
    per layer per step, inside the ``aiter.moe_forward`` custom op that inductor
    cannot fuse into -- hence the hand-written kernel.
    """
    ntok, topk = topk_ids.shape
    num_shared = layout.num_shared
    out_width = topk + num_shared
    n_out = ntok * out_width
    if not _HAS_TRITON or n_out == 0:
        ids, weights = layout.apply_to_topk(
            topk_ids, topk_weights, ep_rank, shared_weight
        )
        return weights, ids

    out_ids = torch.empty(
        (ntok, out_width), dtype=topk_ids.dtype, device=topk_ids.device
    )
    out_w = torch.empty(
        (ntok, out_width), dtype=topk_weights.dtype, device=topk_weights.device
    )

    def grid(meta_kw):
        return (triton.cdiv(n_out, meta_kw["BLOCK"]),)

    _shared_dispatch_kernel[grid](
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        out_ids,
        out_w,
        layout.num_routed_physical,
        layout.routed_slots_per_rank,
        layout.shared_dispatch_id(ep_rank, 0),
        shared_weight,
        topk,
        n_out,
        out_width,
        NUM_SHARED=num_shared,
        BLOCK=256,
    )
    return out_w, out_ids
