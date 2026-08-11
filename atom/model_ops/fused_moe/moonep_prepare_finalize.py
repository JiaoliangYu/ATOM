# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MoonEP prepare/finalize: dispatch straight into the grouped-GEMM layout.

Sibling of ``mori_prepare_finalize.py``, but it does not reproduce mori's
output convention.  The two transports group tokens differently:

======================  ==================================  ==================
                        one dispatched row is               grouping done by
======================  ==================================  ==================
mori                    a (token, destination rank) pair,   the experts kernel,
                        carrying the token's whole topk row  by sorting
MoonEP                  a (token, expert) pair               dispatch itself
======================  ==================================  ==================

MoonEP's dispatch epilogue expands duplicates into their own rows and lays them
out grouped by expert with ``cu_seqlens`` boundaries, which is exactly the shape
a grouped GEMM wants.  Reconstructing mori's convention from it would mean
undoing that grouping so the experts kernel can redo it, so this class hands the
grouped layout over directly instead.

The modular-kernel contract needs no new activation format for this:
``FusedMoEActivationFormat.Standard`` is just ``(num_tokens, hidden_dim)``, and
MoonEP's dispatched rows are a 2-D tensor of that shape.  Only the row
*ordering* differs, and ``run_experts`` below owns the experts step, so nothing
downstream has to know about it.

``dispatch_ids`` is returned as ``None`` for the same reason: ``run_experts``
synthesises its own topk-1 ids naming weight-pool slots.
"""

from __future__ import annotations

import logging

import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig

try:
    from aiter import QuantType
except ImportError:  # pragma: no cover - aiter always present in ATOM
    QuantType = None

logger = logging.getLogger("atom")


class MoonEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Plan-driven EP transport that emits the grouped-by-expert layout."""

    def __init__(
        self,
        moonep_op,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
        ep_group=None,
    ) -> None:
        super().__init__()
        # TBO is refused rather than merely untested.  mori gives each ubatch
        # its own handle via create_handle; MoonEP keeps the plan as per-op
        # state, so with two ubatches sharing one op the second dispatch
        # overwrites the first's plan and combine then reduces against the
        # wrong one -- no crash, just wrong numbers, and nothing downstream
        # would point back here.
        from atom.utils.tbo.ubatching import tbo_active

        if tbo_active():
            raise NotImplementedError(
                "MoonEP EP backend does not support TBO: the plan is per-op "
                "state and two ubatches would overwrite each other's plan, "
                "silently combining against the wrong one. Disable TBO, or use "
                "the mori/aiter EP backend."
            )
        self._op = moonep_op
        self._ep_group = ep_group
        self._pools = None
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.num_local_experts = num_local_experts

    # -- modular-kernel plumbing -----------------------------------------
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        # 2-D (rows, hidden); only the row ordering differs from mori's.
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int:
        return self.max_tokens_per_rank

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def topk_indices_dtype(self) -> torch.dtype:
        return torch.int32

    def output_is_reduced(self) -> bool:
        # Same as mori: combine reduces each token's topk contributions across
        # every rank that ran one of its experts, so finalize returns the
        # finished token and no caller-side all-reduce is owed.
        return True

    def supports_async(self) -> bool:
        # MoonEP needs an all-gather of the expert histogram before it can
        # plan, so there is no useful split point for a comm-stream overlap
        # yet.  Keep it synchronous rather than pretend otherwise.
        return False

    # -- prepare / finalize -----------------------------------------------
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        quant_type=None,
    ) -> mk.PrepareResultType:
        assert (
            not apply_router_weight_on_input
        ), "MoonEP does not support apply_router_weight_on_input=True."
        if a1.dtype != torch.bfloat16:
            raise NotImplementedError(
                "MoonEP dispatch is BF16-only; there is no quantised path. "
                f"Got {a1.dtype}."
            )

        # The op builds the plan internally: local histogram -> all-gather ->
        # GPU planner.  The all-gather is intrinsic to MoonEP (it balances
        # against the *global* expert load) and is a cost mori does not pay.
        dispatched, dispatched_weights, cu_seqlens = self._op.dispatch_grouped(
            a1, topk_weights, topk_ids, decode=not self._is_prefill()
        )

        # Despite the field name, this is the count of *valid rows* in the
        # dispatch buffer, not a per-expert vector: modular_kernel.py:380
        # documents it as the number of leading rows fused_moe is driven over
        # via num_local_tokens, and mori fills it with a scalar for a layout
        # that is not grouped by expert at all.  For MoonEP that is
        # cu_seqlens[-1].  local_group_sizes() still runs, for its check that
        # no group landed here without a prefetch slot.
        self._op.local_group_sizes(cu_seqlens)
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=self._op.valid_rows(),
            expert_num_tokens_cpu=None,
        )

        # dispatch_scale is None on the BF16 path (mori_prepare_finalize.py:207
        # only produces a scale under use_fp8_dispatch).  dispatch_ids is None
        # by the assumption documented at the top of this module.
        return (dispatched, None, expert_tokens_meta, None, dispatched_weights)

    def _is_prefill(self) -> bool:
        """Phase, OR-ed across the EP group and decided once per step.

        It **must** be identical on every rank: dispatch is a collective and the
        two plans have different shapes, so a rank that disagrees either hangs
        or reads the wrong layout. ``forward_context.is_prefill`` is per-rank, so
        it is reduced over the EP group; the answer is cached on the context
        object, which is per forward, so the reduction happens once per step
        rather than once per MoE layer.
        """
        if self._op.decode_plan_available() is False:
            return True
        import torch.distributed as dist

        from atom.utils.forward_context import get_forward_context

        ctx = None
        try:
            ctx = get_forward_context().context
        except Exception:
            return True
        cached = getattr(ctx, "_moonep_is_prefill", None) if ctx is not None else None
        if cached is not None:
            return cached
        local = bool(getattr(ctx, "is_prefill", False)) if ctx is not None else True
        group = getattr(self._ep_group, "device_group", None)
        if group is not None and dist.get_world_size(group) > 1:
            flag = torch.tensor(
                [1 if local else 0],
                dtype=torch.int32,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
            local = bool(flag.item())
        if ctx is not None:
            try:
                ctx._moonep_is_prefill = local
            except Exception:
                pass
        return local

    # -- experts ----------------------------------------------------------
    def run_experts(
        self,
        rows: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        activation=None,
        quant_type=None,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        dtype=None,
        extra_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Run the experts over MoonEP's grouped rows.

        The plain ``fused_moe`` call cannot serve this layout: the last ``B``
        groups are served by *migrated* expert weights living in the prefetch
        segment rather than by this rank's own ``w1``/``w2``, and choosing the
        weight source per group is the whole point of MoonEP's ``E + B``
        layout.  So the weights are re-pointed at the pool and ``fused_moe`` is
        called here instead, with one synthetic topk-1 id per row naming the
        pool slot.  Its sorting pass then reorders an order that is already
        correct, and everything about quantisation, the weight shuffle and the
        GEMM itself stays exactly as ATOM set it up.

        The ``w1``/``w2``/scale arguments are ignored in favour of the pools:
        adopt_weights rebound the layer's parameters onto ``pool.home``, so
        what arrives here is the first ``epn`` slots of the very same storage,
        minus the migration slots this kernel has to see.
        """
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe

        pools = self._pools
        if pools is None:
            raise RuntimeError(
                "adopt_weights() never ran for this layer, so the expert "
                "weights are not in the symmetric heap and peers cannot "
                "prefetch them. It is called from init_prepare_finalize."
            )
        if w1.data_ptr() != pools[0][0].pool.data_ptr():
            raise RuntimeError(
                "the layer's w13_weight no longer aliases the MoonEP pool; "
                "something reassigned it after adopt_weights, and peers would "
                "prefetch stale weights"
            )
        plan = self._op.live_plan()
        if self._op.needs_split():
            # Decode migrates nothing, so there is nothing to prefetch and the
            # P2P weight reads drop out of the step entirely.
            sel = plan.experts_to_copy[self._op.cfg.rank].contiguous()
            for pool, _flat, _sh in pools:
                if pool is not None:
                    pool.prefetch(sel)

        slot_ids = self._op.row_slot_ids()
        out = self._op.get_expert_output_buffer()
        act = activation if activation is not None else ActivationType.Silu
        qt = quant_type if quant_type is not None else QuantType.No

        def experts(lo, hi, seg, num_local_tokens=None):
            """One fused_moe over rows[lo:hi] against a ``seg``-slot slice."""
            if hi <= lo:
                return
            n = hi - lo
            fused_out = fused_moe(
                rows[lo:hi],
                self._slice(pools[0], seg),
                self._slice(pools[1], seg),
                torch.ones((n, 1), dtype=torch.float32, device=rows.device),
                slot_ids[lo:hi],
                None,
                act,
                quant_type=qt,
                w1_scale=self._slice(pools[2], seg),
                w2_scale=self._slice(pools[3], seg),
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                num_local_tokens=num_local_tokens,
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
                bias1=self._slice(pools[4], seg) if pools[4][0] is not None else bias1,
                bias2=self._slice(pools[5], seg) if pools[5][0] is not None else bias2,
                dtype=dtype if dtype is not None else rows.dtype,
                **(extra_kwargs or {}),
            )
            out[lo:hi].copy_(fused_out)

        epn = self._op.cfg.num_experts_per_rank
        if not self._op.needs_split():
            # Decode: nothing migrated, so one call covers every row and the
            # row count stays on device -- no host sync anywhere in the step.
            experts(0, out.shape[0], (0, epn), self._op.valid_rows())
            return out
        home_end, total, nb = self._op.expert_call_split()
        experts(0, home_end, (0, epn))
        experts(home_end, total, (epn, epn + nb))
        return out

    ADOPTED = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_bias",
        "w2_bias",
    )

    def adopt_weights(self, layer) -> None:
        """Move this layer's expert weights into the symmetric heap in place.

        Peers prefetch migrated experts by reading the owner's weights over
        P2P, and ``shmem_ptr_p2p`` can only translate addresses inside the
        symmetric heap -- an ordinary ``torch`` parameter is unreachable.  So
        the weights have to live there.  Copying them in would double the MoE
        weights (~103 GB per rank on V4-Pro), so instead each parameter is
        *rebound* onto the pool's home segment and its original storage is
        dropped.  Steady-state cost is then the weights themselves plus ``B``
        migration slots -- the same shape of cost EPLB pays for its redundant
        replicas, which it gets by sizing the parameter as
        ``num_experts + num_redundant`` up front.

        Must run after ``process_weights_after_loading``: that hook replaces
        ``w13_weight.data`` with the shuffled, quantised tensor, and adopting
        before it would leave the pool holding weights nothing reads and the
        shuffled copy back outside the heap.
        """
        if self._pools is not None:
            return
        pools = []
        for name in self.ADOPTED:
            param = getattr(layer, name, None)
            entry = (None, False, False)
            if param is not None and param.data is not None:
                pool, flat = self._pool_for(param.data)
                # fused_moe reads the pre-shuffled layout off an *attribute*
                # (`isShuffled = getattr(w1, "is_shuffled", False)`), which a
                # plain slice of the pool does not inherit. Losing it makes the
                # kernel read shuffled bytes as unshuffled -- no error, just
                # wrong numbers -- so carry it across explicitly.
                shuffled = bool(getattr(param, "is_shuffled", False))
                # Rebinding frees the original storage, so peak overshoot is
                # one tensor rather than a second copy of the whole model.
                param.data = pool.home.reshape(param.data.shape)
                entry = (pool, flat, shuffled)
            pools.append(entry)
        self._pools = tuple(pools)

    def _pool_for(self, t):
        """Allocate one symmetric ``(epn + B, ...)`` pool and stage ``t`` in.

        Weights arrive as ``[epn, ...]``. Block scales do not: moe_shuffle_scale
        flattens them to 2-D (``[epn * rows_per_expert, cols]``), so the expert
        dimension is folded back out before pooling and folded in again on the
        way to fused_moe. Anything that is neither is a real mismatch, not a
        layout to guess at.
        """
        from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

        cfg = self._op.cfg
        epn = cfg.num_experts_per_rank
        slots = self._op.plan_config.prefetch_slots

        flat = t.shape[0] != epn
        if flat:
            if t.shape[0] % epn != 0:
                raise ValueError(
                    f"cannot index {tuple(t.shape)} by expert: leading dim is "
                    f"neither {epn} nor a multiple of it"
                )
            view = t.reshape(epn, t.shape[0] // epn, *t.shape[1:])
        else:
            view = t
        pool = MoonEPWeightPool(
            rank=cfg.rank,
            world_size=cfg.world_size,
            experts_per_rank=epn,
            prefetch_slots=slots,
            weight_shape=tuple(view.shape[1:]),
            dtype=t.dtype,
            block_num=cfg.dispatch_block_num,
        )
        pool.stage_home(view.contiguous())
        logger.info(
            "MoonEP adopted %s%s into the symmetric heap: %.3f GB for "
            "%d home + %d migration slots",
            tuple(t.shape),
            t.dtype,
            (epn + slots) * view[0].numel() * t.element_size() / 1e9,
            epn,
            slots,
        )
        return pool, flat

    @staticmethod
    def _slice(entry, seg):
        """Slots ``seg`` of a pool, in the shape fused_moe expects.

        Every expert in the returned tensor must be routed to by at least one
        row -- see ``expert_call_split``.
        """
        pool, flat, shuffled = entry
        if pool is None:
            return None
        lo, hi = seg
        t = pool.pool[lo:hi]
        t = t.reshape(-1, *t.shape[2:]) if flat else t
        if shuffled:
            t.is_shuffled = True
        return t

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        num_token = topk_ids.shape[0]
        # MoonEP's combine reads the expert-output buffer in place; the experts
        # kernel must have written into get_expert_output_buffer(), otherwise
        # peers would gather a stale or unrelated allocation.
        result = self._op.combine_grouped(fused_expert_output)
        return result[:num_token]

