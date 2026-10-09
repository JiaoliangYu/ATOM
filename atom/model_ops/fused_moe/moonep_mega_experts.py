# SPDX-License-Identifier: Apache-2.0
"""MoonEP expert balancing inside MegaMoE v2.

Each rank keeps its ``EPR`` resident experts plus ``B`` prefetch slots in one
P2P-readable ``[EPR + B]`` weight window.  Large prefills run a MegaMoEV2
instance over ``R * (EPR + B)`` virtual experts: its prepare kernel already
exchanges the route histogram, so it places hot experts' routes on other
ranks there (MoonEP) and those ranks pull the experts' weights into their
slots before Stage1.  Smaller prefills keep every expert on its owner in the
same instance.  Unified decode runs the plain ``R * EPR`` decode-capacity
instance, which keeps Mega's fixed-slot decode path (exactly 48 experts per
rank).
"""

import logging

import torch

from atom.config import get_current_atom_config
from atom.model_ops.fused_moe.flydsl_mega_experts import (
    _MEGA_DECODE_MTPR,
    _enable_mega_pad_row_mask,
    _select_decode_mtpr,
    get_or_build_mega_moe,
)
from atom.plugin import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

_ADOPTED = ("_mega_w1", "_mega_w1_scale", "_mega_w2", "_mega_w2_scale")


class MoonEPMegaExperts:
    """MoonEP over MegaMoE as a whole-pipeline ``fused_experts`` backend.

    Installed where ``MegaFusedExperts`` would be.  Not an ``nn.Module`` for
    the same reason: it holds the layer, which would make a module cycle.
    """

    @classmethod
    def for_layer(cls, layer: torch.nn.Module, moe, *, model_dim: int, inter_dim: int):
        """Take over ``layer``'s Mega weights on this rank of the EP group."""
        from aiter.dist.parallel_state import get_ep_group

        if getattr(get_current_atom_config(), "eplb_enable", False):
            raise ValueError("ATOM_ENABLE_MOONEP=1 cannot be combined with EPLB")
        # Reading all2all_manager initializes the mori symmetric heap that the
        # weight pools are mapped from.
        ep = get_ep_group()
        am = ep.device_communicator.all2all_manager
        logger.info(
            "MoonEP active over MegaMoE: rank=%d world=%d prefetch_slots=%d",
            am.rank,
            am.world_size,
            envs.MOONEP_PREFETCH_SLOTS,
        )
        return cls(
            layer,
            model_dim=model_dim,
            inter_dim=inter_dim,
            mtpr=moe.max_num_tokens,
            rank=int(am.rank),
            world_size=int(am.world_size),
            num_experts=moe.num_experts,
            prefetch_slots=envs.MOONEP_PREFETCH_SLOTS,
            group=ep.cpu_group,
        )

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        model_dim: int,
        inter_dim: int,
        mtpr: int,
        rank: int,
        world_size: int,
        num_experts: int,
        prefetch_slots: int,
        group=None,
        quant: str = "a8w4",
    ) -> None:
        """``rank``/``world_size`` are positions in ``group``, the EP CPU process
        group the weight pools bootstrap over."""
        if world_size not in (4, 8):
            raise ValueError(f"MoonEP supports EP4 and EP8, got EP{world_size}")
        # Mega allocates on cuda:<rank>, so the EP rank must be the local device.
        if torch.cuda.current_device() != rank:
            raise ValueError(
                f"MoonEP needs EP rank {rank} on cuda:{rank}, "
                f"running on cuda:{torch.cuda.current_device()}"
            )
        if num_experts % world_size:
            raise ValueError("MoonEP requires num_experts divisible by world_size")
        if not 0 < prefetch_slots <= 64:
            raise ValueError(
                f"MoonEP prefetch_slots must be in [1, 64], got {prefetch_slots}"
            )
        self._layer = layer
        self._model_dim = model_dim
        self._inter_dim = inter_dim
        self._mtpr = mtpr
        self._rank = rank
        self._world_size = world_size
        self._experts_per_rank = num_experts // world_size
        self._prefetch_slots = prefetch_slots
        self._quant = quant
        self._group = group
        self._slot_state = None
        self._mask_pad_rows = _enable_mega_pad_row_mask(mtpr)
        self._pool, self._parts = self._adopt(layer)

    def _adopt(self, layer: torch.nn.Module):
        """Move this layer's Mega weights and scales into one P2P-readable pool
        with a ``[EPR + B]`` window per tensor."""

        from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

        epn = self._experts_per_rank
        params, homes, parts, layouts = [], [], [], []
        for name in _ADOPTED:
            param = getattr(layer, name, None)
            if param is None or param.data is None:
                raise ValueError(f"MoonEP needs layer.{name}")
            resident = param.data
            # Scales may be stored flat; the pool still needs them expert-major.
            flat = resident.shape[0] != epn
            if flat and resident.shape[0] % epn:
                raise ValueError(
                    f"cannot index {name} {tuple(resident.shape)} by expert: "
                    f"leading dim is neither {epn} nor a multiple of it"
                )
            home = resident.reshape(epn, -1, *resident.shape[1:]) if flat else resident
            params.append(param)
            homes.append(home)
            parts.append((tuple(home.shape[1:]), resident.dtype))
            layouts.append((flat, bool(getattr(param, "is_shuffled", False))))
        pool = MoonEPWeightPool(
            rank=self._rank,
            world_size=self._world_size,
            experts_per_rank=epn,
            prefetch_slots=self._prefetch_slots,
            parts=parts,
            group=self._group,
        )
        pool.stage_home(homes)
        for param, home in zip(params, pool.home):
            param.data = home.reshape(param.data.shape)
        logger.info(
            "MoonEP adopted %s: %d resident + %d prefetch slots",
            ", ".join(f"{n}{list(s)} {d}" for n, (s, d) in zip(_ADOPTED, parts)),
            epn,
            self._prefetch_slots,
        )
        return pool, tuple(layouts)

    def _view(self, index: int, *, home: bool):
        flat, shuffled = self._parts[index]
        window = (self._pool.home if home else self._pool.local)[index]
        if flat:
            window = window.reshape(-1, *window.shape[2:])
        if shuffled:
            window.is_shuffled = True
        return window

    def _should_balance(self, rows: int) -> bool:
        """Balance prefills whose largest DP rank reaches MOONEP_MIN_PLAN_TOKENS.

        Every input is agreed across the DP group, so all ranks take the same
        instance and the same prepare variant.
        """

        context = get_forward_context().context
        if context is not None and not (
            context.is_prefill or not context.running_tokens_are_unified
        ):
            return False
        threshold = envs.MOONEP_MIN_PLAN_TOKENS
        if threshold <= 0:
            return True
        dp_metadata = getattr(get_forward_context(), "dp_metadata", None)
        tokens = rows if dp_metadata is None else dp_metadata.max_tokens_across_dp
        return tokens >= threshold

    def _mega(self, *, wide: bool, mtpr: int, topk: int):
        """The instance for one window, with this layer's weights bound."""

        experts_per_rank = self._experts_per_rank + (
            self._prefetch_slots if wide else 0
        )
        w1, w1_scale, w2, w2_scale = (
            self._view(i, home=not wide) for i in range(len(_ADOPTED))
        )
        return get_or_build_mega_moe(
            rank=self._rank,
            world_size=self._world_size,
            model_dim=self._model_dim,
            inter_dim=self._inter_dim,
            experts=self._world_size * experts_per_rank,
            topk=topk,
            quant=self._quant,
            mtpr=mtpr,
            swiglu_limit=float(getattr(self._layer, "swiglu_limit", 0.0)),
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            moonep_slots=self._prefetch_slots if wide else 0,
        )

    def _decode_mega(self, topk: int):
        """The resident-window decode instance when this pass may use it.

        Built on every rank in the same order as the wide one, whatever the
        pass, so peers never diverge in symmetric allocations.
        """

        if (
            not envs.ATOM_MEGA_DECODE_FAST_PATH
            or is_plugin_mode()
            or self._mtpr <= _MEGA_DECODE_MTPR
            or self._world_size != 8
            or self._experts_per_rank != 48
        ):
            return None
        from atom.utils.tbo.ubatching import tbo_active

        if tbo_active():
            return None
        narrow = self._mega(wide=False, mtpr=_MEGA_DECODE_MTPR, topk=topk)
        context = get_forward_context().context
        if (
            _select_decode_mtpr(self._mtpr, context, tbo_active=False)
            != _MEGA_DECODE_MTPR
        ):
            return None
        return narrow

    def _fill_slots(self, mega) -> None:
        """Copy the experts prepare placed on this rank into its slots,
        skipping those a slot already holds."""

        placed, prev = mega.moonep_slot_tables()
        self._pool.prefetch(placed[self._rank], prev[self._rank])

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation=None,
        apply_router_weight_on_input: bool = False,
        **_ignored,
    ) -> torch.Tensor:
        from aiter import ActivationType

        if apply_router_weight_on_input:
            raise NotImplementedError(
                "mega does not support apply_router_weight_on_input=True"
            )
        if activation is not None and activation != ActivationType.Silu:
            raise NotImplementedError(
                f"mega hardcodes SwiGLU; got activation={activation}"
            )
        rows = int(hidden_states.shape[0])
        if rows > self._mtpr:
            raise ValueError(f"[moonep-mega] rows={rows} exceeds mtpr={self._mtpr}")

        topk = int(topk_ids.shape[1])
        mega = self._mega(wide=True, mtpr=self._mtpr, topk=topk)
        decode = self._decode_mega(topk)
        balance = self._should_balance(rows)
        kwargs = {}
        if balance or decode is None:
            if self._slot_state is None:
                self._slot_state = mega.new_moonep_slot_state()
            mega.bind_moonep_slot_state(self._slot_state)
            kwargs["moonep_balance"] = balance
            if balance:
                kwargs["after_prepare"] = lambda: self._fill_slots(mega)
        else:
            mega = decode

        wts = topk_weights.to(torch.float32).contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        pad_rows = None
        if self._mask_pad_rows:
            from atom.utils.forward_context import step_pad_rows

            pad_rows = step_pad_rows(rows)
        if pad_rows is not None:
            # Prepare neither counts nor places a -1 slot; combine zeroes it.
            ids = torch.where(pad_rows, -1, ids)
            kwargs["mask_invalid_slots"] = True
        with torch.inference_mode(False), torch.no_grad():
            return mega.forward(hidden_states.contiguous(), wts, ids, **kwargs)
