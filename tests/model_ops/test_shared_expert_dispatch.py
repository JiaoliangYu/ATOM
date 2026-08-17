from types import SimpleNamespace

import pytest
import torch

import atom.model_ops.eplb as eplb_module
import atom.model_ops.moe as moe_module
import atom.model_ops.topK as topK_module
from atom.model_ops.fused_moe.expert_layout import SharedExpertDispatchLayout
from atom.model_ops.moe import FusedMoE, FusedMoEMethodBase, FusedMoEParallelConfig
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)


def _dispatch_layer(*, ep_rank: int = 1) -> SimpleNamespace:
    layout = SharedExpertDispatchLayout(num_routed_physical=8, ep_size=2, num_shared=1)
    ns = SimpleNamespace(
        fuse_shared_into_dispatch=True,
        num_fused_shared_experts=1,
        global_num_experts=8,
        num_redundant_experts=0,
        routed_scaling_factor=2.0,
        shared_dispatch_layout=layout,
        ep_rank=ep_rank,
        layer_id=None,
        use_ep=True,
        expert_map=torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3, 4, -1]),
        expert_mask=torch.empty(0, dtype=torch.int32),
    )
    return ns


def test_map_record_and_dispatch_is_backend_neutral(monkeypatch):
    layer = _dispatch_layer(ep_rank=1)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: False
    )
    # No GPU here; pin to the torch reference.
    monkeypatch.setattr(eplb_module, "_EPLB_HAS_TRITON", False)
    routed_ids = torch.tensor([[0, 4], [3, -1]], dtype=torch.int32)
    routed_weights = torch.tensor([[0.7, 0.3], [0.6, 0.0]])

    weights, ids = FusedMoE.map_record_and_dispatch(layer, routed_weights, routed_ids)

    assert ids.tolist() == [[0, 5, 9], [3, -1, 9]]
    assert torch.equal(weights[:, :2], routed_weights)
    assert weights[:, 2].tolist() == [0.5, 0.5]


def test_rebuild_expert_mask_preserves_dispatch_space_after_eplb_update():
    layer = _dispatch_layer(ep_rank=1)

    FusedMoE.rebuild_expert_mask(layer)

    assert layer.expert_mask.numel() == 10
    assert torch.nonzero(layer.expert_mask, as_tuple=False).flatten().tolist() == [
        5,
        6,
        7,
        8,
        9,
    ]


def test_select_experts_keeps_shared_out_of_router_and_eplb(monkeypatch):
    layer = _dispatch_layer(ep_rank=1)
    captured = {}
    routed_weights = torch.tensor([[0.75, 0.25]])
    routed_ids = torch.tensor([[0, 4]], dtype=torch.int32)

    def fake_select_experts(**kwargs):
        captured["num_fused_shared_experts"] = kwargs["num_fused_shared_experts"]
        return routed_weights, routed_ids

    def fake_map_and_record(received_layer, received_ids):
        # EPLB must see routed ids only.
        assert received_layer is layer
        assert received_ids.shape[1] == 2
        return received_ids

    monkeypatch.setattr(FusedMoE, "select_experts", fake_select_experts)
    monkeypatch.setattr(eplb_module, "eplb_map_and_record_fused", fake_map_and_record)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: True
    )
    # Pins the ordering router -> EPLB -> shared, not the kernel arithmetic.
    monkeypatch.setattr(eplb_module, "_EPLB_HAS_TRITON", False)
    layer.map_record_and_dispatch = lambda weights, ids: (
        FusedMoE.map_record_and_dispatch(layer, weights, ids)
    )

    weights, ids = FusedMoEMethodBase.select_experts_with_record(
        object(),
        layer=layer,
        hidden_states=torch.empty(1, 4),
        router_logits=torch.empty(1, 8),
        top_k=2,
        renormalize=True,
    )

    assert captured["num_fused_shared_experts"] == 0
    assert ids.tolist() == [[0, 5, 9]]
    assert weights.tolist() == [[0.75, 0.25, 1.0]]


def _parallel_config(*, dp_size: int, use_ep: bool) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=8,
        dp_size=dp_size,
        ep_size=8 if use_ep else 1,
        tp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=use_ep,
        local_ep_size=8,
    )


@pytest.mark.parametrize(
    "dp_size, use_ep, has_mori, expected",
    [
        # EP without DP still runs the legacy AITER fusion: no all2all backend,
        # so there is no dispatch space to fold the shared expert into.
        (1, True, True, False),
        (8, True, True, True),
        (8, False, True, False),
        (8, True, False, False),
    ],
)
def test_ep_alone_does_not_imply_all2all(
    monkeypatch, dp_size, use_ep, has_mori, expected
):
    monkeypatch.setattr(moe_module, "_has_module", lambda name: has_mori)

    config = _parallel_config(dp_size=dp_size, use_ep=use_ep)

    assert config.use_all2all_kernels is expected


def _atom_config(*, dp_size: int, dp_attention: bool, switch: bool) -> SimpleNamespace:
    return SimpleNamespace(
        quant_config=SimpleNamespace(exclude_layers=[], quant_dtype=None),
        parallel_config=SimpleNamespace(data_parallel_size=dp_size),
        moe_ep_flatten_tp_across_dp=False,
        enable_dp_attention=dp_attention,
        fuse_shared_expert=switch,
    )


@pytest.mark.parametrize(
    "dp_size, dp_attention, switch, expected",
    [
        # Only the DP + MoRI + dp-attention case is gated by the switch; the
        # legacy AITER fusion must stay reachable everywhere else.
        (1, False, False, True),
        (8, True, False, False),
        (8, True, True, True),
    ],
)
def test_switch_only_gates_the_all2all_case(
    monkeypatch, dp_size, dp_attention, switch, expected
):
    monkeypatch.setattr(topK_module, "_has_module", lambda name: True)
    monkeypatch.setattr(
        topK_module,
        "get_current_atom_config",
        lambda: _atom_config(dp_size=dp_size, dp_attention=dp_attention, switch=switch),
    )

    enabled = is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(None)

    assert enabled is expected
