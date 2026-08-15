from types import SimpleNamespace

import torch

import atom.model_ops.moe as moe_module
from atom.model_ops.fused_moe.expert_layout import SharedExpertDispatchLayout
from atom.model_ops.moe import FusedMoE, FusedMoEMethodBase


def _dispatch_layer(*, ep_rank: int = 1) -> SimpleNamespace:
    layout = SharedExpertDispatchLayout(eplb_num_physical=8, ep_size=2, num_shared=1)
    return SimpleNamespace(
        fuse_shared_into_dispatch=True,
        num_fused_shared_experts=1,
        global_num_experts=8,
        num_redundant_experts=0,
        routed_scaling_factor=2.0,
        shared_dispatch_layout=layout,
        ep_rank=ep_rank,
        use_ep=True,
        expert_map=torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3, 4, -1]),
        expert_mask=torch.empty(0, dtype=torch.int32),
    )


def test_prepare_dispatch_topk_is_backend_neutral(monkeypatch):
    layer = _dispatch_layer(ep_rank=1)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: False
    )
    routed_ids = torch.tensor([[0, 4], [3, -1]], dtype=torch.int32)
    routed_weights = torch.tensor([[0.7, 0.3], [0.6, 0.0]])

    weights, ids = FusedMoE.prepare_dispatch_topk(layer, routed_weights, routed_ids)

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
        assert received_layer is layer
        assert received_ids.shape[1] == 2
        return received_ids

    monkeypatch.setattr(FusedMoE, "select_experts", fake_select_experts)
    monkeypatch.setattr(moe_module, "eplb_map_and_record_fused", fake_map_and_record)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: True
    )
    layer.prepare_dispatch_topk = lambda weights, ids: FusedMoE.prepare_dispatch_topk(
        layer, weights, ids
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
