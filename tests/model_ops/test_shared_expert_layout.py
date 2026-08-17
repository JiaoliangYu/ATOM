from types import SimpleNamespace

import torch

from atom.model_ops.fused_moe.config import FusedMoEConfig
from atom.model_ops.fused_moe.expert_layout import SharedExpertDispatchLayout


def test_layout_separates_routed_and_dispatch_widths():
    layout = SharedExpertDispatchLayout(
        num_routed_physical=256, ep_size=8, num_shared=1
    )

    assert layout.routed_slots_per_rank == 32
    assert layout.slots_per_rank == 33
    assert layout.num_dispatch_slots == 264


def test_backend_config_widens_every_rank_block_by_the_shared_slot():
    parallel = SimpleNamespace(
        tp_size=1,
        dp_size=8,
        ep_size=8,
        tp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=True,
        use_mori_kernels=True,
    )
    config = FusedMoEConfig(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=7168,
        num_local_experts=33,
        num_fused_shared_experts=1,
        moe_parallel_config=parallel,
    )

    assert config.num_local_experts_dispatch == 33
    assert config.num_global_experts_dispatch == 264
    assert config.experts_per_token_dispatch == 9


def test_layout_preserves_routed_owner_and_assigns_local_shared_slot():
    layout = SharedExpertDispatchLayout(num_routed_physical=8, ep_size=2, num_shared=1)

    routed = torch.arange(8, dtype=torch.int32)
    dispatch = layout.routed_to_dispatch(routed)

    assert dispatch.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert [int(x // layout.slots_per_rank) for x in dispatch] == [
        int(x // layout.routed_slots_per_rank) for x in routed
    ]
    assert layout.shared_dispatch_id(0) == 4
    assert layout.shared_dispatch_id(1) == 9


def test_layout_appends_shared_without_rewriting_sentinels():
    layout = SharedExpertDispatchLayout(num_routed_physical=8, ep_size=2, num_shared=1)
    ids = torch.tensor([[0, 4], [3, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.0]])

    out_ids, out_weights = layout.apply_to_topk(
        ids, weights, ep_rank=1, shared_weight=0.5
    )

    assert out_ids.tolist() == [[0, 5, 9], [3, -1, 9]]
    assert torch.equal(out_weights[:, :2], weights)
    assert out_weights[:, 2].tolist() == [0.5, 0.5]
