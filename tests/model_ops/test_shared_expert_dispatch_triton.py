"""The fused Triton path must match the torch reference bit-exactly. Needs a GPU."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.eplb import (
    _map_record_and_dispatch_torch,
    eplb_map_record_and_dispatch,
)
from atom.model_ops.fused_moe.expert_layout import SharedExpertDispatchLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton path needs a GPU"
)


def _layer(ep_rank: int):
    # layer_id=None -> no-EPLB branch.
    return SimpleNamespace(ep_rank=ep_rank, layer_id=None)


@pytest.mark.parametrize("ep_rank", [0, 3, 7])
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_triton_matches_torch_reference(ep_rank, num_tokens):
    torch.manual_seed(1234 + ep_rank + num_tokens)
    num_routed_physical, ep_size, topk = 256, 8, 8
    layout = SharedExpertDispatchLayout(
        num_routed_physical=num_routed_physical, ep_size=ep_size, num_shared=1
    )
    layer = _layer(ep_rank)

    ids = torch.randint(
        0, num_routed_physical, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    # -1 sentinel: floor division sends it to -2 without the in-range guard.
    ids[torch.rand_like(ids, dtype=torch.float) < 0.1] = -1
    weights = torch.rand((num_tokens, topk), dtype=torch.float32, device="cuda")

    w_ref, ids_ref = _map_record_and_dispatch_torch(
        layer, weights.clone(), ids.clone(), layout, 0.4
    )
    w_got, ids_got = eplb_map_record_and_dispatch(
        layer, weights.clone(), ids.clone(), layout, 0.4
    )

    assert ids_got.shape == (num_tokens, topk + 1)
    assert torch.equal(ids_got, ids_ref)
    assert torch.equal(w_got, w_ref)


def test_shared_column_is_this_ranks_constant():
    """The point of the layout: one id per rank, resolving to that rank itself."""
    layout = SharedExpertDispatchLayout(
        num_routed_physical=256, ep_size=8, num_shared=1
    )
    ids = torch.randint(0, 256, (32, 8), dtype=torch.int32, device="cuda")
    weights = torch.rand((32, 8), dtype=torch.float32, device="cuda")

    for ep_rank in range(8):
        _, out_ids = eplb_map_record_and_dispatch(
            _layer(ep_rank), weights.clone(), ids.clone(), layout, 0.4
        )
        shared_col = out_ids[:, -1]
        expected = layout.shared_dispatch_id(ep_rank)
        assert torch.all(shared_col == expected), (ep_rank, shared_col[:4], expected)
        # 33*r + 32 -- the slot right after this rank's 32 routed ones.
        assert expected == 33 * ep_rank + 32


def test_routed_ids_land_on_the_owning_rank():
    """Dispatch ids must resolve to the same rank MoRI would compute."""
    layout = SharedExpertDispatchLayout(
        num_routed_physical=256, ep_size=8, num_shared=1
    )
    ids = torch.arange(256, dtype=torch.int32, device="cuda").reshape(32, 8)
    weights = torch.zeros((32, 8), dtype=torch.float32, device="cuda")

    _, out_ids = eplb_map_record_and_dispatch(_layer(0), weights, ids, layout, 0.4)
    routed = out_ids[:, :8].flatten()
    for physical_id, dispatch_id in enumerate(routed.tolist()):
        # mori: destPe = destExpert / numExpertPerRank (internode.hpp).
        assert dispatch_id // layout.slots_per_rank == physical_id // 32
        assert dispatch_id == layout.routed_to_dispatch(physical_id)


def test_empty_batch_matches_torch():
    """ntok==0 takes the torch early-out; shapes must still line up."""
    layout = SharedExpertDispatchLayout(
        num_routed_physical=256, ep_size=8, num_shared=1
    )
    layer = _layer(ep_rank=2)
    ids = torch.empty((0, 8), dtype=torch.int32, device="cuda")
    weights = torch.empty((0, 8), dtype=torch.float32, device="cuda")

    w_got, ids_got = eplb_map_record_and_dispatch(layer, weights, ids, layout, 0.4)

    assert ids_got.shape == (0, 9)
    assert w_got.shape == (0, 9)
