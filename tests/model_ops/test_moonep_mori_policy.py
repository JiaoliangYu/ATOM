# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contract tests for MoonEP planning layered over MoRI v2 dispatch."""

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs AITER's MoE interfaces")

import torch

import atom.model_ops.fused_moe.mori_v2_prepare_finalize as mpf


class _FakeOp:
    def __init__(self, experts_per_rank: int):
        self.cfg = SimpleNamespace(num_experts_per_rank=experts_per_rank)
        self.calls = []

    def dispatch(self, hidden, weights, scales, indices, *, return_routing):
        self.calls.append((hidden, weights, scales, indices, return_routing))
        return hidden, weights, None, indices, torch.tensor(0), object()


class _FakePolicy:
    def __init__(self, plan):
        self.result = plan
        self.calls = 0

    def plan(self, ids):
        self.calls += 1
        return self.result


def _policy_prepare_finalize(*, prefill: bool):
    obj = mpf.MoonEPPolicyMoriV2PrepareAndFinalize.__new__(
        mpf.MoonEPPolicyMoriV2PrepareAndFinalize
    )
    obj._num_experts = 8
    obj._experts_per_rank = 4
    obj._prefetch_slots = 1
    obj._rank = 0
    obj._world_size = 2
    obj._routing = None
    obj._active_op = None
    obj._active_plan = None
    obj._active_is_prefill = False
    obj._prefill_op = _FakeOp(5)
    obj._decode_op = _FakeOp(4)
    obj._is_prefill = lambda: prefill
    return obj


def test_virtual_masks_map_each_region_to_its_own_weight_rows():
    home, prefetched = mpf._make_virtual_expert_masks(
        rank=1,
        world_size=2,
        experts_per_rank=4,
        prefetch_slots=2,
        device=torch.device("cpu"),
    )
    assert home.tolist() == [0] * 6 + [1, 1, 1, 1, 0, 0]
    assert prefetched.tolist() == [0] * 10 + [1, 1]
    assert (home.cumsum(0) - 1)[6:10].tolist() == [0, 1, 2, 3]
    assert (prefetched.cumsum(0) - 1)[10:12].tolist() == [0, 1]


def test_prefill_dispatch_uses_planned_ids_and_preserves_router_weights():
    obj = _policy_prepare_finalize(prefill=True)
    logical = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    planned = torch.tensor([[0, 9], [2, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32)
    policy = _FakePolicy(
        SimpleNamespace(
            planned_topk_ids=planned,
            num_experts_per_rank=5,
            experts_to_copy=torch.tensor([[-1], [1]], dtype=torch.int32),
        )
    )
    obj._prefill_policy = lambda ids: policy

    result = obj.prepare(
        torch.randn(2, 8),
        weights,
        logical,
        8,
        None,
        False,
        None,
    )

    assert policy.calls == 1
    assert len(obj._prefill_op.calls) == 1
    _, sent_weights, _, sent_ids, return_routing = obj._prefill_op.calls[0]
    assert torch.equal(sent_ids, planned)
    assert torch.equal(sent_weights, weights)
    assert return_routing is True
    assert torch.equal(result[3], planned)
    assert torch.equal(result[4], weights)


def test_decode_is_owner_only_and_never_constructs_prefill_policy():
    obj = _policy_prepare_finalize(prefill=False)
    logical = torch.tensor([[0, 7], [3, 4]], dtype=torch.int64)
    weights = torch.tensor([[0.8, 0.2], [0.55, 0.45]], dtype=torch.float32)
    decode = _FakePolicy(
        SimpleNamespace(
            planned_topk_ids=logical.to(torch.int32), num_experts_per_rank=4
        )
    )
    obj._decode_policy = decode
    obj._prefill_policy = lambda ids: pytest.fail("decode entered PrefillPolicy")

    obj.prepare(torch.randn(2, 8), weights, logical, 8, None, False, None)

    assert decode.calls == 1
    assert len(obj._decode_op.calls) == 1
    assert torch.equal(obj._decode_op.calls[0][3], logical.to(torch.int32))


class _FakePool:
    def __init__(self, home, prefetched):
        self.home = home
        self.prefetched = prefetched
        self.selected = None

    def prefetch(self, selected):
        self.selected = selected.clone()


def test_prefill_experts_use_real_weights_for_home_and_prefetch(monkeypatch):
    obj = _policy_prepare_finalize(prefill=True)
    obj._active_is_prefill = True
    obj._active_plan = SimpleNamespace(
        experts_to_copy=torch.tensor([[6], [-1]], dtype=torch.int32)
    )
    obj._virtual_masks = {}

    home_w1 = torch.empty(4, 8, 8)
    home_w2 = torch.empty(4, 8, 8)
    pf_w1 = torch.empty(1, 8, 8)
    pf_w2 = torch.empty(1, 8, 8)
    p1 = _FakePool(home_w1, pf_w1)
    p2 = _FakePool(home_w2, pf_w2)
    obj._pools = (
        (p1, False, False),
        (p2, False, False),
        (None, False, False),
        (None, False, False),
        (None, False, False),
        (None, False, False),
    )

    calls = []

    def fake_fused_moe(rows, w1, w2, weights, ids, mask, **kwargs):
        calls.append((w1, w2, weights, ids, mask, kwargs))
        return torch.full_like(rows, len(calls), dtype=torch.float32)

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe_module, "fused_moe", fake_fused_moe)
    rows = torch.randn(3, 8)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float32)
    ids = torch.tensor([[0, 4], [1, 2], [4, 3]], dtype=torch.int32)
    out = obj.run_dispatched_experts(
        rows,
        home_w1,
        home_w2,
        topk_weights=weights,
        topk_ids=ids,
        expert_mask=None,
        num_local_tokens=None,
    )

    assert len(calls) == 2
    assert calls[0][0] is home_w1
    assert calls[1][0] is pf_w1
    assert calls[0][2] is weights and calls[1][2] is weights
    assert calls[0][3] is ids and calls[1][3] is ids
    assert calls[0][4].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert calls[1][4].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert p1.selected.tolist() == [6]
    assert p2.selected.tolist() == [6]
    assert torch.equal(out, torch.full_like(rows, 3, dtype=torch.float32))
