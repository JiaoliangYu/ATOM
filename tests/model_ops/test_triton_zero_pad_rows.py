# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU coverage for the MegaMoE pad-row zeroing kernel."""

import pytest
import torch

pytest.importorskip("triton", reason="the row-zeroing helper is a Triton kernel")

from atom.model_ops.fused_moe.flydsl_mega_experts import zero_pad_rows_

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required",
)


@pytest.mark.parametrize("hidden", [17, 1024, 1025, 7168])
def test_zero_pad_rows_matches_reference_and_is_in_place(hidden):
    rows = 11
    out = torch.arange(rows * hidden, device="cuda", dtype=torch.float32).reshape(
        rows, hidden
    )
    out = out.to(torch.bfloat16)
    pad_rows = torch.tensor(
        [
            [False],
            [True],
            [False],
            [True],
            [True],
            [False],
            [False],
            [True],
            [False],
            [False],
            [True],
        ],
        device="cuda",
    )
    expected = out.clone()
    expected[pad_rows[:, 0]] = 0

    returned = zero_pad_rows_(out, pad_rows)
    torch.cuda.synchronize()

    assert returned is out
    assert torch.equal(out, expected)


def test_zero_pad_rows_uses_the_replayed_graph_mask():
    rows, hidden = 8, 1025
    source = torch.arange(rows, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    source = source.unsqueeze(1)
    source = source.expand(rows, hidden).contiguous()
    out = source.clone()
    pad_rows = torch.zeros((rows, 1), dtype=torch.bool, device="cuda")

    zero_pad_rows_(out, pad_rows)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        zero_pad_rows_(out, pad_rows)

    for first_pad_row in (3, 6):
        out.copy_(source)
        pad_rows.copy_(torch.arange(rows, device="cuda").unsqueeze(1) >= first_pad_row)
        graph.replay()
        torch.cuda.synchronize()

        expected = source.clone()
        expected[first_pad_row:] = 0
        assert torch.equal(out, expected)
