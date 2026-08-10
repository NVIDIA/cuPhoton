# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ExchangeBlock must reproduce the authors' reference exchange byte-for-byte.

The reference (reiwanol/nodiff triplet_nodiff_transformer.py @ 274da55) is
deliberately NOT a permutation: on exchanged channels it drops one stream and
duplicates another. These tests pin parity with the reference formula so the
faithful reproduction cannot silently drift (e.g. to a 3-cycle).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cuphoton.xscan.model import ExchangeBlock  # noqa: E402


def _distinct_streams(n_streams: int, channels: int) -> list[torch.Tensor]:
    # x_k[:, c] holds a unique value so a per-channel compare detects any
    # dropped/duplicated stream.
    streams = []
    for k in range(n_streams):
        base = torch.arange(channels, dtype=torch.float32) + 100.0 * (k + 1)
        streams.append(
            base.view(1, channels, 1, 1).expand(1, channels, 2, 2).clone()
        )
    return streams


def _reference_triplet(
    x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, p_swap: int
):
    # Literal per-channel reproduction of the reference ExchangeBlock:
    #   c % (2*p_swap) == 0  (mask_13): (x3, x3, x2)
    #   c % p_swap == 0 else (mask_12): (x2, x1, x1)
    #   otherwise:                      (x1, x2, x3)
    out1, out2, out3 = (torch.zeros_like(t) for t in (x1, x2, x3))
    channels = x1.shape[1]
    for c in range(channels):
        if c % (p_swap * 2) == 0:
            out1[:, c], out2[:, c], out3[:, c] = x3[:, c], x3[:, c], x2[:, c]
        elif c % p_swap == 0:
            out1[:, c], out2[:, c], out3[:, c] = x2[:, c], x1[:, c], x1[:, c]
        else:
            out1[:, c], out2[:, c], out3[:, c] = x1[:, c], x2[:, c], x3[:, c]
    return out1, out2, out3


def test_pair_exchange_is_a_per_channel_involution() -> None:
    channels = 8
    block = ExchangeBlock(p_swap=2)
    x1, x2 = _distinct_streams(2, channels)
    o1, o2 = block(x1, x2)
    for c in range(channels):
        if c % 2 == 0:
            assert o1[0, c, 0, 0] == x2[0, c, 0, 0]
            assert o2[0, c, 0, 0] == x1[0, c, 0, 0]
        else:
            assert o1[0, c, 0, 0] == x1[0, c, 0, 0]
            assert o2[0, c, 0, 0] == x2[0, c, 0, 0]


@pytest.mark.parametrize("p_swap", [0, -1, 1.5, True])
def test_exchange_rejects_invalid_swap_interval(p_swap: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ExchangeBlock(p_swap=p_swap)


@pytest.mark.parametrize("p_swap", [1, 2, 3])
def test_triplet_exchange_matches_reference(p_swap: int) -> None:
    channels = 12
    block = ExchangeBlock(p_swap=p_swap)
    streams = _distinct_streams(3, channels)
    got = block(*streams)
    want = _reference_triplet(*streams, p_swap=p_swap)
    for produced, expected in zip(got, want, strict=True):
        assert torch.equal(produced, expected)


def test_triplet_is_reference_faithful_not_a_permutation() -> None:
    # Pin the exact non-permutation semantics so a future "fix" to a
    # permutation is caught: p_swap=2 -> c%4==0 emits (x3,x3,x2), the
    # remaining even channels emit (x2,x1,x1).
    channels = 8
    block = ExchangeBlock(p_swap=2)
    x1, x2, x3 = _distinct_streams(3, channels)
    o1, o2, o3 = block(x1, x2, x3)
    for c in range(channels):
        got = (o1[0, c, 0, 0], o2[0, c, 0, 0], o3[0, c, 0, 0])
        if c % 4 == 0:
            assert got == (x3[0, c, 0, 0], x3[0, c, 0, 0], x2[0, c, 0, 0])
        elif c % 2 == 0:
            assert got == (x2[0, c, 0, 0], x1[0, c, 0, 0], x1[0, c, 0, 0])
        else:
            assert got == (x1[0, c, 0, 0], x2[0, c, 0, 0], x3[0, c, 0, 0])
