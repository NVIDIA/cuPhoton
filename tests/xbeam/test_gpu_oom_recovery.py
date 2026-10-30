# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio import gpu_execution
from cuphoton.xbeam.adapters.radio.models import SamplingKind
from cuphoton.xbeam.runtime_types import _StationBlocks


@pytest.mark.parametrize("band_count", [3, 5, 6, 7])
@pytest.mark.parametrize("fused_limit", [0, 2])
def test_packed_band_oom_splits_preserve_singleton_bands(
    monkeypatch, band_count, fused_limit
):
    class FakeOutOfMemoryError(MemoryError):
        pass

    xp = SimpleNamespace(
        __name__="cupy",
        cuda=SimpleNamespace(
            memory=SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError)
        ),
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=lambda: None
        ),
    )
    group = [
        (
            (0, index),
            SimpleNamespace(
                datastream=SimpleNamespace(
                    station_id=0, sampling=SamplingKind.REAL
                ),
                frequency=SimpleNamespace(channel_count=2),
                sample_rate_hz=8.0,
                band=SimpleNamespace(identifier=index),
            ),
        )
        for index in range(band_count)
    ]
    expected = {
        key: _StationBlocks(
            np.full((2, 2), key[1] + 1, dtype=np.complex64),
            np.full(2, 1.0),
            np.arange(2.0),
        )
        for key, _data in group
    }

    def fused(_job, selected, _model, _starts, _xp):
        if len(selected) < 2:
            raise ValueError("packed band front-end requires 2..65535 bands")
        if len(selected) > fused_limit:
            raise FakeOutOfMemoryError
        return {key: expected[key] for key, _data in selected}

    def single(_job, data, _model, _starts, _xp):
        return expected[(0, data.band.identifier)]

    monkeypatch.setattr(
        gpu_execution,
        "_ordered_model_runs",
        lambda *_args: [(None, slice(None))],
    )
    monkeypatch.setattr(
        gpu_execution, "_station_blocks_packed_real_group", fused
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks", single)

    actual = gpu_execution._station_blocks_packed_real_group_for_scan(
        None, group, None, np.arange(2.0), xp
    )

    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] is expected[key]


def test_packed_band_oom_releases_failed_buffers_before_retry(monkeypatch):
    class FakeOutOfMemoryError(MemoryError):
        pass

    failed_buffers = []

    def fused(*_args):
        # A front buffer can have been allocated before cuFFT runs out of
        # memory. The exception traceback owns it until the handler exits.
        front_buffer = np.empty(4)
        failed_buffers.append(weakref.ref(front_buffer))
        raise FakeOutOfMemoryError

    def check_released(*_args):
        assert failed_buffers
        assert all(buffer() is None for buffer in failed_buffers)

    xp = SimpleNamespace(
        __name__="cupy",
        cuda=SimpleNamespace(
            memory=SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError)
        ),
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=check_released
        ),
    )
    data = SimpleNamespace(
        datastream=SimpleNamespace(station_id=0, sampling=SamplingKind.REAL),
        frequency=SimpleNamespace(channel_count=2),
        sample_rate_hz=8.0,
    )
    monkeypatch.setattr(
        gpu_execution,
        "_ordered_model_runs",
        lambda *_args: [(None, slice(None))],
    )
    monkeypatch.setattr(
        gpu_execution, "_station_blocks_packed_real_group", fused
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks", check_released)

    gpu_execution._station_blocks_packed_real_group_for_scan(
        None, [((0, 0), data), ((0, 1), data)], None, np.arange(2.0), xp
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("failure", [None, "eligibility", "weights"])
def test_chunk_spectra_released_before_retry_and_next_chunk(
    monkeypatch, packed, failure
):
    class FakeOutOfMemoryError(MemoryError):
        pass

    buffers = []
    starts_seen = []
    products_seen = []
    failed = False

    def assert_released():
        assert all(buffer() is None for buffer in buffers)

    def station_blocks(_job, selected, _scan, starts, _xp, _plans):
        # A retry and a new successful chunk must fit without the old spectra.
        if selected[0][0] == (0, 0):
            assert_released()
            starts_seen.append(tuple(starts))
        result = {}
        for key, _data in selected:
            spectrum = np.full((len(starts), 4), key[1] + 1, np.complex64)
            weight = np.ones(len(starts))
            buffers.extend((weakref.ref(spectrum), weakref.ref(weight)))
            result[key] = _StationBlocks(spectrum, weight, np.arange(4.0))
        return result

    def single(job, data, scan, starts, xp, plan):
        return station_blocks(
            job, [(data.key, data)], scan, starts, xp, plan
        )[data.key]

    def maybe_fail(where):
        nonlocal failed
        if failure == where and not failed:
            failed = True
            temporary = np.empty(4)
            buffers.append(weakref.ref(temporary))
            raise FakeOutOfMemoryError

    def eligibility(blocks, **_kwargs):
        maybe_fail("eligibility")
        return blocks

    def weights(blocks, **_kwargs):
        maybe_fail("weights")
        return blocks.weight

    def launch(groups, *_args, product_kind):
        for products in groups.values():
            for product in products:
                products_seen.append(
                    (product_kind, product.blocks_a.spectrum.copy())
                )

    xp = SimpleNamespace(
        int32=np.int32,
        int64=np.int64,
        cuda=SimpleNamespace(
            memory=SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError)
        ),
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=assert_released
        ),
    )
    scan = SimpleNamespace(identifier="scan")
    band_data = {
        (0, band): SimpleNamespace(
            key=(0, band),
            sample_rate_hz=8.0,
            payload_progress=None,
            payload_plan=None,
            device_payload=object(),
            source_sample_count=8,
        )
        for band in range(2)
    }
    schedule = SimpleNamespace(
        group_starts=np.array([0, 1]),
        group_lengths=np.array([1, 1]),
        block_starts=np.array([0.0, 1.0]),
    )
    chunks = [(0, 2)] if failure else [(0, 1), (1, 2)]
    monkeypatch.setattr(
        gpu_execution, "_automatic_xmac_fusion_width", lambda _xp: 2
    )
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 4)
    monkeypatch.setattr(
        gpu_execution, "_scan_block_schedule", lambda *_args: schedule
    )
    monkeypatch.setattr(
        gpu_execution,
        "_gpu_group_chunks",
        lambda *_args: [
            gpu_execution._GpuGroupChunk("scan", *pair) for pair in chunks
        ],
    )
    monkeypatch.setattr(
        gpu_execution, "_packed_real_group_compatible", lambda *_args: packed
    )
    monkeypatch.setattr(
        gpu_execution,
        "_packed_group_requires_source_replay",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        gpu_execution,
        "_station_blocks_packed_real_group_for_scan",
        station_blocks,
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks_for_scan", single)
    monkeypatch.setattr(gpu_execution, "_apply_fft_eligibility", eligibility)
    monkeypatch.setattr(gpu_execution, "_reported_station_weights", weights)
    monkeypatch.setattr(
        gpu_execution,
        "prepare_visibility_index",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        gpu_execution,
        "visibility_index_view",
        lambda value, index: value[index],
    )
    monkeypatch.setattr(gpu_execution, "_launch_visibility_groups", launch)
    monkeypatch.setattr(
        gpu_execution,
        "_accumulate_reported_weight_adjustments",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        gpu_execution,
        "_reserve_gpu_records",
        lambda *_args: (
            [],
            {
                (kind, group): (group, group)
                for kind in ("cross", "auto")
                for group in range(2)
            },
        ),
    )
    layout = tuple(
        gpu_execution._VisibilityProductSpec(
            (kind,), (0, 2), (0, 1), (0, 0), (0, 1), 1.0, 1, False, True
        )
        for kind in ("cross", "auto")
    )
    job = SimpleNamespace(
        datastreams=[
            SimpleNamespace(
                identifier=0,
                zoom_bands=[
                    SimpleNamespace(
                        parent_band_id=0,
                        start_channel=1,
                        channel_count=2,
                        identifier=2,
                    )
                ],
            )
        ]
    )
    gpu_execution._run_chunked_gpu(
        job,
        xp,
        [(0, scan, [0]), (1, scan, [1])],
        band_data,
        {},
        {},
        {},
        {},
        SimpleNamespace(weight_adjustments=np.zeros(1)),
        None,
        1000,
        None,
        visibility_layout=layout,
    )
    assert starts_seen == (
        [(0.0, 1.0), (0.0,), (1.0,)] if failure else [(0.0,), (1.0,)]
    )
    assert [kind for kind, _spectrum in products_seen] == [
        "cross",
        "auto",
    ] * 2
    for _kind, spectrum in products_seen:
        np.testing.assert_array_equal(spectrum, np.ones((1, 2)))
    assert_released()
