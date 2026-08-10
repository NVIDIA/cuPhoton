# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import kernels, prefetch


@pytest.mark.parametrize(
    ("dtype_char", "itemsize", "expected"),
    [("i2", 2, "f4"), ("i8", 8, "f8")],
)
def test_probe_output_specs_promotes_scaled_integer_images(
    monkeypatch, dtype_char, itemsize, expected
):
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=0,
        heap_span_len=itemsize,
        image_shape=(1,),
        image_dtype_char=dtype_char,
        image_itemsize=itemsize,
        image_bzero=32768.0,
    )
    planned = SimpleNamespace(hdus=[image])
    monkeypatch.setattr(
        prefetch, "_plan_file", lambda *args, **kwargs: planned
    )

    [spec] = prefetch._probe_output_specs("image.fits", (0,))

    assert np.dtype(spec.dtype_char) == np.dtype(expected)


def test_consume_prefetched_group_stores_promoted_image(monkeypatch):
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=0,
        heap_span_len=2,
        image_shape=(1,),
        image_dtype_char="i2",
        image_itemsize=2,
        image_bzero=32768.0,
    )
    item = prefetch.PrefetchedFile(
        path="image.fits",
        file_index=0,
        host_buf=np.zeros(2, dtype=np.uint8),
        hdus=[image],
    )
    outs = [np.zeros((1, 1), dtype=np.float32)]
    scaled = np.array([32768.0], dtype=np.float32)
    monkeypatch.setattr(
        prefetch, "_consume_image", lambda *args, **kwargs: scaled
    )
    monkeypatch.setattr(
        prefetch, "_consume_comp_batch", lambda *args, **kwargs: None
    )

    prefetch._consume_prefetched_group([item], outs, stream=None)

    np.testing.assert_array_equal(outs[0][0], scaled)


def test_consume_image_stages_raw_bytes_before_scaling(monkeypatch):
    class NullStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    fake_cupy = SimpleNamespace(
        dtype=np.dtype,
        empty=np.empty,
        ndarray=np.ndarray,
        uint8=np.uint8,
        float32=np.float32,
        float64=np.float64,
        cuda=SimpleNamespace(Stream=SimpleNamespace(null=NullStream())),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        kernels,
        "byteswap_inplace",
        lambda array, itemsize: array.byteswap(inplace=True),
    )

    def scale(array, bzero, bscale):
        array *= bscale
        array += bzero

    monkeypatch.setattr(kernels, "apply_bzero_bscale", scale)
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=0,
        heap_span_len=2,
        image_shape=(1,),
        image_dtype_char="i2",
        image_itemsize=2,
        image_bzero=32768.0,
    )
    raw = np.array([0], dtype=">i2").view(np.uint8)
    target = np.empty(1, dtype=np.float32)

    result = prefetch._consume_image(image, raw, target, stream=None)

    assert result is target
    np.testing.assert_array_equal(target, np.array([32768.0], dtype="f4"))


def test_native_plan_stages_image_when_output_dtype_is_promoted():
    image = prefetch._HduPlan(
        hdu_index=0,
        kind="image",
        plan=None,
        heap_rel_offsets=None,
        heap_span_start=2880,
        heap_span_len=2,
        image_shape=(1,),
        image_dtype_char="i2",
        image_itemsize=2,
        image_bzero=32768.0,
    )
    planned = prefetch._PlannedFile(
        path="image.fits",
        file_index=0,
        hdus=[image],
        spans=[prefetch._ReadSpan(2880, 2, 0)],
        total_bytes=2,
    )
    outs = [np.empty((1, 1), dtype=np.float32)]

    [converted], [native_plan] = (
        prefetch._native_file_plans_with_direct_outputs([planned], outs)
    )

    assert converted.hdus[0].direct_output is False
    assert converted.total_bytes == 2
    assert native_plan[-1] == []


def test_file_prefetcher_retries_sentinel_until_queue_accepts_it():
    class FullTwiceQueue:
        def __init__(self):
            self.calls = []

        def put(self, item, timeout):
            self.calls.append((item, timeout))
            if len(self.calls) < 3:
                raise queue.Full

    out_queue = FullTwiceQueue()
    producer = prefetch._FilePrefetcher([], (), out_queue)

    producer.run()

    assert [item for item, _ in out_queue.calls] == [
        prefetch._SENTINEL,
        prefetch._SENTINEL,
        prefetch._SENTINEL,
    ]


def test_batch_stream_propagates_error_if_prefetcher_exits_without_item(
    monkeypatch,
):
    class EmptyQueue:
        def __init__(self):
            self.timeouts = []

        def get(self, timeout):
            self.timeouts.append(timeout)
            raise queue.Empty

        def get_nowait(self):
            raise queue.Empty

    class FailedPrefetcher:
        def __init__(self, *args, **kwargs):
            self.error = RuntimeError("prefetch failed")

        def start(self):
            pass

        def is_alive(self):
            return False

        def request_stop(self):
            pass

        def join(self):
            pass

    out_queue = EmptyQueue()
    monkeypatch.setattr(prefetch.queue, "Queue", lambda maxsize: out_queue)
    monkeypatch.setattr(prefetch, "_FilePrefetcher", FailedPrefetcher)
    monkeypatch.setattr(
        prefetch, "configure_kvikio_parallelism", lambda: None
    )
    monkeypatch.setattr(
        prefetch, "_native_batch_components", lambda value: (None, None)
    )
    monkeypatch.setattr(
        prefetch,
        "_probe_output_specs",
        lambda *args, **kwargs: [prefetch._OutputSpec((1,), "f4")],
    )
    monkeypatch.setattr(
        prefetch,
        "_prepare_output_arrays",
        lambda *args, **kwargs: [np.empty((1, 1), dtype=np.float32)],
    )

    with pytest.raises(RuntimeError, match="prefetch failed"):
        prefetch.batch_to_device_stream(["image.fits"], native_batcher=False)

    assert out_queue.timeouts
    assert all(timeout > 0 for timeout in out_queue.timeouts)


def test_batch_stream_does_not_consume_partial_group_after_prefetch_error(
    monkeypatch,
):
    partial = prefetch.PrefetchedFile(
        path="first.fits",
        file_index=0,
        host_buf=np.empty(0, dtype=np.uint8),
        hdus=[],
    )

    class PartialQueue:
        def __init__(self):
            self.items = iter((partial, prefetch._SENTINEL))

        def get(self, timeout):
            return next(self.items)

        def get_nowait(self):
            raise queue.Empty

    class FailedPrefetcher:
        def __init__(self, *args, **kwargs):
            self.error = RuntimeError("second file failed")

        def start(self):
            pass

        def is_alive(self):
            return False

        def request_stop(self):
            pass

        def join(self):
            pass

    output = np.zeros((2, 1), dtype=np.float32)
    consumed = []

    def consume(items, outs, stream):
        consumed.extend(items)
        outs[0][0] = 1.0

    monkeypatch.setattr(
        prefetch.queue, "Queue", lambda maxsize: PartialQueue()
    )
    monkeypatch.setattr(prefetch, "_FilePrefetcher", FailedPrefetcher)
    monkeypatch.setattr(
        prefetch, "configure_kvikio_parallelism", lambda: None
    )
    monkeypatch.setattr(
        prefetch, "_native_batch_components", lambda value: (None, None)
    )
    monkeypatch.setattr(
        prefetch,
        "_probe_output_specs",
        lambda *args, **kwargs: [prefetch._OutputSpec((1,), "f4")],
    )
    monkeypatch.setattr(
        prefetch,
        "_prepare_output_arrays",
        lambda *args, **kwargs: [output],
    )
    monkeypatch.setattr(prefetch, "_consume_prefetched_group", consume)

    with pytest.raises(RuntimeError, match="second file failed"):
        prefetch.batch_to_device_stream(
            ["first.fits", "second.fits"],
            decode_batch_files=2,
            native_batcher=False,
        )

    assert consumed == []
    np.testing.assert_array_equal(output, np.zeros_like(output))
