# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native file reads complete before their worker releases its owners."""

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xdr import nvcomp_batch


@pytest.fixture
def native_io():
    cp = pytest.importorskip("cupy")
    kvikio = pytest.importorskip("kvikio")
    defaults = pytest.importorskip("kvikio.defaults")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    ext = nvcomp_batch._try_get_cpp_ext()
    if ext is None or not hasattr(ext, "NativeBatchBuilder"):
        pytest.skip("native I/O extension unavailable")
    with defaults.set(
        {
            "compat_mode": kvikio.CompatMode.ON,
            "gds_threshold": 0,
            "task_size": 64 * 1024,
            "num_threads": 2,
        }
    ):
        yield cp, ext


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("past_eof", [False, True])
def test_native_span_reads_complete_or_stop_on_eof(
    tmp_path, native_io, direct, past_eof
):
    cp, ext = native_io
    original = np.arange(8192, dtype=np.uint8)
    path = tmp_path / "input.bin"
    path.write_bytes(original.tobytes())
    spans = (
        [(original.size - 4, 8, 0), (0, 32, 8)]
        if past_eof
        else [(7, 256, 0), (1021, 257, 256)]
    )
    nbytes = sum(size for _, size, _ in spans)
    # A slice exercises allocation-base/offset normalization for direct reads.
    allocation = cp.full(nbytes + 34, 251, dtype=cp.uint8)
    target = allocation[17:-17]
    cp.cuda.Stream.null.synchronize()
    plan = (
        (
            str(path),
            0,
            0,
            [],
            [
                (offset, size, int(target.data.ptr) + dest)
                for offset, size, dest in spans
            ],
        )
        if direct
        else (str(path), 0, nbytes, spans)
    )
    builder = ext.NativeBatchBuilder(1, 1, 2, int(cp.cuda.Device().id))
    try:
        builder.submit_batch(0, [plan])
        builder.close_input()
        if past_eof:
            with pytest.raises(RuntimeError, match="EOF|short KvikIO read"):
                builder.next_batch()
        else:
            owner, ptr, size, files = builder.next_batch()
            assert files == [(0, 0)]
            if not direct:
                memory = cp.cuda.UnownedMemory(ptr, size, owner)
                target = cp.ndarray(
                    (size,),
                    dtype=cp.uint8,
                    memptr=cp.cuda.MemoryPointer(memory, 0),
                )
            np.testing.assert_array_equal(
                cp.asnumpy(target),
                np.concatenate(
                    [
                        original[offset : offset + size]
                        for offset, size, _ in spans
                    ]
                ),
            )
            assert builder.next_batch() is None
    finally:
        # The public consumer joins workers before releasing its outputs.
        builder.request_stop()
    if past_eof:
        assert builder.io_stats()["kvikio_reads"] == 0
        if direct:
            # Both spans were queued before the first future failed. The
            # successful sibling must finish before the builder joins.
            np.testing.assert_array_equal(
                cp.asnumpy(target[8:]), original[:32]
            )
    if direct:
        np.testing.assert_array_equal(cp.asnumpy(allocation[:17]), 251)
        np.testing.assert_array_equal(cp.asnumpy(allocation[-17:]), 251)


def test_native_split_read_failure_drains_later_spans(tmp_path, native_io):
    cp, ext = native_io
    original = np.arange(4 * 1024 * 1024, dtype=np.uint8)
    path = tmp_path / "split-input.bin"
    path.write_bytes(original.tobytes())
    # The first pread is split into tasks and fails in its final task. The
    # following large span queues additional tasks using the same file and
    # destination owners; a failed aggregate future does not drain these.
    failed_size = 128 * 1024
    allocation = cp.full(failed_size + original.size, 251, dtype=cp.uint8)
    cp.cuda.Stream.null.synchronize()
    plan = (
        str(path),
        0,
        0,
        [],
        [
            (
                original.size - failed_size // 2,
                failed_size,
                allocation.data.ptr,
            ),
            (0, original.size, allocation.data.ptr + failed_size),
        ],
    )
    builder = ext.NativeBatchBuilder(1, 1, 2, int(cp.cuda.Device().id))
    try:
        builder.submit_batch(0, [plan])
        builder.close_input()
        with pytest.raises(RuntimeError, match="EOF|short KvikIO read"):
            builder.next_batch()
        # Error publication happens after the owned pool is drained, before
        # shutdown joins the native file workers.
        np.testing.assert_array_equal(
            cp.asnumpy(allocation[failed_size:]), original
        )
    finally:
        builder.request_stop()


def test_native_direct_span_checks_allocation_before_read(
    tmp_path, native_io
):
    cp, ext = native_io
    path = tmp_path / "empty.bin"
    # An empty file cannot write any bytes even if the bounds check regresses.
    path.write_bytes(b"")
    allocation = cp.cuda.Memory(64)
    plan = (str(path), 0, 0, [], [(0, allocation.size + 1, allocation.ptr)])
    builder = ext.NativeBatchBuilder(1, 1, 1, int(cp.cuda.Device().id))
    try:
        builder.submit_batch(0, [plan])
        builder.close_input()
        with pytest.raises(
            ValueError, match="span exceeds device allocation"
        ):
            builder.next_batch()
    finally:
        builder.request_stop()
    assert builder.io_stats()["kvikio_reads"] == 0


def test_native_readers_recover_independently_after_failure(
    tmp_path, native_io
):
    cp, ext = native_io
    defaults = pytest.importorskip("kvikio.defaults")
    original = np.arange(1024 * 1024, dtype=np.uint8)
    path = tmp_path / "reuse-input.bin"
    path.write_bytes(original.tobytes())
    outputs = [cp.empty(original.size, dtype=cp.uint8) for _ in range(3)]
    device_id = int(cp.cuda.Device().id)

    # Change the configured size between calls. Within each iteration, one
    # failed builder stops while another remains live, then a new builder
    # starts and both remaining builders must read successfully.
    for threads in (2, 1, 2):
        with defaults.set("num_threads", threads):
            builders = [
                ext.NativeBatchBuilder(1, 1, 1, device_id) for _ in range(2)
            ]
            try:
                builders[0].submit_batch(
                    0,
                    [
                        (
                            str(path),
                            0,
                            0,
                            [],
                            [
                                (
                                    original.size,
                                    original.size,
                                    outputs[0].data.ptr,
                                )
                            ],
                        )
                    ],
                )
                builders[0].close_input()
                with pytest.raises(
                    RuntimeError, match="EOF|short KvikIO read"
                ):
                    builders[0].next_batch()
                builders[0].request_stop()
                builders[0].request_stop()
                builders.append(ext.NativeBatchBuilder(1, 1, 1, device_id))
                for builder, output in zip(
                    builders[1:], outputs[1:], strict=True
                ):
                    builder.submit_batch(
                        0,
                        [
                            (
                                str(path),
                                0,
                                0,
                                [],
                                [(0, original.size, output.data.ptr)],
                            )
                        ],
                    )
                    builder.close_input()
                for builder, output in zip(
                    builders[1:], outputs[1:], strict=True
                ):
                    assert builder.next_batch() is not None
                    np.testing.assert_array_equal(
                        cp.asnumpy(output), original
                    )
                    assert builder.next_batch() is None
            finally:
                for builder in builders:
                    builder.request_stop()
