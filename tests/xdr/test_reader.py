# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import sys
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import prefetch, reader
from cuphoton.xdr.gds import HeapReadHandle
from cuphoton.xdr.reader import GpuCompImageReader, _validate_comp_geometry


@pytest.mark.parametrize("use_loader", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_image_reader_validates_output_before_io(
    monkeypatch, use_loader, contiguous
):
    output = np.zeros((2, 4), dtype=np.int16)
    if not contiguous:
        output = np.zeros((2, 8), dtype=np.int16)[:, ::2]
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            dtype=np.dtype,
            cuda=SimpleNamespace(Stream=SimpleNamespace(null=nullcontext())),
        ),
    )

    def pread(out, *, size, file_offset):
        assert out is output
        assert size == output.nbytes
        assert file_offset == 2880
        calls.append("pread")
        return SimpleNamespace(get=lambda: calls.append("wait"))

    monkeypatch.setattr(
        reader,
        "pread_to_device",
        lambda path, offset, size, out: pread(
            out, size=size, file_offset=offset
        ).get(),
    )
    monkeypatch.setattr(
        reader, "byteswap_inplace", lambda *args: calls.append("byteswap")
    )
    image_reader = reader.GpuImageReader(
        SimpleNamespace(
            header={"NAXIS": 2, "BITPIX": 16, "NAXIS1": 4, "NAXIS2": 2},
            _file=SimpleNamespace(name="image.fits"),
            _data_offset=2880,
        )
    )
    loader = (
        SimpleNamespace(_file=lambda: SimpleNamespace(pread=pread))
        if use_loader
        else None
    )
    if contiguous:
        assert image_reader.read(out=output, loader=loader) is output
        assert calls == ["pread", "wait", "byteswap"]
    else:
        with pytest.raises(
            ValueError, match="out buffer must be C-contiguous"
        ):
            image_reader.read(out=output, loader=loader)
        assert calls == []


@pytest.mark.parametrize("decode_fails", [False, True])
@pytest.mark.parametrize("sync_fails", [False, True])
@pytest.mark.parametrize("owns_loader", [False, True])
def test_comp_reader_retains_buffers_through_completion(
    monkeypatch, decode_fails, sync_fails, owns_loader
):
    class Device:
        def __init__(self, device_id=2):
            self.id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Stream:
        done = False

        def __init__(self):
            self.synchronize_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def synchronize(self):
            self.synchronize_calls += 1
            # Input and caller output must survive even if decode fails.
            assert "device-input" in keepalives[0]
            assert "caller-output" in keepalives[0]
            if sync_fails:
                raise RuntimeError("synchronize failed")
            self.done = True

    class Handle:
        def wait(self):
            return "device-input", "relative-offsets"

    class Loader:
        closed = False

        def load_tiles_async(self, offsets, lengths):
            return Handle()

        def close(self):
            self.closed = True

    loader = Loader()
    stream = Stream()
    quarantine = []
    keepalives = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=SimpleNamespace(null=None), Device=Device
            )
        ),
    )
    monkeypatch.setattr(
        GpuCompImageReader,
        "prepare_plan",
        lambda self, section=None: {
            "sel_abs_offsets": "absolute-offsets",
            "sel_lengths": "lengths",
        },
    )
    monkeypatch.setattr(reader, "GdsHeapLoader", lambda path: loader)

    def decode(d_concat, rel_offsets, plan, **kwargs):
        keepalives.append(kwargs["keepalive"])
        kwargs["keepalive"].append("decode-scratch")
        if decode_fails:
            raise ValueError("decode failed")
        return "decoded"

    monkeypatch.setattr(
        GpuCompImageReader,
        "decode_from_device_heap",
        staticmethod(decode),
    )
    comp_reader = object.__new__(GpuCompImageReader)
    comp_reader.path = "test.fits"
    kwargs = dict(
        loader=None if owns_loader else loader,
        stream=stream,
        out="caller-output",
    )
    if decode_fails:
        with pytest.raises(ValueError, match="decode failed"):
            comp_reader.read(**kwargs)
    elif sync_fails:
        with pytest.raises(RuntimeError, match="synchronize failed"):
            comp_reader.read(**kwargs)
    else:
        assert comp_reader.read(**kwargs) == "decoded"

    assert stream.synchronize_calls == 1
    assert loader.closed is owns_loader
    if sync_fails:
        assert len(quarantine) == 1
        assert quarantine[0].device_id == 2
        assert quarantine[0].keepalive is keepalives[0]
        assert "decode-scratch" in quarantine[0].keepalive
        # Both standalone and batched reads must respect the same gate.
        with pytest.raises(RuntimeError, match="previous XDR GPU work"):
            comp_reader.read(**kwargs)
        assert len(keepalives) == 1
    else:
        assert quarantine == []


@pytest.mark.parametrize(
    ("error_type", "error_stage"),
    [
        (KeyboardInterrupt, "load"),
        (KeyboardInterrupt, "wait"),
        (KeyboardInterrupt, "decode"),
        (SystemExit, "load"),
        (SystemExit, "wait"),
        (SystemExit, "decode"),
        (RuntimeError, "load"),
        (RuntimeError, "wait"),
    ],
)
@pytest.mark.parametrize("owns_loader", [False, True])
@pytest.mark.parametrize("use_stream", [False, True])
def test_comp_reader_early_failure_quarantines_without_waiting(
    monkeypatch, error_type, error_stage, owns_loader, use_stream
):
    error = error_type("failed during " + error_stage)
    quarantine = []
    output = object()
    device_input = object()
    decode_scratch = object()
    failed = False
    partial_owner_refs = []

    class PartialReadOwner:
        pass

    def maybe_fail(stage):
        nonlocal failed
        if stage == error_stage and not failed:
            failed = True
            raise error

    class Device:
        def __init__(self, device_id=2):
            self.id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class Stream:
        done = False
        synchronize_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def synchronize(self):
            self.synchronize_calls += 1
            self.done = True

    class Handle:
        done = False

        def wait(self):
            maybe_fail("wait")
            self.done = True
            return device_input, "relative-offsets"

    handle = Handle()

    class Loader:
        closed = False
        load_calls = 0

        def load_tiles_async(self, offsets, lengths):
            self.load_calls += 1
            if error_stage == "load" and not failed:
                partial_owner = PartialReadOwner()
                partial_owner_refs.append(weakref.ref(partial_owner))
            maybe_fail("load")
            return handle

        def close(self):
            self.closed = True

    loader = Loader()
    loaders = [loader]
    loader_creations = 0

    def make_loader(path):
        nonlocal loader_creations
        loader_creations += 1
        if loader_creations == 1:
            return loader
        new_loader = Loader()
        loaders.append(new_loader)
        return new_loader

    def load_count():
        return sum(item.load_calls for item in loaders)

    stream = Stream()
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=Device, Stream=SimpleNamespace(null=stream)
            )
        ),
    )
    monkeypatch.setattr(reader, "GdsHeapLoader", make_loader)
    monkeypatch.setattr(
        GpuCompImageReader,
        "prepare_plan",
        lambda *args, **kwargs: {
            "sel_abs_offsets": "absolute-offsets",
            "sel_lengths": "lengths",
        },
    )

    def decode(*args, **kwargs):
        if kwargs.get("keepalive") is not None:
            kwargs["keepalive"].append(decode_scratch)
        maybe_fail("decode")
        return output

    monkeypatch.setattr(
        GpuCompImageReader, "decode_from_device_heap", staticmethod(decode)
    )
    comp_reader = object.__new__(GpuCompImageReader)
    comp_reader.path = "test.fits"
    kwargs = dict(
        loader=None if owns_loader else loader,
        stream=stream if use_stream else None,
        out=output,
    )

    with pytest.raises(error_type) as exc_info:
        comp_reader.read(**kwargs)

    assert exc_info.value is error
    error.__traceback__ = None
    del exc_info
    gc.collect()
    assert stream.synchronize_calls == 0
    assert loader.closed is (owns_loader and error_stage == "decode")
    assert len(quarantine) == 1
    completion = quarantine[0]
    assert completion.actively_awaited is False
    assert completion.device_id == 2
    assert output in completion.keepalive
    if error_stage != "load":
        assert handle in completion.keepalive
    if error_stage == "decode":
        assert device_input in completion.keepalive
        if use_stream:
            assert decode_scratch in completion.keepalive
    if error_stage == "load":
        assert len(partial_owner_refs) == 1
        assert partial_owner_refs[0]() is not None

    with pytest.raises(RuntimeError, match="previous XDR GPU work"):
        comp_reader.read(**kwargs)
    assert load_count() == 1
    assert stream.synchronize_calls == 0
    assert quarantine == [completion]

    stream.done = True
    if error_stage != "decode":
        with pytest.raises(RuntimeError, match="previous XDR GPU work"):
            comp_reader.read(**kwargs)
        assert load_count() == 1
        assert quarantine == [completion]
        assert stream.synchronize_calls == 0
        assert loader.closed is False
        handle.done = True
        if error_stage == "load":
            # The load never returned its handle: even an idle CUDA stream
            # cannot establish completion of the unknown partial I/O.
            with pytest.raises(RuntimeError, match="previous XDR GPU work"):
                comp_reader.read(**kwargs)
            assert load_count() == 1
            assert quarantine == [completion]
            assert completion.actively_awaited is False
            assert loader.closed is False
            return

    assert comp_reader.read(**kwargs) is output
    assert load_count() == 2
    assert stream.synchronize_calls == int(use_stream)
    assert quarantine == []


def test_heap_read_handle_done_checks_all_futures_without_waiting():
    class Future:
        get_calls = 0

        def __init__(self, ready):
            self.ready = ready

        def done(self):
            return self.ready

        def get(self):
            self.get_calls += 1

    first = Future(True)
    second = Future(False)
    handle = HeapReadHandle("device-input", "offsets", [first, second])
    assert handle.done is False
    first.ready, second.ready = False, True
    assert handle.done is False
    first.ready = True
    assert handle.done is True
    assert first.get_calls == second.get_calls == 0

    assert handle.wait() == ("device-input", "offsets")
    assert first.get_calls == second.get_calls == 1
    first.ready = second.ready = False
    assert handle.done is True
    assert HeapReadHandle("empty-input", "offsets", []).done is True


def test_inflate_device_heap_uses_native_pool_for_event_keepalive(
    monkeypatch,
):
    class NullStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    captured = {}

    def decompress(*args, **kwargs):
        captured.update(kwargs)
        return "pixels", "offsets"

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(Stream=SimpleNamespace(null=NullStream()))
        ),
    )
    monkeypatch.setattr(reader, "gpu_gzip_decompress_batch", decompress)
    keepalive = []

    result = GpuCompImageReader.inflate_device_heap(
        "device-bytes",
        [0],
        lengths=[20],
        out_bytes=[1],
        keepalive=keepalive,
        header_sizes=[10],
    )

    assert result == ("pixels", "offsets")
    assert captured["header_sizes"] == [10]
    assert captured["use_native_pool"] is True
    assert captured["keepalive"] is keepalive


@pytest.mark.parametrize(
    ("data_shape", "tile_shape"),
    [
        ((100, 100), (0, 64)),
        ((100, 100), (64, 0)),
        ((100, 100), (-1, 64)),
        ((0, 100), (64, 64)),
        ((100, -5), (64, 64)),
    ],
)
def test_comp_geometry_rejects_nonpositive_dims(data_shape, tile_shape):
    with pytest.raises(ValueError, match="must be positive"):
        _validate_comp_geometry(data_shape, tile_shape)


def test_comp_geometry_accepts_positive_dims_case():
    _validate_comp_geometry((100, 100), (64, 64))
