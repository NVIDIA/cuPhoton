# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS manifest and worker-ingress tests; CPU fixtures do not import CUDA."""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from cuphoton.core.artifacts import file_sha256
from cuphoton.xscan import device_pipeline as pipeline
from cuphoton.xscan import dragon_pipeline, pipeline_executor


@pytest.fixture
def inputs(tmp_path):
    image = np.arange(31 * 31, dtype=np.float64).reshape(31, 31) / 17
    image[0, 0] = np.nan
    mask = np.ones(image.shape, dtype=np.uint8)
    mask[0, 0] = 0
    path = tmp_path / "registered.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image),
            fits.ImageHDU(image + 1),
            fits.ImageHDU(mask),
        ]
    ).writeto(path)
    common = {
        "path": str(path),
        "sha256": file_sha256(path),
        "shape": image.shape,
    }
    reference = pipeline.FitsArrayDescriptor(**common, dtype="float64", hdu=1)
    item = pipeline.DevicePipelineItem(
        item_id="registered",
        reference=reference,
        target=replace(reference, hdu=2),
        variance=reference,
        fit_mask=pipeline.FitsArrayDescriptor(**common, dtype="uint8", hdu=3),
        candidates=(
            pipeline.DevicePipelineCandidate("candidate", 15, 15, 0),
        ),
    )
    config = pipeline.DevicePipelineConfig(
        device="cuda:0",
        checkpoint_dir=str(tmp_path),
        checkpoint_sha256="a" * 64,
        feature_schema_path=str(tmp_path / "schema.json"),
        feature_schema_sha256="b" * 64,
        stamp_shape=(11, 11),
        decision_threshold=0.5,
        xpois=pipeline.DeviceXPOISPipelineConfig((3, 3), (0.8,), (0,)),
    )
    return item, config


def test_fits_descriptor_roundtrip_and_relative_manifest(inputs, tmp_path):
    item, config = inputs
    assert pipeline.DevicePipelineItem.from_payload(item.to_payload()) == item
    payload = item.to_payload()
    for role in ("reference", "target", "variance", "fit_mask"):
        payload[role]["path"] = "registered.fits"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": pipeline_executor.PIPELINE_MANIFEST_SCHEMA,
                "configuration": config.to_payload(),
                "items": [payload],
            }
        )
    )
    assert pipeline_executor.load_pipeline_manifest(manifest) == (
        config,
        (item,),
    )
    for update in (
        {"hdu": True},
        {"hdu": -1},
        {"reader": "gds"},
        {"extra": 0},
    ):
        with pytest.raises((TypeError, ValueError)):
            pipeline.FitsArrayDescriptor.from_payload(
                {**item.reference.to_payload(), **update}
            )


def test_fits_preflight_multiple_hdus_count_file_once(inputs):
    item, _ = inputs
    work, identities = dragon_pipeline._preflight_items((item,))
    assert work[0].weight_bytes == Path(item.reference.path).stat().st_size
    files = identities[0]["files"]
    assert len(files) == 1
    assert [
        (image["hdu"], image["roles"]) for image in files[0]["images"]
    ] == [
        (1, ["reference", "variance"]),
        (2, ["target"]),
        (3, ["fit_mask"]),
    ]
    with pytest.raises(ValueError, match="shape or dtype"):
        dragon_pipeline._preflight_items(
            (replace(item, target=replace(item.target, shape=(3, 3))),)
        )
    with pytest.raises(ValueError, match="conflicting FITS"):
        dragon_pipeline._preflight_items(
            (replace(item, target=replace(item.target, reader="xdr")),)
        )


def test_fits_worker_grouped_device_read_and_receipt(inputs, monkeypatch):
    from cuphoton.core import fits_io

    item, config = inputs
    calls = []
    stream = object()
    cpu_read = fits_io.read_fits_images

    def device_read(path, hdus, **options):
        assert options == {"reader": "auto", "device": True, "stream": stream}
        calls.append(tuple(hdus))
        result = cpu_read(path, hdus, reader="astropy")
        return replace(
            result,
            requested_reader="auto",
            fallback_reason="CPU test fixture",
            device=True,
        )

    monkeypatch.setattr(fits_io, "read_fits_images", device_read)
    cp = SimpleNamespace(ndarray=np.ndarray, asarray=np.asarray)
    pipeline._validate_item_contract(item, config=config)
    arrays, reads = pipeline._read_fits_device_inputs(
        item, cp=cp, stream=stream
    )
    assert calls == [(1, 2, 3)]
    np.testing.assert_array_equal(arrays["reference"], arrays["variance"])
    assert arrays["reference"].dtype == np.float64
    assert np.isnan(arrays["reference"][0, 0])
    assert arrays["fit_mask"].dtype == np.bool_
    assert not arrays["fit_mask"][0, 0]
    receipt = pipeline.DevicePipelineTransferReceipt(
        0, 8, 0, fits_reads=reads
    ).to_payload()
    assert receipt["reader_internal_transfers"] == "not-instrumented"
    assert reads[0]["decoded_bytes"] == 31 * 31 * 17
    pipeline._validate_fits_reads(item, receipt["fits_reads"])
    for field, replacement in (
        ("decoded_bytes", 1),
        ("reader", "gds"),
        ("roles", {}),
        ("fallback_reason", None),
    ):
        changed = copy.deepcopy(receipt["fits_reads"])
        changed[0][field] = replacement
        with pytest.raises(ValueError):
            pipeline._validate_fits_reads(item, changed)


def test_fits_mask_rejects_bitplanes_and_changed_content(inputs, monkeypatch):
    from cuphoton.core import fits_io

    item, _ = inputs
    original = fits_io.read_fits_images

    def bad_mask(path, hdus, **options):
        result = original(path, hdus, reader="astropy")
        result.arrays[-1][0, 0] = 2
        return result

    monkeypatch.setattr(fits_io, "read_fits_images", bad_mask)
    cp = SimpleNamespace(ndarray=np.ndarray, asarray=np.asarray)
    with pytest.raises(ValueError, match="only 0 and 1"):
        pipeline._read_fits_device_inputs(item, cp=cp, stream=None)
    bad = replace(
        item,
        reference=replace(item.reference, sha256="f" * 64),
        variance=None,
        target=replace(item.target, sha256="f" * 64),
        fit_mask=None,
    )
    with pytest.raises(RuntimeError, match="changed before"):
        pipeline._read_fits_device_inputs(bad, cp=cp, stream=None)


@pytest.mark.parametrize(
    "error", [RuntimeError("decode failed"), KeyboardInterrupt()]
)
def test_fits_ingress_failure_uses_pipeline_stream_cleanup(
    inputs, monkeypatch, error
):
    item, config = inputs
    synchronized = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def synchronize(self):
            synchronized.append(self)

    context = object.__new__(pipeline.DeviceWorkerContext)
    context.config = config
    context.feature_variance_present = False
    context.cp = SimpleNamespace(asarray=np.asarray)
    context.producer_stream = Stream()
    context.consumer_stream = Stream()
    context._item_lock = threading.Lock()
    context._poisoned_reason = None
    monkeypatch.setattr(
        pipeline.DeviceWorkerContext,
        "_validate_active_device",
        lambda _self: None,
    )

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(pipeline, "_load_pipeline_functions", lambda: None)
    monkeypatch.setattr(pipeline, "_read_fits_device_inputs", fail)
    with pytest.raises(type(error)):
        pipeline.run_device_pipeline_item(item, context)
    assert synchronized == [context.producer_stream, context.consumer_stream]
    assert context._item_lock.acquire(blocking=False)
    context._item_lock.release()


def test_npy_payload_and_accounting_remain_unchanged(tmp_path):
    path = tmp_path / "image.npy"
    array = np.ones((31, 31))
    np.save(path, array)
    descriptor = pipeline.NpyArrayDescriptor(
        str(path), file_sha256(path), array.shape, array.dtype.str
    )
    assert set(descriptor.to_payload()) == {
        "path",
        "sha256",
        "shape",
        "dtype",
    }
    assert (
        "fits_reads"
        not in pipeline.DevicePipelineTransferReceipt(8, 8, 0).to_payload()
    )


@pytest.mark.parametrize("reader", ["astropy", "xdr"])
def test_gpu_fits_ingress_preserves_lossless_pixels(inputs, reader):
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    if count == 0:
        pytest.skip("CUDA unavailable")
    from cuphoton.core import fits_io

    item, _ = inputs
    item = replace(
        item,
        **{
            role: replace(getattr(item, role), reader=reader)
            for role in ("reference", "target", "variance", "fit_mask")
        },
    )
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        arrays, _ = pipeline._read_fits_device_inputs(
            item, cp=cp, stream=stream
        )
    stream.synchronize()
    expected = fits_io.read_fits_images(
        item.reference.path, (1, 2, 3), reader="astropy"
    )
    for role, index in (
        ("reference", 0),
        ("target", 1),
        ("variance", 0),
        ("fit_mask", 2),
    ):
        wanted = expected.arrays[index]
        if role == "fit_mask":
            wanted = wanted.astype(bool)
        np.testing.assert_array_equal(cp.asnumpy(arrays[role]), wanted)
