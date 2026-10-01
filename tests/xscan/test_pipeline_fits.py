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


@pytest.mark.parametrize("reader", [None, "auto", "astropy", "xdr"])
def test_manifest_reader_override_preserves_input_and_records_policy(
    inputs, tmp_path, reader
):
    item, config = inputs
    payload = item.to_payload()
    roles = ("reference", "target", "variance", "fit_mask")
    for role in roles:
        payload[role]["reader"] = "xdr"
    manifest = tmp_path / "manifest.json"
    original = json.dumps(
        {
            "schema": pipeline_executor.PIPELINE_MANIFEST_SCHEMA,
            "configuration": config.to_payload(),
            "items": [payload],
        }
    )
    manifest.write_text(original)
    _, (loaded,) = pipeline_executor.load_pipeline_manifest(
        manifest, fits_reader=reader
    )
    expected = "xdr" if reader is None else reader
    work, identities = dragon_pipeline._preflight_items((loaded,))
    restored = pipeline.DevicePipelineItem.from_payload(work[0].payload)
    assert {getattr(restored, role).reader for role in roles} == {expected}
    assert {
        image["reader"] for image in identities[0]["files"][0]["images"]
    } == {expected}
    assert manifest.read_text() == original


def test_reader_override_only_changes_fits_descriptors(inputs):
    item, _ = inputs
    npy = pipeline.NpyArrayDescriptor(
        path="/variance.npy", sha256="c" * 64, shape=(31, 31), dtype="float64"
    )
    item = replace(item, variance=npy, fit_mask=None)
    payload = item.to_payload()
    payload["reference"]["reader"] = "invalid"
    loaded = pipeline.DevicePipelineItem.from_payload(
        payload, fits_reader="astropy"
    )
    assert loaded.reference.reader == loaded.target.reader == "astropy"
    assert loaded.variance == npy and loaded.fit_mask is None
    assert payload["reference"]["reader"] == "invalid"
    for invalid in ("gds", "", [], True):
        with pytest.raises(ValueError, match="FITS reader"):
            pipeline.DevicePipelineItem.from_payload(
                payload, fits_reader=invalid
            )


def test_benchmark_stage_uses_effective_fits_policy(
    inputs, tmp_path, monkeypatch
):
    from cuphoton.xscan.pipeline_benchmark import runner, stages

    item, config = inputs
    config_path = tmp_path / "configuration.json"
    items_path = tmp_path / "items.json"
    config_path.write_text(json.dumps(config.to_payload()))
    original = json.dumps([item.to_payload()])
    items_path.write_text(original)
    received = []
    monkeypatch.setattr(
        stages, "run_stage", lambda *args: received.append(args)
    )
    runner.run_benchmark(
        SimpleNamespace(
            output=tmp_path / "staged",
            config=config_path,
            items=items_path,
            fits_reader="astropy",
            repeat=1,
            warmup=1,
            timeout=10,
            stage="xpois",
        )
    )
    _, _, (loaded,), _ = received[0]
    assert {
        getattr(loaded, role).reader
        for role in ("reference", "target", "variance", "fit_mask")
    } == {"astropy"}
    assert items_path.read_text() == original


@pytest.mark.parametrize("executor", ["mpi", "dragon"])
def test_pipeline_override_reaches_workload_identity_on_every_rank(
    inputs, tmp_path, monkeypatch, executor
):
    item, config = inputs
    checkpoint = tmp_path / "checkpoint.pt"
    schema = tmp_path / "schema.json"
    checkpoint.write_bytes(b"checkpoint")
    schema.write_text("{}")
    config = replace(
        config,
        checkpoint_sha256=file_sha256(checkpoint),
        feature_schema_sha256=file_sha256(schema),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": pipeline_executor.PIPELINE_MANIFEST_SCHEMA,
                "configuration": config.to_payload(),
                "items": [item.to_payload()],
            }
        )
    )
    original = pipeline_executor.prepare_pipeline_workload((item,), config)

    def run(**kwargs):
        assert kwargs["executor"] == executor
        assert "fits_reader" not in kwargs
        root = kwargs["prepare_workload"](0)
        peer = kwargs["prepare_workload"](1)
        assert root.identity_payload() == peer.identity_payload()
        assert root.manifest_sha256 != original.manifest_sha256
        payload = root.items[0].payload
        for role in ("reference", "target", "variance", "fit_mask"):
            assert payload[role]["reader"] == "astropy"
        return "completed"

    monkeypatch.setattr("cuphoton.core.executors.run_workload", run)
    assert (
        pipeline_executor.run_pipeline_manifest(
            executor=executor,
            manifest_path=manifest,
            output_root=tmp_path / "runs",
            fits_reader="astropy",
        )
        == "completed"
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


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("reader", ["astropy", "auto", "xdr"])
def test_staged_xpois_decodes_all_fits_roles(
    inputs, tmp_path, monkeypatch, mixed, reader
):
    from cuphoton import xpois
    from cuphoton.core import fits_io
    from cuphoton.xscan.pipeline_benchmark import stages

    item, config = inputs
    item = pipeline.DevicePipelineItem.from_payload(
        item.to_payload(), fits_reader=reader
    )
    original = fits_io.read_fits_images
    expected = original(item.reference.path, [1, 2, 1, 3], reader="astropy")
    if mixed:
        replacements = {}
        for role, array in zip(
            ("reference", "target"), expected.arrays[:2], strict=True
        ):
            path = tmp_path / f"{role}.npy"
            np.save(path, array)
            replacements[role] = pipeline.NpyArrayDescriptor(
                str(path), file_sha256(path), array.shape, array.dtype.str
            )
        item = replace(item, **replacements)
    calls = []
    stream = object()

    def device_read(path, hdus, **options):
        calls.append((tuple(hdus), options["reader"]))
        assert options["device"] and options["stream"] is stream
        result = original(path, hdus, reader="astropy")
        return replace(
            result,
            reader="xdr" if reader == "xdr" else "astropy",
            requested_reader=reader,
            device=True,
            fallback_reason="test CPU reader" if reader == "auto" else None,
        )

    monkeypatch.setattr(fits_io, "read_fits_images", device_read)
    solved = []

    def solve(reference, target, _basis, **options):
        solved.append(True)
        for actual, wanted in zip(
            (reference, target, options["variance"], options["fit_mask"]),
            expected.arrays,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, wanted)
        assert options["fit_mask"].dtype == np.bool_
        return SimpleNamespace(
            residual=target - reference,
            kernel_coefficients=np.ones(1),
            background_coefficients=np.zeros(1),
            chi2=1.0,
            dof=1,
            fit_pixel_count=1,
            basis_terms=(),
            solver="test",
            backend="test",
            flux_conserve=False,
        )

    monkeypatch.setattr(xpois, "solve_constant_kernel_device", solve)
    cp = SimpleNamespace(
        ndarray=np.ndarray,
        asarray=np.asarray,
        asnumpy=np.asarray,
        empty=np.empty,
        float64=np.float64,
        float32=np.float32,
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(synchronize=lambda: None),
            get_current_stream=lambda: stream,
        ),
    )
    _, metadata, timings = stages._run_item(
        "xpois", item, config, {"cp": cp, "torch": None}, tmp_path, {}, 0
    )
    assert solved == [True]
    assert calls == [((1, 3) if mixed else (1, 2, 3), reader)]
    assert metadata["fits_reads"][0]["requested_reader"] == reader
    assert metadata["fits_reads"][0]["roles"] == (
        {"variance": 1, "fit_mask": 3}
        if mixed
        else {"reference": 1, "target": 2, "variance": 1, "fit_mask": 3}
    )
    assert timings["read_seconds"] >= 0
