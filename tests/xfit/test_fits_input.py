# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS ingress preserves candidate crops and the ordinary xFit contract."""

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.io import fits

from cuphoton.core.cli import run_component
from cuphoton.xfit import LMConfig, executor, fit_dipoles
from cuphoton.xfit.fits_input import load_fits_input, plan_fits_input
from cuphoton.xfit.io import load_xfit_dataset


def _input(tmp_path, *, mode="difference"):
    image = np.arange(120, dtype=np.float32).reshape(10, 12) / 100
    mask = np.zeros(image.shape, dtype=np.int32)
    mask[3, 3] = 1
    mask[3, 4] = 2
    variance = np.full(image.shape, 2, dtype=np.float32)
    planes = []
    for index in range(3 if mode == "split" else 1):
        filename = f"plane-{index}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(image + index),
                fits.ImageHDU(mask),
                fits.ImageHDU(variance + index),
            ]
        ).writeto(tmp_path / filename)
        planes.append(
            {
                "path": filename,
                "hdu": 1,
                "mask_hdu": 2,
                "variance_hdu": 3,
                "bad_mask_bits": 1,
            }
        )
    data = {
        "schema": "cuphoton.xfit.fits-input/v1",
        "mode": mode,
        "stamp_shape": [3, 5],
        "images": planes,
        "candidates": [
            {"candidate_id": "z", "x": 4, "y": 3},
            {"candidate_id": "a", "x": 6, "y": 5},
        ],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    return path, data, image, mask, variance


@pytest.mark.parametrize("mode", ["difference", "split"])
def test_fits_stamps_keep_order_pixels_masks_and_variances(tmp_path, mode):
    path, _, image, mask, variance = _input(tmp_path, mode=mode)
    dataset = load_xfit_dataset(
        path, model="gaussian", mode=mode, reader="astropy"
    )

    def crops(value):
        return np.stack([value[2:5, 2:7], value[4:7, 4:9]])

    expected = crops(image)
    expected_mask = crops((mask & 1) == 0)
    expected_variance = crops(variance)
    if mode == "split":
        expected = np.stack([expected + i for i in range(3)], axis=1)
        expected_mask = np.stack([expected_mask] * 3, axis=1)
        expected_variance = np.stack(
            [expected_variance + i for i in range(3)], axis=1
        )
    np.testing.assert_array_equal(dataset.images, expected)
    np.testing.assert_array_equal(dataset.mask, expected_mask)
    np.testing.assert_array_equal(dataset.variance, expected_variance)
    assert dataset.images.dtype == np.dtype("float32")
    assert dataset.candidate_id.tolist() == ["z", "a"]
    assert dataset.reader_metadata[0]["section"] == [2, 7, 2, 9]
    assert dataset.reader_metadata[0]["role"] == "difference"
    assert dataset.reader_metadata[0]["path"] == str(
        tmp_path / "plane-0.fits"
    )
    if mode == "split":
        assert [read["role"] for read in dataset.reader_metadata] == [
            "difference",
            "positive",
            "negative",
        ]


def test_fits_planning_does_not_decode_or_initialize_a_gpu(
    tmp_path, monkeypatch
):
    path, _, _, _, _ = _input(tmp_path)
    import cuphoton.xfit.fits_input as source

    monkeypatch.setattr(
        source,
        "read_fits_images",
        lambda *a, **k: pytest.fail("planning decoded image payloads"),
    )
    monkeypatch.setattr(
        executor,
        "collect_gpu_identity",
        lambda *a: pytest.fail("planning initialized a GPU"),
    )
    spec = executor.prepare_xfit_workload(
        input_path=path, chunk_size=1, fit_options={"backend": "cupy"}
    )
    assert len(spec.items) == 2
    assert spec.options_payload["candidate_count"] == 2
    assert spec.options_payload["result_dtype"] == "float32"
    assert len(spec.options_payload["input_sources"]) == 1


def test_worker_reads_fits_only_after_binding_and_requests_device(
    tmp_path, monkeypatch
):
    path, _, _, _, _ = _input(tmp_path)

    bound = []
    monkeypatch.setattr(
        executor,
        "collect_gpu_identity",
        lambda backend: bound.append(backend) or {},
    )
    original = executor.load_xfit_dataset
    calls = []

    def load(path, **kwargs):
        assert bound == ["cupy"]
        calls.append(kwargs.copy())
        return original(
            path, **{**kwargs, "device": False, "reader": "astropy"}
        )

    monkeypatch.setattr(executor, "load_xfit_dataset", load)
    items, options = executor.plan_xfit_chunks(
        path, chunk_size=1, fit_options={"fits_reader": "xdr"}
    )
    worker = executor.XFitWorker(options)
    assert calls[0]["device"] is True
    assert calls[0]["reader"] == "xdr"
    assert len(items) == 2
    worker.close()


def test_source_hash_change_is_rejected_even_with_unchanged_stat(tmp_path):
    path, _, _, _, _ = _input(tmp_path)
    _, options = executor.plan_xfit_chunks(path, chunk_size=1, fit_options={})
    source = tmp_path / "plane-0.fits"
    stat = source.stat()
    with fits.open(source, mode="update") as hdus:
        hdus[1].data[0, 0] += 1
    assert source.stat().st_size == stat.st_size
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="referenced FITS input changed"):
        executor._load_input(options)


def test_empty_npz_planning_keeps_validation_error(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(
        path,
        candidate_id=np.empty(0, dtype=np.int64),
        images=np.empty((0, 3, 5), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="at least one candidate"):
        executor.plan_xfit_chunks(path, chunk_size=1, fit_options={})


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda d: d.update(stamp_shape=[2, 5]), "odd"),
        (lambda d: d["candidates"][0].update(x=0), "outside"),
        (lambda d: d["candidates"][0].update(x=4.2), "integer"),
        (lambda d: d["candidates"][0].update(candidate_id="a"), "unique"),
        (
            lambda d: d["candidates"][0].update(candidate_id=1),
            "all integers or all strings",
        ),
        (lambda d: d.update(alignment="auto"), "unsupported fields"),
        (
            lambda d: d["images"][0].update(bad_mask_bits=1 << 32),
            "dtype width",
        ),
        (lambda d: d.update(mode="split"), "requires 3"),
        (lambda d: d.update(initial=[[1], [2]]), "8 parameters"),
    ],
)
def test_manifest_rejects_ambiguous_or_invalid_inputs(
    tmp_path, mutation, error
):
    path, data, _, _, _ = _input(tmp_path)
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        plan_fits_input(path, model="gaussian")


def test_masked_nonfinite_pixels_preserve_existing_validation(tmp_path):
    path, _, _, _, _ = _input(tmp_path)
    image_path = tmp_path / "plane-0.fits"
    with fits.open(image_path, mode="update") as hdus:
        hdus[1].data[3, 3] = np.nan
        hdus[3].data[3, 3] = -1
    load_fits_input(path, reader="astropy", model="gaussian")
    with fits.open(image_path, mode="update") as hdus:
        hdus[2].data[3, 3] = 0
    with pytest.raises(ValueError, match="finite at included"):
        load_fits_input(path, reader="astropy", model="gaussian")


@pytest.mark.parametrize("reader", ["auto", "astropy"])
def test_npz_and_fits_cli_keep_identical_numeric_outputs(
    tmp_path, capsys, monkeypatch, reader
):
    path, _, _, _, _ = _input(tmp_path)
    dataset = load_fits_input(path, reader="astropy", model="gaussian")
    npz = tmp_path / "stamps.npz"
    np.savez(
        npz,
        candidate_id=dataset.candidate_id,
        images=dataset.images,
        mask=dataset.mask,
        variance=dataset.variance,
    )
    import cuphoton.xfit.fits_input as source

    original = source.read_fits_images

    def read(*args, **kwargs):
        assert kwargs["reader"] == "astropy"
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "read_fits_images", read)
    for label, input_path in (("fits", path), ("npz", npz)):
        assert (
            run_component(
                "xfit",
                [
                    "fit-dipoles",
                    "--input",
                    str(input_path),
                    "--fits-reader",
                    reader,
                    "--model",
                    "gaussian",
                    "--backend",
                    "numpy",
                    "--max-evaluations",
                    "2",
                    "--output-dir",
                    str(tmp_path / label),
                ],
            )
            == 0
        )
        capsys.readouterr()
    left = pq.read_table(tmp_path / "fits/fits.parquet")
    right = pq.read_table(tmp_path / "npz/fits.parquet")
    assert left.schema == right.schema
    for name in left.column_names:
        a, b = left[name].to_numpy(), right[name].to_numpy()
        if a.dtype.kind in "fiu":
            np.testing.assert_array_equal(a, b)
        else:
            assert a.tolist() == b.tolist()
    with (
        np.load(tmp_path / "fits/fit-arrays.npz") as left,
        np.load(tmp_path / "npz/fit-arrays.npz") as right,
    ):
        assert left.files == right.files
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])


def test_fits_distributed_finalizer_preserves_ordinary_outputs(
    tmp_path, monkeypatch
):
    path, _, _, _, _ = _input(tmp_path)
    original = executor.load_xfit_dataset
    monkeypatch.setattr(executor, "collect_gpu_identity", lambda _: {})

    def read(path, **kwargs):
        dataset = original(
            path, **{**kwargs, "device": False, "reader": "astropy"}
        )
        if kwargs.get("device"):
            # Simulate a bound xDR worker and retain the real host audit read.
            dataset = replace(
                dataset,
                reader_metadata=tuple(
                    {**receipt, "reader": "xdr", "location": "device"}
                    for receipt in dataset.reader_metadata
                ),
            )
        return dataset

    monkeypatch.setattr(
        executor,
        "load_xfit_dataset",
        read,
    )
    monkeypatch.setattr(
        executor,
        "fit_dipoles",
        lambda *args, **kwargs: replace(
            fit_dipoles(*args, **{**kwargs, "backend": "numpy"}),
            backend="cupy",
            device="cuda:0",
        ),
    )
    spec = executor.prepare_xfit_workload(
        input_path=path,
        chunk_size=1,
        fit_options={"backend": "cupy", "max_evaluations": 2},
    )
    worker = spec.worker_factory(spec.options_payload)
    output = tmp_path / "round"
    records = []
    for item in spec.items:
        worker.run_item(item, output / "items" / item.item_id)
        records.append({"item_id": item.item_id, "status": "success"})
    result = spec.finalize_round(output, records)
    assert result["candidate_count"] == 2
    assert pq.read_table(output / "scientific/fits.parquet")[
        "candidate_id"
    ].to_pylist() == ["z", "a"]
    summary = json.loads((output / "scientific/summary.json").read_text())
    assert "fits_reads" not in summary["inputs"]
    assert (
        summary["inputs"]["fits_finalizer_audit_reads"][0]["reader"]
        == "astropy"
    )
    receipts = summary["inputs"]["fits_worker_setup_reads"]
    assert receipts == result["fits_worker_setup_reads"]
    assert len(receipts) == 2
    assert all(row["reads"][0]["reader"] == "xdr" for row in receipts)
    worker.close()


def test_manifest_initial_and_stamp_basis_are_preserved(tmp_path):
    path, data, _, _, _ = _input(tmp_path)
    data["initial"] = [[-1, -1, 1, 1, 2], [-2, 0, 2, 0, 3]]
    basis = np.full((5, 5), 0.04, dtype=np.float32)
    fits.PrimaryHDU(basis).writeto(tmp_path / "psf.fits")
    data["stamp_basis"] = {"path": "psf.fits", "hdu": 0}
    path.write_text(json.dumps(data))
    dataset = load_fits_input(path, model="stamp", reader="astropy")
    np.testing.assert_array_equal(dataset.initial, data["initial"])
    np.testing.assert_array_equal(dataset.stamp_basis, basis)
    assert len(dataset.input_sources) == 2


@pytest.mark.skipif(
    os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="GPU tests disabled"
)
@pytest.mark.parametrize("compressed", [False, True])
def test_gpu_fits_stamps_remain_on_device_and_match_host(
    tmp_path, compressed
):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("a CUDA device is required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("a CUDA device is required")
    path, _, _, _, _ = _input(tmp_path)
    if compressed:
        source = tmp_path / "plane-0.fits"
        with fits.open(source) as hdus:
            arrays = [hdu.data.copy() for hdu in hdus[1:]]
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                *(
                    fits.CompImageHDU(
                        array, compression_type="GZIP_2", quantize_level=0
                    )
                    for array in arrays
                ),
            ]
        ).writeto(source, overwrite=True)
    expected = load_fits_input(path, reader="astropy", model="gaussian")
    actual = load_fits_input(
        path, reader="xdr", model="gaussian", device=True
    )
    for name in ("images", "mask", "variance"):
        assert isinstance(getattr(actual, name), cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)), getattr(expected, name)
        )
    assert actual.reader_metadata[0]["reader"] == "xdr"
    assert actual.reader_metadata[0]["location"] == "device"
    baseline = fit_dipoles(
        expected.images,
        model="gaussian",
        mask=expected.mask,
        variance=expected.variance,
        backend="cupy",
        config=LMConfig(max_evaluations=2),
    )
    accelerated = fit_dipoles(
        actual.images,
        model="gaussian",
        mask=actual.mask,
        variance=actual.variance,
        backend="cupy",
        config=LMConfig(max_evaluations=2),
    )
    for name in ("parameters", "covariance", "residuals", "converged"):
        np.testing.assert_array_equal(
            getattr(accelerated, name), getattr(baseline, name)
        )


def test_candidate_copy_failure_retains_decoded_owners_until_sync(
    monkeypatch,
):
    from cuphoton.xdr import prefetch
    from cuphoton.xfit.fits_input import _candidate_stamps

    result = SimpleNamespace(arrays=(np.ones((3, 3)), np.zeros((3, 3))))
    plan = SimpleNamespace(stamp_shape=(3, 3), centers=((1, 1),))
    copied = []

    def stack(values):
        if copied:
            raise RuntimeError("candidate copy failed")
        copied.append(np.stack(values))
        return copied[-1]

    sync = []

    def synchronize(handle, *, synchronize_stream):
        assert synchronize_stream
        assert handle.keepalive[0] is result
        assert handle.keepalive[1]["hdu"] is copied[0]
        sync.append(True)

    monkeypatch.setattr(prefetch, "_synchronize_gpu_batch", synchronize)
    ap = SimpleNamespace(
        stack=stack,
        cuda=SimpleNamespace(
            get_current_stream=lambda: object(),
            runtime=SimpleNamespace(getDevice=lambda: 0),
        ),
    )
    with pytest.raises(RuntimeError, match="candidate copy failed"):
        _candidate_stamps(
            result, ["hdu", "variance_hdu"], plan, offset=(0, 0), ap=ap
        )
    assert sync == [True]
