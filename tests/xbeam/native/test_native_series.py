# SPDX-License-Identifier: Apache-2.0
"""Native extent admission and optional-build behavior need no GPU."""

import numpy as np
import pytest

from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series
from cuphoton.xbeam.adapters.arrays import ArraySource
from cuphoton.xbeam.native.series import _configuration


def write_input(path, *, dtype="float32", chunks=None, count=128):
    import h5py

    with h5py.File(
        path, "w", alignment_threshold=1, alignment_interval=4096
    ) as f:
        for name in ("a", "b"):
            ds = f.create_dataset(
                name, data=np.arange(count, dtype=dtype), chunks=chunks
            )
            ds.attrs["sample_rate_hz"] = 64.0
            ds.attrs["origin_seconds"] = 12.5
        f.create_dataset("valid", data=np.ones(count, dtype=bool))
    return path


@pytest.mark.parametrize(
    "dtype,code,wide",
    [
        ("float32", 0, False),
        ("float64", 1, True),
        ("complex64", 2, False),
        ("complex128", 3, True),
    ],
)
def test_native_extent_descriptors(tmp_path, dtype, code, wide):
    path = write_input(tmp_path / "input.h5", dtype=dtype)
    spec = CorrelationSpec((("a", "b"),), 16, backend="native")
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths={"a": "valid"}
    ) as source:
        config = _configuration(source, spec)
        assert config["wide"] == wide
        assert config["inputs"][0]["dtype"] == code
        assert config["inputs"][0]["offset"] > 0
        assert config["inputs"][0]["mask_offset"] > 0
        assert config["inputs"][1]["mask_offset"] == -1
        assert config["pairs"] == [(0, 1)]
        assert config["inode"] == path.stat().st_ino
        assert config["sample_count"] == 128


@pytest.mark.parametrize("options", [{"chunks": (32,)}, {"dtype": "int16"}])
def test_native_rejects_layouts_without_fallback(tmp_path, options):
    path = write_input(tmp_path / "input.h5", **options)
    with Hdf5Source(path, {"a": "a", "b": "b"}) as source:
        with pytest.raises(ValueError, match="contiguous"):
            list(
                correlate_series(
                    source,
                    CorrelationSpec((("a", "b"),), 16, backend="native"),
                )
            )


def test_native_missing_build_is_actionable(tmp_path, monkeypatch):
    from cuphoton.xbeam import native

    path = write_input(tmp_path / "input.h5")
    monkeypatch.setattr(native, "cuda_available", lambda: False)
    with Hdf5Source(path, {"a": "a", "b": "b"}) as source:
        with pytest.raises(RuntimeError, match="CUPHOTON_XBEAM_NATIVE_CUDA"):
            list(
                correlate_series(
                    source,
                    CorrelationSpec((("a", "b"),), 16, backend="native"),
                )
            )


def test_native_rejects_non_hdf5_sources():
    source = ArraySource({"a": np.ones(16)}, sample_rate_hz=1)
    with pytest.raises(TypeError, match="Hdf5Source"):
        list(
            correlate_series(
                source, CorrelationSpec((("a", "a"),), 4, backend="native")
            )
        )


def test_radio_backend_does_not_admit_native_series():
    from cuphoton.xbeam.commands import (
        BackendInvariant,
        SeriesBackendInvariant,
    )

    assert "native" in SeriesBackendInvariant._set
    assert "native" not in BackendInvariant._set


def test_series_cli_admits_native_and_fails_before_output_without_build(
    tmp_path, monkeypatch
):
    import json
    from io import StringIO

    from cuphoton.core.cli import run_component
    from cuphoton.xbeam import native

    path = write_input(tmp_path / "input.h5")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "input": str(path),
                "datasets": {"a": "a", "b": "b"},
                "correlation": {"pairs": [["a", "b"]], "window_samples": 16},
            }
        )
    )
    output = tmp_path / "output.h5"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(native, "cuda_available", lambda: False)
    error = StringIO()
    status = run_component(
        "xbeam",
        [
            "correlate-series",
            str(config),
            "--backend",
            "native",
            "--output",
            str(output),
        ],
        err=error,
    )
    assert status != 0
    assert "CUPHOTON_XBEAM_NATIVE_CUDA" in error.getvalue()
    assert not output.exists()
