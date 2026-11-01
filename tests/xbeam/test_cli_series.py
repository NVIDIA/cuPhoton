# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line behavior for xBeam series."""

import json

import numpy as np
import pytest


def test_hdf5_adapter_and_cli(tmp_path):
    h5py = pytest.importorskip("h5py")
    from cuphoton.core.cli import run_component
    from cuphoton.xbeam.adapters.hdf5 import Hdf5Source

    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as output:
        for name in ["a", "b"]:
            ds = output.create_dataset(
                name,
                data=np.arange(64, dtype=np.float32),
                chunks=True,
                compression="gzip",
            )
            ds.attrs.update(sample_rate_hz=4.0, origin_seconds=2.0)
        output.create_dataset("mask", data=np.ones(64, dtype=bool))
    config = dict(
        input="input.h5",
        datasets={"a": "a", "b": "b"},
        validity={"a": "mask"},
        correlation=dict(pairs=[["a", "b"]], window_samples=8, output="both"),
    )
    config_path = tmp_path / "series.json"
    config_path.write_text(json.dumps(config))
    output_path = tmp_path / "output.h5"
    assert (
        run_component(
            "xbeam",
            [
                "correlate-series",
                str(config_path),
                "--output",
                str(output_path),
            ],
        )
        == 0
    )
    with h5py.File(output_path) as output:
        assert output.attrs["complete"]
        assert len(output["results"]) == 8
        assert output["results/0/spectrum"].dtype == np.complex64
        assert output["results/0"].attrs["origin_seconds"] == 2
    assert (
        run_component(
            "xbeam",
            [
                "correlate-series",
                str(config_path),
                "--output",
                str(output_path),
            ],
        )
        == 1
    )
    with h5py.File(path, "a") as output:
        output["b"].attrs["origin_seconds"] = 3
    with pytest.raises(ValueError, match="common sample grid"):
        Hdf5Source(path, {"a": "a", "b": "b"})
