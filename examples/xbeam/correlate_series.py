# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create independent synthetic series, then demonstrate delay estimation.

Run: python examples/xbeam/correlate_series.py /tmp/series-example
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series
from cuphoton.xbeam.adapters.hdf5 import write_series_hdf5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--backend", choices=["numpy", "cupy"], default="numpy"
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    a = rng.normal(size=8192).astype(np.float32)
    b = np.zeros_like(a)
    b[7:] = a[:-7]
    with h5py.File(args.directory / "signals.h5", "x") as output:
        for name, samples in (("a", a), ("b", b)):
            dataset = output.create_dataset(name, data=samples, chunks=True)
            dataset.attrs["sample_rate_hz"] = 1024.0
            dataset.attrs["origin_seconds"] = 0.0
    config = {
        "input": "signals.h5",
        "datasets": {"a": "a", "b": "b"},
        "correlation": {
            "pairs": [["a", "b"]],
            "window_samples": 1024,
            "windows_per_result": 4,
            "output": "both",
        },
    }
    (args.directory / "series.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    with Hdf5Source(
        args.directory / "signals.h5", config["datasets"]
    ) as source:
        spec = CorrelationSpec(**config["correlation"], backend=args.backend)

        def measured():
            for result in correlate_series(source, spec):
                peak = result.lag_samples[np.argmax(np.abs(result.lag[0]))]
                print(
                    f"result {result.result_index}: B delay = {peak} samples"
                )
                yield result

        write_series_hdf5(args.directory / "results.h5", measured())


if __name__ == "__main__":
    main()
