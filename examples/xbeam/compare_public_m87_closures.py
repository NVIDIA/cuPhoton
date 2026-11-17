# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare xBeam closures with eht-imaging on public 2017 M87 Stokes I data.

See docs/xbeam/public-m87-closures.md for data provenance and interpretation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import sys
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from cuphoton.xbeam.adapters.radio.closure import compute_closures
from cuphoton.xbeam.adapters.radio.output import OutputRecord

FILENAME = "SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits"
DATA_URL = (
    "https://data.cyverse.org/dav-anon/iplant/home/shared/commons_repo/"
    "curated/EHTC_FirstM87Results_Apr2019/uvfits/" + FILENAME
)
DATA_SHA256 = (
    "697af2bb3bbf732115108ffefacd3e59e307f38fe685c3c4579146b0bd661298"
)
REFERENCE_VERSION = "1.3.2"


def seconds(hours: float) -> Decimal:
    """Use the same exact timestamp conversion on both comparison paths."""
    return Decimal(str(float(hours))) * 3600


def compare_observation(obs, *, tolerance: float = 1e-12) -> dict:
    """Check all finite, available maximal-set closures without averaging.

    This consumes an eht-imaging Stokes I observation, not raw recordings.
    The loader, frequency averaging, Stokes conversion and requested station
    tuples are shared inputs; the closure arithmetic is independent.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if obs.polrep != "stokes" or obs.timetype != "UTC":
        raise ValueError("comparison requires Stokes I data and UTC times")
    data = obs.data
    if not len(data) or not np.all(
        np.isfinite(data["vis"])
        & (np.abs(data["vis"]) > 0)
        & np.isfinite(data["sigma"])
        & (data["sigma"] > 0)
        & np.isfinite(data["tint"])
        & (data["tint"] > 0)
    ):
        raise ValueError("comparison requires finite nonzero valid inputs")

    # The reference computes all available instantaneous closures. It has no
    # SNR cut and no closure-amplitude debiasing, matching xBeam's arithmetic.
    phases = obs.c_phases(
        vtype="vis", mode="all", count="max", ang_unit="rad", snrcut=0.0
    )
    amplitudes = obs.c_amplitudes(
        vtype="vis",
        mode="all",
        count="max",
        ctype="logcamp",
        debias=False,
        snrcut=0.0,
    )
    reference = {}
    triangles = set()
    quadrangles = set()
    for kind, table, field in (
        ("phase", phases, "cphase"),
        ("log_amplitude", amplitudes, "camp"),
    ):
        for row in table:
            if kind == "phase":
                stations = tuple(str(row[f"t{i}"]) for i in (1, 2, 3))
                triangles.add(stations)
            else:
                # ehtim uses |12 34| / |14 23|. Swap its last two stations
                # to match xBeam's |ab cd| / |ac bd| definition.
                stations = tuple(str(row[f"t{i}"]) for i in (1, 2, 4, 3))
                quadrangles.add(stations)
            key = (kind, seconds(row["time"]), stations)
            if key in reference:
                raise ValueError("duplicate closure in reference output")
            reference[key] = float(row[field])
    if not len(phases) or not len(amplitudes):
        raise ValueError("comparison needs both triangles and quadrangles")

    records = [
        OutputRecord(
            antenna_a=str(row["t1"]),
            antenna_b=str(row["t2"]),
            mjd=int(obs.mjd),
            seconds_of_day=seconds(row["time"]),
            source_id=str(obs.source),
            configuration_id="EHT-2017",
            output_frequency_id="lo-band-averaged",
            polarization_a="I",
            polarization_b="I",
            weight=1.0,
            # Geometry is unused by closures. Obsdata has no w coordinate;
            # do not imply that this adapter supplies validated UVW metadata.
            u_m=float("nan"),
            v_m=float("nan"),
            w_m=float("nan"),
            spectrum=np.array([row["vis"]], dtype=np.complex128),
        )
        for row in data
    ]
    closures = compute_closures(
        records, triangles=sorted(triangles), quadrangles=sorted(quadrangles)
    )
    observed = {
        (item.kind, item.key.seconds_of_day, item.stations): float(
            item.values[0]
        )
        for item in closures
        if item.valid[0]
    }
    if observed.keys() != reference.keys():
        missing = len(reference.keys() - observed.keys())
        extra = len(observed.keys() - reference.keys())
        raise AssertionError(
            f"closure coverage mismatch: {missing=}, {extra=}"
        )

    maximum_error = {"phase": 0.0, "log_amplitude": 0.0}
    for key, expected in reference.items():
        difference = observed[key] - expected
        if key[0] == "phase":
            difference = np.arctan2(np.sin(difference), np.cos(difference))
        error = float(abs(difference))
        if not np.isfinite(error) or error > tolerance:
            raise AssertionError(f"closure mismatch at {key}: {error=}")
        maximum_error[key[0]] = max(maximum_error[key[0]], error)

    return {
        "visibility_records": len(data),
        "integration_times": len(np.unique(data["time"])),
        "closure_phases": len(phases),
        "log_closure_amplitudes": len(amplitudes),
        "unavailable_closures": len(closures) - len(observed),
        "max_phase_error_rad": maximum_error["phase"],
        "max_log_amplitude_error": maximum_error["log_amplitude"],
        "tolerance": tolerance,
    }


def verify_data(path: Path, *, download: bool = False) -> None:
    """Download only the fixed public file and require its exact checksum."""
    if download and not path.exists():
        # Verify before writing; the release file is only 650,880 bytes.
        with urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
            raise ValueError(
                "downloaded M87 file does not match release hash"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            target.write(payload)
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != DATA_SHA256:
        raise ValueError("M87 input does not match the pinned release file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uvfits", type=Path, help=f"local path to {FILENAME}")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the fixed public file if the path does not exist",
    )
    args = parser.parse_args()
    version = importlib.metadata.version("ehtim")
    if version != REFERENCE_VERSION:
        parser.error(
            f"install ehtim=={REFERENCE_VERSION} for this comparison"
        )
    verify_data(args.uvfits, download=args.download)
    # ehtim's informational output goes to stderr, leaving stdout as JSON.
    with contextlib.redirect_stdout(sys.stderr):
        import ehtim as eh

        obs = eh.obsdata.load_uvfits(str(args.uvfits), polrep="stokes")
        summary = compare_observation(obs)
    summary.update(
        {
            "data_url": DATA_URL,
            "data_sha256": DATA_SHA256,
            "ehtim_version": version,
            "numpy_version": np.__version__,
            "cuphoton_version": importlib.metadata.version("cuphoton"),
            "source": obs.source,
            "mjd": obs.mjd,
        }
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
