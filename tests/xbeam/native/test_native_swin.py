# SPDX-License-Identifier: Apache-2.0
"""Native SWIN serialization must preserve the observable byte contract."""

import importlib.util
import io
import os
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.swin import SwinRecord, write_record


@pytest.fixture(scope="module")
def binding():
    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("build the binding and set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    assert len(files) == 1
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("channels", [0, 1, 3, 257, 4099])
@pytest.mark.parametrize("polarization", ["XX", "XY", "RR", "LL", "XL"])
def test_exact_swin_bytes(binding, channels, polarization, tmp_path):
    rng = np.random.default_rng(42)
    spectrum = rng.normal(size=channels) + 1j * rng.normal(size=channels)
    spectrum = spectrum.astype(np.complex64)
    reference = SwinRecord(
        258,
        60000,
        42.5,
        2,
        3,
        7,
        polarization,
        0,
        0.875,
        -123456.75,
        1.125,
        99999.25,
        spectrum,
    )
    out = io.BytesIO()
    write_record(out, reference)
    actual = binding.encode_visibility(
        reference.baseline,
        reference.mjd,
        reference.seconds,
        reference.configuration_id,
        reference.source_id,
        reference.frequency_id,
        polarization.encode("ascii"),
        reference.pulsar_bin,
        reference.weight,
        reference.u_m,
        reference.v_m,
        reference.w_m,
        spectrum.view(np.float32),
    )
    assert actual == out.getvalue()
    path = tmp_path / "visibility.swin"
    binding.write_visibility_bytes(path, actual)
    assert path.read_bytes() == actual
    with pytest.raises(RuntimeError, match="cannot create SWIN output"):
        binding.write_visibility_bytes(path, b"do not overwrite")
    assert path.read_bytes() == actual


@pytest.mark.parametrize("baseline", [0, 1, 256, 65536, -1])
def test_invalid_baselines(binding, baseline):
    with pytest.raises(RuntimeError, match="baseline"):
        binding.encode_visibility(
            baseline,
            60000,
            0,
            0,
            0,
            0,
            b"RR",
            0,
            1,
            0,
            0,
            0,
            np.zeros(2, dtype=np.float32),
        )


def test_reject_nul_filename(binding, tmp_path):
    with pytest.raises(RuntimeError, match="NUL"):
        binding.write_visibility_bytes(
            str(tmp_path / "bad") + "\x00tail", b""
        )
