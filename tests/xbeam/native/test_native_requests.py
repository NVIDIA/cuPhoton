# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
import random
from fractions import Fraction
from pathlib import Path

import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    buffered_request_cursor,
)


@pytest.fixture(scope="module")
def binding():
    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    assert len(files) == 1
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "ratio,n,alignment",
    [
        (Fraction(4), 262144, 128),
        (Fraction(1, 8), 32, 32),
        (Fraction(3, 2), 36, 8),
        (Fraction(2, 3), 32, 24),
    ],
)
def test_native_cursors_exact(binding, ratio, n, alignment):
    geometry = BufferedRequestGeometry(ratio, n, 50, alignment, 2)
    rng = random.Random(61)
    local_values = [-(2**60), 2**60, -n * 50, -n, -4, -1, 0, 7, 1023]
    local_values += [rng.randrange(-n * 51, n * 51) for _ in range(200)]
    for local in local_values:
        for drift in (0.0, 8e-7, -8e-7):
            expected = buffered_request_cursor(
                local,
                drift,
                geometry,
                sample_rate=4096000000.0,
            )
            if expected is not None and any(
                not -(2**63) <= value < 2**63
                for value in (
                    expected.origin_bytes,
                    expected.skipped_ffts,
                    expected.decoder_delta_samples,
                )
            ):
                with pytest.raises(RuntimeError, match="int64"):
                    binding.buffered_cursor(
                        local,
                        drift,
                        ratio.numerator,
                        ratio.denominator,
                        n,
                        50,
                        alignment,
                        2,
                        4096000000.0,
                    )
                continue
            actual = binding.buffered_cursor(
                local,
                drift,
                ratio.numerator,
                ratio.denominator,
                n,
                50,
                alignment,
                2,
                4096000000.0,
            )
            if expected is None:
                assert actual is None
            else:
                assert actual == (
                    expected.origin_bytes,
                    expected.skipped_ffts,
                    expected.decoder_delta_samples,
                ), (ratio, n, local, drift, actual, expected)


@pytest.mark.parametrize("drift", [float("nan"), float("inf")])
def test_nonfinite_drift(binding, drift):
    with pytest.raises(RuntimeError, match="finite"):
        binding.buffered_cursor(
            0, drift, 4, 1, 262144, 50, 128, 2, 4096000000.0
        )


def test_large_coordinate_overflow_is_checked(binding):
    with pytest.raises(RuntimeError, match="int64"):
        binding.buffered_cursor(2**63 - 1, 0, 1, 8, 32, 50, 32, 2, 128)
