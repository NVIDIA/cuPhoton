# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xdr import nvcomp_batch


@pytest.mark.parametrize("flag", [0x08, 0x10])
def test_parse_gzip_header_rejects_unterminated_optional_string(flag):
    head = bytearray(16)
    head[0:2] = b"\x1f\x8b"
    head[3] = flag
    head[10:] = b"abcdef"

    with pytest.raises(ValueError, match="truncated gzip header"):
        nvcomp_batch._parse_gzip_header_len(bytes(head))


@pytest.mark.parametrize(
    ("buffer_size", "offset", "length"),
    [(15, 0, 15), (20, 10, 16)],
)
def test_header_probe_rejects_tile_before_device_gather(
    buffer_size, offset, length
):
    class DeviceBuffer:
        size = buffer_size

        def __getitem__(self, key):
            raise AssertionError("device gather must not run")

    with pytest.raises(ValueError, match="gzip tile header probe"):
        nvcomp_batch._compute_header_sizes_from_device(
            DeviceBuffer(),
            np.array([offset], dtype=np.int64),
            np.array([length], dtype=np.int64),
        )


def test_gpu_library_preload_orders_kvikio_dependency_first(
    monkeypatch, tmp_path
):
    package_dirs = {}
    for module, library in (
        ("nvidia.libnvcomp", "libnvcomp.so.5"),
        ("rapids_logger", "librapids_logger.so"),
        ("libkvikio", "libkvikio.so"),
    ):
        package_dir = tmp_path / module.replace(".", "-")
        lib_dir = package_dir / "lib64"
        lib_dir.mkdir(parents=True)
        (lib_dir / library).touch()
        package_dirs[module] = package_dir

    loaded = []
    monkeypatch.setattr(
        nvcomp_batch,
        "_package_dir",
        lambda module: package_dirs.get(module),
    )
    monkeypatch.setattr(
        nvcomp_batch,
        "_load_shared_library",
        lambda path: loaded.append(path.name),
    )

    nvcomp_batch._preload_gpu_package_libraries()

    assert loaded == [
        "libnvcomp.so.5",
        "librapids_logger.so",
        "libkvikio.so",
    ]
