# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def test_xdr_uses_cuphoton_namespace():
    root = Path(__file__).resolve().parents[2]

    assert (root / "pyproject.toml").is_file()
    package = root / "src" / "cuphoton" / "xdr"
    assert (package / "__init__.py").is_file()
    assert not (root / "xdr").exists()


def test_native_extension_uses_xdr_namespace_case():
    root = Path(__file__).resolve().parents[2]
    package = root / "src" / "cuphoton" / "xdr"

    setup_source = (package / "setup_package.py").read_text()
    extension_source = (package / "src" / "nvcomp_batch_ext.cpp").read_text()

    assert 'name="cuphoton.xdr._nvcomp_batch_ext"' in setup_source
    assert "namespace xdr_gpu {" in extension_source


def test_xdr_build_controls_use_product_prefix():
    root = Path(__file__).resolve().parents[2]
    package = root / "src" / "cuphoton" / "xdr"
    sources = "\n".join(
        path.read_text()
        for path in (
            root / "setup.py",
            package / "mock_storage.py",
            package / "nvcomp_batch.py",
            package / "setup_package.py",
            package / "src" / "build.sh",
            package / "src" / "memory_manager.cpp",
        )
    )
    expected = {
        "CUPHOTON_XDR_BUILD_EXT",
        "CUPHOTON_XDR_CFITSIO_ROOT",
        "CUPHOTON_XDR_NVCOMP_LIB_DIR",
        "CUPHOTON_XDR_KVIKIO_LIB_DIR",
        "CUPHOTON_XDR_RAPIDS_LOGGER_LIB_DIR",
        "CUPHOTON_XDR_PINNED_POOL_MAX_BYTES",
        "CUPHOTON_XDR_DEVICE_POOL_MAX_BYTES",
        "CUPHOTON_XDR_MOCK_STORAGE",
        "CUPHOTON_XDR_SILENCE_CPP_FALLBACK",
    }

    assert all(name in sources for name in expected)
