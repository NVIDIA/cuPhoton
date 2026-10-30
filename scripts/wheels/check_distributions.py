#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the native cuPhoton wheel matrix and source distribution."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ABIS = ("cp312", "cp313", "cp314")
ARCHITECTURES = ("x86_64", "aarch64")
LICENSES = {"LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/CCCL-3.1.2.txt"}
CCCL_LICENSE_SHA256 = (
    "f96f51edda77fb9897de29d924d615e6bd153f4969db85fc6b7a22d7a624631e"
)
# Complete notice blocks from pybind11 3.0.4 LICENSE and the CUB/LLVM
# sections in CuPy 14.1.1's cupy/_core/include/cupy/_cccl/LICENSE.
NATIVE_NOTICE_SHA256 = {
    "pybind11 license": (
        "83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb"
    ),
    "CUB license": (
        "f1fc812e3325b255db010dd0d11e9ae0166eab00ee2d1497093ede413cd19b52"
    ),
    "LLVM exception": (
        "19352480e53ee7d6ab21ab8f47da3e1ab07561f341d5748e0b5e1644cd192409"
    ),
}
SOURCES = {
    "setup.py",
    "pyproject.toml",
    "src/cuphoton/xdr/setup_package.py",
    "src/cuphoton/xdr/src/nvcomp_batch_ext.cpp",
    "src/cuphoton/xdr/src/nvcomp_batch_ext.h",
    "src/cuphoton/xdr/src/memory_manager.cpp",
    "src/cuphoton/xdr/src/io.cpp",
    "src/cuphoton/xdr/src/build.sh",
    "src/cuphoton/xfit/setup_package.py",
    "src/cuphoton/xfit/src/native_ext.cpp",
    "src/cuphoton/xfit/src/native_api.h",
    "src/cuphoton/xfit/src/native.cu",
    "src/cuphoton/xfit/src/native_norm.cu",
    "src/cuphoton/xfit/src/native_jacobian.h",
    "src/cuphoton/xfit/src/native_jacobian.cu",
    "src/cuphoton/xfit/src/native_jacobian_lean.cu",
    "scripts/wheels/prepare_cfitsio.sh",
    "scripts/wheels/patches/cfitsio-version-string.patch",
    "scripts/wheels/install_build_dependencies.sh",
    "scripts/wheels/build-requirements.txt",
    "scripts/wheels/repair_wheel.py",
    "scripts/wheels/test_installed.py",
    "scripts/wheels/check_distributions.py",
}
BINARY = re.compile(r"\.(?:so(?:\.\d+)*|a|o|pyd|dll|dylib|whl|pyc|pyo)$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_native_notices(content, artifact):
    blocks = re.findall(
        rb"```text\n(.*?)```", content.replace(b"\r\n", b"\n"), re.DOTALL
    )
    digests = {hashlib.sha256(block).hexdigest() for block in blocks}
    for name, digest in NATIVE_NOTICE_SHA256.items():
        require(
            digest in digests,
            f"{artifact}: missing or altered full {name} notice",
        )


def check_cccl_license(content, artifact):
    require(
        hashlib.sha256(content).hexdigest() == CCCL_LICENSE_SHA256,
        f"{artifact}: missing or altered full CCCL license notice",
    )


def check_xfit_libraries(content, artifact):
    """Require dynamic CUDA/cuBLAS linkage in the delivered xFit ELF."""
    with tempfile.TemporaryDirectory(prefix="cuphoton-elf-") as directory:
        extension = Path(directory) / "xfit.so"
        extension.write_bytes(content)
        result = subprocess.run(
            ["readelf", "--dynamic", str(extension)],
            capture_output=True,
            text=True,
            check=False,
        )
    require(result.returncode == 0, f"{artifact}: invalid xFit ELF library")
    needed = set(re.findall(r"\(NEEDED\).*\[([^\]]+)\]", result.stdout))
    require(
        {"libcudart.so.13", "libcublas.so.13"} <= needed,
        f"{artifact}: xFit must dynamically link CUDA 13 and cuBLAS 13",
    )
    require(
        not any("/" in name for name in needed),
        f"{artifact}: xFit contains an absolute library dependency",
    )


def check_metadata(content, version, artifact, license_expression):
    metadata = BytesParser().parsebytes(content)
    require(metadata["Name"] == "cuphoton", f"{artifact}: wrong project name")
    require(metadata["Version"] == version, f"{artifact}: version mismatch")
    require(
        metadata.get_all("License-Expression") == [license_expression],
        f"{artifact}: expected License-Expression: {license_expression}",
    )
    python_bounds = (metadata["Requires-Python"] or "").replace(" ", "")
    require(
        set(python_bounds.split(",")) == {">=3.12", "<3.15"},
        f"{artifact}: Requires-Python must be >=3.12,<3.15",
    )
    require(
        {"io", "gpu"} <= set(metadata.get_all("Provides-Extra", [])),
        f"{artifact}: missing io/gpu extras",
    )


def check_wheel(path):
    match = re.fullmatch(
        r"cuphoton-([^-]+)-(?:\d[^-]*-)?(cp31[234])-\2-(.+)\.whl", path.name
    )
    require(match is not None, f"{path.name}: unexpected wheel filename/ABI")
    version, abi, platform = match.groups()
    platforms = platform.split(".")
    architecture = re.search(
        r"(?:^|\.)manylinux_2_28_(x86_64|aarch64)(?:\.|$)", platform
    )
    require(
        architecture is not None, f"{path.name}: missing manylinux_2_28 tag"
    )
    arch = architecture.group(1)
    require(
        all(
            re.fullmatch(rf"manylinux(?:_\d+_\d+|20\d+)_{arch}", tag)
            for tag in platforms
        ),
        f"{path.name}: inconsistent platform architecture",
    )
    info = f"cuphoton-{version}.dist-info/"
    native = {
        f"cuphoton/{component}/{module}.cpython-{abi[2:]}-{arch}-linux-gnu.so"
        for component, module in (
            ("xdr", "_nvcomp_batch_ext"),
            ("xfit", "_native_ext"),
        )
    }
    with zipfile.ZipFile(path) as archive:
        files = set(archive.namelist())
        required = native | {info + "METADATA", info + "WHEEL"}
        required.update(info + "licenses/" + name for name in LICENSES)
        require(
            required <= files,
            f"{path.name}: missing {sorted(required - files)}",
        )
        check_metadata(
            archive.read(info + "METADATA"),
            version,
            path.name,
            "Apache-2.0 AND CFITSIO AND BSD-3-Clause "
            "AND Apache-2.0 WITH LLVM-exception",
        )
        wheel = BytesParser().parsebytes(archive.read(info + "WHEEL"))
        require(
            wheel["Root-Is-Purelib"] == "false", f"{path.name}: pure wheel"
        )
        require(
            set(wheel.get_all("Tag", []))
            == {f"{abi}-{abi}-{tag}" for tag in platforms},
            f"{path.name}: WHEEL tags disagree with filename",
        )
        cfitsio = {
            name
            for name in files
            if re.fullmatch(
                r"cuphoton\.libs/libcfitsio-[0-9a-f]{8,}\.so(?:\.\d+)*", name
            )
        }
        require(
            len(cfitsio) == 1,
            f"{path.name}: expected one renamed CFITSIO library",
        )
        require(
            b"CFITSIO 4.7.0\0" in archive.read(next(iter(cfitsio))),
            f"{path.name}: bundled CFITSIO is missing its 4.7.0 identity",
        )
        binaries = {name for name in files if BINARY.search(name)}
        require(
            binaries == native | cfitsio,
            f"{path.name}: unexpected bundled binaries: "
            f"{sorted(binaries - native - cfitsio)}",
        )
        notices = archive.read(info + "licenses/THIRD_PARTY_NOTICES.md")
        check_native_notices(notices, path.name)
        check_cccl_license(
            archive.read(info + "licenses/licenses/CCCL-3.1.2.txt"),
            path.name,
        )
        require(
            b"Permission to freely use, copy, modify, and distribute"
            in notices,
            f"{path.name}: missing CFITSIO notice",
        )
        xfit = next(
            name for name in native if name.startswith("cuphoton/xfit/")
        )
        check_xfit_libraries(archive.read(xfit), path.name)
    return (abi, arch), version


def check_sdist(path):
    match = re.fullmatch(r"cuphoton-(.+)\.tar\.gz", path.name)
    require(match is not None, f"{path.name}: unexpected source archive name")
    version = match.group(1)
    prefix = f"cuphoton-{version}/"
    with tarfile.open(path, "r:gz") as archive:
        files = {
            member.name.removeprefix(prefix)
            for member in archive
            if member.isfile()
        }
        required = SOURCES | LICENSES | {"PKG-INFO"}
        require(
            required <= files,
            f"{path.name}: missing {sorted(required - files)}",
        )
        binaries = sorted(name for name in files if BINARY.search(name))
        require(
            not binaries,
            f"{path.name}: source archive contains binaries: {binaries}",
        )
        check_native_notices(
            archive.extractfile(prefix + "THIRD_PARTY_NOTICES.md").read(),
            path.name,
        )
        check_cccl_license(
            archive.extractfile(prefix + "licenses/CCCL-3.1.2.txt").read(),
            path.name,
        )
        check_metadata(
            archive.extractfile(prefix + "PKG-INFO").read(),
            version,
            path.name,
            "Apache-2.0",
        )
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--arch", choices=ARCHITECTURES, action="append")
    parser.add_argument("--version", help="Required release version")
    args = parser.parse_args()
    try:
        require(
            args.directory.is_dir(),
            f"Not a distribution directory: {args.directory}",
        )
        expected = {
            (abi, arch)
            for abi in ABIS
            for arch in (args.arch or ARCHITECTURES)
        }
        found, versions = set(), set()
        for path in sorted(args.directory.glob("*.whl")):
            key, version = check_wheel(path)
            require(
                key not in found, f"Duplicate wheel for {key}: {path.name}"
            )
            found.add(key)
            versions.add(version)
        require(
            found == expected,
            f"Wheel matrix mismatch: missing {sorted(expected - found)}, "
            f"unexpected {sorted(found - expected)}",
        )
        sdists = list(args.directory.glob("*.tar.gz"))
        require(
            len(sdists) == 1,
            f"Expected one source archive, found {len(sdists)}",
        )
        versions.add(check_sdist(sdists[0]))
        require(
            len(versions) == 1,
            f"Inconsistent artifact versions: {sorted(versions)}",
        )
        version = versions.pop()
        require(
            args.version is None or args.version == version,
            f"Expected version {args.version}, found {version}",
        )
    except (
        ValueError,
        OSError,
        KeyError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as error:
        parser.exit(1, f"Distribution check failed: {error}\n")
    print(
        f"Validated {len(found)} native wheels and one source archive "
        f"for cuPhoton {version}"
    )


if __name__ == "__main__":
    main()
