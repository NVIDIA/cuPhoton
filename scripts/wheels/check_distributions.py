#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the native cuPhoton wheel matrix and source distribution."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ABIS = ("cp312", "cp313", "cp314")
ARCHITECTURES = ("x86_64", "aarch64")
LICENSES = {"LICENSE", "THIRD_PARTY_NOTICES.md"}
SOURCES = {
    "setup.py",
    "pyproject.toml",
    "src/cuphoton/xdr/setup_package.py",
    "src/cuphoton/xdr/src/nvcomp_batch_ext.cpp",
    "src/cuphoton/xdr/src/nvcomp_batch_ext.h",
    "src/cuphoton/xdr/src/memory_manager.cpp",
    "src/cuphoton/xdr/src/io.cpp",
    "src/cuphoton/xdr/src/build.sh",
    "scripts/wheels/prepare_cfitsio.sh",
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
    native = (
        "cuphoton/xdr/_nvcomp_batch_ext."
        f"cpython-{abi[2:]}-{arch}-linux-gnu.so"
    )
    with zipfile.ZipFile(path) as archive:
        files = set(archive.namelist())
        required = {native, info + "METADATA", info + "WHEEL"}
        required.update(info + "licenses/" + name for name in LICENSES)
        require(
            required <= files,
            f"{path.name}: missing {sorted(required - files)}",
        )
        check_metadata(
            archive.read(info + "METADATA"),
            version,
            path.name,
            "Apache-2.0 AND CFITSIO",
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
        binaries = {name for name in files if BINARY.search(name)}
        require(
            binaries == {native} | cfitsio,
            f"{path.name}: unexpected bundled binaries: "
            f"{sorted(binaries - {native} - cfitsio)}",
        )
        require(
            b"Permission to freely use, copy, modify, and distribute"
            in archive.read(info + "licenses/THIRD_PARTY_NOTICES.md"),
            f"{path.name}: missing CFITSIO notice",
        )
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
