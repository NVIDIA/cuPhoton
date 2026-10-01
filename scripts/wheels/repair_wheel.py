#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bundle CFITSIO and retain the declared CUDA/RAPIDS wheel dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# These libraries are supplied by the I/O extra, not copied into cuPhoton.
EXTERNAL_LIBRARIES = (
    "libcudart.so.13",
    "libnvcomp.so.5",
    "libkvikio.so",
    "librapids_logger.so",
)


def library_directories() -> list[str]:
    roots = [Path(os.environ["CUPHOTON_XDR_CFITSIO_ROOT"])]
    for name in (
        "nvidia.cu13",
        "nvidia.libnvcomp",
        "libkvikio",
        "rapids_logger",
    ):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"Missing native build package: {name}")
        roots.extend(Path(path) for path in spec.submodule_search_locations)
    return [
        str(directory)
        for root in roots
        for directory in (root / "lib64", root / "lib")
        if directory.is_dir()
    ]


def include_bundled_license(wheel: Path) -> None:
    """Update the repaired artifact's license and regenerate its RECORD."""
    wheel = wheel.resolve()
    with tempfile.TemporaryDirectory(
        prefix=".cuphoton-license-", dir=wheel.parent
    ) as temporary:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "unpack",
                "-d",
                temporary,
                str(wheel),
            ],
            check=True,
        )
        metadata_files = list(Path(temporary).glob("*/*.dist-info/METADATA"))
        if len(metadata_files) != 1:
            raise RuntimeError("Expected one wheel METADATA file")
        metadata = metadata_files[0]
        content = metadata.read_bytes()
        original = b"License-Expression: Apache-2.0"
        if content.partition(b"\n\n")[0].splitlines().count(original) != 1:
            raise RuntimeError(
                "Expected Apache-2.0 license before bundling CFITSIO"
            )
        metadata.write_bytes(
            content.replace(original, original + b" AND CFITSIO", 1)
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                "-d",
                temporary,
                str(metadata.parent.parent),
            ],
            check=True,
        )
        (Path(temporary) / wheel.name).replace(wheel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    architecture = platform.machine()
    if architecture not in {"x86_64", "aarch64"}:
        parser.error(f"Unsupported wheel architecture: {architecture}")

    environment = os.environ.copy()
    directories = library_directories()
    if environment.get("LD_LIBRARY_PATH"):
        directories.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(directories)
    command = [
        sys.executable,
        "-m",
        "auditwheel",
        "repair",
        "--plat",
        f"manylinux_2_28_{architecture}",
        "--wheel-dir",
        str(args.destination),
    ]
    for library in EXTERNAL_LIBRARIES:
        command.extend(("--exclude", library))
    command.append(str(args.wheel))
    subprocess.run(command, check=True, env=environment)
    output_prefix = args.wheel.name.rsplit("-", 1)[0]
    repaired = list(args.destination.glob(f"{output_prefix}-*.whl"))
    if len(repaired) != 1:
        raise RuntimeError(f"Expected one repaired wheel, found: {repaired}")
    with zipfile.ZipFile(repaired[0]) as archive:
        bundled = [
            name
            for name in archive.namelist()
            if ".libs/" in name and ".so" in Path(name).name
        ]
    if len(bundled) != 1 or not Path(bundled[0]).name.startswith(
        "libcfitsio-"
    ):
        raise RuntimeError(
            "Expected only a private CFITSIO library in the repaired wheel, "
            f"found: {bundled}"
        )
    include_bundled_license(repaired[0])


if __name__ == "__main__":
    main()
