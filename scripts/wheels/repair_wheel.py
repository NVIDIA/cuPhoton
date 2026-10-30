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
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# These libraries are supplied by the I/O extra, not copied into cuPhoton.
EXTERNAL_LIBRARIES = (
    "libcudart.so.13",
    "libcublas.so.13",
    "libcublasLt.so.13",
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
        original = b"License-Expression: Apache-2.0 AND MIT"
        if content.partition(b"\n\n")[0].splitlines().count(original) != 1:
            raise RuntimeError(
                "Expected Apache-2.0 AND MIT before adding native licenses"
            )
        metadata.write_bytes(
            content.replace(
                original,
                original + b" AND CFITSIO AND BSD-3-Clause "
                b"AND Apache-2.0 WITH LLVM-exception",
                1,
            )
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


def check_xfit_linkage(wheel: Path) -> None:
    """Require the CUDA 13 SONAMEs that the runtime loader supplies."""
    with zipfile.ZipFile(wheel) as archive:
        extensions = [
            name
            for name in archive.namelist()
            if re.fullmatch(r"cuphoton/xfit/_native_ext\.[^/]+\.so", name)
        ]
        if len(extensions) != 1:
            raise RuntimeError("Expected one native xFit extension")
        with tempfile.TemporaryDirectory(
            prefix="cuphoton-linkage-"
        ) as scratch:
            extension = Path(scratch) / Path(extensions[0]).name
            extension.write_bytes(archive.read(extensions[0]))
            result = subprocess.run(
                ["readelf", "--dynamic", str(extension)],
                check=True,
                capture_output=True,
                text=True,
            )
    needed = set(re.findall(r"\(NEEDED\).*?\[(.*?)\]", result.stdout))
    required = {"libcudart.so.13", "libcublas.so.13"}
    if not required <= needed:
        raise RuntimeError(
            "Native xFit must link to external CUDA 13 libraries; "
            f"missing {sorted(required - needed)} in {sorted(needed)}"
        )


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
    check_xfit_linkage(repaired[0])
    include_bundled_license(repaired[0])


if __name__ == "__main__":
    main()
