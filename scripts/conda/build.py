# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build native conda packages from a versioned cuPhoton source archive."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/conda"))
    parser.add_argument("--python", choices=("3.12", "3.13", "3.14"))
    args = parser.parse_args()
    archive = args.sdist.resolve()
    output = args.output_dir.resolve()
    if any(output.glob("linux-*/cuphoton-*.conda")):
        parser.error(
            "Use an output directory without existing cuPhoton packages"
        )
    # Conda records build prefixes in its recipe, bytecode, and relocation
    # metadata. Keep user-specific checkout/output paths out of the package.
    with tempfile.TemporaryDirectory(
        prefix="cuphoton-conda-", dir="/tmp"
    ) as temporary:
        stage = Path(temporary)
        build_output = stage / "output"
        with tarfile.open(archive) as source:
            source.extractall(stage / "archive", filter="data")
        (source_dir,) = (stage / "archive").iterdir()
        metadata = email.parser.Parser().parsestr(
            (source_dir / "PKG-INFO").read_text()
        )
        if metadata["Name"] != "cuphoton" or not metadata["Version"]:
            raise ValueError("Expected a versioned cuphoton source archive")
        version = metadata["Version"]
        source_dir.rename(stage / "source")
        shutil.copytree(stage / "source/packaging/conda", stage / "recipe")
        environment = os.environ.copy()
        environment["CUPHOTON_CONDA_VERSION"] = version
        # Solving/testing native planning does not require a physical GPU.
        environment.setdefault("CONDA_OVERRIDE_CUDA", "13.0")
        command = [
            "rattler-build",
            "build",
            "--recipe",
            str(stage / "recipe"),
            "--output-dir",
            str(build_output),
            "--no-config",
            "--channel",
            "rapidsai",
            "--channel",
            "conda-forge",
            "--test",
            "native",
        ]
        if args.python:
            command.extend(("--variant", f"python={args.python}"))
        subprocess.run(command, env=environment, check=True)
        packages = sorted(
            build_output.glob(f"linux-*/cuphoton-{version}-*.conda")
        )
        expected_count = 1 if args.python else 3
        if len(packages) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} cuPhoton packages, "
                f"found {len(packages)}"
            )
        receipt = {
            "version": version,
            "source_archive": archive.name,
            "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "packages": {},
        }
        for package in packages:
            relative_path = package.relative_to(build_output)
            destination = output / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package, destination)
            receipt["packages"][str(relative_path)] = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
        (output / "provenance.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
