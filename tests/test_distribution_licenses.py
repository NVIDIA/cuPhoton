# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check license contents in the archives that distribution CI accepts."""

from __future__ import annotations

import importlib.util
import io
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_distributions", ROOT / "scripts/wheels/check_distributions.py"
)
assert SPEC is not None and SPEC.loader is not None
distributions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distributions)
NOTICES = (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes()
CCCL_LICENSE = (ROOT / "licenses/CCCL-3.1.2.txt").read_bytes()
PYBIND11_NOTICE = next(
    block
    for block in re.findall(rb"```text\n(.*?)```", NOTICES, re.DOTALL)
    if b"Wenzel Jakob" in block
)


@pytest.fixture(scope="module")
def xfit_elf(tmp_path_factory):
    """A real ELF with the same external SONAME contract as native xFit."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("ELF dependency checks require a C compiler")
    directory = tmp_path_factory.mktemp("xfit-elf")
    source = directory / "stub.c"
    source.write_text("void stub(void) {}\n")
    libraries = []
    for name in ("libcudart.so.13", "libcublas.so.13"):
        library = directory / name
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(library),
                f"-Wl,-soname,{name}",
            ],
            check=True,
        )
        libraries.append(str(library))
    extension = directory / "xfit.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(extension),
            "-Wl,--no-as-needed",
            *libraries,
        ],
        check=True,
    )
    return extension.read_bytes()


def make_archive(tmp_path, kind, notices, expression=None, xfit_elf=b""):
    version = "0.1.4"
    expression = expression or (
        "Apache-2.0 AND CFITSIO AND BSD-3-Clause"
        " AND Apache-2.0 WITH LLVM-exception"
        if kind == "wheel"
        else "Apache-2.0"
    )
    metadata = (
        f"Name: cuphoton\nVersion: {version}\n"
        f"License-Expression: {expression}\n"
        "Requires-Python: >=3.12,<3.15\n"
        "Provides-Extra: io\nProvides-Extra: gpu\n"
    ).encode()
    if kind == "wheel":
        tag = "cp312-cp312-manylinux_2_28_x86_64"
        path = tmp_path / f"cuphoton-{version}-{tag}.whl"
        info = f"cuphoton-{version}.dist-info/"
        files = {
            info + "METADATA": metadata,
            info + "WHEEL": (
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n"
            ).encode(),
            info + "licenses/LICENSE": (ROOT / "LICENSE").read_bytes(),
            info + "licenses/THIRD_PARTY_NOTICES.md": notices,
            info + "licenses/licenses/CCCL-3.1.2.txt": CCCL_LICENSE,
            (
                "cuphoton/xdr/_nvcomp_batch_ext."
                "cpython-312-x86_64-linux-gnu.so"
            ): b"",
            (
                "cuphoton/xfit/_native_ext.cpython-312-x86_64-linux-gnu.so"
            ): xfit_elf,
            "cuphoton.libs/libcfitsio-12345678.so.10.0.0": b"",
        }
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
    else:
        path = tmp_path / f"cuphoton-{version}.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name in sorted(
                distributions.SOURCES | distributions.LICENSES | {"PKG-INFO"}
            ):
                content = {
                    "PKG-INFO": metadata,
                    "THIRD_PARTY_NOTICES.md": notices,
                    "licenses/CCCL-3.1.2.txt": CCCL_LICENSE,
                }.get(name, b"source\n")
                member = tarfile.TarInfo(f"cuphoton-{version}/{name}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return path


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archives_include_complete_pybind11_notice(tmp_path, kind, xfit_elf):
    path = make_archive(tmp_path, kind, NOTICES, xfit_elf=xfit_elf)
    check = getattr(distributions, f"check_{kind}")
    check(path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "removed",
    [
        PYBIND11_NOTICE,
        b"Copyright (c) 2016 Wenzel Jakob",
        b"and/or other materials provided with the distribution.",
        b"EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.",
    ],
    ids=["whole-notice", "copyright", "binary-condition", "disclaimer"],
)
def test_archives_reject_incomplete_pybind11_notice(tmp_path, kind, removed):
    path = make_archive(tmp_path, kind, NOTICES.replace(removed, b""))
    check = getattr(distributions, f"check_{kind}")
    with pytest.raises(ValueError, match="full pybind11 license notice"):
        check(path)


def test_native_wheel_requires_pybind11_license_expression(tmp_path):
    path = make_archive(
        tmp_path, "wheel", NOTICES, expression="Apache-2.0 AND CFITSIO"
    )
    with pytest.raises(ValueError, match="expected License-Expression"):
        distributions.check_wheel(path)


def test_xfit_elf_requires_external_cuda_libraries(xfit_elf):
    distributions.check_xfit_libraries(xfit_elf, "native xFit")
    missing_cublas = xfit_elf.replace(b"libcublas.so.13", b"libcublas.so.99")
    with pytest.raises(ValueError, match="dynamically link"):
        distributions.check_xfit_libraries(missing_cublas, "native xFit")


def test_cccl_notice_rejects_truncated_license():
    with pytest.raises(ValueError, match="full CCCL license notice"):
        distributions.check_cccl_license(CCCL_LICENSE[:200], "artifact")
