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
import sys
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


def make_archive(
    tmp_path,
    kind,
    notices,
    expression=None,
    cfitsio=b"CFITSIO 4.7.0\0",
    xfit_elf=b"",
    cccl_license=CCCL_LICENSE,
):
    version = "0.1.4"
    expression = expression or (
        "Apache-2.0 AND CFITSIO AND BSD-3-Clause "
        "AND Apache-2.0 WITH LLVM-exception"
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
            info + "licenses/licenses/CCCL-3.1.2.txt": cccl_license,
            (
                "cuphoton/xdr/_nvcomp_batch_ext."
                "cpython-312-x86_64-linux-gnu.so"
            ): b"",
            (
                "cuphoton/xfit/_native_ext.cpython-312-x86_64-linux-gnu.so"
            ): xfit_elf,
            "cuphoton.libs/libcfitsio-12345678.so.10.0.0": cfitsio,
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
                    "licenses/CCCL-3.1.2.txt": cccl_license,
                }.get(name, b"source\n")
                member = tarfile.TarInfo(f"cuphoton-{version}/{name}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return path


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archives_include_complete_pybind11_notice(tmp_path, kind, request):
    xfit_elf = request.getfixturevalue("xfit_elf") if kind == "wheel" else b""
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


@pytest.mark.parametrize(
    "expression",
    [
        "Apache-2.0 AND CFITSIO",
        "Apache-2.0 AND CFITSIO AND BSD-3-Clause",
    ],
)
def test_native_wheel_requires_header_license_expression(
    tmp_path, expression
):
    path = make_archive(tmp_path, "wheel", NOTICES, expression=expression)
    with pytest.raises(ValueError, match="expected License-Expression"):
        distributions.check_wheel(path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "removed, name",
    [
        (b"Copyright (c) 2010-2011, Duane Merrill.", "CUB license"),
        (b"without complying", "LLVM exception"),
    ],
)
def test_archives_reject_incomplete_cccl_notices(
    tmp_path, kind, removed, name
):
    path = make_archive(tmp_path, kind, NOTICES.replace(removed, b""))
    check = getattr(distributions, f"check_{kind}")
    with pytest.raises(ValueError, match=f"full {name} notice"):
        check(path)


def test_native_wheel_requires_xfit_extension(tmp_path):
    path = make_archive(tmp_path, "wheel", NOTICES)
    with zipfile.ZipFile(path) as archive:
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.startswith("cuphoton/xfit/_native_ext.")
        }
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match=r"missing .*xfit/_native_ext"):
        distributions.check_wheel(path)


@pytest.mark.parametrize("library", ["libcudart", "libcublas", "libcublasLt"])
def test_native_wheel_rejects_vendored_cuda_libraries(tmp_path, library):
    path = make_archive(tmp_path, "wheel", NOTICES)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(f"cuphoton.libs/{library}-12345678.so.13", b"")
    with pytest.raises(ValueError, match="unexpected bundled binaries"):
        distributions.check_wheel(path)


@pytest.fixture(scope="module")
def xfit_elf(tmp_path_factory):
    """A real ELF with the same external SONAME contract as native xFit."""
    if sys.platform != "linux" or shutil.which("readelf") is None:
        pytest.skip("ELF dependency checks require Linux and readelf")
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


@pytest.mark.parametrize("library", ["libcudart", "libcublas"])
def test_wheel_rejects_wrong_cuda_linkage(tmp_path, xfit_elf, library):
    broken = xfit_elf.replace(
        f"{library}.so.13".encode(), f"{library}.so.99".encode()
    )
    path = make_archive(tmp_path, "wheel", NOTICES, xfit_elf=broken)
    with pytest.raises(ValueError, match="dynamically link"):
        distributions.check_wheel(path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archives_reject_truncated_cccl_license(tmp_path, kind, request):
    xfit_elf = request.getfixturevalue("xfit_elf") if kind == "wheel" else b""
    path = make_archive(
        tmp_path,
        kind,
        NOTICES,
        xfit_elf=xfit_elf,
        cccl_license=CCCL_LICENSE[:200],
    )
    with pytest.raises(ValueError, match="full CCCL license notice"):
        getattr(distributions, f"check_{kind}")(path)


@pytest.mark.parametrize("missing", [None, "libcudart", "libcublas"])
def test_wheel_repair_requires_both_cuda_libraries(
    tmp_path, xfit_elf, missing
):
    spec = importlib.util.spec_from_file_location(
        "repair_wheel", ROOT / "scripts/wheels/repair_wheel.py"
    )
    assert spec is not None and spec.loader is not None
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    if missing is not None:
        xfit_elf = xfit_elf.replace(
            f"{missing}.so.13".encode(), f"{missing}.so.99".encode()
        )
    path = make_archive(tmp_path, "wheel", NOTICES, xfit_elf=xfit_elf)
    if missing is None:
        repair.check_xfit_linkage(path)
    else:
        with pytest.raises(RuntimeError, match=f"missing .*{missing}.so.13"):
            repair.check_xfit_linkage(path)


@pytest.mark.parametrize("cfitsio", [b"", b"CFITSIO 4.6.4\0"])
def test_native_wheel_requires_cfitsio_identity(tmp_path, cfitsio):
    path = make_archive(tmp_path, "wheel", NOTICES, cfitsio=cfitsio)
    with pytest.raises(ValueError, match="CFITSIO is missing its 4.7.0"):
        distributions.check_wheel(path)
