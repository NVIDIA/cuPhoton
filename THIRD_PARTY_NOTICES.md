# Third-party notices

cuPhoton does not intentionally vendor third-party source code. Python runtime,
optional-development, and build requirements are declared in
[`pyproject.toml`](pyproject.toml). Optional distributed runtimes, native
requirements, and the CFITSIO library bundled in Linux wheels are documented
below. `uv.lock` records the reproducible resolution for the project Python
dependency profiles, including the optional DragonHPC and mpi4py packages.
The site-provided MPI implementation is not part of that lock. Build-system requirements are resolved separately
by the PEP 517 build frontend and are not locked by `uv.lock`; they are labeled
`not locked` below.

Git-derived package versions use the MIT-licensed build tools
`setuptools-scm==10.3.4` and its `vcs-versioning` dependency. They are not
included in the installed runtime dependencies. Native wheel builds pin both
tools in `scripts/wheels/build-requirements.txt`; isolated source builds
resolve build requirements separately from `uv.lock`.

cuPhoton uses `uv` to resolve Python distributions from the registries
recorded in `uv.lock` (currently the Python Package Index). NVIDIA-authored
CUDA packages are distributed as NVIDIA wheels through Python package
indexes; some are governed by NVIDIA SDK terms rather than an open-source
license. Consult each publisher's governing terms and the license and notice
files included in the installed distribution; package metadata can be
incomplete.

## Direct Python dependency inventory

The locked versions below reflect `uv.lock`. Compound expressions and
component caveats are retained where binary wheels contain material under
more than one license.

| Profile | Declared requirement | Locked version(s) | License or component caveat | Upstream | Distribution |
| --- | --- | --- | --- | --- | --- |
| `build` | `setuptools>=83.0.0` | `not locked` | `MIT` | [setuptools](https://github.com/pypa/setuptools) | `build frontend / PyPI` |
| `build` | `wheel` | `not locked` | `MIT` | [wheel](https://github.com/pypa/wheel) | `build frontend / PyPI` |
| `base` | `astropy>=6.1.4` | `8.0.0` | `BSD-3-Clause` | [Astropy](https://github.com/astropy/astropy) | `uv / PyPI` |
| `base` | `h5py>=3.10` | `3.16.0` | `BSD-3-Clause`; binary wheels may include HDF5 under its HDF5 license | [h5py](https://github.com/h5py/h5py) | `uv / PyPI` |
| `base` | `numexpr>=2.10` | `2.14.1` | `MIT` | [NumExpr](https://github.com/pydata/numexpr) | `uv / PyPI` |
| `base` | `numpy>=2.0,<2.6` | `2.4.6` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | [NumPy](https://github.com/numpy/numpy) | `uv / PyPI` |
| `base` | `pandas>=2.2` | `3.0.3` | `BSD-3-Clause` | [pandas](https://github.com/pandas-dev/pandas) | `uv / PyPI` |
| `photometry` | `photutils>=3.0` | `3.0.0` | `BSD-3-Clause` | [Photutils](https://github.com/astropy/photutils) | `uv / PyPI` |
| `base` | `pyarrow>=23.0` | `24.0.0` | `Apache-2.0`; binary distributions include Arrow and third-party notices | [Apache Arrow](https://github.com/apache/arrow) | `uv / PyPI` |
| `base` | `PyYAML>=6.0` | `6.0.3` | `MIT` | [PyYAML](https://github.com/yaml/pyyaml) | `uv / PyPI` |
| `base` | `scipy>=1.13` | `1.18.0` | `BSD-3-Clause`; distributions include separately licensed components | [SciPy](https://github.com/scipy/scipy) | `uv / PyPI` |
| `torch` | `torch>=2.13,<3` | `2.13.0` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | [PyTorch](https://github.com/pytorch/pytorch) | `uv / PyPI` |
| `viz` | `bokeh>=3.9` | `3.9.1` | `BSD-3-Clause` | [Bokeh](https://github.com/bokeh/bokeh) | `uv / PyPI` |
| `viz` | `pillow>=10.4` | `12.3.0` | `MIT-CMU` | [Pillow](https://github.com/python-pillow/Pillow) | `uv / PyPI` |
| `viz` | `tornado>=6.5.10` | `6.5.10` | `Apache-2.0` | [Tornado](https://github.com/tornadoweb/tornado) | `uv / PyPI` |
| `io`, `gpu` | `cupy-cuda13x[ctk]>=14,<15` | `14.1.1` | `MIT`; the `ctk` extra installs separately licensed NVIDIA CUDA component wheels | [CuPy](https://github.com/cupy/cupy) | `uv / PyPI` |
| `io`, `gpu` | `kvikio-cu13==26.6.*` | `26.6.0` | `Apache-2.0` | [KvikIO](https://github.com/rapidsai/kvikio) | `uv / PyPI` |
| `io`, `gpu` | `libkvikio-cu13==26.6.*` | `26.6.0` | `Apache-2.0` | [KvikIO](https://github.com/rapidsai/kvikio) | `uv / PyPI` |
| `gpu` | `numba>=0.61,<0.66` | `0.65.1` | `BSD-2-Clause` | [Numba](https://github.com/numba/numba) | `uv / PyPI` |
| `gpu` | `numba-cuda[cu13]>=0.30,<0.31` | `0.30.3` | `BSD-2-Clause` | [Numba-CUDA](https://github.com/NVIDIA/numba-cuda) | `uv / PyPI` |
| `io`, `gpu` | `nvidia-libnvcomp-cu13==5.2.*` | `5.2.0.13` | NVIDIA License Agreement for Software Development Kits; no SPDX expression declared | [nvCOMP](https://developer.nvidia.com/nvcomp) | `uv / PyPI; NVIDIA SDK wheel` |
| `io`, `gpu` | `nvidia-nvcomp-cu13==5.2.*` | `5.2.0.13` | NVIDIA License Agreement for Software Development Kits; no SPDX expression declared | [nvCOMP](https://developer.nvidia.com/nvcomp) | `uv / PyPI; NVIDIA SDK wheel` |
| `native build` | `pybind11==3.0.4` | `build recipe` | `BSD-3-Clause` | [pybind11](https://github.com/pybind/pybind11) | `uv / PyPI` |
| `gpu` | `torch>=2.13,<3` | `2.13.0` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | [PyTorch](https://github.com/pytorch/pytorch) | `uv / PyPI` |
| `cutile` | `cuda-tile>=1.6,<2` | `1.6.0` | `Apache-2.0` | [CUDA Tile](https://github.com/NVIDIA/cutile-python) | `uv / PyPI` |
| `cutile` | `cupy-cuda13x[ctk]>=14,<15` | `14.1.1` | `MIT`; the `ctk` extra installs separately licensed NVIDIA CUDA component wheels | [CuPy](https://github.com/cupy/cupy) | `uv / PyPI` |
| `dev` | `setuptools>=83.0.0` | `83.0.0` | `MIT` | [setuptools](https://github.com/pypa/setuptools) | `uv / PyPI` |
| `dev` | `pre-commit>=4.0` | `4.6.0` | `MIT` | [pre-commit](https://github.com/pre-commit/pre-commit) | `uv / PyPI` |
| `dev` | `pytest>=8.3` | `9.1.1` | `MIT` | [pytest](https://github.com/pytest-dev/pytest) | `uv / PyPI` |
| `dev` | `ruff>=0.15.12` | `0.15.20` | `MIT` | [Ruff](https://github.com/astral-sh/ruff) | `uv / PyPI` |

## Optional distributed runtime inventory

The Dragon executor requires DragonHPC. MPI collective aggregation requires
`mpi4py` and an MPI implementation. The `dragon` extra declares
`dragonhpc>=0.14.2,<0.15`; the `mpi` extra declares `mpi4py>=4.1.2,<5`.
Both are installed from upstream distributions and recorded in `uv.lock`;
neither is bundled in the cuPhoton wheel. The MPI implementation remains
site-provided. The inventoried versions do not establish compatibility with every Python version,
transport, or cluster configuration. See the
[XPOIS launch documentation](docs/components/xpois.md#launch-with-dragon)
for a Dragon launch example.

| Runtime | Inventoried versions | License and upstream notices | Use and distribution |
| --- | --- | --- | --- |
| DragonHPC (`dragonhpc`; Python import `dragon`) | `0.14.1`, `0.14.2` | [MIT](https://github.com/DragonHPC/dragon/blob/0.14.2/LICENSE); Hewlett Packard Enterprise Development LP. Its Python dependencies and native libraries need their own inventory; see below. | ProcessGroup workers, placement, and communication for the Dragon executor. Installed separately from upstream; no Dragon source or binaries are bundled in cuPhoton. |
| `mpi4py` | `4.1.2` | [BSD-3-Clause](https://github.com/mpi4py/mpi4py/blob/4.1.2/LICENSE.rst); Lisandro Dalcin. | Python MPI bindings for collective aggregation. Installed separately and linked to the selected MPI implementation; not bundled in cuPhoton. |
| Open MPI | `4.1.6` | [BSD-3-Clause-Open-MPI and component notices](https://github.com/open-mpi/ompi/blob/v4.1.6/LICENSE). MPI implementations and their system dependencies have separate licenses. | External MPI launcher and runtime. The `cuphoton-openmpi-rank-exec` helper reads Open MPI rank variables; the MPI executor also supports scheduler-launched ranks. No MPI implementation is bundled in cuPhoton. |

### Dragon dependencies

The released `dragonhpc==0.14.2` wheel declares these base Python requirements.
The resolved versions and license identifiers below come from an inspected
Linux/Python 3.12 environment and its wheel metadata. These are a reference
inventory, not cuPhoton dependency pins. PyYAML also appears in cuPhoton's
base requirements.

| Dragon requirement | Reference version | License and upstream notice | Notes |
| --- | --- | --- | --- |
| `cloudpickle>=3.0.0` | `3.1.2` | [BSD-3-Clause](https://github.com/cloudpipe/cloudpickle/blob/7576fff24b9769432f76cc6d2c01282583ee87a9/LICENSE) | Python serialization. |
| `pyyaml>=6.0.2` | `6.0.3` | [MIT](https://github.com/yaml/pyyaml/blob/49790e73684bebad1df05ef8d828fa12f685bffb/LICENSE) | YAML parsing. |
| `psutil>=5.9.0` | `7.2.2` | [BSD-3-Clause](https://github.com/giampaolo/psutil/blob/9eea97dd6f1d16ea33f5144c8925f1ce7a0688e1/LICENSE) | Process and system information. |
| `pycapnp>=2.0.0,<2.2.0` | `2.1.0` | [BSD-2-Clause](https://github.com/capnproto/pycapnp/blob/3a3adfb5f1a8d1b52c98e4984f38ceb5b89a94a6/LICENSE.md) | Python bindings; also inspect the Cap'n Proto code included in the native extension. |
| `paramiko>=3.5.1` | `5.0.0` | [LGPL-2.1](https://github.com/paramiko/paramiko/blob/710cc5c02e2ded370d8d24e261e2baa8317a20fa/LICENSE) | SSH support; its cryptographic dependencies carry separate licenses. |
| `shtab>=1.6.0` | `1.12.1` | [MPL-2.0](https://github.com/tqdm/shtab/blob/e1ab40616e77298204a61e1aad50d6f6e90e9217/LICENCE) | Shell completion. |

The same environment resolved the following additional Python packages
through Paramiko. Optional extras and other environments can resolve a
different dependency tree.

| Package | Reference version | License and upstream notice | Required by |
| --- | --- | --- | --- |
| `bcrypt` | `5.0.0` | [Apache-2.0](https://github.com/pyca/bcrypt/blob/main/LICENSE) | Paramiko. |
| `cryptography` | `50.0.1` | [Apache-2.0 OR BSD-3-Clause](https://github.com/pyca/cryptography/blob/main/LICENSE); native OpenSSL and Rust components require their own inventory. | Paramiko. |
| `invoke` | `3.0.3` | [BSD-2-Clause](https://github.com/pyinvoke/invoke/blob/main/LICENSE) | Paramiko. |
| `PyNaCl` | `1.6.2` | [Apache-2.0](https://github.com/pyca/pynacl/blob/main/LICENSE); the wheel also includes an [ISC notice for libsodium](https://github.com/jedisct1/libsodium/blob/master/LICENSE). | Paramiko. |
| `cffi` | `2.1.1` | [MIT-0](https://github.com/python-cffi/cffi/blob/main/LICENSE); inspect native libffi separately. | `cryptography`, PyNaCl. |
| `pycparser` | `3.0` | [BSD-3-Clause](https://github.com/eliben/pycparser/blob/main/LICENSE) | `cffi`. |

Dragon's MIT license does not apply to this entire dependency tree. In
particular, Paramiko and shtab have the distinct terms listed above. Optional
Dragon extras and transitive Python dependencies depend on the installation.
Dragon wheels also contain native Dragon and transport libraries; inventory
their bundled and linked components for the selected artifact and transport.
Use the license and notice files from the exact installed distributions when
preparing a deployment or redistribution inventory.

The native release build pins its tools and CUDA 13.0 SDK inputs in
[`scripts/wheels/build-requirements.txt`](scripts/wheels/build-requirements.txt).
Those inputs are separate from the runtime lock. cuFile is requested explicitly
through `cuda-toolkit[cufile]>=13,<14` in the `io` extra. GPU runtime shared
libraries remain in their upstream distributions and are not copied into
cuPhoton wheels.

## Native dependency inventory

| Package | Version or version range | License identifier | Upstream | Use in cuPhoton | Distribution |
| --- | --- | --- | --- | --- | --- |
| `CFITSIO` | Release wheels bundle `4.7.0`, built with reentrant support. Source builds require a reentrant system or user-provided library. | [`CFITSIO`](https://spdx.org/licenses/CFITSIO.html) | [NASA HEASARC CFITSIO](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/fitsio.html) | FITS header, HDU, binary-table, and heap-descriptor parsing used to construct native read plans for `cuphoton.xdr`. CFITSIO does not perform the GDS data transfer or GPU decompression. | Linux wheels include a privately renamed shared library in `cuphoton.libs`; its copyright and warranty disclaimer follow below. The source archive includes a checksum-pinned download/build recipe, not CFITSIO source. |

### CFITSIO copyright and license notice

The following notice is reproduced from `licenses/License.txt` in the
CFITSIO 4.7.0 distribution bundled in release wheels:

```text
Copyright (Unpublished--all rights reserved under the copyright laws of
the United States), U.S. Government as represented by the Administrator
of the National Aeronautics and Space Administration.  No copyright is
claimed in the United States under Title 17, U.S. Code.

Permission to freely use, copy, modify, and distribute this software
and its documentation without fee is hereby granted, provided that this
copyright notice and disclaimer of warranty appears in all copies.

DISCLAIMER:

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND,
EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO,
ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE
DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE
SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NASA BE LIABLE FOR ANY
DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR
CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY
CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY,
CONTRACT, TORT , OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY
PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED
FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR
SERVICES PROVIDED HEREUNDER.
```

## Transitive dependencies and reproduction

For the Python dependency profiles declared in `pyproject.toml`, `uv.lock`
records transitive dependencies, registry URLs, artifact hashes, platform
markers, and versions. Generate a human-readable appendix across supported
Python versions and platforms with:

```bash
uv tree --locked --all-groups --universal
```

For a distributed installation, also capture the packages installed in the
runtime interpreter. Set `RUNTIME_PYTHON` to the Python executable used by the
Dragon workers or MPI ranks, then run:

```bash
uv pip freeze --python "$RUNTIME_PYTHON"
uv pip tree --python "$RUNTIME_PYTHON"
```

Record the selected runtime artifact versions and hashes, enabled extras,
transitive licenses, and native MPI/transport/system libraries alongside that
package inventory. The lockfile and Python package list do not enumerate
libraries supplied by the host or bundled inside third-party wheels.

Reproduce the CUDA 13 development environment with:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
```

For a CPU-only development environment, replace `gpu` with `torch`.

## cuPhoton distribution contents

`make build` creates a source distribution containing cuPhoton's extension
sources and the pinned native build recipe. `make wheels` builds the native
Linux wheels from that archive. Each wheel includes the XDR extension and a
privately renamed CFITSIO shared library, with the license notice above. The
source archive and wheels include `LICENSE` and this notice file.

DragonHPC, `mpi4py`, MPI, and CUDA runtime libraries are installed separately.
A development source build links against the available CFITSIO and CUDA
libraries. Distributors must retain the licenses and notices for the libraries
included in their artifacts.
