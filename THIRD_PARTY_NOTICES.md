# Third-party notices

The xRay structured-root backend adapts AMRVW.jl as described below. Python runtime,
optional-development, and build requirements are declared in
[`pyproject.toml`](pyproject.toml). Optional distributed runtimes, native
requirements, compiled header code, and the CFITSIO library bundled in Linux
wheels are documented below. `uv.lock` records the reproducible resolution for
the project Python dependency profiles and the `native-build` and `notebook`
groups, including the optional DragonHPC and mpi4py packages.
The optional NOOA investigation profile is also recorded in the lock; its
Python dependencies and the system bubblewrap executable are installed
separately and are not bundled in cuPhoton artifacts.
The site-provided MPI implementation is not part of that lock. Isolated
PEP 517 builds resolve build-system requirements separately from `uv.lock`;
their `build` rows are labeled `not locked` below.

Git-derived package versions use the MIT-licensed build tools
[`setuptools-scm==10.3.4`](https://pypi.org/project/setuptools-scm/10.3.4/)
and its [`vcs-versioning==2.5.0`](https://pypi.org/project/vcs-versioning/2.5.0/)
dependency in native wheel builds. They are not
included in the installed runtime dependencies. Native wheel builds pin both
tools in `scripts/wheels/build-requirements.txt`; checkout builds use the
locked `native-build` group. Isolated source builds resolve build requirements
separately from `uv.lock`.

cuPhoton uses `uv` to resolve Python distributions from the registries
recorded in `uv.lock` (currently the Python Package Index). NVIDIA-authored
CUDA packages are distributed as NVIDIA wheels through Python package
indexes; some are governed by NVIDIA SDK terms rather than an open-source
license. Consult each publisher's governing terms and the license and notice
files included in the installed distribution; package metadata can be
incomplete.

## xRay structured-root backend

`src/cuphoton/xray/src/structured_roots.cpp` and
`src/cuphoton/xray/src/structured_roots.cu` adapt mathematical
routines from [AMRVW.jl](https://github.com/jverzani/AMRVW.jl), revision
`8cb7afc20abd9dff4216380faa3a149cce16421f`. These files carry `Apache-2.0 AND MIT`: the MIT notice
applies to the adapted portions, and Apache-2.0 applies to the NVIDIA-authored
validation, error handling and CUDA transport code. The separately authored
double-double refinement is Apache-2.0. The CPU solver and refinement are
compiled into the optional xRay extension. The CUDA source is included as a
package resource and compiled when the structured GPU backend is selected.

The MIT License (MIT)

Copyright (c) 2019 John Verzani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Direct Python dependency inventory

The locked versions below reflect `uv.lock`. Compound expressions and
component caveats are retained where binary wheels contain material under
more than one license.

| Profile | Declared requirement | Locked version(s) | License or component caveat | Upstream | Distribution |
| --- | --- | --- | --- | --- | --- |
| `build` | `setuptools>=84.2.1` | `not locked` | `MIT` | [setuptools](https://github.com/pypa/setuptools) | `build frontend / PyPI` |
| `build` | `wheel` | `not locked` | `MIT` | [wheel](https://github.com/pypa/wheel) | `build frontend / PyPI` |
| `native-build` | `setuptools-scm==10.3.4` | `10.3.4` | `MIT` | [setuptools-scm](https://github.com/pypa/setuptools-scm) | `uv / PyPI; checkout build tools` |
| `native-build` | `wheel` | `0.48.0` | `MIT` | [wheel](https://github.com/pypa/wheel) | `uv / PyPI; checkout build tools` |
| `native-build` | `pybind11>=3.0,<4` | `3.1.0` | `BSD-3-Clause` | [pybind11](https://github.com/pybind/pybind11) | `uv / PyPI; checkout build tools` |
| `notebook` | `jupyterlab>=4,<5` | `4.6.5` | `BSD-3-Clause` | [JupyterLab](https://github.com/jupyterlab/jupyterlab) | `uv / PyPI; notebook tools` |
| `notebook` | `ipykernel>=6,<8` | `7.4.0` | `BSD-3-Clause` | [IPython kernel](https://github.com/ipython/ipykernel) | `uv / PyPI; notebook tools` |
| `base` | `astropy>=7.2.0` | `8.0.0` | `BSD-3-Clause` | [Astropy](https://github.com/astropy/astropy) | `uv / PyPI` |
| `base` | `h5py>=3.10` | `3.16.0` | `BSD-3-Clause`; binary wheels may include HDF5 under its HDF5 license | [h5py](https://github.com/h5py/h5py) | `uv / PyPI` |
| `catalog` | `numexpr>=2.10` | `2.14.1` | `MIT` | [NumExpr](https://github.com/pydata/numexpr) | `uv / PyPI` |
| `base` | `numpy>=2.0,<2.6` | `2.4.6` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | [NumPy](https://github.com/numpy/numpy) | `uv / PyPI` |
| `catalog` | `pandas>=2.2` | `3.0.3` | `BSD-3-Clause` | [pandas](https://github.com/pandas-dev/pandas) | `uv / PyPI` |
| `photometry` | `photutils>=3.0` | `3.0.0` | `BSD-3-Clause` | [Photutils](https://github.com/astropy/photutils) | `uv / PyPI` |
| `base` | `pyarrow>=23.0` | `24.0.0` | `Apache-2.0`; binary distributions include Arrow and third-party notices | [Apache Arrow](https://github.com/apache/arrow) | `uv / PyPI` |
| `ximage`, `dev` | `pydantic>=2.11,<3` | `2.14.1` | `MIT` | [Pydantic](https://github.com/pydantic/pydantic) | `uv / PyPI` |
| `ximage` | `zarr>=3,<4` | `3.4.1` | `MIT` | [Zarr](https://github.com/zarr-developers/zarr-python) | `uv / PyPI` |
| `rubin` | `lsst-images[https]==30.0.10` | `30.0.10` | `BSD-3-Clause`; optional dependency tree includes `starlink-pyast` under `LGPL-3.0-or-later` | [LSST Images](https://github.com/lsst/images) | `uv / PyPI` |
| `xlens-inference` | `httpx>=0.28,<1` | `0.28.1` | `BSD-3-Clause` | [HTTPX](https://github.com/encode/httpx) | `uv / PyPI; also selected by xlens-nooa` |
| `xlens-nooa` | `nooa==0.0.12` | `0.0.12` | `Apache-2.0`; upstream distribution includes its own third-party notices | [NVIDIA OO Agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents) | `uv / PyPI; Linux, Python 3.12/3.13 only` |
| `base` | `PyYAML>=6.0` | `6.0.3` | `MIT` | [PyYAML](https://github.com/yaml/pyyaml) | `uv / PyPI` |
| `base` | `scipy>=1.13` | `1.18.0` | `BSD-3-Clause`; distributions include separately licensed components | [SciPy](https://github.com/scipy/scipy) | `uv / PyPI` |
| `torch` | `torch>=2.13,<3` | `2.13.0` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | [PyTorch](https://github.com/pytorch/pytorch) | `uv / PyPI` |
| `torch`, `gpu`, `dev`, `docs` | `filelock>=4.2.1` | `4.2.10` | `MIT` | [filelock](https://github.com/tox-dev/filelock) | `uv / PyPI; shared dependency of PyTorch and virtualenv` |
| `viz` | `bokeh>=3.9` | `3.9.1` | `BSD-3-Clause` | [Bokeh](https://github.com/bokeh/bokeh) | `uv / PyPI` |
| `viz` | `pillow>=13.0.0` | `13.0.0` | `MIT-CMU` | [Pillow](https://github.com/python-pillow/Pillow) | `uv / PyPI` |
| `viz` | `tornado>=6.6` | `6.6` | `Apache-2.0` | [Tornado](https://github.com/tornadoweb/tornado) | `uv / PyPI` |
| `io`, `gpu` | `cuda-pathfinder>=1.5.6,<2` | `1.5.6` | `Apache-2.0` | [CUDA Python](https://github.com/NVIDIA/cuda-python/tree/main/cuda_pathfinder) | `uv / PyPI; loaded separately for native xFit CUDA library discovery` |
| `io`, `gpu` | `cuda-bindings>=13,<14` | `13.3.1` | `LicenseRef-NVIDIA-SOFTWARE-LICENSE` | [CUDA Python](https://github.com/NVIDIA/cuda-python) | `uv / PyPI` |
| `io`, `gpu` | `cupy-cuda13x[ctk]>=14,<15` | `14.1.1` | `MIT`; its bundled CCCL headers have the component licenses below, and the `ctk` extra installs separately licensed NVIDIA CUDA component wheels | [CuPy](https://github.com/cupy/cupy) | `uv / PyPI` |
| `io`, `gpu` | `kvikio-cu13==26.6.*` | `26.6.0` | `Apache-2.0` | [KvikIO](https://github.com/rapidsai/kvikio) | `uv / PyPI` |
| `io`, `gpu` | `libkvikio-cu13==26.6.*` | `26.6.0` | `Apache-2.0`; its bundled BS::thread_pool header is MIT-licensed | [KvikIO](https://github.com/rapidsai/kvikio) | `uv / PyPI; shared library installed separately; header code compiled into xDR` |
| `io`, `gpu` | `nvidia-libnvcomp-cu13==5.2.*` | `5.2.0.13` | NVIDIA License Agreement for Software Development Kits; no SPDX expression declared | [nvCOMP](https://developer.nvidia.com/nvcomp) | `uv / PyPI; NVIDIA SDK wheel` |
| `io`, `gpu` | `nvidia-nvcomp-cu13==5.2.*` | `5.2.0.13` | NVIDIA License Agreement for Software Development Kits; no SPDX expression declared | [nvCOMP](https://developer.nvidia.com/nvcomp) | `uv / PyPI; NVIDIA SDK wheel` |
| `wheel build` | `pybind11==3.0.4` | `build recipe` | `BSD-3-Clause` | [pybind11](https://github.com/pybind/pybind11) | `uv / PyPI` |
| `gpu` | `torch>=2.13,<3` | `2.13.0` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | [PyTorch](https://github.com/pytorch/pytorch) | `uv / PyPI` |
| `cutile` | `cuda-tile>=1.6,<2` | `1.6.0` | `Apache-2.0` | [CUDA Tile](https://github.com/NVIDIA/cutile-python) | `uv / PyPI` |
| `cutile` | `cupy-cuda13x[ctk]>=14,<15` | `14.1.1` | `MIT`; its bundled CCCL headers have the component licenses below, and the `ctk` extra installs separately licensed NVIDIA CUDA component wheels | [CuPy](https://github.com/cupy/cupy) | `uv / PyPI` |
| `dev`, `native-build` | `setuptools>=84.2.1` | `84.2.1` | `MIT` | [setuptools](https://github.com/pypa/setuptools) | `uv / PyPI` |
| `dev` | `pre-commit>=4.0` | `4.6.0` | `MIT` | [pre-commit](https://github.com/pre-commit/pre-commit) | `uv / PyPI` |
| `dev` | `virtualenv>=21.15.4` | `21.15.4` | `MIT` | [virtualenv](https://github.com/pypa/virtualenv) | `uv / PyPI; pre-commit environment creation` |
| `dev` | `pytest>=8.3` | `9.1.1` | `MIT` | [pytest](https://github.com/pytest-dev/pytest) | `uv / PyPI` |
| `dev` | `ruff>=0.15.12` | `0.15.20` | `MIT` | [Ruff](https://github.com/astral-sh/ruff) | `uv / PyPI` |
| `dev` | `mypy==1.19.1` | `1.19.1` | `MIT` | [mypy](https://github.com/python/mypy) | `uv / PyPI` |
| `dev` | `types-PyYAML>=6.0.12` | `6.0.12.20260906` | `Apache-2.0` | [typeshed](https://github.com/python/typeshed) | `uv / PyPI` |
| `dev` | `coverage>=7.10.6,<8` | `7.16.2` | `Apache-2.0` | [Coverage.py](https://github.com/coveragepy/coveragepy/tree/7.16.2) | `uv / PyPI; development only` |

## Optional NOOA investigation runtime

The `xlens-nooa` extra installs NOOA and its declared Python dependencies from
upstream distributions. cuPhoton loads NOOA in an isolated worker for iterative
evidence queries and Python execution. NOOA source, model weights and dependency
package trees are not bundled in cuPhoton wheels or source archives.

NOOA 0.0.12 is Apache-2.0 licensed. Its declared dependencies include AnyIO,
HTTPX, LiteLLM, MessagePack, OpenInference instrumentation, Pydantic and urllib3;
their dependencies and licenses are separate. In particular:

| Component | Reference version | License and upstream notice | Use |
| --- | --- | --- | --- |
| NOOA | `0.0.12` | [Apache-2.0](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/LICENSE); NVIDIA Corporation & Affiliates | Agent loop in the worker. Its distribution includes additional third-party notices. |
| LiteLLM | `1.109.0` | [MIT](https://github.com/BerriAI/litellm/blob/main/LICENSE); its distribution also contains tokenizer assets with their own notices | NOOA dependency, installed separately. Model requests in this integration are brokered by cuPhoton's HTTP client. |
| OpenInference LiteLLM instrumentation | `0.1.55` | [Apache-2.0](https://github.com/Arize-ai/openinference/blob/main/LICENSE) | NOOA dependency, installed separately; the cuPhoton adapter suppresses NOOA's automatic tracing initialization. |
| bubblewrap | Site-provided | [LGPL-2.0-or-later](https://github.com/containers/bubblewrap/blob/main/COPYING) | Separate Linux executable used to start the isolated worker. It is not linked into or bundled with cuPhoton. |

The Python versions above reflect the inspected distributions and lock. Consult
each installed distribution's license files and third-party notices for its
complete contents. Bubblewrap comes from the user's operating-system provider;
its package and system dependencies are outside `uv.lock`. See the
[xLens installation and execution requirements](docs/components/xlens.md#configure-optional-investigation).

## Optional distributed runtime inventory

The Dragon executor requires DragonHPC. MPI collective aggregation requires
`mpi4py` and an MPI implementation. The `dragon` extra declares
`dragonhpc>=0.14.2,<0.15`; the `mpi` extra declares `mpi4py>=4.1.2,<5`.
Both are installed from upstream distributions and recorded in `uv.lock`;
neither is bundled in the cuPhoton wheel. The MPI implementation remains
site-provided. The inventoried versions do not establish compatibility with every Python version,
transport, or cluster configuration. See the
[xPois launch documentation](docs/components/xpois.md#launch-with-dragon)
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

Paramiko is a Python SSH library used by Dragon's SSH orchestration tools.
Dragon loads it through Python imports at runtime; it is not statically linked
into cuPhoton's native extension. cuPhoton does not modify or bundle Paramiko
source or binaries. Paramiko is installed separately with Dragon's dependencies,
and its upstream wheel includes Python source and its LGPL license.

shtab is a Python library used by Dragon to generate shell completions. cuPhoton
does not modify or bundle shtab. The separately installed `shtab==1.12.1` wheel
includes its Python source files and MPL-2.0 license. The corresponding source
archive is available from the [shtab 1.12.1 release](https://pypi.org/project/shtab/1.12.1/#files).

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
| `CFITSIO` | The local wheel recipe bundles `4.7.0`, built with reentrant support. Source builds require a reentrant system or user-provided library. | [`CFITSIO`](https://spdx.org/licenses/CFITSIO.html) | [NASA HEASARC CFITSIO](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/fitsio.html) | FITS header, HDU, binary-table, and heap-descriptor parsing used to construct native read plans for `cuphoton.xdr`. CFITSIO does not perform the GDS data transfer or GPU decompression. | Linux wheels include a privately renamed shared library in `cuphoton.libs`; its copyright and warranty disclaimer follow below. The source archive includes a checksum-pinned download/build recipe, not CFITSIO source. |
| `pybind11` | Native wheel builds use `3.0.4`. | `BSD-3-Clause` | [pybind11](https://github.com/pybind/pybind11/tree/v3.0.4) | C++ bindings for the xDR extension. | Header code is compiled into the native extension; its copyright and license notice follow below. No separate pybind11 runtime package is bundled. |
| `KvikIO` | `26.6.0` headers from the pinned `libkvikio-cu13` build wheel. | `Apache-2.0` | [KvikIO](https://github.com/rapidsai/kvikio) | File I/O and thread-pool wrappers for native xDR. | Header code is compiled into the xDR extension. The KvikIO shared library is installed separately; the included BS::thread_pool header has the MIT notice below. |
| `BS::thread_pool` | `4.1.0`, included in the pinned KvikIO headers. | `MIT` | [BS::thread_pool 4.1.0](https://github.com/bshoshany/thread-pool/tree/v4.1.0) | Worker scheduling used by KvikIO in native xDR. | Header code is compiled into the xDR extension; its copyright and full license notice follow below. |
| `CCCL` (CUB, Thrust, libcu++) | `3.1.2` headers from the pinned `cupy-cuda13x==14.1.1` build wheel. | `BSD-3-Clause` (CUB), `Apache-2.0` (Thrust), `Apache-2.0 WITH LLVM-exception` (libcu++) | [CCCL 3.1.2](https://github.com/NVIDIA/cccl/tree/v3.1.2) | CUB reductions, selection and sorting, with supporting Thrust/libcu++ headers, for native xFit and core imaging. | Header code is compiled into the xFit and core imaging extensions. The applicable notices follow below; no CCCL source tree or runtime library is bundled. |

The complete upstream CCCL 3.1.2 license is also included as
[`licenses/CCCL-3.1.2.txt`](licenses/CCCL-3.1.2.txt) in source, wheel and
Conda distributions. Native Conda builds use the same pinned CuPy headers,
CUDA 13.0 compiler and cuBLAS 13.0.2.14 floor.

### pybind11 copyright and license notice

The following notice is reproduced from the
[pybind11 3.0.4 LICENSE](https://github.com/pybind/pybind11/blob/v3.0.4/LICENSE).
The final upstream paragraph refers to
[pybind11's contribution guidance](https://github.com/pybind/pybind11/blob/v3.0.4/.github/CONTRIBUTING.md).

```text
Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please also refer to the file .github/CONTRIBUTING.md, which clarifies licensing of
external contributions to this project including patches, pull requests, etc.
```

### BS::thread_pool copyright and license notice

KvikIO supplies the `BS_thread_pool.hpp` 4.1.0 header, whose code
is compiled into the native xDR extension. The following notice is reproduced
from the [BS::thread_pool 4.1.0 LICENSE.txt](https://github.com/bshoshany/thread-pool/blob/v4.1.0/LICENSE.txt).

```text
MIT License

Copyright (c) 2024 Barak Shoshany

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### CCCL compiled header notices

Native xFit and core imaging compile CUB reductions, selection and sorting
from the CCCL 3.1.2 headers in `cupy-cuda13x==14.1.1`, with supporting
Thrust and libcu++ headers. The CUB notice and LLVM exception
below are reproduced from that wheel's `cupy/_core/include/cupy/_cccl/LICENSE`,
also available in the [CCCL 3.1.2 LICENSE](https://github.com/NVIDIA/cccl/blob/v3.1.2/LICENSE).
The complete Apache License 2.0 text applicable to Thrust and libcu++ is
included in cuPhoton's [LICENSE](LICENSE); libcu++ adds the LLVM exception.

#### CUB copyright and license notice

```text
Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

#### libcu++ LLVM exception to Apache License 2.0

```text
As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
```

### CFITSIO copyright and license notice

The bundled library includes a local patch that adds a readable version
string. The patch is included in the cuPhoton source distribution.

The following notice is reproduced from `licenses/License.txt` in the
CFITSIO 4.7.0 distribution bundled by the local wheel recipe:

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
Linux wheels from that archive. Each wheel includes native extensions and a
privately renamed CFITSIO shared library. The xDR extension contains compiled
pybind11, KvikIO and BS::thread_pool header code; the xFit and core imaging
extensions contain compiled CCCL header code. The source archive and wheels
include `LICENSE` and this notice file, including the CFITSIO, pybind11,
BS::thread_pool, CUB, and LLVM exception notices above.

DragonHPC, `mpi4py`, MPI, and CUDA runtime libraries are installed separately.
A development source build links against the available CFITSIO and CUDA
libraries. Distributors must retain the licenses and notices for the libraries
included in their artifacts.
