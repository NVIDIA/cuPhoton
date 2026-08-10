# Third-party notices

cuPhoton does not intentionally vendor third-party source code. Python runtime,
optional-development, and build requirements are declared in
[`pyproject.toml`](pyproject.toml). Native system requirements are documented
separately below. `uv.lock` records the reproducible resolution for the project
Python dependency profiles. Build-system requirements are resolved separately
by the PEP 517 build frontend and are not locked by `uv.lock`; they are labeled
`not locked` below.

cuPhoton uses `uv` to resolve Python distributions from the registries
recorded in `uv.lock` (currently the Python Package Index). NVIDIA-authored
CUDA packages are distributed as NVIDIA wheels through Python package
indexes; some are governed by NVIDIA SDK terms rather than an open-source
license. Consult each publisher's governing terms and the license and notice
files included in the installed distribution; package metadata can be
incomplete.

## Direct Python dependency inventory

The locked versions below reflect `uv.lock`. `scipy` resolves to 1.17.1 on
Python 3.11 and 1.18.0 on Python 3.12 or later. Compound expressions and
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
| `base` | `photutils>=3.0` | `3.0.0` | `BSD-3-Clause` | [Photutils](https://github.com/astropy/photutils) | `uv / PyPI` |
| `base` | `pyarrow>=23.0` | `24.0.0` | `Apache-2.0`; binary distributions include Arrow and third-party notices | [Apache Arrow](https://github.com/apache/arrow) | `uv / PyPI` |
| `base` | `PyYAML>=6.0` | `6.0.3` | `MIT` | [PyYAML](https://github.com/yaml/pyyaml) | `uv / PyPI` |
| `base` | `scipy>=1.13` | `1.17.1, 1.18.0` | `BSD-3-Clause`; distributions include separately licensed components | [SciPy](https://github.com/scipy/scipy) | `uv / PyPI` |
| `torch` | `torch>=2.13,<3` | `2.13.0` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | [PyTorch](https://github.com/pytorch/pytorch) | `uv / PyPI` |
| `viz` | `bokeh>=3.9` | `3.9.1` | `BSD-3-Clause` | [Bokeh](https://github.com/bokeh/bokeh) | `uv / PyPI` |
| `viz` | `pillow>=10.4` | `12.3.0` | `MIT-CMU` | [Pillow](https://github.com/python-pillow/Pillow) | `uv / PyPI` |
| `gpu` | `cupy-cuda13x[ctk]>=14,<15` | `14.1.1` | `MIT`; the `ctk` extra installs separately licensed NVIDIA CUDA component wheels | [CuPy](https://github.com/cupy/cupy) | `uv / PyPI` |
| `gpu` | `kvikio-cu13>=26.6,<27` | `26.6.0` | `Apache-2.0` | [KvikIO](https://github.com/rapidsai/kvikio) | `uv / PyPI` |
| `gpu` | `libkvikio-cu13>=26.6,<27` | `26.6.0` | `Apache-2.0` | [KvikIO](https://github.com/rapidsai/kvikio) | `uv / PyPI` |
| `gpu` | `numba>=0.61,<0.66` | `0.65.1` | `BSD-2-Clause` | [Numba](https://github.com/numba/numba) | `uv / PyPI` |
| `gpu` | `numba-cuda[cu13]>=0.30,<0.31` | `0.30.3` | `BSD-2-Clause` | [Numba-CUDA](https://github.com/NVIDIA/numba-cuda) | `uv / PyPI` |
| `gpu` | `nvidia-libnvcomp-cu13>=5.2,<6` | `5.2.0.13` | NVIDIA License Agreement for Software Development Kits; no SPDX expression declared | [nvCOMP](https://developer.nvidia.com/nvcomp) | `uv / PyPI; NVIDIA SDK wheel` |
| `gpu` | `nvidia-nvcomp-cu13>=5.2,<6` | `5.2.0.13` | NVIDIA License Agreement for Software Development Kits; no SPDX expression declared | [nvCOMP](https://developer.nvidia.com/nvcomp) | `uv / PyPI; NVIDIA SDK wheel` |
| `gpu` | `pybind11>=2.12,<4` | `3.0.4` | `BSD-3-Clause` | [pybind11](https://github.com/pybind/pybind11) | `uv / PyPI` |
| `gpu` | `torch>=2.13,<3` | `2.13.0` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | [PyTorch](https://github.com/pytorch/pytorch) | `uv / PyPI` |
| `hdf5` | `legate>=25.1,<27` | `26.6.0` | `Apache-2.0` | [Legate](https://github.com/nv-legate/legate) | `uv / PyPI` |
| `cutile` | `cuda-tile>=1.4` | `1.4.0` | `Apache-2.0` | [CUDA Tile](https://github.com/NVIDIA/cutile-python) | `uv / PyPI` |
| `cutile` | `cupy-cuda13x[ctk]>=14,<15` | `14.1.1` | `MIT`; the `ctk` extra installs separately licensed NVIDIA CUDA component wheels | [CuPy](https://github.com/cupy/cupy) | `uv / PyPI` |
| `dev` | `pre-commit>=4.0` | `4.6.0` | `MIT` | [pre-commit](https://github.com/pre-commit/pre-commit) | `uv / PyPI` |
| `dev` | `pytest>=8.3` | `9.1.1` | `MIT` | [pytest](https://github.com/pytest-dev/pytest) | `uv / PyPI` |
| `dev` | `ruff>=0.15.12` | `0.15.20` | `MIT` | [Ruff](https://github.com/astral-sh/ruff) | `uv / PyPI` |

## Native system dependency inventory

| Package | Version or version range | License identifier | Upstream | Use in cuPhoton | Distribution |
| --- | --- | --- | --- | --- | --- |
| `CFITSIO` | No numeric version constraint is currently enforced; release validation used `4.6.4`. A thread-safe/reentrant build is required. | [`CFITSIO`](https://spdx.org/licenses/CFITSIO.html) | [NASA HEASARC CFITSIO](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/fitsio.html) | FITS header, HDU, binary-table, and heap-descriptor parsing used to construct native read plans for `cuphoton.xdr`. CFITSIO does not perform the GDS data transfer or GPU decompression. | System- or user-provided native library linked by the `cuphoton.xdr` extension; CFITSIO source is not vendored. A distributor that bundles CFITSIO must retain its copyright notice and warranty disclaimer. |

### CFITSIO copyright and license notice

The following notice is reproduced from the CFITSIO 4.6.4 distribution used
for release validation:

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

For project dependency profiles, `uv.lock` is the complete machine-readable
transitive inventory, including registry URLs, artifact hashes, platform
markers, and versions. Generate a human-readable appendix for a review record
with:

```bash
uv tree --locked --all-groups
```

Reproduce the CUDA 13 development environment with:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
```

For a CPU-only development environment, replace `gpu` with `torch`.
