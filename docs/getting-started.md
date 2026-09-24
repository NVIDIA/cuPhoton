# Getting started

## Requirements

cuPhoton supports CPython 3.12 through 3.14 on Linux for the base, GPU, CPU
PyTorch, and visualization profiles. CPU workflows do not require CUDA. The
GPU profile targets CUDA 13 and requires a compatible NVIDIA driver. The
experimental cuTile profile supports the same Python versions. Dragon
currently requires Python 3.12 or 3.13.

Install [uv](https://docs.astral.sh/uv/) before working from a checkout. uv is
the supported environment and lock-file tool; editable pip installation is
also available for integration into an existing environment.

## Install a release

After the first PyPI release is published, install it with the commands below.
Until then, use the [checkout instructions](#clone-and-select-a-profile).

```bash
python -m pip install cuphoton
python -m pip install 'cuphoton[io]'  # Native GPU FITS loading
```

The Linux x86-64 and ARM64 wheels include the XDR extension and private
CFITSIO. `io` installs the CUDA 13 runtime dependencies; no compiler or local
CUDA toolkit is needed. GPU execution still requires a compatible NVIDIA
driver. See [XDR](components/xdr.md) for GPUDirect Storage requirements.

Install `cuphoton[photometry]` for source detection, background estimation,
and aperture photometry. It uses Photutils, which currently requires a C
compiler on ARM64. The broader `gpu` profile includes `io` and `photometry`.
Free-threaded Python and Windows/macOS wheels are not provided.

## Clone and select a profile

```bash
git clone https://github.com/NVIDIA/cuPhoton.git
cd cuPhoton
```

For CUDA 13 development and visualization:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
```

For CPU development, including PyTorch workflows:

```bash
uv sync --locked --extra dev --extra torch --extra viz --extra photometry
```

The base package supports CPU data inspection and NumPy/SciPy workflows:

```bash
uv sync --locked
```

The extras are composable:

| Extra | Adds |
| --- | --- |
| `dev` | pytest, Ruff, pre-commit, and packaging checks |
| `photometry` | Photutils background, detection, and aperture routines |
| `io` | CuPy, KvikIO, cuFile, and nvCOMP for native XDR |
| `torch` | CPU-capable PyTorch |
| `gpu` | `io`, `photometry`, CUDA 13 PyTorch, and Numba-CUDA |
| `cutile` | experimental `cuda.tile` and its CuPy bridge |
| `mpi` | mpi4py bindings; an MPI runtime and launcher are also required |
| `dragon` | DragonHPC; Python 3.12 or 3.13 |
| `viz` | Bokeh and Pillow |

For cuTile backend development:

```bash
uv sync --locked --extra dev --extra gpu --extra cutile
```

## Optional runtimes

Extras compose: use `cuphoton[gpu,mpi]` or `cuphoton[gpu,dragon]` for the
corresponding distributed GPU executor. From a checkout:

```bash
# Choose the MPI profile:
uv sync --locked --extra gpu --extra mpi
# Or the Dragon profile:
uv sync --locked --python 3.13 --extra gpu --extra dragon
```

The `mpi` extra installs mpi4py, not an MPI implementation. Load the site's
MPI module or install a compatible MPI runtime and launcher, then verify
`.venv/bin/python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'`
from a checkout, or use the activated environment's Python for a wheel install.
Use the same runtime with `mpiexec` on every node. Follow
[mpi4py's installation guide](https://mpi4py.readthedocs.io/en/stable/install.html)
for site-specific builds. MPI file aggregation does not import mpi4py.

DragonHPC 0.14.2 provides Linux x86-64 and ARM64 wheels for Python 3.12 and
3.13, but no Python 3.14 wheel. Installing `[dragon]` on Python 3.14 fails
instead of silently omitting Dragon. Use the installed `dragon` launcher and
the [distributed launch examples](components/xpois.md#launch-with-dragon).
Both executors need the same environment on every participating node.

The `cutile` extra installs cuda-tile 1.6 or newer and CuPy on Python
3.12–3.14. For kernel compilation in this setup, use a CUDA 13.2 or newer
toolkit with `tileiras`, `ptxas`, and NVVM. The compiler is separate:
cuTile's upstream `[tileiras]` extra requires `cuda-toolkit>=13.2`, while
PyTorch 2.13 pins that Python metapackage to 13.0.3. Requesting both compiler
and PyTorch extras in one environment cannot currently resolve. Use a
system toolkit for cuTile compilation alongside the pip GPU runtime
dependencies; see the [cuTile setup guide](https://docs.nvidia.com/cuda/cutile-python/quickstart.html).

Select the matching compiler tools for the process, for example:

```bash
PATH=/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 \
  uv run --locked --extra gpu --extra cutile cuphoton --version
```

Use the same environment when running cuTile workloads. Selecting the
compiler does not require adding the toolkit's libraries to `LD_LIBRARY_PATH`.

## Verify the checkout

```bash
uv run python -c 'import cuphoton; print(cuphoton.__version__)'
uv run cuphoton xray doctor
uv run python examples/run_quickstarts.py --profile cpu
```

`cuphoton xray doctor` reports optional GPU and visualization capabilities. The
quickstart summary reports the backend and device actually used by every
selected component.

## Editable pip installation

The repository uses standard Python package metadata. From an activated
environment:

```bash
python -m pip install -e '.[dev,torch,viz,photometry]'
```

or, for CUDA 13:

```bash
python -m pip install -e '.[dev,gpu,viz]'
```

The lock file applies to uv-managed checkout environments. A downstream pip or
conda project should record its own complete dependency resolution.

## Next steps

Run the [component quickstarts](quickstarts.md), choose a workflow from the
[CLI index](cli.md), and review the [data contracts](data-artifacts.md) before
using observational data.
