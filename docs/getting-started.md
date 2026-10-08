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

Version 0.1.3 is a source-only release. Use the
[checkout instructions](#clone-and-select-a-profile) below, then select the
extras needed for your workflow. Native FITS loading requires an explicit
[xDR source build](components/xdr.md#install).

Select the `photometry` extra for source detection, background estimation,
and aperture photometry. It uses Photutils, which currently requires a C
compiler and Python development headers on ARM64. The broader `gpu` profile
includes `io` and `photometry`.

## Clone and select a profile

```bash
git clone --branch v0.1.3 https://github.com/NVIDIA/cuPhoton.git
cd cuPhoton
```

Omit `--branch v0.1.3` to work with the current development sources.

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
| `dev` | pytest, Ruff, mypy, pre-commit, and packaging checks |
| `photometry` | Photutils background, detection, and aperture routines |
| `io` | CuPy, KvikIO, cuFile, and nvCOMP for native xDR |
| `torch` | CPU-capable PyTorch |
| `gpu` | `io`, `photometry`, CUDA 13 PyTorch, and Numba-CUDA |
| `cutile` | experimental `cuda.tile` and its CuPy bridge |
| `mpi` | mpi4py bindings; an MPI runtime and launcher are also required |
| `dragon` | DragonHPC; Python 3.12 or 3.13 |
| `viz` | Bokeh and Pillow |

`uv sync` installs an editable Python package from the checkout. Selecting
`io` or `gpu` installs the runtime dependencies but does not compile xDR.
For native FITS loading, follow the
[xDR source build instructions](components/xdr.md#native-extension-availability).
The shared FITS reader can use Astropy without that extension; an explicit
`--fits-reader xdr` requires the extension and a working GPU.

When changing profiles, include every extra you want to retain in `uv sync`.
See [uv's syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/)
for exact synchronization behavior. After syncing, `uv run --no-sync` runs
commands in that prepared environment.

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
from the checkout.
Use the same runtime with `mpiexec` on every node. Follow
[mpi4py's installation guide](https://mpi4py.readthedocs.io/en/stable/install.html)
for site-specific builds. xPois's `--aggregation-mode files` does not import
mpi4py; shared component MPI executors require it.

DragonHPC 0.14.2 provides Linux x86-64 and ARM64 wheels for Python 3.12 and
3.13, but no Python 3.14 wheel. Installing `[dragon]` on Python 3.14 fails
instead of silently omitting Dragon. Use the installed `dragon` launcher and
the [distributed launch guide](distributed-execution.md).
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

After selecting a development profile above:

```bash
uv run --no-sync python -c 'import cuphoton; print(cuphoton.__version__)'
uv run --no-sync cuphoton xray doctor
uv run --no-sync python examples/run_quickstarts.py --profile cpu
```

The xScan quickstart requires PyTorch, included in both development profiles
above. With the GPU profile, require GPU execution with:

```bash
uv run --no-sync python examples/run_quickstarts.py --require-gpu \
  --output-dir quickstart-gpu-output
```

`cuphoton xray doctor` reports xRay's optional GPU and visualization
dependencies. The quickstart summary reports the backend and device actually
used by every selected component. Quickstart success does not establish native
xDR availability: xRep's automatic FITS reader can use Astropy. Verify native
loading separately with the [xDR guide](components/xdr.md).

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
