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

Choose a published version from
[Releases](https://github.com/NVIDIA/cuPhoton/releases) and follow its
installation notes for available packages or a source checkout. For current
development sources, use the checkout instructions below. Native FITS loading
from a checkout requires an explicit [xDR source build](components/xdr.md#install).

Select the `photometry` extra for source detection, background estimation,
and aperture photometry. It uses Photutils, which currently requires a C
compiler and Python development headers on ARM64. The broader `gpu` profile
includes `io` and `photometry`.

## Clone and select a profile

```bash
git clone https://github.com/NVIDIA/cuPhoton.git
cd cuPhoton
```

This selects the development branch. To use a particular release, add
`--branch TAG` to the clone command, replacing `TAG` with its release tag.
Use the documentation in that checkout for the selected version.

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
| `catalog` | Pandas and NumExpr for xScan catalog and registry ingestion |
| `photometry` | Photutils background, detection, and aperture routines |
| `io` | CuPy, KvikIO, cuFile, and nvCOMP for native xDR |
| `torch` | CPU-capable PyTorch |
| `ximage` | PyTorch, photometry, Pydantic configuration, and Zarr bundles |
| `rubin` | Offline Rubin image archive reader |
| `gpu` | `io`, `photometry`, and CUDA 13 PyTorch |
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
make sync-dragon
# Choose Python 3.13 instead:
make sync-dragon DRAGON_PYTHON=3.13
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
The Dragon target selects `gpu`, `viz`, and `dragon`. Add `--extra catalog`
to an explicit `uv sync` command when using xScan's catalog/registry inputs,
or `--extra ximage` for xImage workflows. Include every desired extra when
syncing an existing environment.

### Dragon with pip

[`requirements-dragon.txt`](../requirements-dragon.txt) is generated from
`uv.lock` for the `gpu`, `viz`, and `dragon` extras. It contains pinned runtime
dependencies and hashes for Linux on Python 3.12 or 3.13. Use the requirements
file from the same release or revision as cuPhoton, in a fresh environment:

```bash
python -m pip install --require-hashes -r requirements-dragon.txt
```

Then install the matching cuPhoton wheel with `python -m pip install --no-deps`
followed by the wheel's path. The requirements file contains dependencies;
the wheel supplies cuPhoton and its native extensions. A source installation
still needs the [native build prerequisites](components/xdr.md#install).
The NVIDIA driver and site configuration remain host prerequisites. On ARM64,
Photutils requires a source build and its compiler prerequisites described above.

Regenerate the file after changing `pyproject.toml` and running `uv lock`:

```bash
make requirements-dragon
make requirements-check
```

CI checks that the committed export matches the lock and resolves it on
supported Dragon interpreters. The export excludes development, notebook,
and native-build groups. For another workflow, use `uv export --locked` with
its extras rather than editing the generated requirements file.

### cuTile compiler

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

The lock file applies to uv-managed checkout environments. Pip users can use
the [generated Dragon requirements](#dragon-with-pip) or export their chosen
extras with `uv export --locked --format requirements.txt --no-emit-project`.
Downstream environments with different dependencies should record their own
complete resolution.

## Notebook and native build tools

The `notebook` and `native-build` dependency groups contain checkout tools
resolved in `uv.lock`. Add `--group notebook` when preparing an environment
for JupyterLab, then launch it without resyncing:

```bash
uv sync --locked --extra gpu --extra viz --group notebook
uv run --no-sync jupyter lab
```

The xDR build helper installs the locked `native-build` group into the chosen
interpreter and preserves its installed runtime extras. The group contains
Python build tools; CUDA headers, a C++ compiler, and CFITSIO are separate
[native prerequisites](components/xdr.md#native-extension-availability).

## Next steps

Run the [component quickstarts](quickstarts.md), choose a workflow from the
[CLI index](cli.md), and review the [data contracts](data-artifacts.md) before
using observational data.
