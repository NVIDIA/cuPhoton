# Getting started

## Requirements

cuPhoton supports CPython 3.12 through 3.14 on Linux for the base, GPU, CPU
PyTorch, and visualization profiles. CPU workflows do not require CUDA. The
GPU profile targets CUDA 13 and requires a compatible NVIDIA driver. Python
3.12 or 3.13 is required for the experimental cuTile profile.

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
| `viz` | Bokeh and Pillow |

For cuTile backend development:

```bash
uv sync --locked --python 3.12 \
  --extra dev --extra gpu --extra cutile
```

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
