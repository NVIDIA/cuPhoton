# Getting started

## Requirements

cuPhoton supports Python 3.11 through 3.14 on Linux for the base, GPU, CPU
PyTorch, and visualization profiles. CPU workflows do not require CUDA. The
GPU profile targets CUDA 13 and requires a compatible NVIDIA driver. The locked
Legate-backed HDF5 profile is unavailable on Python 3.14. Python 3.12 or 3.13
is required for the experimental cuTile profile.

Install [uv](https://docs.astral.sh/uv/) before working from a checkout. uv is
the supported environment and lock-file tool; editable pip installation is
also available for integration into an existing environment.

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
uv sync --locked --extra dev --extra torch --extra viz
```

The base package is sufficient for CPU data inspection and workflows that do
not use PyTorch or optional review views:

```bash
uv sync --locked
```

The extras are composable:

| Extra | Adds |
| --- | --- |
| `dev` | pytest, Ruff, pre-commit, and packaging checks |
| `torch` | CPU-capable PyTorch |
| `gpu` | CUDA 13 PyTorch, CuPy, and Numba-CUDA |
| `hdf5` | Legate-backed HDF5 dataset loading |
| `cutile` | experimental `cuda.tile` and its CuPy bridge |
| `viz` | Bokeh and Pillow |

Use the cuTile profile only when developing that backend:

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
python -m pip install -e '.[dev,torch,viz]'
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
