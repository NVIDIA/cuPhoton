# cuPhoton

cuPhoton is a collection of GPU-accelerated reference workflows for
astronomical imaging and X-ray trace analysis. It is intended for research
teams that want working implementations they can run, inspect, and adapt to
their own instruments and data products. It is not a stable application
framework, and the scientific assumptions in each workflow must be checked
against the target use case.

All Python APIs are in the `cuphoton.*` namespace:

- `cuphoton.core` provides the shared command-line, application-context,
  logging, and invariant framework.
- `cuphoton.xdr` loads FITS image HDUs through GPU-native GDS and nvCOMP
  paths and exposes optional Legate-backed HDF5 loading.
- `cuphoton.xfit` performs batched nonlinear least-squares fits for sampled
  stamp and analytic Gaussian dipole models.
- `cuphoton.xpois` fits matching kernels and performs optimal image
  subtraction.
- `cuphoton.xscan` prepares transient datasets and trains, evaluates, and
  reviews real/bogus classifiers, with explicitly selected, candidate-keyed
  xFit fusion.
- `cuphoton.xrep` provides xRep (xReproject) for placing FITS images on common
  WCS grids.
- `cuphoton.xray` extracts and analyzes X-ray detector traces.

cuPhoton releases are currently alpha-quality. The curated Python
exports and structured run artifacts are the intended integration points, but
interfaces may evolve as additional institutions adapt the workflows.

## Start here

[uv](https://docs.astral.sh/uv/) manages the supported development
environments and lock file. On a Linux system with a CUDA 13-capable driver:

```bash
git clone https://github.com/NVIDIA/cuPhoton.git
cd cuPhoton
uv sync --locked --extra dev --extra gpu --extra viz
uv run python examples/run_quickstarts.py
```

The quickstart creates synthetic inputs, runs each science component, and
writes results to `quickstart-output/`. Its JSON summary records the requested
profile, the backend and device selected for each component, and the artifact
paths. The default `auto` profile prefers a GPU and reports when it falls back
to CPU.

For a deterministic CPU run:

```bash
uv sync --locked --extra dev --extra torch --extra viz
uv run python examples/run_quickstarts.py --profile cpu
```

Use `--require-gpu` when a CPU fallback should be an error. See
[Quickstarts](docs/quickstarts.md) for individual components and output
contracts.

## Choose a workflow

| Goal | Component | First command |
| --- | --- | --- |
| Load FITS image HDUs directly to GPU arrays | xDataReader | `uv run cuphoton xdr benchmark-fits --help` |
| Fit dipole models to image stamps | xFit | `uv run cuphoton xfit --help` |
| Match PSFs and subtract two images | XPOIS | `uv run cuphoton xpois --help` |
| Build and assess a real/bogus classifier | XScan | `uv run cuphoton xscan --help` |
| Reproject images onto a shared sky grid | xRep (xReproject) | `uv run cuphoton xrep --help` |
| Analyze X-ray detector traces | XRay | `uv run cuphoton xray --help` |

## Installation profiles

The base install contains the shared CPU data and scientific stack. Optional
extras are deliberately separated by purpose:

Python 3.11 through 3.14 is supported on Linux for the base, GPU, CPU PyTorch,
and visualization profiles. The locked Legate-backed HDF5 profile is
unavailable on Python 3.14. The experimental cuTile profile remains limited to
Python 3.12 and 3.13.

| Extra | Use |
| --- | --- |
| `dev` | Tests, formatting, linting, and build tools |
| `torch` | PyTorch workflows that can be forced to CPU execution |
| `gpu` | CUDA 13 PyTorch, CuPy, Numba-CUDA, KvikIO, and nvCOMP backends |
| `hdf5` | Legate-backed HDF5 dataset loading |
| `cutile` | Experimental `cuda.tile` backend on Python 3.12 or 3.13 |
| `viz` | Bokeh reviews and Pillow image outputs |

Typical editable installs are:

```bash
# CPU development
python -m pip install -e '.[dev,torch,viz]'

# CUDA 13 development
python -m pip install -e '.[dev,gpu,viz]'
```

The cuTile profile is separate because it has a narrower Python and toolchain
compatibility range:

```bash
uv sync --locked --python 3.12 --extra dev --extra gpu --extra cutile
```

Only CUDA 13 dependency variants are supported by this release.

xDataReader's GPU FITS path additionally needs a natively built extension
that is not included in prebuilt wheels. Build it from a source checkout with
`bash src/cuphoton/xdr/src/build.sh` (see
[docs/components/xdr.md](docs/components/xdr.md)).

## Python and command-line interfaces

Import through the namespaced modules:

```python
from cuphoton.xdr import batch_to_device
from cuphoton.xdr import load_hdf5
from cuphoton.xfit import GaussianDipoleModel, fit_dipoles
from cuphoton.xpois import solve_separable_kernel
from cuphoton.xrep import ReprojectionSpec, reproject_array
```

cuPhoton installs one executable:

```text
cuphoton
```

Run `cuphoton --help` for the component list, `cuphoton <group> --help` for a
group's commands, and `cuphoton <group> help <command>` for detailed options.
The same interface is available through `python -m cuphoton`.

## Data boundary

cuPhoton works with caller-supplied local files. Depending on the component,
these may include FITS, HDF5, NumPy, CSV, or Parquet products. The package
does not authenticate to observatory services, acquire data rights, or install
survey pipeline stacks. Keep credentials, restricted datasets, trained
weights, and generated runs outside the repository.

See [Data and artifact contracts](docs/data-artifacts.md) before adapting a
workflow to new products.

## Development

```bash
uv lock --check
make lint
make test-cpu
uv build
```

See [Contributing](CONTRIBUTING.md) for the full development workflow and
[Support](SUPPORT.md) for the project's best-effort support model.

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Quickstarts](docs/quickstarts.md)
- [Architecture](docs/architecture.md)
- [Adapting the workflows](docs/adapting-workflows.md)
- [Command-line index](docs/cli.md)
- [Troubleshooting](docs/troubleshooting.md)

## License and citation

cuPhoton is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[Third-party notices](THIRD_PARTY_NOTICES.md). Cite the software as described
in [CITATION.md](CITATION.md).
