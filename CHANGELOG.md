# Changelog

## 0.1.3

These changes are relative to 0.1.2. Version 0.1.3 is available on
[PyPI](https://pypi.org/project/cuphoton/0.1.3/).

### Breaking changes

- Python 3.11 is no longer supported. Use Python 3.12, 3.13, or 3.14.
- Photutils is now optional and is no longer installed by `pip install cuphoton`.
  Install `cuphoton[photometry]` for CPU photometry or `cuphoton[gpu]` for the
  combined GPU and photometry dependencies.
- xDR's Legate-backed HDF5 loader has been removed. Use `h5py` for local HDF5
  reads and explicitly transfer arrays to a GPU when needed. xRay continues
  to read its documented HDF5 layouts; see
  [HDF5 migration](docs/components/xdr.md#hdf5-migration).

### Installation and development

- Native Linux x86-64 and ARM64 wheels include the xDR extension and a
  private, thread-safe CFITSIO library for CPython 3.12, 3.13, and 3.14.
  Release packages are distributed through PyPI.
  Source and editable installs still require an explicit native build.
- The `io`, `gpu`, `cutile`, `mpi`, and `dragon` extras separate GPU I/O,
  numerical backends, and distributed runtimes. This release supports CUDA
  13; cuTile needs a compatible TileIR compiler. Dragon requires Python
  3.12 or 3.13. MPI requires a site-provided implementation and launcher.
- Development checks now include mypy, C/C++/CUDA formatting, and ShellCheck
  alongside Ruff and the portable `make hooks` setup.

See [Getting started](docs/getting-started.md) and
[Packaging](docs/packaging.md) for installation and build details.

### Imaging workflows

- Shared FITS inspection and reading select Astropy or xDR independently of
  the compute backend, preserving HDU, dtype, scaling, and reader provenance.
  xPois and xRep use this interface, as do selected xScan data adapters and
  the device pipeline. Explicit xDR requests require supported inputs;
  automatic selection records an Astropy fallback reason.
- xFit accepts FITS candidate manifests with image, variance, mask, and
  region selections in addition to NPZ batches. The `fit_dipoles_device` API
  retains results and diagnostics on the GPU; ordinary fits and CLI runs
  return host results. The optional cuTile backend fits analytic Gaussian
  models. Gaussian model evaluation avoids computing unused derivatives.
- xRep can reproject image stacks onto an existing FITS WCS grid.
- xPois adds spatially varying alternating-linear-least-squares matching
  and an experimental Gaussian-polynomial Python API, with CPU and CuPy
  implementations. Its constant-kernel path supports device results,
  fixed-kernel residual variance propagation, and cuTile normal equations.
  Fits can omit review
  artifacts, and subtraction diagnostics summarize standardized residuals.
- xScan supports candidate-keyed xFit feature bundles and device feature
  conversion, explicit checkpoint inference policies, and direct FITS
  candidate scoring through its CUDA `predict_fits` Python API.
- A reusable device context connects constant-kernel xPois, Gaussian xFit,
  and triplet xScan inference. Shared Dragon/MPI executors support persistent
  workers for image pairs, candidate chunks, and inference minibatches.
  Benchmark rounds preserve separate warmup and measurement artifacts.
- xDR retains buffers until GPU work completes, permits independent reads
  during GDS waits, and exports benchmark results as JSON.

See the [component guides](docs/README.md#workflow-guides),
[data contracts](docs/data-artifacts.md), and
[pipeline stage benchmark](docs/components/pipeline-stage-benchmark.md).

### xRay

- Optional damped-cosine iterative fitting complements linear prediction.
  Fit artifacts include status-aware diagnostics and optional P2 ridge
  regularization.
- Detector processing batches eligible row traces and FFTs. Detector maps
  retain the row-within-tile fit granularity; each row result is repeated
  across that tile's columns.
- Detector artifacts support lossless `tile-rows` storage for compact spectra
  and bounded pixel reads through the shared loader. Dense NPY output remains
  the default for existing consumers.

See the [xRay guide](docs/xray/README.md) for numerical contracts and options.
