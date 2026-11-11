# Changelog

## Unreleased

### xLens

- Restore NOOA as an opt-in `xlens-nooa` extra for iterative evidence queries
  and isolated Python on Linux with Python 3.12/3.13 and bubblewrap. Basic
  inference remains the default. This supersedes the earlier development
  removal of NOOA and restores `backend`, `max_iterations`, `max_tool_calls`
  and `max_evidence_bytes` configuration.
- Add an explicit Responses API transport alongside Chat Completions, with
  documentation for optional `gpt-6-rosalind` configuration. Model access and
  endpoint compatibility depend on the user's provider; a live Rosalind
  integration has not been qualified.

### Breaking changes

- Move Pandas and NumExpr into the `catalog` extra for xScan ingestion,
  and Pydantic into the `ximage` extra for xImage configuration. Select
  these extras for the corresponding workflows.
- Remove the Numba-CUDA xPois backend and Numba-CUDA-MLIR xFit backend,
  including their dependencies, installation extra, CLI options, and
  pipeline configuration choices. Use CuPy, native CUDA, or cuTile for GPU
  execution. CPU execution remains available.

### Installation and development

- Add `make sync-dragon` and a generated, hashed `requirements-dragon.txt`
  for Python 3.12/3.13 GPU environments, including catalog dependencies.
  CI checks the export against `uv.lock`.
- Lock notebook and native Python build tools in dependency groups. The xDR
  source-build helper preserves installed runtime dependencies when rebuilding.

### xFit solver errata

- The default `v0.2` solver can require substantially more evaluations on
  slowly converging fits. Reaching the evaluation cap returns an
  unconverged result with invalid uncertainties, including for some fits
  that previously reported convergence. Inspect termination status and
  uncertainty validity when comparing results or timings. An explicit
  larger `LMConfig(max_evaluations=...)` budget or `--max-evaluations`
  permits more work without guaranteeing convergence.
- The `v0.1.3` policy can report convergence after a small actual cost
  reduction while further improvement remains, and its unscaled rank and
  covariance calculations depend on parameter units. Its intermediates use
  the selected compute dtype: FP32 for float32 fits and FP64 for float64
  fits. FP32 intermediates can lose small curvature. Its dense covariance
  propagation can erase small derived uncertainties through cancellation.
  For comparisons that require this earlier policy, select
  `LMConfig(solver_version="v0.1.3")` in Python or
  `cuphoton xfit fit-dipoles ... --solver-version v0.1.3` in the CLI.
  `v0.2` remains the default; its scaling and stopping checks address those
  limitations. Both policies record the selected version and require
  checking the fit and its uncertainty diagnostics. See
  [solver version selection](docs/components/xfit.md#choosing-the-solver-version).

### xBeam

- Correlate uniformly sampled signals from arrays or HDF5 with NumPy or
  CuPy, producing cross-spectra and lag correlations.
- Compute radio visibilities from two-bit, non-legacy VDIF files with
  bounded input windows, GPU execution, time sharding and validated output
  assembly.
- Use the shared `cuphoton xbeam` CLI for correlation, VDIF inspection,
  sidecar creation and radio shard merging. See the
  [xBeam guide](docs/xbeam/README.md) for installation and examples.

## 0.1.3

These changes are relative to 0.1.2. Version 0.1.3 is a source-only release;
follow the [checkout instructions](docs/getting-started.md#clone-and-select-a-profile).

### Breaking changes

- Python 3.11 is no longer supported. Use Python 3.12, 3.13, or 3.14.
- Photutils is now optional and is no longer part of the base installation.
  Select the `photometry` extra for CPU photometry or `gpu` for the
  combined GPU and photometry dependencies.
- xDR's Legate-backed HDF5 loader has been removed. Use `h5py` for local HDF5
  reads and explicitly transfer arrays to a GPU when needed. xRay continues
  to read its documented HDF5 layouts; see
  [HDF5 migration](docs/components/xdr.md#hdf5-migration).

### Installation and development

- Local native package build tooling covers Linux x86-64 and ARM64 for
  CPython 3.12, 3.13, and 3.14. The wheel recipe includes the xDR extension
  and a private, thread-safe CFITSIO library. Native FITS loading from
  source or editable installs requires an explicit native xDR build;
  Astropy provides the Python-only FITS reader.
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
- xDR defaults to fused FITS pixel restoration and prefers native Gzip decoding, with
  automatic selection of compatible decompression hardware. CLI flags and
  Python options select separate restoration kernels, aligned raw DEFLATE,
  or CUDA decompression explicitly; see [runtime choices](docs/components/xdr.md#runtime-choices).

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
