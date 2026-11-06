# cuPhoton documentation

cuPhoton is organized as a set of reference workflows. Use the README's
[workflow chooser](../README.md#choose-a-workflow) to match your inputs to a
component and see what it produces. The [component overview](../README.md#components)
explains the scientific operations.

For a view across components, read
[how the science components fit together](architecture.md#how-the-science-components-fit-together)
and [what an exposure contains](architecture.md#what-an-exposure-contains).
Then run a synthetic quickstart and read the contract for the component you
intend to adapt.

The [changelog](../CHANGELOG.md) records release history and migration
requirements. [Releases](https://github.com/NVIDIA/cuPhoton/releases)
provides installation notes for published versions.

## First run

- [User guide](https://nvidia.github.io/cuPhoton/): installation, a first image
  difference, and task guides with executable examples.
- [Getting started](getting-started.md): prerequisites and installation
  profiles.
- [Packaging](packaging.md): local wheel and Conda builds, source builds,
  and package qualification procedures.
- [Quickstarts](quickstarts.md): data-independent CPU and GPU smoke runs.
- Imaging walkthrough:
  [notebook](../examples/imaging-pipeline/run_imaging_pipeline.ipynb) and
  [script](../examples/imaging-pipeline/run_imaging_pipeline.py) with setup for
  synthetic FITS loading, alignment, subtraction, dipole fitting, and plotting;
  requires a CUDA 13-capable NVIDIA GPU and xDR's native extension.
- [Command-line index](cli.md): the umbrella executable and command groups.

## Workflow guides

- [Core](components/core.md): shared CLI and configuration behavior.
- [cuPhoton xLens](components/xlens.md): scientific monitoring, offline
  dashboards and optional evidence-based investigation through institution-hosted
  models.
- [xDataReader](components/xdr.md): GPU FITS loading and the shared
  Astropy/xDR reader interface, with [runtime compression controls](components/xdr.md#runtime-choices)
  for CLI, Python and manifests; native GDS depends on the storage setup.
- [xFit](components/xfit.md): batched nonlinear least-squares dipole fitting.
- [xImage](components/ximage.md): multi-exposure reconstruction and PSF calibration.
- [xPois](components/xpois.md): kernel fitting and image subtraction.
- [xScan](components/xscan.md): transient datasets, classification,
  evaluation, and review.
- [xRep (xReproject)](components/xrep.md): FITS/WCS reprojection.
- [xRay](xray/README.md): X-ray trace and detector analysis.
- [xBeam](xbeam/README.md): time-series correlation and radio visibilities,
  with [array and HDF5 inputs](xbeam/time-series.md).

## Adapting and validating

- [Architecture](architecture.md): package boundaries and execution model.
- [Distributed architecture](distributed.md): work assignment, GPU ownership,
  data movement and completion.
- [Dragon architecture](dragon.md): process placement, GPU sharing with MPS,
  queues and worker lifecycle.
- [MPI architecture](mpi.md): rank binding, collective coordination and file mode.
- [Distributed launch guide](distributed-execution.md): prepare a pipeline
  manifest and launch MPI or Dragon workers using Slurm or SSH host files.
- [Dragon transport and performance](dragon-performance.md): transport selection,
  queue placement experiments and timing boundaries.
- [Pipeline stage benchmark](components/pipeline-stage-benchmark.md): compare
  a persistent GPU pipeline with separate stages and inspect transfer costs.
- [Adapting the workflows](adapting-workflows.md): a practical extension
  process.
- [Data and artifact contracts](data-artifacts.md): shapes, files, metadata,
  and provenance.
- [Uncertainty quantification](uncertainty-quantification.md): the error bars
  of transient catalogs, their terms, configuration, and checks on your data.
- [Troubleshooting](troubleshooting.md): environment, CUDA, data, and CLI
  failures.
- [Glossary](glossary.md): project and astronomy terminology.
- [Scientific references](references.md): methods represented in the code.

Project-wide policies are in [Contributing](../CONTRIBUTING.md),
[Governance](../GOVERNANCE.md), [Support](../SUPPORT.md), and
[Security](../SECURITY.md).

## Building the user guide

The Quarto book in [`guide/`](guide/) renders with Python 3.12 and Quarto
1.9.36. From the repository root:

```sh
uv sync --locked --python 3.12 --group docs --extra ximage --extra catalog
CUPHOTON_REPO="$PWD" QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render docs/guide
```

The documentation environment uses CPU PyTorch. Open
`docs/guide/_book/index.html` to read the result. Generated HTML, sample
outputs, and Quarto's execution cache are ignored by Git.

The `user guide` workflow builds pull requests and publishes successful
`main` builds to GitHub Pages. It also supports a manual run on `main`.
Each clean CI build executes the CPU examples from the checked-out source.
