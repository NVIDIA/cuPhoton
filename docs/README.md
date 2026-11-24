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
- [Model Sky](https://nvidia.github.io/cuPhoton/main/model-sky.html),
  [Model Scene](https://nvidia.github.io/cuPhoton/main/model-scene.html) and
  [Model Alerts](https://nvidia.github.io/cuPhoton/main/model-alerts.html)
  are interactive notebooks in the development guide. Explore simulated sky
  measurements, scene reconstruction and transient classification in a browser.
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

Build the complete guide for a committed revision with Python 3.12 and the
[documentation developer tools](../dev/README.md). Run these commands from
the repository root and use a new output directory:

```sh
uv sync --project dev --locked --python 3.12
uv run --project dev --locked cuphoton dev docs-build \
  --ref HEAD --output /tmp/cuphoton-guide
uv run --project dev --locked cuphoton dev docs-check /tmp/cuphoton-guide
```

Open `/tmp/cuphoton-guide/index.html`. The builder runs the CPU examples with
the revision's locked dependencies and Quarto version. For revisions that
declare notebook chapters, it retrieves their saved GPU results, verifies
them and renders the chapters in the same book. Pull requests build a guide
preview. Successful pushes to `main` publish the development guide. A formal
release keeps the chapters declared by its own source revision.

When editing the CPU chapters in `docs/guide`, a direct Quarto render provides a
shorter local preview with output in `docs/guide/_book`:

```sh
uv sync --locked --python 3.12 --group docs --extra ximage --extra catalog
CUPHOTON_REPO="$PWD" QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render docs/guide
```

## Updating a notebook chapter

Run and review the full reference study on a CUDA 13 system. Each example's
publishing guide describes the required profile, scientific checks and local
HTML preview. Follow the instructions for [Model Sky](../examples/model-sky/PUBLISHING.md),
[Model Scene](../examples/model-scene/PUBLISHING.md) or
[Model Alerts](../examples/model-alerts/PUBLISHING.md).

Prepare the public files before uploading them. Set `REVISION` to the full Git
commit used for GPU execution, `REFERENCE` to the reference run's output
directory and `PUBLIC` to a new directory. For Model Sky:

```sh
uv run --project dev --locked python -m cuphoton_dev.notebook_export \
  --name model-sky --revision "$REVISION" --reference "$REFERENCE" \
  --output "$PUBLIC"
```

For Model Alerts, use `--name model-alerts`. For Model Scene, use
`--name model-scene` and add `--scene-downloads /path/to/preview/guide/_book/source`
to use the preview's sanitized notebook and receipt. Preparation checks the
original execution hash, source revision and public notebook contents. Inspect
the public files, including text output and warnings, before recording them.

```sh
uv run --project dev --locked python -m cuphoton_dev.notebooks record \
  --name model-sky --revision "$REVISION" \
  --notebook "$PUBLIC/model-sky.executed.ipynb" \
  --receipt "$PUBLIC/model-sky-execution.json"
```

This command uploads content-addressed assets to the repository's
`docs-notebook-assets` prerelease using the authenticated `gh` account.
Use the entry it prints to update [`guide/notebooks.json`](guide/notebooks.json).
Review the pinned source revision, notebook digest and receipt digest in the
source PR. After committing the update, run the complete guide build and check
the new chapter's figures, downloads, navigation and source links.

The trusted publisher stores the large rendered HTML and notebook downloads
alongside the execution inputs, under SHA-256 asset names. Its `gh-pages`
archive retains the hashes. Each deployment restores and checks the declared
files for every retained version. Keep those assets available for subsequent
deployments. The [publishing runbook](../dev/README.md#publish-bootstrap-and-recover)
covers release publication, retries and live verification.
