# cuPhoton

cuPhoton is a collection of GPU-accelerated reference workflows for
astronomical imaging and X-ray trace analysis. It is intended for research
teams that want working implementations they can run, inspect, and adapt to
their own instruments and data products. The curated Python exports and
structured run artifacts provide integration points for those workflows.

## Choose a workflow

| If you have... | Use... | You get... |
| --- | --- | --- |
| FITS files whose pixels you need on a GPU | [xDataReader](docs/components/xdr.md) | Decoded and scaled device arrays |
| Images with different pixel-to-sky mappings | [xRep](docs/components/xrep.md) | Images resampled onto a common sky grid |
| An aligned reference image and a new observation | [XPOIS](docs/components/xpois.md) | A matched reference and a difference image |
| Small cutouts containing positive/negative residuals | [xFit](docs/components/xfit.md) | Fitted positions, amplitudes, shapes, uncertainties, and fit status |
| Candidate cutouts to score, or reviewed examples to train on | [XScan](docs/components/xscan.md) | Real/bogus scores, evaluation metrics, and review material |
| X-ray detector images sampled over experimental delay | [XRay](docs/xray/README.md) | Signal traces and maps of fitted oscillations |

Each component can be used independently. An optical workflow can combine
loading, reprojection, subtraction, and candidate analysis; XRay handles a
separate kind of experimental data. See [how the components fit
together](docs/architecture.md#how-the-science-components-fit-together) for
the data flow and the adapters needed between stages.

## Start here

[uv](https://docs.astral.sh/uv/) manages the supported development
environments and lock file. On a Linux system with a CUDA 13-capable driver:

```bash
git clone https://github.com/NVIDIA/cuPhoton.git
cd cuPhoton
uv sync --locked --extra dev --extra gpu --extra viz
uv run python examples/run_quickstarts.py
```

The quickstart creates synthetic inputs for xRep, XPOIS, xFit, XScan, and
XRay, then writes results to `quickstart-output/`. xDataReader has a separate
[native GPU setup](docs/components/xdr.md#install). The JSON summary records
the requested profile, the backend and device selected for each component,
and the artifact paths. The default `auto` profile prefers a GPU and reports
when it falls back to CPU.

For a deterministic CPU run:

```bash
uv sync --locked --extra dev --extra torch --extra viz
uv run python examples/run_quickstarts.py --profile cpu
```

Use `--require-gpu` to require GPU execution. See
[Quickstarts](docs/quickstarts.md) for individual components and output
contracts.

For a synthetic imaging walkthrough from FITS loading through alignment,
subtraction, dipole fitting, and plotting, run the
[notebook](examples/imaging-pipeline/run_imaging_pipeline.ipynb) or equivalent
[script](examples/imaging-pipeline/run_imaging_pipeline.py). Both include setup
instructions and require a CUDA 13-capable NVIDIA GPU and XDR's native extension.

## Components

### xDataReader: load pixels into GPU memory

xDataReader (`cuphoton.xdr`) reads selected images from local FITS files,
decodes supported compression, and applies byte-order and scaling rules to
produce CuPy arrays. Use it to feed a GPU workflow while retaining the headers
and scientific metadata in your application. Its native FITS extension
requires a source build. Whether reads use native GPUDirect Storage depends
on the storage and driver configuration.

### xRep: put images on the same sky grid

xRep, short for xReproject (`cuphoton.xrep`), resamples images so corresponding
pixels refer to the same sky positions. It uses the supplied World Coordinate
System (WCS) mapping and a chosen output grid. This handles differences in
pixel scale, rotation, and position; matching the images' blur is a separate
operation. Outputs include the reprojected arrays, optional masks, and grid
metadata.

### XPOIS: subtract a matched reference to reveal changes

XPOIS (`cuphoton.xpois`) starts with aligned reference and target images. It
fits a convolution kernel and background correction to match their blur,
brightness scale, and background, then subtracts the matched reference.
Unchanged sources should largely cancel, leaving a difference image for
candidate detection. Use the saved matching model and diagnostics to assess
alignment, pixel quality, and the fit alongside candidate residuals.

### xFit: turn a candidate's shape into measurements

A dipole is a positive lobe beside a negative lobe in a difference image;
a small positional mismatch or a moving source can produce one. xFit
(`cuphoton.xfit`) fits models to batches of small image cutouts, called
stamps. It returns positions, amplitudes, and shape parameters with residuals,
uncertainties, and fit status. These measurements describe the candidate and
can optionally become inputs to an XScan classifier.

### XScan: score candidates and collect review labels

XScan (`cuphoton.xscan`) trains and evaluates real/bogus classifiers and runs
inference on candidate stamps. Models use search and template images,
optionally a difference image, and explicitly selected xFit features.
Real/bogus scores help prioritize plausible detections over artifacts for
further scientific classification. XScan also produces review material for
inspecting predictions and collecting labels. Training requires reviewed
labels and suitable train/validation/test splits.

### XRay: measure oscillations in detector signals

XRay (`cuphoton.xray`) analyzes X-ray detector measurements sampled over
experimental delay. In a pump/probe experiment, one pulse excites a sample
and another measures its response after a controlled delay. XRay extracts
and normalizes signal traces from selected detector regions, then uses linear
prediction to estimate their oscillatory modes. Outputs include frequencies,
amplitudes, fit diagnostics, and detector review maps. Interpreting those
measurements requires the experiment's geometry and calibration.

All Python code is in the `cuphoton.*` namespace. The shared
[Core](docs/components/core.md) (`cuphoton.core`) provides command discovery,
application context, logging, and invariant checks. Each science component
owns its algorithms and data contracts. The [glossary](docs/glossary.md)
explains the scientific and file-format terms used here.

## Installation profiles

The base install contains the shared CPU data and scientific stack. Optional
extras are deliberately separated by purpose:

Python 3.11 through 3.14 is supported on Linux for the base, GPU, CPU PyTorch,
and visualization profiles. The experimental cuTile profile supports Python
3.12 and 3.13.

| Extra | Use |
| --- | --- |
| `dev` | Tests, formatting, linting, and build tools |
| `torch` | PyTorch workflows that can be forced to CPU execution |
| `gpu` | CUDA 13 PyTorch, CuPy, Numba-CUDA, KvikIO, and nvCOMP backends |
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

xDataReader's GPU FITS path uses a native extension built from source.
From a source checkout, build the extension with
`bash src/cuphoton/xdr/src/build.sh` (see
[docs/components/xdr.md](docs/components/xdr.md)).

## Python and command-line interfaces

Import through the namespaced modules:

```python
from cuphoton.xdr import batch_to_device
from cuphoton.xfit import GaussianDipoleModel, fit_dipoles
from cuphoton.xpois import solve_separable_kernel
from cuphoton.xrep import ReprojectionSpec, reproject_array
```

cuPhoton provides one Python console entry point:

```text
cuphoton
```

Run `cuphoton --help` for the component list, `cuphoton <group> --help` for a
group's commands, and `cuphoton <group> help <command>` for detailed options.
The same interface is available through `python -m cuphoton`.

Installations also include `cuphoton-openmpi-rank-exec`, a low-level launch
helper that binds one Open MPI rank to one visible GPU before Python starts.
Invoke it through `mpirun`.

## Data boundary

cuPhoton works with caller-supplied local files. Depending on the component,
these may include FITS, HDF5, NumPy, CSV, or Parquet products. Applications
handle observatory access, data permissions, and survey-specific preparation,
then pass local products into cuPhoton. Store credentials, datasets, trained
weights, and generated runs in application-managed locations.

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
