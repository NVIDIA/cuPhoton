# xImage

`cuphoton.ximage` reconstructs a common scene from aligned astronomical
exposures using supplied PSFs, variance planes, exposure times, and masks.
It provides a squared-error solver and robust-loss variants, with PyTorch
execution on CPU or CUDA 13 GPUs.

xImage implements a PyTorch adaptation of the
[ImageMM reconstruction approach](https://doi.org/10.3847/1538-3881/adfb72)
described by Sukurdeep et al. (2025). For publications using xImage
reconstruction, cite both ImageMM and the cuPhoton version used; see the
[citation guide](../../CITATION.md#ximage) and
[scientific references](../references.md#image-restoration-and-super-resolution).

PSF calibration is a separate operation: Adam or multiplicative updates fit
per-exposure kernels to supplied PSF stamps using a Gaussian reference.
Reconstruction holds its supplied kernels fixed. It does not estimate scene
and PSF jointly from the science images.

## Synthetic example

Install the component dependencies and create a small, deterministic bundle:

```bash
uv sync --locked --extra ximage
uv run python examples/ximage/make_synthetic_bundle.py --output data
uv run cuphoton ximage data-validate --base data
uv run cuphoton ximage reconstruct-l2 \
  --config examples/ximage/reconstruct-l2.example.yaml
uv run cuphoton ximage reconstruct-rho \
  --config examples/ximage/reconstruct-rho.example.yaml
```

The example generates three noisy exposures of synthetic point sources.
Its configurations reconstruct a central 48-by-48 ROI within the 64-by-64
images and disable masking. With `padding: VALID`, the input must include
the surrounding pixels required by the PSF support: four pixels beyond each
ROI edge for the example's native 9-by-9 PSFs. Outputs
are written under `runs/`; an existing named run is preserved unless
`output.overwrite` is explicitly enabled. Configuration paths resolve from
the working directory.

Set `device: cpu` or `device: cuda` to require that device. `device: auto`
selects CUDA when available and otherwise uses CPU. Float32 and float64 are
supported; each run records the resolved device, precision, and timings.

## Inputs and reconstruction

Bundles use NPY directories, HDF5, or Zarr stores. Image, variance, and sky
arrays have axes `(batch, y, x, exposure)`; PSFs have axes
`(psf_y, psf_x, channel, exposure)`. The current solvers require one PSF
channel; evaluation supports one image batch. Exposures must describe the
same registered sky region.
See [Data and artifact contracts](../data-artifacts.md#ximage-reconstruction-bundles)
for all arrays, mask conventions, and output fields.

```bash
cuphoton ximage data-inspect --store data --artifacts images,psfs
cuphoton ximage psf-solve-adam \
  --config examples/ximage/psf-adam.example.yaml
cuphoton ximage psf-solve-mm \
  --config examples/ximage/psf-mm.example.yaml
```

PSF calibration supports L1 or L2 objectives with Adam and L2 with
multiplicative updates. To reconstruct with calibrated kernels, set
`psf_path` in the reconstruction configuration to the saved `psfs.npy`.
With `psf_path: null`, reconstruction uses the bundle's PSFs.

Each completed run contains `config.yaml`, `summary.json`, numeric result
arrays under `artifacts/`, and numeric convergence traces under `trace/`.
With `VALID` padding, the saved reconstruction includes PSF context around
the selected ROI; evaluation projects and crops it to the observation ROI.

Run directories and local Rubin manifests retain input file paths, exposure
identifiers, sky coordinates, and caller-provided metadata needed to trace
the computation. Benchmark reports also record hardware details. Review
these generated files before sharing them; they are not anonymized exports.

ROI selection, exposure selection, exposure batching, and optional tiling
limit the work performed per run. Consult `cuphoton ximage --help` and the
annotated example configurations for command options.

## Evaluation and timing

```bash
cuphoton ximage evaluate-reconstruction \
  --run-dir runs/reconstruct-l2-example
cuphoton ximage benchmark-devices \
  --config examples/ximage/reconstruct-l2.example.yaml \
  --workflow reconstruct-l2 --devices cpu,cuda --warmup 1 --repeats 3
```

Evaluation reports image similarity, background statistics, and aperture
photometry against an equally weighted mean or median coadd of valid,
exposure-time-normalized pixels. Masked or invalid-variance samples are
excluded; pixels with no valid contributors are zero. Supply
`--pipeline-coadd-fits`, `--catalog-parquet`, or `--reference-catalog-csv`
for external comparisons.
A coadd and reconstruction made from the same exposures share noise, and
similarity or sharpness alone does not establish reconstruction accuracy.
Catalog comparisons also depend on registration, flux calibration, source
selection, and the reference catalog. Evaluate those assumptions for the
intended scientific measurement; no automatic scientific acceptance decision
is produced.

The device benchmark times complete workflow calls, including input loading
and artifact writing, after the requested warmup runs. It records individual
timings and hardware metadata. `--evaluate` adds reconstruction metrics;
elementwise numerical parity requires a separate array comparison.

## Local Rubin products

The `rubin-*` commands inspect caller-supplied FITS and catalog files, build
manifests and assemble reconstruction bundles. `rubin-build-store` can align
visit images using their WCS; inspect its mask and interpolation settings
for the selected products. Inspect the
per-exposure `psf_source` metadata as well: when no native PSF is available,
the builder uses a Gaussian derived from exposure summary moments, or a
unit impulse (`delta_fallback`) if those moments are unavailable. These
fallback kernels are approximations; supply an appropriate PSF model for
scientific reconstruction.
The build result reports `psf_source_counts` and warns about unit-impulse
placeholders. The generated `sky` plane is zero-filled, not a sky estimate.
The optional `rubin` dependency extra enables the `lsst-images` readers.
Its HTTPS support is needed to interpret archive schema identifiers even
when reading local files.
Data acquisition and authentication remain the caller's responsibility.

Evaluating coadds or sky-coordinate catalogs against a reference-WCS-aligned
bundle requires its local reference FITS file and valid WCS headers.

Pipeline coadds carry resampling and combination effects. They can serve as
comparison images; treating them as independent raw exposures requires an
appropriate noise and PSF model. Coadd-derived reference catalogs inherit
the same image noise and are useful for relative checks.

## Python API

```python
from cuphoton.ximage import load_ximage_bundle, validate_bundle

with load_ximage_bundle("data") as bundle:
    issues = validate_bundle(bundle)
    if issues:
        raise ValueError(issues)
    images = bundle.images
```

HDF5 and Zarr arrays remain lazy while the bundle is open. Materialize any
arrays needed after closing it. Workflow functions `run_psf_solve` and
`run_reconstruction` are available from `cuphoton.ximage.workflows`.
