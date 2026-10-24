# xImage

`cuphoton.ximage` reconstructs a common nonnegative scene from registered
frames using supplied per-frame point-spread functions (PSFs), noise
variances, and good-pixel masks. It provides weighted least-squares and
robust-loss reconstruction with PyTorch on CPU or CUDA 13 GPUs.

The reconstruction model assumes:

- Frames are already registered to a common pixel grid. Use
  [xRep](xrep.md) when resampling is needed before reconstruction.
- Each frame has one known PSF, fixed across the field. Reconstructing
  separate tiles can approximate field variation when each tile is supplied
  an appropriate local PSF; the solver does not fit a spatial PSF field.
- The reconstructed intensity is nonnegative, as required by the
  multiplicative update.
- Noise is Gaussian with known pixel variances. The L2 solver minimizes
  the corresponding variance-weighted squared error.

xImage implements a PyTorch adaptation of the
[ImageMM reconstruction approach](https://doi.org/10.3847/1538-3881/adfb72)
described by Sukurdeep et al. (2025). For publications using xImage
reconstruction, cite both ImageMM and the cuPhoton version used; see the
[citation guide](../../CITATION.md#ximage) and
[scientific references](../references.md#image-restoration-and-super-resolution).
The robust example uses the paper's Huber loss. The additional `cauchy`,
`bisquare`, and `damped_rl` losses and the `scalar_per_exposure` background
model are experimental extensions. Their use requires validation for the
intended data and measurement.

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

Bundles use NPY directories, HDF5, or Zarr stores. Images and variances have
axes `(batch, y, x, frame)`; PSFs have axes
`(psf_y, psf_x, channel, frame)`. Each PSF is the image of a point source,
including when loaded from a PSF calibration run. The current solvers
require one PSF channel; evaluation supports one image batch. Frames must
describe the same registered scene.

An optional boolean `masks` array has the same shape as the images, with
`True` selecting a good pixel. `mask.mode: provided` uses these masks and
`mask.mode: none` ignores them. Missing masks admit all pixels. Supply finite image values and strictly
positive finite variances. Optional `exp_times` defaults to
ones; supplied values divide image intensities and their squares divide
variances. The auxiliary `sky` and `mcgauss` arrays are optional.
Convert legacy twelve-plane masks explicitly through
`cuphoton.ximage.adapters.legacy` before using the generic bundle.
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
  --run-dir runs/reconstruct-l2-example --metrics-backend auto
cuphoton ximage benchmark-devices \
  --config examples/ximage/reconstruct-l2.example.yaml \
  --workflow reconstruct-l2 --devices cpu,cuda --warmup 1 --repeats 3
```

Image metrics use CuPy when a CUDA device is available, otherwise NumPy/SciPy.
Select `--metrics-backend cpu` or `cupy` to require an engine; the same option
applies to `benchmark-devices --evaluate`. Install `--extra ximage --extra gpu`
for CuPy support. FFT sharpness, sky statistics, finite coverage, PSNR and
SSIM run in float64; partial images retain finite-support comparisons and
unavailable Fourier sharpness. Photometry and WCS alignment remain on CPU.
Reports record `metrics_backend` separately from the reconstruction backend.
Fourier sharpness takes the logarithm of FFT magnitudes, which amplifies
roundoff near zero coefficients. Such spectra can give different sharpness
values across backends; select `cpu` when comparing against CPU baselines.

Generic image similarity and background-statistic functions are available
from `cuphoton.ximage.metrics`, including `psnr`, `ssim`,
`fourier_sharpness`, and `image_metrics`. Use `evaluate_image_metrics`
from the same module to select a CPU or CuPy backend for a pair of images.
Run evaluation also supports the
Rubin adapter's aperture photometry and catalog comparisons. It compares
against an equally weighted mean or median coadd of valid,
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

## Worked adapter: local Rubin products

The `cuphoton.ximage.adapters.rubin` package converts local Rubin products
to the generic frame, PSF, variance, and good-pixel-mask contract. Its
`rubin-*` commands inspect caller-supplied FITS and catalog files, build
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
