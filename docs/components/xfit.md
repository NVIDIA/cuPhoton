# xFit

`cuphoton.xfit` provides batched nonlinear least-squares fitting for
astronomical dipole models. Its low-level Levenberg--Marquardt solver retains
NumPy or CuPy arrays; the high-level fit result exposes portable NumPy arrays
and records the resolved backend, device, and dtype.

## Quickstart

Run the deterministic analytic-Gaussian fit on CPU:

```bash
uv run python examples/run_quickstarts.py \
  --component xfit --profile cpu \
  --output-dir /tmp/xfit-quickstart
```

Use `--require-gpu` in place of `--profile cpu` to require the CUDA 13 CuPy
path. Inspect the generated `summary.json`; an automatic run may use NumPy
when no usable CUDA device is available.

## Models and Python API

`GaussianDipoleModel` fits
`amplitude, sigma_x, sigma_y, theta, x_pos, y_pos, x_neg, y_neg`.
The two sigma parameters are strictly positive pixel standard deviations and
are optimized in log space. Returned Gaussian orientations are canonicalized
to `[-pi/2, pi/2)`; exchanging `sigma_x` and `sigma_y` with a quarter-turn
describes the same ellipse.
`StampDipoleModel` fits `x_pos, y_pos, x_neg, y_neg, flux` using a supplied
sampled stamp/PSF basis. Stamp evaluation supports bilinear,
bilinear-vignetted, and finite-volume integration paths.

```python
import numpy as np

from cuphoton.xfit import GaussianDipoleModel, fit_dipoles

model = GaussianDipoleModel((17, 21), dtype=np.float64)
truth = np.asarray(
    [[7.0, 1.7, 1.3, 0.2, -2.3, -1.8, 2.5, 1.4]],
    dtype=np.float64,
)
images = model.evaluate(truth, mode="difference")
result = fit_dipoles(
    images,
    model=model,
    initial=truth + 0.1,
    backend="auto",
)
```

Difference mode accepts `(batch, y, x)` images. Split mode accepts
`(batch, 3, y, x)` ordered as difference, positive, and negative. Optional
masks and variances select and weight fitted pixels. Results include fit
status, evaluation counts, valid-pixel coverage, fitted and zero-signal
chi-square statistics, covariance, standard errors, and an explicit
uncertainty-validity reason. Nonfinite image or variance values are accepted
only where the mask excludes that pixel. Interpret residual entries at pixels
with finite image values.

Python inputs may be NumPy arrays, CuPy arrays, or array-like values accepted
by the resolved backend. `backend="auto"` can therefore transfer a host input
to CuPy when CUDA is usable. In split mode, an auxiliary `(batch, y, x)` array
applies one mask or variance plane to every channel of each candidate. If the
batch size is three, `(3, y, x)` keeps that per-candidate meaning; use the
explicit `(1, 3, y, x)` shape for per-plane values.

An explicit `backend="cutile"` uses one `cuda.tile` CTA per Gaussian fit to
form its weighted 8-by-8 normal equations directly. Select this backend
explicitly and install the `cuphoton[cutile]` extra on Linux with Python
3.12 through 3.14. Final rank and covariance diagnostics use the analytic
Jacobian and a singular-value factorization. Sampled-stamp fits stay on the
NumPy or CuPy backends. The Tile backend requires analytic derivatives and
rejects finite-difference fitting.

The `cutile` extra installs cuda-tile 1.6 or newer and CuPy. Kernel compilation
also requires a compatible CUDA toolkit. Follow the shared
[cuTile compiler setup](../getting-started.md#optional-runtimes) to configure
the compiler alongside the GPU dependencies. GPU support depends on the
compiler version.

Compare warmed end-to-end Gaussian fits with:

```bash
uv run --locked --python 3.12 --extra gpu --extra cutile \
  python examples/xfit/benchmark_gaussian.py --dtype float64
```

Split mode uses diagonal per-plane weights. When the difference plane is
derived from the positive and negative planes, those residuals are correlated;
statistical calibration of the reported covariance requires a caller-supplied
weighting model that accounts for that dependence. `uncertainty_valid` reports
numerical and rank validity. The xScan feature adapter requires difference-mode
xFit runs.

## CLI and artifacts

The CLI accepts pickle-free NPZ inputs whose arrays are numeric or Unicode:

```bash
uv run cuphoton xfit data-inspect --input /path/to/dipoles.npz
uv run cuphoton xfit data-validate \
  --input /path/to/dipoles.npz --model gaussian --mode difference
uv run cuphoton xfit fit-dipoles \
  --input /path/to/dipoles.npz \
  --output-dir /path/to/new-fit-run \
  --model gaussian --mode difference --backend auto \
  --compute-dtype input
```

An input archive requires `candidate_id` and `images`. It may also contain
`initial`, `mask`, `variance`, and `stamp_basis`. Object arrays, `.npy` files,
and pickle-backed inputs are rejected. A successful fit writes
`summary.json`, `effective-config.yaml`, `fits.parquet`, and
`fit-arrays.npz`; residuals remain numeric arrays within the NPZ
archive.

### FITS images and candidate positions

To fit directly from aligned FITS images, pass a JSON candidate manifest
instead of preparing an NPZ of stamps:

```json
{
  "schema": "cuphoton.xfit.fits-input/v1",
  "mode": "difference",
  "stamp_shape": [51, 51],
  "images": [
    {
      "path": "difference.fits",
      "hdu": 1,
      "mask_hdu": 2,
      "variance_hdu": 3,
      "bad_mask_bits": 15
    }
  ],
  "candidates": [
    {"candidate_id": "source-a", "x": 100, "y": 200},
    {"candidate_id": "source-b", "x": 300, "y": 400}
  ]
}
```

```bash
cuphoton xfit fit-dipoles --input candidates.json \
  --model gaussian --mode difference --backend cupy \
  --fits-reader auto --output-dir fit-run
```

FITS paths resolve relative to the manifest. HDUs are zero-based integer
indices. Candidate centers are zero-based integer `(x, y)` pixels;
`stamp_shape` is odd positive `[height, width]`. Candidate IDs retain their
order and must be unique integers or unique strings. Stamps that extend
outside an image are rejected.

Difference mode reads one supplied difference plane. Split mode reads exactly
three supplied planes in **difference, positive, negative** order. All planes
must already be aligned and have matching dimensions. This route performs
no registration, image subtraction, candidate detection or background removal.

Mask and variance HDUs are optional. A mask pixel is included when none of
its `bad_mask_bits` are set; omit that field to exclude every nonzero mask
pixel. The example value `15` selects bits 0–3; choose the bits appropriate
for your own data. No instrument-specific mask policy is assumed. Split mode
requires variance on every plane or on none. Nonfinite values retain the
ordinary xFit validation rules described above.

Optional `initial` is a numeric array with one parameter row per candidate,
using the same stamp-local model coordinates as NPZ inputs. Sampled-stamp
models accept `"stamp_basis": {"path": "psf.fits", "hdu": 0}`.

`--fits-reader auto` uses Astropy for NumPy fits and selects xDR for supported
GPU-fit inputs when its dependencies are available. `astropy` selects CPU
FITS decoding; `xdr` requires the xDR route and reports unsupported inputs.
GPU fits crop and retain stamps on device. The reader uses one bounding
rectangle enclosing all candidate stamps where supported. Automatic reads
of uncompressed cutouts use Astropy sections; explicit xDR reads those
images in full before cropping. Widely separated candidates can therefore
read much of an image even when the stamps are small.
Selecting xDR does not assert native GPUDirect Storage use. Run artifacts
record the manifest and source-file hashes, selected reader and any automatic
fallback. Existing NPZ loading is unaffected by this option.

`--xdr-postprocess auto|fused|separate` selects xDR postprocessing for FITS
reads. Python `load_xfit_dataset` accepts the equivalent
`xdr_options={"postprocess": "separate"}`. FITS manifests may specify
`xdr_options` at the top level and on individual image or `stamp_basis`
descriptors. Per-image choices override the manifest default; explicit
Python/CLI choices override only supplied keys and survive worker dispatch.

Input archives contain candidate identifiers and exact image pixels. Fit
artifacts contain identifiers, hashes, parameters, uncertainties, covariance,
and optional residuals. Confirm that the underlying data and metadata are cleared
for release before publishing these artifacts.

For the sampled-stamp model, choose `--stamp-evaluation bilinear`,
`bilinear-vignetted`, or `finite-volume` and provide `stamp_basis` in the
input archive. Solver controls include `--f-tol`, `--x-tol`, `--g-tol`,
`--max-evaluations`, and `--use-finite-difference`. Gaussian fits use their
analytic Jacobian unless finite differences are requested; sampled-stamp
fits always use finite differences and record that resolved choice in the
effective configuration. `--x-tol` is an absolute step-norm tolerance, so
its convergence criterion stays consistent when a periodic parameter drifts
to a large equivalent value.

`--compute-dtype input` preserves the input floating dtype. Select `float32`
or `float64` to run the solver at an explicit precision while preserving the
input arrays and their recorded per-candidate hashes. Float64 is preferable for
ill-conditioned observational fits when the additional compute cost is
acceptable.

See [Data and artifact contracts](../data-artifacts.md#xfit-dipole-batches)
for the stable shapes and output fields.

## Distributed fitting

`fit-dipoles --executor dragon|mpi` distributes independent candidate chunks
across GPUs. The default `--executor local` retains the original batch fit.
Distributed fitting requires `--backend cupy` or `--backend cutile`, a shared
filesystem for the input and output, and the same installed environment on
every worker. `--chunk-size` sets candidates per task independently of worker
count. Keep it fixed for matched comparisons; candidate IDs and input order
are restored in the merged artifacts.
The task count must be at least the MPI rank count. Dragon uses the smaller
of the requested worker count and task count.

The same executor commands accept a FITS candidate manifest. Planning checks
headers, candidate bounds and source hashes without decoding image pixels.
Each bound worker reads the candidate region and retains its stamps during
worker setup, before measured rounds. The first ordinary artifact finalization
reads host stamps for per-candidate input hashes after the timed worker phase.
The coordinator retains these stamps for later rounds and verifies referenced
file hashes before reuse. Setting `retain_input=False` in the workload API
instead reloads the host stamps each round.
Worker item receipts retain `fits_setup_reads`; the merged scientific
summary separates `fits_worker_setup_reads` from
`fits_finalizer_audit_reads`. Each read identifies its source and plane role;
the audit receipt's `reused` flag distinguishes retained data from a new read.
These setup/finalization reads are separate from reported worker timing;
include them explicitly when measuring a complete ingestion-to-result run.
All referenced FITS files must be visible to every worker and the coordinator.

Under an allocation with Dragon configured, run one warmup and two measured
passes with workers retained across all three passes:

```bash
dragon .venv/bin/cuphoton xfit fit-dipoles \
  --executor dragon --max-workers 8 \
  --input /shared/dipoles.npz --model gaussian --backend cupy \
  --chunk-size 256 --output-dir /shared/results/xfit-dragon \
  --warmup-rounds 1 --measure-rounds 2
```

With Open MPI, use the installed rank wrapper to narrow GPU visibility before
Python starts. The parent mask must list allocated GPUs in local-rank order:

```bash
: "${CUDA_VISIBLE_DEVICES:?must enumerate the allocated GPUs}"
mpirun -n 8 --map-by slot --bind-to none -x CUDA_VISIBLE_DEVICES \
  .venv/bin/cuphoton-openmpi-rank-exec -- \
  .venv/bin/cuphoton xfit fit-dipoles \
  --executor mpi --input /shared/dipoles.npz --model gaussian --backend cupy \
  --chunk-size 256 --output-dir /shared/results/xfit-mpi \
  --warmup-rounds 1 --measure-rounds 2
```

The output directory must be new. Its basename is the run ID: 1–128 ASCII
letters, digits, dots, underscores or hyphens, starting with a letter or digit.
Each pass retains normal xFit artifacts
under `rounds/<round-id>/scientific/`, including warmup passes. Without round
flags, a single pass writes them under `scientific/`. The execution summary
records placement, item receipts and round timing; scientific merging and
validation occur after the timed worker phase.

## Opt-in observational-data checks

The observational checks read caller-supplied FITS files and create every
injection in memory.

The public ZTF check uses an independently shifted empirical difference PSF
on quiet regions of a real subtraction image. Download the two products from
NASA/IPAC IRSA, retaining these local names:

```bash
mkdir -p /tmp/xfit-ztf
curl -fL -o /tmp/xfit-ztf/difference.fits.fz \
  'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/0411/467847/ztf_20180411467847_000535_zr_c11_o_q3_scimrefdiffimg.fits.fz'
curl -fL -o /tmp/xfit-ztf/difference-psf.fits \
  'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/0411/467847/ztf_20180411467847_000535_zr_c11_o_q3_diffimgpsf.fits'
CUPHOTON_XFIT_ZTF_DIR=/tmp/xfit-ztf make test-xfit-real
```

Set `CUPHOTON_XFIT_REAL_GPU=1` and run with the repository GPU extra to add
NumPy/CuPy parity. The test verifies the source SHA-256 values before fitting.

The same ZTF products drive an astrometric-offset recovery check. A
science/template registration offset `delta` turns one point source into the
exact dipole `flux * (PSF(x) - PSF(x - delta))`, so the fitted
`StampDipoleModel` lobe-separation vector estimates `delta` directly. The
check builds the shifted template lobe with the real `cuphoton.xrep`
Lanczos-3 resampling path on quiet real backgrounds and verifies both fit
regimes: with the offset resolved (`delta` at or above the PSF width), the
offset vector itself is recovered; in the unresolved regime the flux and
separation are degenerate along `flux * delta`, and only that dipole moment
and the offset direction are asserted. Deterministic synthetic variants of
the same protocol run unconditionally in `tests/xfit`.

Sub-pixel offset work should supply an oversampled stamp basis
(`StampDipoleModel(..., scale=1/oversample)`); a native-resolution basis
interpolates bilinearly and biases diagonal sub-pixel offsets by several
tenths of a pixel. Initialization matters near the degeneracy: the windowed
image first moment estimates `flux * delta`, so starting on the valley at a
fixed trial separation, and keeping the better chi-square of two such
starts, converges where peak-based starts stall.

An optional Rubin check accepts a Parquet candidate inventory through
`CUPHOTON_XFIT_RUBIN_METADATA`. Each row must identify a local difference
FITS path, pixel center, stamp size, and pipeline `candidate_isDipole` value.
The check reads `IMAGE`, `MASK`, and `VARIANCE` directly from FITS. That
pipeline flag supports dipole smoke tests. Real/bogus training requires
human-reviewed labels.
