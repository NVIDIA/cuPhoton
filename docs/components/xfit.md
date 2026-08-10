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
only where the mask excludes that pixel; residual entries for excluded
nonfinite image values are not meaningful.

Python inputs may be NumPy arrays, CuPy arrays, or array-like values accepted
by the resolved backend. `backend="auto"` can therefore transfer a host input
to CuPy when CUDA is usable. In split mode, an auxiliary `(batch, y, x)` array
applies one mask or variance plane to every channel of each candidate. If the
batch size is three, `(3, y, x)` keeps that per-candidate meaning; use the
explicit `(1, 3, y, x)` shape for per-plane values.

Split mode uses diagonal per-plane weights. When the difference plane is
derived from the positive and negative planes, those residuals are correlated;
the reported split-mode covariance is therefore not statistically calibrated
unless the caller's weighting model accounts for that dependence. The XScan
feature adapter accepts difference-mode xFit runs only.
`uncertainty_valid` establishes numerical and rank validity; it does not
override this split-plane calibration caveat.

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

These inputs and outputs are data-bearing, not privacy-sanitized. Input
archives contain candidate identifiers and exact image pixels; fit artifacts
contain identifiers, hashes, parameters, uncertainties, covariance, and
optional residuals. Do not publish them unless the underlying data and
metadata are cleared for release.

For the sampled-stamp model, choose `--stamp-evaluation bilinear`,
`bilinear-vignetted`, or `finite-volume` and provide `stamp_basis` in the input
archive. Solver controls include `--f-tol`, `--x-tol`, `--g-tol`,
`--max-evaluations`, and `--use-finite-difference`. Gaussian fits use their
analytic Jacobian unless finite differences are requested; sampled-stamp fits
always use finite differences and record that resolved choice in the effective
configuration.
`--x-tol` is an absolute step-norm tolerance; unlike a parameter-relative
criterion, it cannot declare convergence merely because a periodic parameter
has drifted to a large equivalent value.

`--compute-dtype input` preserves the input floating dtype. Select `float32`
or `float64` to run the solver at an explicit precision without changing the
input arrays or their recorded per-candidate hashes. Float64 is preferable for
ill-conditioned observational fits when the additional compute cost is
acceptable.

See [Data and artifact contracts](../data-artifacts.md#xfit-dipole-batches)
for the stable shapes and output fields.

## Opt-in observational-data checks

No observational fixture is committed. The external checks read caller-owned
FITS files and create every injection in memory.

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
pipeline flag is useful for a dipole smoke test, but is not a human-reviewed
real/bogus label and must not be treated as training truth.
