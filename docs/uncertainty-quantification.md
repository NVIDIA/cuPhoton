# Uncertainty quantification for transient catalogs

The [transient chain](../examples/transient-chain/README.md) reports an error
bar with every position and flux it measures. This guide explains the
following:

- Where each error bar comes from
- How to configure the error propagation
- How to read the catalog columns
- How to use the underlying library functions without the chain
- How to check the error bars on your own data

The error budget is experimental. Its column names, configuration keys and
terms can change between releases.

## Overview

The chain measures three quantities for each candidate, listed in the
following table.

| Quantity | Columns | What it locates or measures |
|---|---|---|
| Dipole center | `center_x`, `center_y` | the midpoint of the two lobes of a dipole |
| Source position | `source_x`, `source_y` | the lobe that holds the source, whichever its sign |
| PSF flux | `psf_flux` | forced point-spread-function photometry at the source |

Positions are in pixels relative to the candidate pixel (`x`, `y`). Fluxes are
in image units.

xFit's fit covariance gives the uncertainty of each quantity when the target
image's variance plane describes all the noise in the difference image. A
difference image breaks that assumption in several ways. The template brings
its own noise. Resampling and kernel convolution correlate neighboring pixels.
The subtraction kernel is itself fitted, and the alignment of the two images is
uncertain. The chain models each effect as a separate term, a standard
uncertainty in the quantity's units, and adds the terms in quadrature:

```text
<q>_err = sqrt(fit^2 + corr^2 + kfit^2 + kmatch^2 + mis^2)
```

`<q>_err` is the error bar to quote. For a position it is one standard
deviation per axis, and for a flux it is one standard deviation of the flux.
The chain also writes each term to its own column. You can rebuild the error
bar with any term left out.

The workflow has three steps:

1. Run the chain with the `uq` configuration block enabled, which is the
   default.
   Worker step 8c computes every term for every candidate and writes the terms
   and their sum to the catalog.
2. Read `<q>_err` for the quantity you need, and keep the rows where it is
   finite. Each item's `summary.json` records which terms went into the error
   bars and why any term was left out.
3. Check the error bars on your data. The repeat test compares the scatter of
   static sources across repeated exposures with their error bars.

## Quick start

From the root of the checkout, install cuPhoton with the `gpu` extra and the
example. The example's
[prerequisites](../examples/transient-chain/README.md#prerequisites) give the
details.

```bash
uv pip install -e '.[gpu]' -e examples/transient-chain
```

Run one detector-exposure on one GPU. `config-generic.yaml` describes a FITS
target with an inverse-variance weight file and a template on its own pixel
grid. Its template has no variance plane. The corr term is left out until you
add one, as described in [Use your own survey data](#use-your-own-survey-data).
The [selections file](../examples/transient-chain/README.md#the-selections-file)
lists the input files.

```bash
python examples/transient-chain/launch.py --executor local \
    --config examples/transient-chain/config-generic.yaml \
    --selections sel.json --output-dir out --only N4/123456
```

Without `--checkpoint-dir` the chain stops after xFit. The error bars do not
need the classifier. Read the PSF flux with its error bar and its terms:

```python
import numpy as np
import pandas as pd

catalog = pd.read_parquet("out/catalog.parquet")
terms = ["fit", "corr", "kfit", "kmatch", "mis"]
columns = ["psf_flux", "psf_flux_err"] + [f"psf_flux_err_{t}" for t in terms]
valid = np.isfinite(catalog["psf_flux_err"])
print(catalog.loc[valid, ["candidate_id", *columns]])
```

With several exposures of the same detector, run the repeat test on the
per-item catalogs:

```bash
python examples/transient-chain/repeat_catalogs.py out/items/*/catalog.parquet \
    --out repeat.json
```

## Sources of uncertainty

The following table lists each term of the error budget, its cause, and the
function that computes it.

| Term | Column | Cause | Quantities | Computed with |
|---|---|---|---|---|
| fit | `<q>_err_fit` | pixel noise of the target image | all | `cuphoton.xfit` fit covariance and forced photometry |
| corr | `<q>_err_corr` | template noise, correlated between pixels by resampling and the kernel | all | `cuphoton.xpois.correlation_factor`, `cuphoton.xrep.resampling_weights_at` |
| kfit | `<q>_err_kfit` | uncertainty of the fitted kernel and background | all | `cuphoton.xpois.kernel_fit_covariance`, `kernel_fit_variance` |
| kmatch | `<q>_err_kmatch` | template flux the kernel did not match | PSF flux | the chain, at the stars of the refine stage |
| mis | `<q>_err_mis` | uncertainty of the template-to-target alignment | positions | the chain, from the refine stage's misalignment model |
| sky | `sky_err_arcsec` | absolute astrometry of the exposure | reported apart | the `uq.sky_astrometry_arcsec` input |

A term that does not apply to a quantity is 0 in that quantity's column.

### Fit

xFit weights each pixel by the target variance plane and propagates its
parameter covariance to the derived quantities, cross terms included. For the
dipole center and source position, refer to
[Derived quantities and their uncertainties](components/xfit.md#derived-quantities-and-their-uncertainties).
For the flux, refer to [Forced PSF photometry](components/xfit.md#forced-psf-photometry).
The flux error is conditional on the position and the PSF width. The chain
therefore takes the width from `xfit.psf_sigma_px` or from the image's stars,
never from the candidate fits.

A difference stamp can hold one source, such as a brightening or a fading
source. The data then do not support one of the dipole's lobes. Step 8b refits it with the one-lobe
model of [Single sources](components/xfit.md#single-sources) and marks the row
`fit_model = "single"`. The dipole fit places its spare lobe on noise,
which inflates the position covariance.

Without a target variance plane, xFit scales the covariance by the fit's
reduced chi-square. The terms that need the plane are left out.

### Correlation

The template's noise is not in the target variance plane. Resampling and
convolution also correlate the template's pixels. Both effects make a weighted
sum of difference pixels, such as a PSF flux, vary more than the diagonal
variance predicts. The ratio of the two variances is the correlation factor
`f`. The term is the following:

```text
<q>_err_corr = <q>_err_fit * sqrt(max(f, 1) - 1)
```

The chain computes `f` in closed form at each candidate from the template's
variance plane on the target grid, the subtraction kernel, and the resampling
weights of the template build. For details, refer to
[Correlation factor of weighted sums](components/xpois.md#correlation-factor-of-weighted-sums)
and to the resampling weights in the xRep [Python API](components/xrep.md#python-api).

The closed form models resampling of independent template pixels through the
kernel. A coadd's own pixel correlations escape it. The chain therefore also
measures the factor once per exposure at up to `uq.correlation_positions`
empty positions with `cuphoton.xpois.empirical_correlation_factor`. The
exposure adopts the measured factor where it exceeds 1 and exceeds the median
closed form at the same positions by more than two standard errors. A
candidate then uses it where it also exceeds the candidate's own closed form.
Such rows carry `uq_corr_measured`.

### Kernel fit

xPois fits the subtraction kernel and background. Their uncertainty adds
variance to every difference pixel. The added variance is largest where a
candidate sits on a bright template source.
`cuphoton.xpois.kernel_fit_covariance` computes the covariance of the fit
coefficients once per item. `kernel_fit_variance` propagates it to each
candidate's weighted sum. For details, refer to
[Kernel-fit covariance and the variance it adds](components/xpois.md#kernel-fit-covariance-and-the-variance-it-adds).

### Kernel mismatch

A kernel that does not match the template's PSF everywhere leaves part of a
static star's flux in the difference image. At the stars that the refine
stage matched, the chain fits `eta`, the fraction of template flux left beyond
the noise. The flux term is `eta` times the kernel-matched template flux at the
candidate. Positions get 0 for this term.

### Misalignment

The refine stage fits an affine map between the template and the target from
matched stars. Its misalignment model gives the remaining position variance
across the detector. Each quantity gets a share of that variance:

- The dipole center gets a quarter of it. The center is the mean of two lobes,
  and each lobe moves with its own image.
- The lobe of a resolved dipole gets the variance unscaled.
- The lobe of a single source gets the variance times `lambda^2`, with
  `lambda = (D + T) / D`. `D` is the lobe's forced flux in the difference image,
  and `T` is the kernel-matched template's forced flux at the same position.
  A faint change on a bright source moves with that source.

Without a PSF width, `lambda` is 1. The chain caps `|lambda|` at 10, and
`uq.sensitivity_capped` counts the capped rows.

### Sky astrometry

The absolute astrometry of the exposure is an input, not a measurement. Set
`uq.sky_astrometry_arcsec` and `uq.sky_astrometry_source` together. The chain
copies them to `sky_err_arcsec` and `sky_err_source` and never adds them to
the pixel-frame error bars.

### Masked pixels

A masked pixel near a candidate does not cost it the error bar. The weighted
sums and the correlation factor run over the usable pixels of the window, and
the chain flags the row `uq_window_masked`. A quantity's `<q>_err` is NaN in
two cases. The first is when fewer than half of its window's pixels are
usable. The second is when its correlation factor is undefined, for example on
a grid that rotates against the template's grid.
`uq_valid` is true where every quantity of the row has a defined error bar.

## Catalog columns

The error budget adds the columns in the following table to every item's
catalog. For the full column table, refer to the example's
[catalog columns](../examples/transient-chain/README.md#catalog-columns).

| Column | Meaning |
|---|---|
| `<q>_err` | the error bar to quote, NaN where the quantity's window is not valid |
| `<q>_err_fit` | xFit's standard uncertainty alone |
| `<q>_err_corr`, `<q>_err_kfit`, `<q>_err_kmatch`, `<q>_err_mis` | the other terms, 0 where a term does not apply |
| `center_xy_cov`, `positive_xy_cov` | x-y covariance from xFit's fit covariance |
| `source_x`, `source_y` with `_err_fit` and `_err` | the measured lobe whichever its sign |
| `fit_model` | `"dipole"`, or `"single"` for a one-lobe refit |
| `uq_valid`, `uq_window_masked`, `uq_corr_measured` | validity and provenance flags of the row |
| `sky_err_arcsec`, `sky_err_source` | the external astrometric uncertainty and its source |

`q` is one of `center_x`, `center_y`, `positive_x`, `positive_y` and
`psf_flux`. `source_*_err_fit` and `source_*_err` copy the `positive_*`
values. There are no `source_*` term columns. A fading row without a
one-lobe refit takes the negative lobe with NaN error bars.

To see how much a term contributes, rebuild an error bar with that term left
out. `q` is one of the five quantities above. The rebuilt value does not
reapply the validity rule. Use it only on rows where `<q>_err` is finite.

```python
def error_without(catalog, q, term):
    kept = [t for t in terms if t != term]
    return np.sqrt(sum(catalog[f"{q}_err_{t}"] ** 2 for t in kept))
```

## Configuration

The `uq` block of the chain configuration controls the error budget. The
following block shows the defaults.

```yaml
uq:
  enabled: true               # false: <q>_err equals <q>_err_fit and every other term is 0
  kernel_fit: true            # compute the kfit term
  max_lobe_reach_px: 32       # largest center-to-lobe distance with a center error bar
  measure_correlation: true   # measure the correlation factor at empty positions
  correlation_positions: 1000 # empty positions for that measurement, at least 20
  sky_astrometry_arcsec: null # external astrometric uncertainty per axis
  sky_astrometry_source: null # where that value comes from
  single_source_refit: true   # refit single sources with the one-lobe model
```

Each term also needs inputs from the survey and from earlier stages, listed
in the following table. A term without its inputs is left out, and
`summary.json` lists it under `uq.excluded` with the reason.

| Term | Needs |
|---|---|
| fit | nothing beyond the fit. Without a target variance plane xFit scales the covariance by the reduced chi-square |
| corr | target and template variance planes and a PSF width |
| kfit | a target variance plane, a PSF width and `uq.kernel_fit: true` |
| kmatch | a target variance plane, a PSF width and at least 5 stars matched by the refine stage |
| mis | a misalignment model, which needs `align.refine: true`, an affine refinement, a star finder other than `daofind`, and a template reprojected or mosaicked onto the target grid |

The refine stage runs only on a template that the chain reprojects or
mosaics onto the target grid (`on_target_grid: false`, or Rubin
`template_coadds`). A template already on the target grid gives no stars.
The chain then leaves out kmatch and mis. The PSF width comes from
`xfit.psf_sigma_px`, or from the refine stage's stars when that key is null.
Without a width, the `psf_flux` columns are NaN except on one-lobe rows.

## Use your own survey data

The terms that use the template's noise need its variance plane. With the
`generic` adapter, name the variance file or HDU of the target and of the
template.

```yaml
survey:
  name: generic
  params:
    target:
      path_key: image
      variance_path_key: weight
      variance_kind: inverse_variance   # variance | inverse_variance | none
    template:
      path_key: template
      variance_path_key: template_variance
      variance_kind: variance
      on_target_grid: false             # reproject the template through its WCS
      pieces: correlated                # correlated | independent
```

Each selection entry then names the template variance file under the
`template_variance` key. `pieces` sets how the variances of overlapping
template pieces combine. Use `correlated` for patches cut from one coadd, and
`independent` for separate reference frames.

The `rubin` adapter reads the variance HDU of the visit. With
`coadd_variance_hdu` set, it also reads the variance HDU of each coadd patch. A
new adapter provides the template variance through its template provider. For
details, refer to [Other surveys](../examples/transient-chain/README.md#other-surveys).

## Read the summary

Each item's `summary.json` records how its error bars were built:

```python
import json

summary = json.load(open("out/items/N4-v123456/summary.json"))
summary["uq_terms"]                 # terms that went into <q>_err
summary["uq"]["excluded"]           # term -> reason it was left out
summary["uq"]["correlation"]        # closed-form and measured factors, and which was used
summary["uq"]["kernel_mismatch"]    # eta and the number of stars it was fitted on
```

`uq` is null for an item with no candidates. With `uq.enabled: false`, it has
no correlation, kernel-mismatch, or window entries. Otherwise, it also records
the following counts:

- Rows with an invalid window (`invalid_window`)
- Rows with a masked window (`masked_window`)
- Rows with an undefined correlation factor (`undefined_factor`)
- Rows with a capped sensitivity (`sensitivity_capped`)
- Single-source refits (`single_candidates`, `single_fitted`, `single_model`)

## Use the components on their own

The library functions behind the terms work without the chain. The following
example builds the error bar of one PSF flux from a template, a target, and
their variance planes. All four are NumPy arrays of shape `(n, n)` on the CPU.
The template maps to the target grid by a shift `(dx, dy)`. `positions` holds
the `(x, y)` target pixel of each source, with shape `(N, 2)`.

First, resample the template onto the target grid with xRep and read the
resampling weights at the sources:

```python
import numpy as np

from cuphoton.xrep import (
    BBox,
    ReprojectionSpec,
    prepare_reprojection,
    reproject_masked_array,
    resampling_weights_at,
)

spec = ReprojectionSpec(
    mapping=lambda xy: xy + np.array([dx, dy]),  # target (x, y) -> template pixels
    output_bbox=BBox(0, 0, n, n),
    interpolation="lanczos3",
)
prepared = prepare_reprojection(spec, source_shape=template.shape)
resampled = reproject_masked_array(
    template, template_variance, template_mask, spec,
    backend="cpu", prepared=prepared,
)
taps = resampling_weights_at(
    spec, positions[:, 0], positions[:, 1],
    prepared=prepared, source_shape=template.shape,
)
```

Second, fit the subtraction kernel with xPois and measure the flux with xFit.
This gives the fit term:

```python
from cuphoton.xfit import forced_psf_photometry
from cuphoton.xpois import GaussianBasisComponent, solve_constant_kernel, window_stamps

fit = solve_constant_kernel(
    resampled.image,
    target,
    [GaussianBasisComponent(1.0, 0), GaussianBasisComponent(2.0, 0)],
    kernel_shape=(9, 9),
    variance=target_variance,
    backend="cpu",
)
size = 21
stamps = window_stamps(fit.residual, positions, size)
stamp_variance = window_stamps(target_variance, positions, size)
offset = positions - np.rint(positions)
photometry = forced_psf_photometry(stamps, stamp_variance, offset, psf_sigma)
u_fit = photometry.flux_error
```

Third, write the flux estimator as weights on the difference pixels and carry
the template noise and the kernel-fit uncertainty to it:

```python
from cuphoton.xpois import (
    correlation_factor,
    kernel_fit_covariance,
    kernel_fit_variance,
    psf_weights,
)

psf = psf_weights(psf_sigma, size, offset=offset)
weights = psf / stamp_variance
weights /= (weights * psf).sum(axis=(1, 2), keepdims=True)

factor = correlation_factor(
    weights,
    kernel=fit.kernel,
    target_variance=target_variance,
    reference_variance=resampled.variance,
    resampling_weights=taps.weights,
    positions=positions,
    residual_variance=target_variance,
    is_translation=taps.is_translation,
).factor
u_corr = u_fit * np.sqrt(np.maximum(factor, 1.0) - 1.0)

covariance = kernel_fit_covariance(
    fit,
    reference=resampled.image,
    target_variance=target_variance,
    reference_variance=resampled.variance,
    resampling_weights=taps.weights[0],
)
u_kfit = np.sqrt(
    kernel_fit_variance(
        covariance, reference=resampled.image, positions=positions, weights=weights
    )
)

flux_error = np.sqrt(u_fit**2 + u_corr**2 + u_kfit**2)
```

`residual_variance=target_variance` makes the factor relative to the variance
that xFit used. The factor then includes the template noise as well as its
correlations. For a position, replace the PSF weights with
`psf_weights(..., derivative="x")` or `"y"`, normalized the same way.
`kernel_fit_covariance` uses one resampling stamp for every fit pixel. This
approximation holds where the mapping is close to a translation.

You can also use each component alone:

- In xFit, `derived_dipole_quantities`, `forced_psf_photometry`, and
  `fit_single_sources` give the fit term for stamps that you cut yourself.
- In xPois, `correlation_factor` and `kernel_fit_variance` apply to any
  weighted sum of a difference image. `empirical_correlation_factor` tells
  you whether a survey's variance plane already includes the correlation.
- In xRep, `reproject_masked_array` propagates a diagonal variance plane.
  `resampling_weights_at` returns the weights that a noise model needs.

## Check the error bars

### Repeat test

A static source lands at the same sky position in every exposure of a
detector. Its scatter divided by its error bars gives a chi-square per degree
of freedom near 1 when the error bars are correct. Survey pipelines call this
repeatability. `repeat_catalogs.py` links each detector's detections across
its exposures. It then scores three positions, each on the classes of
candidates whose columns describe it:

- The dipole center (`center_*`)
- The positive lobe (`positive_*`)
- The measured lobe of either sign (`source_*`)

Run it on the per-item catalogs:

```bash
python examples/transient-chain/repeat_catalogs.py out/items/*/catalog.parquet \
    --out repeat.json --link-px 2 --duplicate-px 1
```

For each quantity, the report gives two statistics. The first is the pooled
chi-square per degree of freedom. The second is the median per-group ratio,
which is 1 for correct error bars at any group size. The report computes both
statistics for six arms. An arm is one version of the error bar:
`<q>_err`, `<q>_err_fit` alone, or `<q>_err` with one of the corr, kfit,
kmatch, or mis terms removed. Read the arms together:

- A statistic well above 1 means the error bars are too small. The term whose
  removal raises the statistic most contributes most to the error bar.
- A statistic below 1 means the error bars are too large, or that a term
  describes noise shared by every exposure.
- The template is the same in every exposure. Its noise therefore never
  scatters between them. An arm that includes the template noise can score below 1 by
  that share even with correct error bars.
- A bias common to all exposures, such as a template position error, moves
  every detection alike and passes the test unseen.

For how positions reach the sky and how the script links and classifies
detections, refer to
[Repeat test across repeated exposures](../examples/transient-chain/README.md#repeat-test-across-repeated-exposures).

### Reference catalog

`compare_catalogs.py` matches the chain's candidates to a reference catalog
within 1.5 pixels and reports the differences.

```bash
python examples/transient-chain/compare_catalogs.py --ours out/catalog.parquet \
    --reference reference.csv --out report.json
```

The report gives position and parameter differences, not normalized
residuals. To test the error bars against a reference that carries its own
errors, divide each difference by the quadrature sum of both catalogs'
errors.

### Coverage on injected sources

If you inject sources of known position and flux into your images, you can
measure coverage directly. Coverage is the fraction of injected sources whose
true value lies within `1.96 * <q>_err` of the measured value. It is near 0.95
for correct error bars, and the spread of `(measured - true) / <q>_err` is
near 1. Hold out some exposures from any tuning, and score coverage on those
exposures.

## Limits

- The error bars are local and first order. They hold while the uncertainties
  are small compared with the PSF width and the lobe separation. For
  overlapping lobes, the fit measures only the center and the product of
  amplitude and separation.
- The fit term and the closed-form correlation factor assume the supplied
  variance planes describe the pixel noise.
- The closed-form factor needs a grid mapping that is locally a translation.
  On a rotated grid, the factor is undefined and the quantity's `<q>_err` is
  NaN.
- The kernel mismatch term covers the PSF flux only. A candidate on template
  light that the kernel did not match can have a position scatter that its
  error bars do not include.
- The sky astrometry is an input. The chain does not measure it.
- The repeat test cannot see noise or bias shared by every exposure of a
  detector.

## Related

- [Transient chain](../examples/transient-chain/README.md), including
  [How the error bar is built](../examples/transient-chain/README.md#how-the-error-bar-is-built)
  for the full definition of every term.
- [xFit](components/xfit.md), [xPois](components/xpois.md) and
  [xRep](components/xrep.md) for the library functions.
- [Glossary](glossary.md) for the terms used here.
