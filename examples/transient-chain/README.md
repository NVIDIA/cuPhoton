# A survey's images on the GPU, end to end, under one persistent worker per GPU

One worker per GPU takes a target image from FITS bytes to a scored candidate catalog without
writing an intermediate file. A survey adapter reads the target and template planes to device
memory through the FITS reader you choose (xDR GPU decode, Astropy, or automatic); xRep, XPOIS,
the detector, xFit and XScan then work on device arrays. Headers, control data, diagnostics
and the final catalog use host memory as detailed in the per-stage table below. cuPhoton's
executors place the workers locally, under Dragon, or under MPI.

The example uses the cuPhoton development APIs in this checkout. Install the library and
example together from the same checkout. The shipped `config.yaml` holds the parameters
of the file-based reference chain on Rubin DP1; with a different `survey:` block the same
chain runs on other FITS surveys.

```
launch.py                   CLI: --executor local|dragon|mpi, --set block.key=value overrides
config.yaml                 every stage's parameters, validated at load; io.reader and survey.name live here
transient_chain/config.py      the config schema: defaults, required keys, value checks, --set overrides
transient_chain/fitsio.py      PlaneReader: auto | xdr | astropy on top of cuphoton.core.fits_io, with read receipts
transient_chain/surveys/       SurveyAdapter protocol; rubin.py (DP1 layout), generic.py (any FITS pair by parameters)
transient_chain/worker.py      the persistent worker: read -> template -> refine -> mask -> XPOIS -> detect -> stamps -> xFit -> XScan
transient_chain/ops.py         stage operations on NumPy or CuPy arrays (star finder, affine, hotspots, fit mask, stamps)
transient_chain/uq.py          the error budget per candidate: the terms behind <q>_err (step 8c)
transient_chain/catalog.py     the catalog column contract (same columns and dtypes for every item)
transient_chain/workload.py    selections -> WorkItems -> WorkloadSpec; run_local, run_distributed
compare_catalogs.py         match two catalogs by position and report differences
repeat_catalogs.py          link one detector's detections across repeated exposures and test the error bars
tests/                      host tests: ops against cuPhoton's reference code, config, adapters on synthetic FITS, catalog schema
```

## Choosing the processing path

Three choices are yours; the defaults pick the best path the environment supports.

| Choice | Where | Options | Default |
|---|---|---|---|
| FITS reader | `io.reader` | `auto`: xDR GPU decode when the file is a lossless GZIP tile-compressed or raw image and the GPU dependencies are present, Astropy otherwise; `xdr`: insist on the GPU decoder, fail before reading pixels when a file cannot take it (RICE, scaled, quantized, BLANK); `astropy`: host decode, then copy to the device | `auto` |
| Executor | `--executor` | `local`: this process, one GPU; `dragon`: one persistent worker per GPU, placed and audited by Dragon; `mpi`: one worker per rank under `mpirun` | required |
| Star finder | `align.star_finder` | `gpu`: device peaks with windowed centroids; `daofind`: photutils DAOStarFinder on host copies (the reference chain's path) | `gpu` |

Every item's `summary.json` records which reader each file took (`diagnostics.read.receipts`:
reader, fallback reason, HDUs, compression, bytes), so a run's provenance states the path, not
the request.

## What stays on the GPU and what does not

| Step | Device arrays | Host traffic |
|---|---|---|
| Read | target image, mask, variance and every template plane; GZIP tiles inflated by nvCOMP under xDR | FITS headers (WCS, mask-plane cards); under `astropy` the decoded planes cross once |
| Template | Lanczos-3 reprojection of each source onto the target grid, mean over sources, bad bits OR-ed | the WCS mapping on a coarse grid (Astropy is CPU-only; xRep expands it on the device) |
| Refine | each image median-subtracted and smoothed once; star finding on both images and the rebuilt template; per-star centroid errors and widths | two star lists of at most `align.max_stars` rows and two low-threshold peak lists (counted in `control_bytes`); affine least squares, misalignment model, PSF width |
| Mask | bad pixels to NaN; compact-source fit mask | ranking of a few hundred peaks |
| XPOIS | `solve_constant_kernel_device`, residual | chi2, dof, condition number |
| Detect | hotspot rule (threshold, 4-connected label, per-component peak and centroid) | the hotspot table |
| Stamps | search, template = search minus difference, difference; per-stamp noise-match factor | none |
| xFit | `fit_dipoles_device` on float64 difference stamps, weighted by the target variance, with masked and nonfinite pixels excluded | none |
| Budget | the kernel-fit covariance once per item, then per candidate the correlation windows, forced fluxes and kernel-fit terms | candidate positions for the misalignment model, kept-star fluxes and variances for the kernel-mismatch solve, and validity flags |
| XScan | DLPack handoff of the scaled stamps to Torch, `predict_tensors` | none |
| End | one packed download: positions, fit parameters, uncertainties, logits, probabilities | `catalog.parquet` and `summary.json` per item |

`summary.json` splits host traffic into `transfers.terminal_bytes` (the catalog),
`control_bytes` (star lists, hotspot table, candidate filter, and budget control
payloads downloaded to the host) and `diagnostic_bytes`
(cross-correlation windows), and records `intermediate_files: 0`.

After the affine fit, the refine stage measures the stars matched on the rebuilt template
(`ops.measure_matched_stars`). A star with another peak above 5 sigma within one box width
(`align.star_box_px` pixels) in either image is dropped: a neighbor moves the two centroids by
different amounts and inflates a chip-wide rms. For each remaining pair it measures the standard
error of both centroids, propagated from the pixel variance through the windowed centroid, and
the target star's second-moment width; a pair whose centroid error is not finite is dropped and
counted (`stars_nonfinite_dropped`).

The misalignment model (`ops.fit_misalignment`) is fitted only in affine mode and only with the
`gpu` star finder, whose centroids the error model describes (`daofind` centroids are skipped with
`reason: "daofind centroids"`). It is the affine fit with its covariance from the centroid noise,
the excess residual as a quadratic field across the image (one constant per axis with fewer than
30 stars) and outliers clipped at 5 sigma. `variance(x, y)` includes the squared remaining affine and
quadratic displacement because this diagnostic fit does not correct the template or catalog
positions. The worker keeps the model for the later stages. The summary under
`diagnostics.align.refine.misalignment` records `model` (`"polynomial"`, `"rms"`, or `null` with
a `reason`), the star counts, the residual rms and the excess and image-center sigma in pixels.
The template has no variance plane, so its centroid noise is counted as excess.

When at least one kept target star has a finite width, `ops.psf_width` computes the clipped
median and its standard error, written under the same diagnostics key as `psf_width`. When
`xfit.psf_sigma_px` is null that width is the one the uncertainty stage uses, and `summary.json`
records it as `psf_sigma_px` with `psf_sigma_source: "stars"` (`"config"` when the configuration
set it, `null` when neither was available and the `psf_flux` columns are NaN).

## Prerequisites

1. cuPhoton with the `gpu` extra (CuPy, xDR dependencies, Torch, photutils) plus `dragon`
   or `mpi` for those executors; Python 3.12 or 3.13 (Dragon has no 3.14 wheel). Follow
   `docs/getting-started.md` for development environment setup and
   `docs/components/xdr.md` for the xDR native extension. From the root of this development
   checkout, install both packages in editable mode:
   ```bash
   uv pip install -e '.[gpu]' -e examples/transient-chain
   ```
   Use `'.[gpu,dragon]'` or `'.[gpu,mpi]'` for the corresponding executor.
   The example itself needs only `pyyaml`, `pandas`, `pyarrow`, `astropy` and `scipy`, all of which
   cuPhoton already requires.
   The host tests run without a GPU, from this directory, with the `dev` and `photometry`
   extras installed:
   ```bash
   CUDA_VISIBLE_DEVICES= pytest tests
   ```
2. A selections file (next section) and the survey's FITS files.
3. Optional: an image-only triplet XScan checkpoint with `image_size` equal to `stamps.stamp_size`
   (51 in the shipped config). Checkpoints with xFit fusion features are rejected: this worker
   scores stamps only. Without `--checkpoint-dir` the chain stops after xFit and the classifier
   columns are NaN or false.

## The selections file

`{"selections": [ ... ]}` with one object per target image (a bench `{"shards": [...]}` is
accepted too). The keys depend on `survey.name`.

For `rubin` (the default): `visit`, `detector`, `band`, `visit_image`, and either
`template_coadds` (a list of coadd patch files, mosaicked onto the visit grid) or `template_fits`
(a template already on the visit grid). `date`, `tract`, `patch` and `physical_filter` are
copied into the catalog when present.

For `generic`: the keys named in `survey.params` (see "Other surveys").

One entry of each form:

```json
{
  "selections": [
    {
      "visit": 2024121000367,
      "detector": "R22_S00",
      "band": "r",
      "physical_filter": "r_03",
      "date": 20241210,
      "tract": 7850,
      "patch": 17,
      "visit_image": "/data/dp1/visit_image/2024121000367/visit_image_R22_S00.fits",
      "template_coadds": [
        "/data/dp1/template_coadd/7850/17/r/template_coadd.fits",
        "/data/dp1/template_coadd/7850/18/r/template_coadd.fits"
      ]
    },
    {
      "expnum": 123456,
      "ccdnum": "N4",
      "band": "g",
      "image": "/data/decam/c4d_123456_N4_ooi_g.fits",
      "weight": "/data/decam/c4d_123456_N4_oow_g.fits",
      "mask": "/data/decam/c4d_123456_N4_ood_g.fits",
      "template": "/data/decam/templates/field_42_N4_g.fits"
    }
  ]
}
```

The second entry is for `survey.name: generic` with identity keys `expnum`, `ccdnum`, `band`,
`target.path_key: image` and `template.path_key: template`.

Every entry is validated before any GPU work: a missing key or a missing file is reported with
the entry index and the first missing path. `--only CCD` or `--only CCD/EXPOSURE` (repeatable)
selects entries by identity (`R22_S00`, `R22_S00/2024121000367`).

## Run

Local, one GPU, no launcher (smoke test or parity run):

```bash
python examples/transient-chain/launch.py --executor local --selections sel.json \
    --checkpoint-dir /path/to/run --output-dir out --only R22_S00/2024121000367
```

Dragon, one node, one worker per GPU. The script must be started by the `dragon` launcher
(`-s` single node, `-t tcp -o tcp` the transport; see the library's `docs/dragon.md`):

```bash
dragon -s -t tcp -o tcp examples/transient-chain/launch.py --executor dragon \
    --selections sel.json --checkpoint-dir /path/to/run --output-dir runs --name probe --max-workers 4
```

MPI, one prebound GPU per rank (Open MPI; `cuphoton-openmpi-rank-exec` is installed with the
library; `docs/distributed-execution.md` has the Slurm form). Under a one-task Slurm allocation
you may need `--allow-run-as-root --oversubscribe`, and preloading the venv's `libnccl.so.2`
keeps Torch from binding to an older NCCL the MPI stack loaded first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -n 4 --map-by slot --bind-to none -x CUDA_VISIBLE_DEVICES \
    cuphoton-openmpi-rank-exec -- python examples/transient-chain/launch.py --executor mpi \
    --selections sel.json --checkpoint-dir /path/to/run --output-dir runs --name probe-mpi
```

Override any config value without editing the file: `--set io.reader=astropy`,
`--set align.star_finder=daofind`, `--set xfit.solver.max_evaluations=400`. Add
`--warmup-rounds 1 --measure-rounds 2` (dragon and mpi) for the executor's benchmark layout
(outputs under `rounds/`). Flags that do not apply to the chosen executor are rejected.

Output per item: `items/<CCD>-v<EXPOSURE>/{catalog.parquet,summary.json}`; per run or round:
`catalog.parquet` (all items), the executor's `summary.json`, and under `local` the executor's
per-item `records/`. All three executors print the same block on exit: items, failed item ids,
candidate count, catalog path.

## Configuration

`config.yaml` is validated at load: unknown blocks or keys are rejected, required keys are
named, odd sizes and enumerations are checked, and every default lives in
`transient_chain/config.py`, not in the worker. The blocks:

| Block | What it controls | Keys worth knowing |
|---|---|---|
| `io` | FITS reader | `reader: auto\|xdr\|astropy` |
| `survey` | which adapter reads the files | `name: rubin\|generic`, `params` (adapter-specific; see below) |
| `align` | template refinement | `refine`, `star_finder`, `star_fwhm_px`, `star_threshold_sigma`, `max_stars`, `star_box_px`, `centroid_iterations`, `match_radius_px`, `verify_radius_px`, `min_affine_matches`, `min_shift_matches`, `mapping_grid_step`, `xcorr_window_px` |
| `subtract` | XPOIS and the fit mask | `kernel_height`, `kernel_width`, `basis_sigmas`, `basis_degrees`, `background_degree`, `flux_conserve`, `template_bad_dilation_px`, `auto_stamp_*` (mask selection), `auto_background_filter_size`, `auto_peak_filter_size`, `auto_compact_core_size` |
| `detect` | hotspot rule | `threshold_sigma`, `min_pixels` |
| `stamps` | stamp cutting | `stamp_size` (odd; equals the checkpoint's `image_size`) |
| `xfit` | dipole fits, quality flags, forced photometry | `mode`, `compute_dtype`, `solver` (LMConfig: `f_tol`, `x_tol`, `g_tol`, `max_evaluations`, `initial_damping`, `damping_increase`, `damping_decrease`), `psf_sigma_px` |
| `uq` | the error budget behind `<q>_err` | `enabled`, `kernel_fit` (the kernel fit's own term, one covariance per item), `max_lobe_reach_px` (largest center-to-lobe distance with a center error bar; default 32), `sky_astrometry_arcsec` and `sky_astrometry_source` (set together; written to `sky_err_arcsec` and `sky_err_source`) |
| `classify` | XScan | `inference_policy: strict\|checkpoint`, `apply_noise_match`, `noise_reference_sigma`, `decision_threshold`, `batch_size` |

`classify.noise_reference_sigma` is a property of the checkpoint, not of the survey: it is the
median robust sigma of the search stamps the classifier was trained on; the value in
`config.yaml` belongs to the checkpoint the chain was developed with. Change it when you change
checkpoints, or set
`apply_noise_match: false` to score unscaled stamps.

## Other surveys

The chain downstream of the read is survey-free. What a survey contributes is an adapter: how to
turn one selection entry into target planes (image, optional bad mask, optional variance, WCS),
a template provider, and an identity (`exposure_id`, `ccd_id`, `band`, extras). Two adapters ship.

`rubin` reads IMAGE, MASK and VARIANCE from HDUs 1, 2 and 3 of the visit file, IMAGE and MASK
from HDUs 1 and 2 of each coadd patch, resolves bad pixels with the LSST stack's `MP_<NAME>`
cards under the `hsc-masklite` policy, and mosaics the patches through their WCS. Each of these
is a parameter (`image_hdu`, `mask_hdu`, `variance_hdu`, `coadd_image_hdu`, `coadd_mask_hdu`,
`mask_policy`, `variance_kind`, `extra_keys`).

`generic` describes any FITS pair by parameters. A DECam-style layout with the image in the
primary HDU, a separate inverse-variance weight file, a separate mask file, and a template on
another pixel grid:

```yaml
survey:
  name: generic
  params:
    identity: {exposure_id: expnum, ccd_id: ccdnum, band: band}   # selection keys
    extra_keys: [field]                                           # copied into the catalog
    target:
      path_key: image
      image_hdu: 0              # index, EXTNAME, or null for the first 2D image
      mask_path_key: mask       # null = same file as the image
      mask_hdu: 0               # null = this survey has no mask plane
      variance_path_key: weight
      variance_hdu: 0           # null = no variance (XPOIS then fits unweighted)
      variance_kind: inverse_variance   # variance | inverse_variance | none
    template:
      path_key: template
      image_hdu: 0
      mask_hdu: null
      on_target_grid: false     # false: reproject the reference through its WCS onto the target grid
    mask_policy: strict         # none | strict (any nonzero bit) | hsc-masklite | {planes: [BAD, SAT]} | {bits: 5}
```

A plain image pair with no mask and no variance needs only `identity` and the two path keys;
the defaults do the rest. Candidate ids are `cu-<exposure_id>-<ccd_id>-<index>`, and the
catalog has the columns `exposure_id`, `ccd_id`, `band` plus the adapter's extras (for Rubin: `visit`,
`detector`, `date`, `tract`, `patch`, `physical_filter`).

Adding a third survey is one file in `transient_chain/surveys/` implementing `required_keys`,
`identity`, `input_paths`, `payload`, `load_target` and `template`, registered in
`surveys/__init__.py`. A template provider that is not a mosaic returns `refinable = False` and
the refine stage skips itself.

## Catalog columns

Every item writes the same columns with the same dtypes, whether it found zero candidates or a
thousand and whether a checkpoint was given, so per-item catalogs merge without type promotion.

| column | dtype | stage | meaning |
|---|---|---|---|
| `candidate_id` | object | detect | `cu-<exposure_id>-<ccd_id>-<index>` |
| `x`, `y` | int64 | detect | zero-based pixel of the rounded hotspot centroid |
| `peak_abs_sigma` | float64 | detect | peak abs(residual) / robust sigma within the hotspot |
| `pixel_count` | int64 | detect | pixels of the hotspot above threshold |
| `peak_value` | float64 | detect | residual at the peak pixel (image units) |
| `sign` | int64 | detect | +1 brightening, -1 fading |
| `noise_match_factor` | float32 | stamps | scale applied to the stamp triplet before scoring (1 when disabled) |
| `amplitude`, `sigma_x`, `sigma_y`, `theta`, `x_pos`, `y_pos`, `x_neg`, `y_neg` and `*_err` | float64 | xfit | Gaussian dipole parameters and standard errors |
| `converged`, `uncertainty_valid` | bool | xfit | solver status |
| `chi_square`, `reduced_chi_square` | float64 | xfit | fit quality on the difference stamp |
| `evaluations`, `valid_pixel_count` | int64 | xfit | solver bookkeeping |
| `quality_flags` | int64 | xfit | bits of `cuphoton.xfit.DipoleQualityFlag`; 0 when none is set |
| `center_x`, `center_y`, `positive_x`, `positive_y` and `*_err_fit`, `*_err`, `center_xy_cov`, `positive_xy_cov` | float64 | xfit, uq | lobe midpoint and positive-flux lobe in pixels relative to (`x`, `y`). `<q>_err_fit`: standard uncertainty from xFit's covariance alone. `<q>_err`: the error bar to quote, every term in quadrature, NaN where the quantity's window is not valid (see below). `*_xy_cov`: x-y covariance from xFit's covariance |
| `separation`, `lobe_flux` and `*_err` | float64 | xfit | distance between the lobes and each lobe's total flux, with standard uncertainties |
| `psf_flux`, `psf_flux_err_fit`, `psf_flux_err` | float64 | xfit, uq | forced PSF photometry at the source. `psf_flux_err_fit`: conditional uncertainty from xFit alone. `psf_flux_err`: the error bar to quote, NaN where its window is not valid. All three NaN without a variance plane or a PSF width from `xfit.psf_sigma_px` or the refine stage's stars |
| `<q>_err_corr`, `<q>_err_kfit`, `<q>_err_kmatch`, `<q>_err_mis` for `q` in `center_x`, `center_y`, `positive_x`, `positive_y`, `psf_flux` | float64 | uq | the terms of `<q>_err` beyond `<q>_err_fit`, each a standard uncertainty in the quantity's units (next section); 0 where a term does not apply or could not be computed for the item; `<q>_err_corr` NaN where the factor is undefined |
| `uq_valid` | bool | uq | every `<q>_err` of the row is defined; false where any quantity's window holds a masked pixel or its correlation factor is undefined (that quantity's `<q>_err` is NaN, the others stay defined, the term columns keep their values) |
| `sky_err_arcsec`, `sky_err_source` | float64, object | uq | the external astrometric uncertainty per axis and its provenance, copied from `uq.sky_astrometry_arcsec` and `uq.sky_astrometry_source` (NaN and None when null); never added to the pixel-frame error bars |
| `logit`, `probability` | float32 | xscan | classifier output (NaN without a checkpoint) |
| `decision` | bool | xscan | `probability >= classify.decision_threshold` (false without a checkpoint) |
| `ra`, `dec` | float64 | terminal | degrees from the header WCS at the integer candidate pixel, not the xFit sub-pixel position |
| `exposure_id`, `ccd_id`, `band` | object | identity | generic identity, plus the adapter's extras |
| `astrometry` | object | identity | how `ra`/`dec` were derived (`header-wcs-approx` for Rubin: the header WCS, unrefined) |

`transient_chain.catalog.markdown_table()` prints this table from the schema.

## How the error bar is built

For `center_x`, `center_y`, `positive_x`, `positive_y` and `psf_flux`, `<q>_err` is the error bar
to quote: the quadrature of `<q>_err_fit` and the four terms below, each written to its own column
so a user can rebuild the bar with any term left out (step 8c, `transient_chain/uq.py`). Every
term is a standard uncertainty in the quantity's units, pixels for a position and image units for
the flux.

The terms that depend on the reference's noise (`corr`, `kfit`) are computed for a weighted sum of
residual pixels that stands in for xFit's estimator: the PSF-weighted sum at the photometry
position for the flux (the forced-photometry estimator), the PSF-derivative-weighted sum at a lobe
divided by the lobe flux for the lobe's position (a shift in pixels), and the mean of the two
lobes' shifts for the center, built as one weight stamp over both lobes so the covariance between
them is kept. Each stamp is centered on the sub-pixel position it measures; a window covers
`2 ceil(5 psf_sigma) + 1` px (`uq.window_px` in `summary.json`, `uq.center_window_px` for the
center's stamp over both lobes).

- `fit` (`<q>_err_fit`): xFit's covariance, weighted by the target variance plane; the template's
  noise is not in that plane (below).
- `corr` (`<q>_err_corr`): `<q>_err_fit * sqrt(f - 1)`, `f` the correlation factor of the
  quantity's weighted sum (`cuphoton.xpois.correlation_factor`, floored at 1): the resampling and
  the kernel correlate the reference's pixels. The factor is measured against the target variance
  used by xFit, so it includes both the reference's diagonal noise and its correlations.
  It needs the reference's variance plane on the target grid and the resampling
  weights (`cuphoton.xrep.resampling_weights_at` with the template's `ReprojectionSpec`); this
  version's template providers carry no variance plane, so the term is 0 and `summary.json` lists
  it under `uq.excluded`. Where the grid mapping is not locally a translation the factor is
  undefined (`uq.rotated_grid` counts those rows).
- `kfit` (`<q>_err_kfit`): the subtraction kernel's own uncertainty carried to the same weighted sum,
  `cuphoton.xpois.kernel_fit_covariance` once per item over the fit pixels (the sandwich form when a
  reference variance plane exists, else the plain form) and `kernel_fit_variance` per candidate.
  Largest where a candidate sits on a bright reference source. The position terms are 0 where
  the lobe flux is not usable. Without a target variance plane the term is excluded because its
  noise scale is unknown.
  `uq.kernel_fit: false` skips it.
- `kmatch` (`<q>_err_kmatch`, flux only): `eta` times the kernel-matched reference flux `T` at the
  photometry position. `eta` is the fraction of a static star's reference flux that the subtraction
  leaves in the residual beyond noise: at the stars the refine stage kept, the residual's forced
  flux `D` has predicted variance `uD^2 f + kfit` when the kernel is right, and `eta^2` is the
  extra variance, in units of `T^2`, that puts the median of `D^2` over that prediction at the
  chi-square(1) median (a variable star or two does not move a median). Positions get 0: a
  kernel-mismatch term for the position of a residual on a bright reference source is a planned
  addition.
- `mis` (`<q>_err_mis`, positions only): the misalignment model's `variance(x, y)` at the position.
  The center of a resolved dipole gets 1/4 of it: each lobe moves with its own image, and the
  center is their mean. The positive lobe of a candidate with one supported lobe gets `lambda^2`
  times it, `lambda = (D + T) / D` with `D` the residual's forced flux and `T` the matched
  reference's at the lobe (a faint change on a bright source moves with the source, not with the
  residual); `|lambda|` is capped at 10 and `uq.sensitivity_capped` counts the rows it capped. A
  resolved dipole's positive lobe gets `lambda = 1`. For a fading single source the positive lobe is
  the unsupported one, so `lambda^2` scales columns that describe a lobe that is not there; use
  `psf_flux` for it. 0 without a misalignment model (refinement off, a shift-only refinement, or the
  `daofind` star finder).
- `sky` (`sky_err_arcsec`): the exposure's absolute astrometry, an input the chain does not measure;
  reported with its provenance and kept out of the pixel-frame columns.

Validity is per quantity: `<q>_err` is NaN where the quantity's window (both lobes' windows for a
center) holds a masked or non-finite pixel, or where its correlation factor is undefined; the term
columns keep their values (`<q>_err_corr` is NaN with the factor). Invalid original measurement positions and nonfinite total uncertainties also leave their
quantity's error undefined. `uq_valid` is true where every
quantity of the row has a finite total uncertainty and is valid, so a candidate near a masked block can carry a valid `positive_x_err`
and a NaN `psf_flux_err` in the same row with `uq_valid` false. A rule for windows with masked
pixels is a planned addition. A term the item could not compute is written as 0 and `summary.json`
records the terms that went into the bars (`uq_terms`) with the reason for each one left out
(`uq.excluded`), the kernel-fit form, the kernel-mismatch fit at the stars (`eta`, the stars used
and the median residual-to-reference flux ratio), the window sizes and the counts of rows with an
invalid window, an undefined factor and a capped sensitivity. `uq.enabled: false` writes
`<q>_err = <q>_err_fit`, zero terms and `uq_valid` true.

The xFit standard errors come from the target's variance plane. The template's noise, carried
through the subtraction kernel, is not in that variance, so the standard errors understate the
difference image's noise by that amount. Without a variance plane, xFit scales each covariance by
the fit's reduced chi-square, which estimates the noise from the stamp's residuals.

`quality_flags` marks fits that a valid covariance does not exclude: widths outside 0.5 to 2.5 times
the PSF width (xFit's default range when none is known), lobes outside the stamp, no lobe at the
candidate, overlapping lobes, and an unsupported lobe. A candidate with only
`NEGATIVE_LOBE_UNSUPPORTED` is a brightening single source and one with only
`POSITIVE_LOBE_UNSUPPORTED` a fading one; for them `psf_flux` is measured at the supported lobe and is
the quantity to use, and the dipole columns describe a lobe that is not there. Every other candidate's
`psf_flux` is measured at the positive-flux lobe, including when the lobes overlap. For overlapping
lobes, only the fitted center coordinates and the product of amplitude and separation are
identifiable; neither the separation nor the individual lobe flux is measured independently.
This is a limit on interpreting the dipole fit, not a switch of the forced-photometry position.
The PSF is a circular Gaussian of width `xfit.psf_sigma_px`, or of
the width measured from the refine stage's stars when that key is null (`summary.json` records
which, see above); the width must come from a measurement independent of the candidate fits, and
without one the `psf_flux` columns are NaN. `psf_flux_err` assumes the position and the PSF are fixed and
correct. The position is the fit's, measured on the same stamp, so the uncertainty is conditional on
it; a fitted dipole width used as the PSF would make it far too small.

## Compare with a reference catalog

```bash
python examples/transient-chain/compare_catalogs.py --ours out/catalog.parquet \
    --reference reference.csv --out report.json
```

The reference is a CSV or parquet catalog with pixel positions and the same detector-exposure
identity columns. Candidates are matched within 1.5 px, one reference row per candidate, each
reference row counted once (duplicates are reported, not counted). Exact agreement with a
DAOStarFinder-aligned chain is not expected under the `gpu` star finder: the alignment differs
at the sub-pixel level and the residual with it; `--set align.star_finder=daofind` matches
that alignment.

## Repeat test across repeated exposures

```bash
python examples/transient-chain/repeat_catalogs.py out/items/*/catalog.parquet --out repeat.json
```

Without a reference catalog, a source that does not move should land at the same place in every
exposure of a detector, so its scatter divided by its error bars gives a chi-square per degree of
freedom near 1 when the bars are right (survey pipelines call this repeatability). The script
scores the chain against itself from the catalogs it wrote for a detector's repeated exposures;
several detectors are linked separately and pooled.

Positions reach the sky in two steps. The refine affine from the `summary.json` beside each
per-item catalog (identity when there is none) moves each sub-pixel position into the reference
frame. The catalog's own `ra`/`dec` at the integer candidate pixel then anchors it: per exposure
an affine pixel-to-tangent-plane map is fitted to those anchors (at least 10 of them, in a layout
that is not a line; exposures that fail are listed under `detectors.<id>.exposures_without_anchor_fit`),
and a detection's sky position is its anchor's plus the linear part of that map applied to the
offset from the integer pixel. The header WCS's distortion stays in the anchors, and dithers and
rotations between exposures are not scatter. `detectors.<id>.anchors.<exposure>` records each
exposure's detections, the transform applied (`identity`, `shift` or `affine`) and the anchor fit
(anchors, residual rms in arcsec, arcsec per pixel).

Detections are linked across exposures within `--link-px` (default 2) after duplicates within an
exposure are dropped within `--duplicate-px` (default 1, keeping the highest `peak_abs_sigma`). A
group seen in fewer than two exposures, or twice in one, is dropped.

Two quantities are scored, each on the classes whose columns describe the source. The classes come
from `quality_flags`, `converged` and `uncertainty_valid`: `resolved` (two supported lobes),
`overlapping`, `single` (a brightening source: only the positive lobe supported), `fading` (only
the negative lobe supported), `failed` (width out of range or no lobe at the candidate) and
`invalid` (no usable covariance). The dipole center (`center_*`) is scored on resolved and
overlapping candidates; the source position (`positive_*`) on resolved and single ones. A fading
single's source is its negative lobe, which the catalog does not carry as a position, so that class
is in neither quantity.

For each quantity, overall and per class, the report gives two statistics: the pooled chi-square
per degree of freedom, and the median per-group ratio (the median over groups of chi-square over
the chi-square median for the group's degrees of freedom, 1 for correct bars at any group size).
Both are given for six arms: `<q>_err`, `<q>_err_fit` alone, and `<q>_err` with each of the corr,
kfit, kmatch and mis terms removed in quadrature. Every arm uses the same rows, those whose
`<q>_err` is defined on both axes in groups that keep two detections. The per-axis variances are
along the pixel axes, scaled to arcsec by the anchor fit, and are paired with the tangent-plane
offsets without rotation, which matters only where the two axes' variances differ. The JSON holds
every number; the printed table shows quantity, class, groups, arm, chi-square per degree of
freedom and the median ratio.

What the test cannot see, stated in its output: the reference's noise is the same in every
exposure of a detector and never scatters between them, so every arm keeps a term the test does
not measure and a chi-square below 1 by that share is still consistent; and a bias common to all
exposures (a reference position error, a systematic of the fit) moves every detection alike and
passes unseen.

## Known issues

- In this development checkout, device reprojection propagates mask bits with a bilinear
  2x2 footprint even when the image is sampled with Lanczos-3 (`xrep/backends/cupy_backend.py` does not pass
  the spec's interpolation to the mask kernel; the host path does). A bad coadd pixel 2 to 3
  source pixels from an output center can therefore contaminate the template value without
  setting a bad bit. `subtract.template_bad_dilation_px: 3` is the stopgap until the library
  passes the interpolation through; the shipped config keeps 0 to match the reference chain.
- Stage timings in `summary.json` use the names `read, template, refine, mask, xpois, detect,
  stamps, xfit, uncertainty, uq, xscan, terminal_d2h, write`. `xfit` is the fit alone,
  `uncertainty` the quality flags, propagated uncertainties and forced photometry after it, and
  `uq` the error budget; data captured before this version used `headers`, `xdr_read` and
  `mosaic`.
