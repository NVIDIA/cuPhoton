# A survey's images on the GPU, end to end, under one persistent worker per GPU

One worker per GPU takes a target image from FITS bytes to a scored candidate catalog without
writing an intermediate file. A survey adapter reads the target and template planes to device
memory through the FITS reader you choose (xDR GPU decode, Astropy, or automatic); xRep, XPOIS,
the detector, xFit and XScan then work on device arrays; the catalog is the only thing that
comes back. cuPhoton's executors place the workers locally, under Dragon, or under MPI.

The example builds on `cuphoton` 0.1.3 and later. The shipped `config.yaml` holds the
parameters of the file-based reference chain on Rubin DP1; with a different `survey:` block
the same chain runs on other FITS surveys.

```
launch.py                   CLI: --executor local|dragon|mpi, --set block.key=value overrides
config.yaml                 every stage's parameters, validated at load; io.reader and survey.name live here
transient_chain/config.py      the config schema: defaults, required keys, value checks, --set overrides
transient_chain/fitsio.py      PlaneReader: auto | xdr | astropy on top of cuphoton.core.fits_io, with read receipts
transient_chain/surveys/       SurveyAdapter protocol; rubin.py (DP1 layout), generic.py (any FITS pair by parameters)
transient_chain/worker.py      the persistent worker: read -> template -> refine -> mask -> XPOIS -> detect -> stamps -> xFit -> XScan
transient_chain/ops.py         stage operations on NumPy or CuPy arrays (star finder, affine, hotspots, fit mask, stamps)
transient_chain/catalog.py     the catalog column contract (same columns and dtypes for every item)
transient_chain/workload.py    selections -> WorkItems -> WorkloadSpec; run_local, run_distributed
compare_catalogs.py         match two catalogs by position and report differences
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
| XScan | DLPack handoff of the scaled stamps to Torch, `predict_tensors` | none |
| End | one packed download: positions, fit parameters, uncertainties, logits, probabilities | `catalog.parquet` and `summary.json` per item |

`summary.json` splits host traffic into `transfers.terminal_bytes` (the catalog),
`control_bytes` (star lists, hotspot table, candidate filter) and `diagnostic_bytes`
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
30 stars) and outliers clipped at 5 sigma, a limit that keeps every Gaussian residual while a
blend, tens of sigma out, still goes. `variance(x, y)` includes the squared remaining affine and
quadratic displacement because this diagnostic fit does not correct the template or catalog
positions. The worker keeps the model for the later stages. The summary under
`diagnostics.align.refine.misalignment` records `model` (`"polynomial"`, `"rms"`, or `null` with
a `reason`), the star counts, the residual rms and the excess and image-center sigma in pixels.
The template has no variance plane, so its centroid noise is counted as excess.

The kept target stars always give the PSF width (`ops.psf_width`: clipped median of the widths
with its standard error), written under the same diagnostics key as `psf_width`. When
`xfit.psf_sigma_px` is null that width is the one the uncertainty stage uses, and `summary.json`
records it as `psf_sigma_px` with `psf_sigma_source: "stars"` (`"config"` when the configuration
set it, `null` when neither was available and the `psf_flux` columns are NaN).

## Prerequisites

1. cuPhoton 0.1.3 or later with the `gpu` extra (CuPy, xDR dependencies, Torch, photutils) plus `dragon`
   or `mpi` for those executors; Python 3.12 or 3.13 (Dragon has no 3.14 wheel). Follow the
   library's `docs/getting-started.md` for the install that matches your release (checkout or
   wheels) and `docs/components/xdr.md` for the xDR native extension. Then, from the library checkout:
   ```bash
   uv pip install -e examples/transient-chain --no-deps
   ```
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
| `center_x`, `center_y`, `positive_x`, `positive_y` and `*_err_fit`, `*_err`, `center_xy_cov`, `positive_xy_cov` | float64 | xfit | lobe midpoint and positive-flux lobe in pixels relative to (`x`, `y`). `<q>_err_fit`: standard uncertainty from xFit's covariance alone. `<q>_err`: the error bar to quote (see below). `*_xy_cov`: x-y covariance from xFit's covariance |
| `separation`, `lobe_flux` and `*_err` | float64 | xfit | distance between the lobes and each lobe's total flux, with standard uncertainties |
| `psf_flux`, `psf_flux_err_fit`, `psf_flux_err` | float64 | xfit | forced PSF photometry at the source. `psf_flux_err_fit`: conditional uncertainty from xFit alone. `psf_flux_err`: the error bar to quote (see below). All three NaN without `xfit.psf_sigma_px` or a variance plane |
| `logit`, `probability` | float32 | xscan | classifier output (NaN without a checkpoint) |
| `decision` | bool | xscan | `probability >= classify.decision_threshold` (false without a checkpoint) |
| `ra`, `dec` | float64 | terminal | degrees from the header WCS at the integer candidate pixel, not the xFit sub-pixel position |
| `exposure_id`, `ccd_id`, `band` | object | identity | generic identity, plus the adapter's extras |
| `astrometry` | object | identity | how `ra`/`dec` were derived (`header-wcs-approx` for Rubin: the header WCS, unrefined) |

`transient_chain.catalog.markdown_table()` prints this table from the schema.

For `center_x`, `center_y`, `positive_x`, `positive_y` and `psf_flux`, `<q>_err` is the error bar
to quote and grows as uncertainty terms are added to the chain: this version includes only the xFit
covariance term, so `<q>_err` equals `<q>_err_fit`. The `*_err` columns of the raw dipole parameters
(`amplitude`, `sigma_x`, `sigma_y`, `theta`, `x_pos`, `y_pos`, `x_neg`, `y_neg`), `separation` and
`lobe_flux` are xFit's term alone and do not grow.

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

## Known issues

- At cuPhoton 0.1.3 the device reprojection propagates mask bits with a bilinear 2x2 footprint
  even when the image is sampled with Lanczos-3 (`xrep/backends/cupy_backend.py` does not pass
  the spec's interpolation to the mask kernel; the host path does). A bad coadd pixel 2 to 3
  source pixels from an output center can therefore contaminate the template value without
  setting a bad bit. `subtract.template_bad_dilation_px: 3` is the stopgap until the library
  passes the interpolation through; the shipped config keeps 0 to match the reference chain.
- Stage timings in `summary.json` use the names `read, template, refine, mask, xpois, detect,
  stamps, xfit, uncertainty, xscan, terminal_d2h, write`. `xfit` is the fit alone and
  `uncertainty` the quality flags, propagated uncertainties and forced photometry after it; data captured before this version used `headers`,
  `xdr_read` and `mosaic`.
