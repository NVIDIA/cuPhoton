# Data and artifact contracts

This page describes the stable concepts a workflow adapter should preserve.
Individual commands may accept additional fields; use `<command> help
<subcommand>` for the complete option contract.

## General rules

- Array dimensions are listed in storage order.
- Image coordinates use `(y, x)` array order unless an option explicitly asks
  for `(x, y)` pixel coordinates.
- Inputs must be finite unless a command documents a NaN policy.
- Variance arrays represent variance, not standard deviation, and must be
  positive on fitted pixels.
- Boolean fit masks use `True` for selected pixels. Instrument bit masks must
  be translated with the command's mask policy.
- Keep units and coordinate frames in FITS headers or adjacent metadata; NumPy
  arrays do not carry them.

## xFit dipole batches

xFit reads `.npz` archives with pickle disabled. The archive requires a unique
one-dimensional `candidate_id` array and floating-point `images`. Difference
images have shape `(batch, y, x)`; split images have shape
`(batch, 3, y, x)` with channels ordered as difference, positive, and
negative. Optional `initial` parameters use `(batch, parameter)`. Optional
`mask` and `variance` arrays must align with the fitted image pixels;
variances must be positive wherever the mask selects a pixel. The
sampled-stamp model additionally consumes a numeric `stamp_basis` array.
For split inputs, `(batch, y, x)` auxiliaries are per-candidate and broadcast
over all three planes. This interpretation also wins when `batch == 3`; use
`(1, 3, y, x)` to express per-plane auxiliaries without ambiguity.

The command rejects standalone `.npy` inputs, object arrays, and archives that
require pickle. It does not download or bundle observational data.

A successful `fit-dipoles` run contains:

| File | Meaning |
| --- | --- |
| `summary.json` | requested and resolved model/backend, input and compute dtypes, device, input-archive hash, input counts, artifact paths, and artifact hashes |
| `effective-config.yaml` | validated options used for the fit |
| `fits.parquet` | one row per candidate with an exact input-stamp hash, parameters, status, convergence, valid-pixel coverage, fitted and zero-signal chi-square statistics, and uncertainties |
| `fit-arrays.npz` | candidate indices and IDs, covariance matrices, and residual arrays without pickled objects |

Covariance uses supplied variances when present and residual scaling
otherwise. Non-converged or rank-deficient fits retain status information and
report invalid uncertainties explicitly instead of publishing finite-looking
errors.

## XScan HSC NPY inputs

XScan accepts an HSC NPY directory directly or as `HSC_npy` beneath a supplied
base directory. The directory contains `metadata.json` and these arrays:

| File | Axis order | Meaning |
| --- | --- | --- |
| `images.npy` | `(batch, y, x, exposure)` | observed images |
| `variances.npy` | `(batch, y, x, exposure)` | per-pixel variances |
| `psfs.npy` | `(psf_y, psf_x, channel, exposure)` | exposure PSFs |
| `exp_times.npy` | `(exposure,)` | exposure times |
| `masks.npy` | optional `(batch, y, x, mask_class, exposure)` | mask planes |
| `sky.npy` | optional `(batch, y, x, exposure)` | sky estimates |

Spatial dimensions and exposure counts must agree across corresponding
arrays. Treat the directory as caller-supplied data and keep it outside the
repository.

## XPOIS image pairs

`reference` and `target` are two-dimensional arrays with the same shape. An
optional variance image and input masks must also match that shape. Kernels
have odd height and width. An explicit NPY fit mask is boolean or binary and
uses `True`/`1` for pixels included in the weighted solve.

A successful fit writes `summary.json` and these arrays under `artifacts/`:

| File | Meaning |
| --- | --- |
| `kernel.npy` | fitted matching kernel |
| `matched.npy` | convolved reference plus fitted background |
| `residual.npy` | target minus matched image |
| `fit_mask.npy` | pixels used by the solve |
| `background.npy` | fitted differential background |

Auto-stamp selection also writes metadata describing the selected regions.
Benchmark runs add timings and numerical comparisons.

## XScan datasets

A prepared dataset directory contains:

| File | Shape or format |
| --- | --- |
| `search.npy` | `(sample, y, x)` floating-point stamps |
| `template.npy` | `(sample, y, x)` floating-point stamps |
| `difference.npy` | optional `(sample, y, x)` triplet channel |
| `labels.npy` | `(sample,)` binary labels |
| `split.npy` | `(sample,)`; train `0`, validation `1`, test `2` |
| `metadata.jsonl` | one JSON object per sample when metadata is available |
| `summary.json` | dataset kind, shapes, split counts, and saved files |

All image arrays have the same shape and all first dimensions equal the label
count. Pair models consume search and template channels; triplet models also
require `difference.npy`. Validate label provenance before training; smoke or
unlabeled placeholder rows are not training labels.

Training runs contain an effective config, checkpoint and metric artifacts,
and a summary that records the selected device and split. Review annotations
are separate append-only artifacts and should retain reviewer and source-run
provenance.

### xFit feature bundles for XScan

`cuphoton xscan data-export-xfit-input` writes a pickle-free `.npz` containing
numeric or Unicode arrays: unique `candidate_id` and exact float `images` rows from
`difference.npy`. It is the default bridge into `cuphoton xfit fit-dipoles`;
construct an archive manually only when the fit also needs masks, variances,
initial parameters, or a stamp basis.

`cuphoton xscan data-build-xfit-features` joins a completed difference-mode
xFit run to a prepared XScan dataset by `candidate_id` and writes a new bundle
directory. It verifies the xFit artifact hashes and binds every matched row to
the exact XScan `difference.npy` stamp by a dtype-, shape-, and content-aware
hash:

| File | Meaning |
| --- | --- |
| `candidate-id.npy` | numeric or Unicode candidate IDs in exact XScan metadata order, loaded with pickle disabled |
| `features.npy` | row-aligned float32 feature matrix, loaded read-only through a memory map so split datasets and data-loader workers share the same file-backed values |
| `input-image-sha256.npy` | fixed-width ASCII SHA-256 values for the exact difference stamps, loaded with pickle disabled |
| `schema.json` | ordered feature contract, bounded transforms, source SHA-256 hashes, missing policy, and join diagnostics |

The candidate array must exactly match XScan metadata order when loaded.
Duplicate XScan rows may reuse one fit only when the duplicate stamps, splits,
and split groups are identical; duplicate xFit candidate IDs are rejected.
The default `missing_policy: error` requires every dataset row to have a fit.
`indicator` instead emits finite zero features with
`fit_present=0` for unmatched rows. Invalid or non-converged fits set their
validity gates to zero and do not expose invalid parameter-derived values.
Every load recomputes the current dataset's difference-stamp hashes. Training
checkpoints retain the exact schema and feature artifact identity, and
inference records the independently validated identity of its target bundle.

These generated artifacts are data-bearing, not privacy-sanitized. The xFit
input contains exact pixels and candidate identifiers; fit and feature
artifacts retain identifiers, hashes, residuals or derived fit values; and run
summaries can retain resolved local paths. Do not publish them unless their
source data and metadata are cleared for release.

Version 1 uses fixed, bounded transforms rather than statistics fitted to the
training, validation, or test rows. A training config opts in with top-level
`use_xfit_features: true` plus `xfit_feature_dir`;
`model.xfit_feature_names` is then populated from and locked to the bundle
schema. The optional fusion controls are
`xfit_hidden_dim`, `xfit_dropout`, and `xfit_modality_dropout`. Inference and
evaluation of a fusion checkpoint require both `--use-xfit-features` and
`--xfit-feature-dir`. Omitting both preserves the image-only path; an
image-only checkpoint rejects xFit features. Indicator-policy evaluation also
rejects a validation/evaluated-split `fit_present` coverage difference greater
than 0.05 unless the caller explicitly supplies
`--allow-xfit-coverage-mismatch`.

These artifacts do not create labels. Rubin `candidate_isDipole` and
placeholder labels must not be interpreted as real/bogus truth. When reviewed
labels become available, make train/validation/test partitions group-aware by
DiaObject or an equivalent stable source identity.

## xRep FITS and arrays

CLI inputs are two-dimensional FITS images with a celestial WCS. Optional masks
must align with the source image. Python array workflows pair a 2D array with a
`ReprojectionSpec` containing the source-to-grid mapping, output bounding box,
interpolation, fill value, and mask policy.

Single-image runs write `artifacts/reprojected.npy`, an optional `mask.npy`, and
`summary.json` with backend, interpolation, grid reference, pixel scale,
bounding box, and timings. FITS output is optional. Stack runs use
`reprojected_stack.npy` and an optional mask stack on one shared grid.

## XRay HDF5 and trace products

XRay probes an HDF5 file before selecting a supported image cube, scan/delay
axis, entry counts, and intensity-normalization fields. On/off cubes must use
the same schema and image shape. Detector ROIs are supplied as `(x, y)` origins
and `(width, height)` dimensions; generated NumPy images remain row-major.

Trace NPZ files contain one-dimensional time/delay and signal arrays. Detector
artifact runs persist a manifest plus detector-wide frequency, amplitude, and
related NumPy arrays. Distributed runs add a plan, shard manifests, per-shard
logs, and a merged manifest. Preserve the ROI, excluded rows, normalization,
fit parameters, shard ranges, and package/hardware details with published
results.

## Run-directory hygiene

Run directories are generated artifacts, not source. Store them outside the
checkout when practical, do not commit them, and do not treat a standalone HTML
view as the only result. A portable run should contain enough structured
configuration and metadata to regenerate its derived review files.
