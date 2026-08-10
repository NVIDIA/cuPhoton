# XScan

`cuphoton.xscan` packages transient image stamps, trains and evaluates
PyTorch real/bogus classifiers, and creates numeric or Bokeh review artifacts.
The umbrella CLI group is `cuphoton xscan`.

XScan is CLI-first because dataset provenance, split controls, and run
artifacts are part of the reproducible workflow. Internal model modules are
extension points, not a broad stable API.

## Install and smoke test

```bash
# CUDA 13
uv sync --locked --extra dev --extra gpu --extra viz
uv run python examples/run_quickstarts.py --component xscan --require-gpu

# CPU
uv sync --locked --extra dev --extra torch --extra viz
uv run python examples/run_quickstarts.py \
  --component xscan --profile cpu
```

The quickstart builds deterministic stamps and runs a small one-epoch train and
evaluation path. `device: auto` prefers PyTorch CUDA and reports a PyTorch CPU
fallback.

## Prepared dataset contract

Pair models use `search.npy` and `template.npy`; triplet models also require
`difference.npy`. Arrays have shape `(sample, y, x)`. `labels.npy` and
`split.npy` are one-dimensional and use the same sample count. Metadata is one
JSON object per sample when present. See
[Data and artifact contracts](../data-artifacts.md#xscan-datasets).

Inspect and validate a prepared dataset before training:

```bash
uv run cuphoton xscan data-inspect --dataset-dir /path/to/dataset
uv run cuphoton xscan data-validate --dataset-dir /path/to/dataset
uv run cuphoton xscan data-check-training-labels \
  --dataset-dir /path/to/dataset \
  --require-ok
```

The dataset builders under `examples/xscan/` translate caller-supplied
NumPy, FITS, CSV, Parquet, or registry products into that contract. Paths in
the examples are placeholders.

## Optional xFit fusion

Export the exact difference stamps, fit them, then build a portable,
row-aligned feature bundle:

```bash
uv run cuphoton xscan data-export-xfit-input \
  --dataset-dir /path/to/dataset \
  --output /path/to/xfit-input.npz

uv run cuphoton xfit fit-dipoles \
  --input /path/to/xfit-input.npz \
  --output-dir /path/to/xfit-run \
  --model gaussian --mode difference --backend auto \
  --compute-dtype float64

uv run cuphoton xscan data-build-xfit-features \
  --dataset-dir /path/to/dataset \
  --xfit-run-dir /path/to/xfit-run \
  --output-dir /path/to/xfit-features \
  --missing-policy error
```

The exporter preserves the `difference.npy` dtype and values, rejects
nonfinite unmasked stamps, and safely collapses duplicate candidate IDs only
when their stamps and split identity agree. Use an explicitly constructed
xFit archive when masks, variances, initial parameters, or a sampled PSF basis
are required. `--compute-dtype float64` keeps the row hashes bound to the
original survey stamps while solving in float64, which is recommended for
ill-conditioned observational fits. Use `input` or `float32` when that
precision/performance tradeoff is intentional.

The exported input, feature bundle, checkpoints, and run summaries are
data-bearing, not privacy-sanitized. Depending on the stage, they retain exact
pixels, candidate identifiers, per-stamp hashes, fit values, and resolved
local paths. Do not publish generated artifacts unless their source data and
metadata are cleared for release.

The builder accepts difference-mode xFit runs and joins `fits.parquet` to
XScan metadata by `candidate_id`; array position is never used as the join
key. It also verifies that each fit row was computed from the exact
`difference.npy` stamp, including dtype and shape, and validates the hashes
recorded by the xFit run. Pair and triplet XScan models can both consume this
same difference-fit sidecar. The new output directory contains
standalone `candidate-id.npy`, `features.npy`, and
`input-image-sha256.npy` arrays plus `schema.json`. The arrays are
pickle-free and memory-mappable; the schema records their hashes, the ordered
feature names, transforms, and join diagnostics. The default
`--missing-policy error` rejects an incomplete join. Use
`--missing-policy indicator` only when missing fits are expected: affected
rows have every feature set to zero, including `fit_present` and run-level
indicators such as `variance_weighted`. Present but invalid fits retain valid
run-level diagnostics, set validity gates to zero, and zero fit-parameter
features.
The feature bundle retains row hashes and is rebound to the current
`difference.npy` every time it is loaded. Converting float32 stamps to float64
or changing a pixel after bundle construction is rejected. Repeated candidate
IDs may reuse one fit only when their difference stamps, split, and split
group are identical.

Features use versioned, fixed bounded transforms for fit validity, residual
improvement, uncertainty, dipole geometry, and Gaussian shape. No means,
standard deviations, thresholds, or other scaling parameters are estimated
from the dataset, so building a bundle cannot leak validation or test-set
statistics into training.

Enable fusion with the paired top-level training settings
`use_xfit_features` and `xfit_feature_dir`:

```yaml
dataset_dir: /path/to/dataset
use_xfit_features: true
xfit_feature_dir: /path/to/xfit-features
model:
  xfit_hidden_dim: 32
  xfit_dropout: 0.0
  xfit_modality_dropout: 0.0
```

Training populates `model.xfit_feature_names` from the validated bundle and
stores the exact ordered names in the checkpoint. An explicitly configured
list must match the bundle. The scalar head adds a gated residual logit to the
image model; its final layer starts at zero, so fusion starts from the image
logits. `xfit_modality_dropout` can train robustness to an unavailable fit.
The checkpoint and run summary also retain the training feature and schema
SHA-256 identity. Inference accepts a target-specific bundle with the same
versioned feature contract, rebinds it to that target dataset, and records its
own artifact identity.

Pass the feature bundle when using a fusion checkpoint:

```bash
uv run cuphoton xscan infer-real-bogus \
  --run-dir /path/to/run \
  --dataset-dir /path/to/dataset \
  --use-xfit-features \
  --xfit-feature-dir /path/to/xfit-features

uv run cuphoton xscan evaluate-real-bogus \
  --run-dir /path/to/run \
  --dataset-dir /path/to/dataset \
  --use-xfit-features \
  --xfit-feature-dir /path/to/xfit-features
```

`use_xfit_features: true` and `--use-xfit-features` make fusion an explicit
choice; each must be paired with its `xfit_feature_dir` or
`--xfit-feature-dir` location. Fusion checkpoints require a matching bundle,
while image-only checkpoints reject one. Leaving both settings out preserves
the existing image-only dataset, model, and checkpoint path exactly.

For bundles built with `--missing-policy indicator`, evaluation also compares
fit availability (`fit_present`) in the evaluated split with the validation
split used to select the threshold. A difference greater than 0.05 is rejected
by default because it can change calibration. Use
`--allow-xfit-coverage-mismatch` only after reviewing that distribution shift;
the override is recorded in the evaluation summary.

## Training and evaluation

```bash
uv run cuphoton xscan train-inada-pair \
  --config examples/xscan/train-pair.example.yaml

uv run cuphoton xscan train-inada-triplet \
  --config examples/xscan/train-triplet.example.yaml

uv run cuphoton xscan infer-real-bogus \
  --run-dir /path/to/run \
  --dataset-dir /path/to/dataset

uv run cuphoton xscan evaluate-real-bogus \
  --run-dir /path/to/run \
  --dataset-dir /path/to/dataset
```

Use fixed, group-aware splits that keep related samples from crossing train,
validation, and test sets. Record the seed, model config, selected checkpoint,
device, label source, and dataset summary. Do not train on smoke placeholders
or unlabeled rows. In particular, Rubin `candidate_isDipole` flags and
placeholder `label.npy` values are not real/bogus truth. A future labeled
evaluation should use reviewed labels and split by DiaObject, or an equivalent
stable source group, before model selection. Training rejects `split_group`
values that cross splits and also rejects cross-split Rubin DiaObject IDs when
those fields are present.

The `*.blackwell.example.yaml` files demonstrate throughput-oriented settings
for recent NVIDIA GPUs. They are starting points, not universal performance
recommendations.

## Review workflow

```bash
uv run cuphoton xscan review-queue \
  --run-dir /path/to/run \
  --dataset-dir /path/to/dataset \
  --split test \
  --output-dir /path/to/review

uv run cuphoton xscan review-bokeh --review-dir /path/to/review
uv run cuphoton xscan review-aggregate \
  --review-dir /path/to/review \
  --output-report /path/to/review/aggregation.json

# Serve the standalone raw-product or Alard--Lupton review applications.
uv run cuphoton xscan review-raw-compare --help
uv run cuphoton xscan review-alard-lupton --help
```

Numeric JSON/CSV queues and append-only annotations are the durable review
artifacts. Bokeh pages and contact sheets are derived views. Keep reviewer and
source-run provenance when merging annotations.

## Local data adapters and reproduction commands

`cuphoton xscan --help` lists prepared/raw builders, local FITS registry
adapters, controlled pair/triplet reproduction workflows, and entity-review
commands. Keep data acquisition and authentication outside the package, pass
explicit local paths, and preserve source identifiers in untracked run
metadata.
