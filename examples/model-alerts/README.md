<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Alerts — Which changes deserve attention?

Generate independent reference/search exposures, detect positive xPois
residuals, and train fresh native xScan pair/triplet classifiers. Follow the
notebook from electron counts through labels, grouped splits, operating
points, mistakes and a human review queue. Expanded science panels derive
the assumptions; terminology links support hover, keyboard focus and touch.

Every image, source, detector defect and noise realization is generated from
declared analytic models. No survey imagery, calibration products, old demo
datasets or pretrained checkpoints are inputs. The classifier results apply
to this modeled population. They do not establish survey performance or
calibrated astrophysical probabilities.

## Execute in a clean environment

Use Linux, Python 3.12 or newer supported by cuPhoton, and a CUDA 13-capable
GPU with its driver. From a complete cuPhoton source checkout:

```sh
uv venv --python 3.12 .venv-model-alerts
uv pip install --python .venv-model-alerts/bin/python -e '.[gpu]'
uv pip install --python .venv-model-alerts/bin/python \
  -r examples/model-alerts/requirements.txt
```

Use the appropriate CUDA 13 PyTorch index for your platform if it is needed
by your package installation. Keep these example dependencies optional;
ordinary cuPhoton installations do not install them. Bokeh, Jupyter and
Matplotlib are used for visualization/execution; Beautiful Soup supports the
repository's documentation checks. No third-party source is vendored.

Choose an output directory outside the checkout that does not yet exist.
Every session sharing a GPU must use the same advisory lock. This process
retains that lock for the complete scientific workload:

```sh
export CUPHOTON_MODEL_ALERTS_GPU_LOCK=/absolute/path/shared/gpu0.lock
.venv-model-alerts/bin/python examples/model-alerts/run.py \
  --profile quick-smoke --output /absolute/path/new-api-smoke \
  --gpu-lock "$CUPHOTON_MODEL_ALERTS_GPU_LOCK"
.venv-model-alerts/bin/python examples/model-alerts/run.py \
  --profile full-reference --output /absolute/path/new-full-reference \
  --gpu-lock "$CUPHOTON_MODEL_ALERTS_GPU_LOCK"
```

Create the shared lock's parent directory first. Check actual GPU ownership
as well as the lock before starting; do not interrupt another workload.
Provision dependencies before blocking network access for scientific
execution. Generation, training and inference make no network requests.

Quick smoke uses small field counts and one training epoch. It exercises the
APIs and export path and cannot substantiate scientific performance. Full
reference starts with 180 training and 60 validation fields, reserves 60 test
fields, and adds 60 shifted fields. Every field has three epochs. Development
counts double if the primary validation field-bootstrap half-width exceeds
five percentage points. All reference models complete 12 training epochs;
minimum validation loss selects the checkpoint. Validation fields are reused
for checkpoint and threshold selection, so the precision rule is a conditional
development heuristic. Runtime and artifact sizes follow the actual counts.

The runner snapshots source, executes a fresh notebook kernel and preserves
stdout, warnings and figures. It writes `model-alerts.executed.ipynb`,
`model-alerts-source.zip`, `execution.json`, and the separate `study/` data,
weights and scientific receipts. The committed notebook stays output-free.
`CUPHOTON_MODEL_ALERTS_OUTPUT` and `CUPHOTON_MODEL_ALERTS_PROFILE` are set by the
runner for its kernel; a direct notebook session must set them and the shared
lock explicitly.

## Scientific controls and frozen evaluation

The ontology precedes training. Unmatched detections stay negative; movers
are separate astrophysical challenges. Ambiguous associations remain counted
in coverage, separately from overlapping causes. Injection retention separates
losses from detection, eligibility exclusions and classification. Training excludes those two special categories from primary
binary fitting and evaluation states the same coverage boundary.

Native xScan validation and label-provenance checks guard independent fields.
All shared template noise, persistent defects and epochs remain together.
Model inputs contain only measured image channels or explicitly enabled
measured xFit features. Scaling is fitted on training data. Pair/triplet use
the same candidate pool and three initialization seeds. Controls include
SNR-only and measured-statistics baselines, one fixed training-label
permutation, nested development learning curves and a rejected invalid split.
xFit fusion uses native exact-candidate/stamp correspondence checks and
reports failed-fit coverage. Raw nonfinite and repeated-extreme fractions
are measured by binary class to check trivial generator signatures.

Each model/seed uses the highest inclusive score threshold achieving at least
95% validation recall, including tied scores. The final endpoint is test FPR
at that frozen threshold; achieved test and shift recall are reported
separately. Field bootstrap preserves epoch dependence and pairs comparisons
on the same fields. Seed outcomes are shown individually. Both evaluation
populations include family, SNR and overlapping-cause breakdowns. Local
registration offsets vary by star; brightening hosts are randomized among
non-mismatch stars. The development receipt counts bipolar registration
residuals. All generator mixture amplitudes are declared in `Design`.

`study/freeze.json` precedes reserved-field generation. Never retune after
unblinding. If a bug invalidates final evaluation, retain that attempt and set
fresh disjoint `CUPHOTON_MODEL_ALERTS_TEST_SEED_BASE` and
`CUPHOTON_MODEL_ALERTS_SHIFT_SEED_BASE` blocks for a new full reference. Smoke
uses different default blocks from the full reference. Unexpected permutation
predictability stops full execution before freeze for diagnosis.

## Static interactions

| Live feature | Static equivalent or local operation |
| --- | --- |
| Image panels, linked pan/zoom and tapped pixels | Reused xScan Bokeh image/inspector primitives with browser callbacks |
| Raw versus display transformations | Browser-only transforms; exact raw values retained |
| Candidate navigation and family/outcome filters | Saved deterministic gallery of at most 96 candidates |
| Score/SNR and failure inspection | Complete evaluation score plot and stratified failure gallery |
| Threshold and queue choices | Saved scores for all eligible evaluation candidates |
| Prevalence changes | Fixed-class-conditional-mixture projection from complete TPR/FPR |
| Annotation persistence | Optional local `cuphoton xscan review-bokeh --review-dir /path/to/generated-review` after preparing a queue from this generated dataset |
| Training, inference or new data | Local execution only; browser controls do not rerun them |

The gallery is selected by round-robin confusion category, SNR bin and family,
then candidate-ID order. Empty categories remain empty. Metrics and queue
projections use complete evaluation arrays, never gallery counts. The HTML
contains gallery pixels and evaluation statistics; full training arrays and
weights are separate artifacts.

The source reuses Model Sky's public notebook presentation code and the
existing xScan raw-comparison/Alard-Lupton display helpers. It does not read
live demo fixtures or open a Bokeh server. Existing server viewer behavior
is unchanged. See [PUBLISHING.md](PUBLISHING.md) for Quarto export, source/hash
verification, downloadable bundles and versioned guide publication.

## Check source and interactions

```sh
PYTHONPATH=src .venv-model-alerts/bin/python -m pytest \
  examples/model-alerts/tests
CUPHOTON_MODEL_ALERTS_CHROMIUM=/absolute/path/to/chromium \
  PYTHONPATH=src .venv-model-alerts/bin/python -m pytest \
  examples/model-alerts/tests/test_static_browser.py
ruff check examples/model-alerts
```

The browser regression runs standalone HTML with network APIs blocked.
Complete qualification also covers the final rendered chapter on desktop and
mobile, themes, DPR, keyboard/touch, source links and a versioned static route.
