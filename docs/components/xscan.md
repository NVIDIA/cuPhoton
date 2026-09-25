# XScan

`cuphoton.xscan` packages transient image stamps, trains and evaluates
PyTorch real/bogus classifiers, and creates numeric or Bokeh review artifacts.
The umbrella CLI group is `cuphoton xscan`.

XScan is CLI-first because dataset provenance, split controls, and run
artifacts are part of the reproducible workflow. Internal model modules
provide extension points for custom workflows, but are not a broad stable API.

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
when their stamps and split identity agree. Optional `--variance` and `--mask`
accept NPY arrays with exactly the same `(sample, y, x)` shape as
`difference.npy`, in the original dataset row order. The exporter selects the
same unique rows from every array and rejects duplicate IDs with conflicting
auxiliary planes, including values at excluded pixels. Nonzero mask values
include pixels; zero excludes them. Images must be finite and variance must
be finite and strictly positive at included pixels. Excluded image and
variance pixels can retain nonfinite values. Masks must be finite everywhere.

```bash
uv run cuphoton xscan data-export-xfit-input \
  --dataset-dir /path/to/dataset \
  --variance /path/to/variance.npy --mask /path/to/inclusion-mask.npy \
  --image-unit electron \
  --output /path/to/weighted-input.npz > /path/to/export-summary.json
```

Variance must describe the exported images in their squared units. If images
have been rescaled by a factor, rescale their variance by the factor
squared. `--image-unit` labels the existing values; the JSON summary records
that label, the corresponding variance unit, source filenames and SHA-256
hashes, and the output archive hash. Supplying variance enables xFit's
variance-weighted chi-square and the two variance-dependent scalar features.
Supply variance estimates and convert instrument bit masks to inclusion
masks before export.

By default, the exporter re-hashes each source after copying and removes its
new archive if a source changed. For large inputs guaranteed to remain
immutable throughout export, `--skip-source-rehash` skips this second source
hash pass (`verify_sources_after_copy=False` in the Python API). Initial
source hashes and the output archive hash are always recorded. The summary's
`source_hash_verification` is `before_and_after_copy` by default or
`before_copy_only` with the opt-out. Use the default verification to detect
source changes during export.

Use an explicitly constructed xFit archive when initial parameters or a
sampled PSF basis are required. `--compute-dtype float64` keeps the row hashes
bound to the original stamps while solving in float64, which is recommended for
ill-conditioned observational fits. Use `input` or `float32` when that
precision/performance tradeoff is intentional.

The exported input, feature bundle, checkpoints, and run summaries retain exact
pixels, candidate identifiers, per-stamp hashes, fit values, and resolved
local paths, depending on the stage. Confirm that their source data and metadata
are cleared for release before publishing generated artifacts.

The builder accepts difference-mode xFit runs and joins `fits.parquet` to
XScan metadata by `candidate_id`. It also verifies that each fit row was
computed from the exact `difference.npy` stamp, including dtype and shape,
and validates the hashes recorded by the xFit run. Pair and triplet XScan
models can both consume this same difference-fit sidecar. The new output
directory contains standalone `candidate-id.npy`, `features.npy`, and
`input-image-sha256.npy` arrays plus `schema.json`. The arrays are
pickle-free and memory-mappable; the schema records their hashes, the
ordered feature names, transforms, and join diagnostics. The default
`--missing-policy error` rejects an incomplete join. Use
`--missing-policy indicator` only when missing fits are expected: affected
rows have every feature set to zero, including `fit_present` and run-level
indicators such as `variance_weighted`. Present but invalid fits retain
valid run-level diagnostics, set validity gates to zero, and zero
fit-parameter features. The feature bundle retains row hashes and is rebound
to the current `difference.npy` every time it is loaded. Converting float32
stamps to float64 or changing a pixel after bundle construction is rejected.
Repeated candidate IDs may reuse one fit only when their difference stamps,
split, and split group are identical.

Features use versioned, fixed bounded transforms for fit validity, residual
improvement, uncertainty, dipole geometry, and Gaussian shape. These transforms
are independent of dataset statistics, preserving the separation of training,
validation, and test sets during bundle construction.

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

For repeated inference calls, use `infer-real-bogus --num-workers 0` to avoid
starting loader processes. Omit the option to retain the checkpoint's worker
count, or pass a positive count for parallel loading. The override applies to
a copy of the inference settings and leaves the checkpoint unchanged; zero
workers also disables persistent workers. `--batch-size` controls inference
batches independently and defaults to `32`.

The inference `summary.json` records the batch size and resolved performance
settings. Keep the batch size fixed when comparing scores across worker counts.

Use fixed, group-aware splits that keep related samples from crossing train,
validation, and test sets. Record the seed, model config, selected checkpoint,
device, label source, and dataset summary. Use reviewed real/bogus labels for
training and evaluation; Rubin `candidate_isDipole` flags and placeholder
`label.npy` values serve smoke tests. Split by DiaObject, or an equivalent
stable source group, before model selection. Training rejects `split_group`
values that cross splits and also rejects cross-split Rubin DiaObject IDs when
those fields are present.

The `*.blackwell.example.yaml` files demonstrate throughput-oriented settings
for recent NVIDIA GPUs. Tune these starting points for your hardware and
dataset.

## Distributed inference

`infer-real-bogus --executor dragon|mpi` scores a fixed split across GPUs.
Each worker loads the checkpoint once and retains it across optional warmup
and measured passes. Training and evaluation commands retain their existing
local behavior. The default `--executor local` also preserves the original
inference output location.

Inputs, checkpoint, optional xFit feature bundle and output directory must
reside on a filesystem shared by all workers. `--batch-size` retains its
ordinary inference meaning. `--task-batches` groups whole minibatches into
tasks; only the final task can contain the original final partial batch.
Keep both values fixed when comparing worker counts. The merge restores
selected-split order, original sample indices, labels and candidate metadata,
and uses the same host probability calculation as local inference.
The task count must be at least the MPI rank count. Dragon uses the smaller
of the requested worker count and task count.

Distributed inference defaults to `--num-workers 0`, independently of the
checkpoint's loader setting. An explicit positive value creates persistent
loader processes during worker setup; the loader and parsed metadata are
reused across tasks and rounds. Workers retain inputs and model state, so
their initial load is outside the timed rounds.

Under a configured Dragon allocation:

```bash
dragon .venv/bin/cuphoton xscan infer-real-bogus \
  --executor dragon --max-workers 8 \
  --run-dir /shared/model --dataset-dir /shared/dataset --split test \
  --batch-size 32 --task-batches 16 --num-workers 0 \
  --output-dir /shared/results/inference-dragon \
  --warmup-rounds 1 --measure-rounds 2
```

With Open MPI, the rank wrapper narrows GPU visibility before Python starts.
The parent mask must list allocated GPUs in local-rank order:

```bash
: "${CUDA_VISIBLE_DEVICES:?must enumerate the allocated GPUs}"
mpirun -n 8 --map-by slot --bind-to none -x CUDA_VISIBLE_DEVICES \
  .venv/bin/cuphoton-openmpi-rank-exec -- \
  .venv/bin/cuphoton xscan infer-real-bogus \
  --executor mpi --run-dir /shared/model --dataset-dir /shared/dataset \
  --split test --batch-size 32 --task-batches 16 --num-workers 0 \
  --output-dir /shared/results/inference-mpi \
  --warmup-rounds 1 --measure-rounds 2
```

Distributed inference requires a new `--output-dir`. Its basename is the run
ID: 1–128 ASCII letters, digits, dots, underscores or hyphens, starting with
a letter or digit. Each pass writes merged
logits, labels, probabilities, sample indices and a summary under
`rounds/<round-id>/scientific/`; warmup outputs are retained too. Without
round flags, the single pass uses `scientific/`. The model directory remains
unchanged. Execution receipts and timing are separate from these scientific
outputs, and merging and validation occur after the timed worker phase.

## Persistent XPOIS, xFit and XScan pipeline

The Python API in `cuphoton.xscan.device_pipeline` runs complete image pairs
through constant-kernel XPOIS, stamp extraction, Gaussian difference-mode
xFit, feature conversion and triplet XScan inference. A `DeviceWorkerContext`
loads the model once and accepts serial jobs on one CUDA device. This path
requires CUDA 13, CuPy and Torch (`uv sync --locked --extra gpu`).

Use a triplet fusion checkpoint and the exact `schema.json` from its training
xFit feature bundle. The schema must describe unmasked, unweighted Gaussian
difference fits with the configured stamp shape. Pipeline inference disables
AMP, TF32, compilation and cuDNN benchmarking through an explicit checkpoint
policy. This policy does not enable PyTorch's deterministic-algorithm mode.

Prepare descriptors from caller-owned NPY images; this example uses an
existing 63-pixel checkpoint and an interior candidate in images of at least
95 by 95 pixels. Change the candidate coordinates and kernel settings for
your data. Optional item `variance` and `fit_mask` descriptors apply to XPOIS;
xFit consumes unweighted difference stamps.

```python
from pathlib import Path

import numpy as np

from cuphoton.core.artifacts import file_sha256
from cuphoton.xscan.device_pipeline import (
    DevicePipelineCandidate,
    DevicePipelineConfig,
    DevicePipelineItem,
    DeviceWorkerContext,
    DeviceXPOISPipelineConfig,
    NpyArrayDescriptor,
    decode_device_pipeline_evidence,
    run_device_pipeline_item,
)


def describe(path):
    path = Path(path).resolve()
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    return NpyArrayDescriptor(
        path=str(path), sha256=file_sha256(path),
        shape=tuple(array.shape), dtype=array.dtype.str,
    )


checkpoint = Path("/path/to/training-run").resolve()
schema = Path("/path/to/training-features/schema.json").resolve()
config = DevicePipelineConfig(
    device="cuda:0",
    checkpoint_dir=str(checkpoint),
    checkpoint_sha256=file_sha256(checkpoint / "checkpoint.pt"),
    feature_schema_path=str(schema),
    feature_schema_sha256=file_sha256(schema),
    stamp_shape=(63, 63),
    decision_threshold=0.5,
    xpois=DeviceXPOISPipelineConfig(
        kernel_shape=(15, 15), basis_sigmas=(1.5,), basis_degrees=(0,),
    ),
)
items = tuple(
    DevicePipelineItem(
        item_id=f"pair-{index}",
        reference=describe(reference), target=describe(target),
        candidates=(DevicePipelineCandidate(
            candidate_id=f"candidate-{index}",
            center_x=47, center_y=47, source_index=0,
        ),),
    )
    for index, (reference, target) in enumerate([
        ("/path/to/reference-0.npy", "/path/to/target-0.npy"),
        ("/path/to/reference-1.npy", "/path/to/target-1.npy"),
    ])
)
```

For direct execution, import Torch before CuPy calls CUDA, bind both libraries
to the configured device, and retain the context between items:

```python
import torch
import cupy as cp

torch.cuda.set_device(config.device_id)
cp.cuda.Device(config.device_id).use()
context = DeviceWorkerContext.initialize(config)
results = [run_device_pipeline_item(item, context) for item in items]
arrays = decode_device_pipeline_evidence(
    results[0].scientific_evidence, config=config,
)
```

For Dragon, use the descriptor preparation in a fresh coordinator process
without importing Torch or CuPy there. Run the following entry point under
your site's installed Dragon launcher, with all input, checkpoint, schema
and output paths accessible on every worker:

```python
from cuphoton.xscan.dragon_pipeline import run_dragon_device_pipeline

if __name__ == "__main__":
    batch = run_dragon_device_pipeline(
        items=items, config=config,
        output_root=Path("/path/to/pipeline-runs"), max_workers=1,
    )
    if batch.status != "success":
        raise RuntimeError(f"Pipeline failed: {batch.run_dir}")
    print(batch.run_dir)
```

Dragon places one worker on each selected GPU and maps it to local `cuda:0`.
Each worker reuses its context for complete image-pair jobs; increase
`max_workers` to use more GPUs. Candidate order is preserved within each
item. Workers validate placement before loading Torch and probing CuPy.
Results include predictions, configuration/input hashes and compact
scientific evidence; Dragon saves each result in
`items/<item_id>/summary.json` and checks it again in the coordinator.
The evidence decoder returns the 22 named arrays for comparison.

The `run-pipeline` command accepts the same descriptors with either Dragon or
MPI. Save the configuration and items above as a manifest:

```python
import json

Path("pipeline.json").write_text(json.dumps({
    "schema": "cuphoton.xscan.pipeline-manifest/v1",
    "configuration": config.to_payload(),
    "items": [item.to_payload() for item in items],
}, indent=2) + "\n")
```

Paths may be absolute or relative to the manifest. All nodes must see the
same source, inputs and output directory. Use your site's launcher settings;
these examples run one warmup pass and two measured passes:

```bash
dragon cuphoton xscan run-pipeline --executor dragon \
  --manifest pipeline.json --output-dir runs --name dragon-pipeline \
  --max-workers 8 --warmup-rounds 1 --measure-rounds 2

mpiexec -n 8 -x CUDA_VISIBLE_DEVICES cuphoton-openmpi-rank-exec -- \
  cuphoton xscan run-pipeline --executor mpi \
  --manifest pipeline.json --output-dir runs --name mpi-pipeline \
  --warmup-rounds 1 --measure-rounds 2
```

The MPI example requires Open MPI and the allocation's visible GPU list on
each node. Other MPI launchers must bind each rank to exactly one GPU before
starting Python; do not use the Open MPI wrapper with MPICH. Each rank uses
local `cuda:0`, and duplicate physical GPU assignments fail validation.

Both executors initialize one worker context and reuse it across all items
and rounds. Warmup outputs remain under `rounds/warmup-*`; measured outputs
remain under `rounds/measure-*`. Each round retains ordinary item results and
audit evidence. The parent `summary.json` separates readiness, batch time,
artifact validation and cleanup. Batch time includes dispatch, input loading,
numerical work and worker output publication; coordinator audits follow that
timer. It is not kernel-only time. Failed rounds or cleanup invalidate the
reported statistics. Omitting both round flags runs a single pass.

The pipeline retains device owners through the blocking terminal copy and
synchronizes failed work before reuse. Failed cleanup makes the context
unusable. Transfer receipts count pipeline-owned uploads and the packed
terminal download; internal XPOIS/xFit control transfers are outside that
count. Timings are host elapsed times, so distinguish model initialization,
first-item compilation and warmed context reuse when comparing runs.

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
