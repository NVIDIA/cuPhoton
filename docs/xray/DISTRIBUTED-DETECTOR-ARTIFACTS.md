# Distributed detector artifacts

`cuphoton xray detector-artifact-distributed` divides a detector ROI into
x-axis shards and either plans or launches the same `detector-artifacts`
worker command for each shard. It supports local GPU assignment and Slurm
array scripts. Add `--artifact-layout tile-rows` to store one spectrum per
x tile row in each worker and the merged output. The default `dense` layout
writes pixel-shaped NPY arrays for existing NumPy consumers. Use
`--artifact-layout dense` to select it explicitly. Resume identities
distinguish the layouts, so changing the option rebuilds existing shards.
Merging requires all shards to use the same layout and preserves that layout.

The examples below select compact output explicitly. Use the shared
[`load_detector_array` reader](README.md#single-node-hdf5-workflow) for bounded
ROI reads from either format. The compact spectral files and their
`spectral-layout.json` must stay together.

## Inspect an in-memory dry run

`dry-run` builds the plan in memory and prints it with the rendered local/Slurm
scripts as JSON. It reads the input files to determine shapes and fingerprints,
but does not launch workers or write run files:

```bash
uv run cuphoton xray detector-artifact-distributed \
  --h5dir /path/to/hdf5 \
  --fon run-on.h5 \
  --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001 \
  --run-label run-001 \
  --shard-count 4 \
  --gpus 4 \
  --artifact-layout tile-rows \
  --executor dry-run \
  --json > /tmp/run-001-dry-run.json
```

Review the global ROI, tile shape, shard ranges, worker commands,
normalization, fit parameters, and concurrency.

From Python, put the layout in `detector_options` when building the plan:

```python
from cuphoton.xray.detector_distributed import (
    build_detector_artifact_distributed_plan,
    run_detector_artifact_distributed,
)

plan = build_detector_artifact_distributed_plan(
    h5dir="/path/to/hdf5",
    fon="run-on.h5",
    foff="run-off.h5",
    output_dir="/path/to/artifacts/run-001",
    run_label="run-001",
    shard_count=4,
    gpus=4,
    detector_options={"artifact_layout": "tile-rows"},
)
preview = run_detector_artifact_distributed(
    plan=plan, executor="dry-run", submit=False, merge=True, resume=True,
)
print(preview.payload["local_script"])
```

Use `detector_options={"artifact_layout": "dense"}` for dense output. The plan
carries the choice to every local or Slurm worker before resume decisions.
After reviewing the plan, set `executor="local", submit=True` in the runner
call to launch local workers with the same merge and resume settings.

## Reuse detector normalization

Each worker needs a normalization shift from the full detector, even when
fitting only one shard. Compute it once on the CPU to avoid repeating that
full input scan in every worker:

```bash
uv run cuphoton xray detector-artifact-normalize \
  --h5dir /path/to/hdf5 --fon run-on.h5 --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001-normalization --json
```

Add `--normalization-cache /path/to/artifacts/run-001-normalization` to each
planning or launch command below. The cache contains `normalization.json`
and `normalization.npz`. Use the same input pair, `--drop-leading` and
zero-offset settings for cache creation and workers; incompatible caches
are rejected. If selecting `--zero-offset-index` manually, set it in both
commands and interpret it after the leading samples have been dropped.

## Persist a local plan

Use `--executor local` to write the plan. Worker execution starts when you add
`--submit`:

```bash
uv run cuphoton xray detector-artifact-distributed \
  --h5dir /path/to/hdf5 \
  --fon run-on.h5 \
  --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001 \
  --run-label run-001 \
  --shard-count 4 \
  --gpus 4 \
  --artifact-layout tile-rows \
  --executor local \
  --json
```

The plan is written to:

```text
/path/to/artifacts/run-001/_distributed/run-001/plan.json
```

`plan.json` is an operational run file containing local input and output paths
plus expanded worker commands. Keep it with private run artifacts, or redact
those fields before sharing it. Published artifact manifests identify HDF5
and shard paths with path digests.

## Execute locally

Rerun the same plan options with `--submit`. `--merge` merges successful shards
after all workers finish. `--resume` skips a shard when its arrays are
complete and its recorded plan, shard, input, package, and detector-option
identity matches the current plan:

```bash
uv run cuphoton xray detector-artifact-distributed \
  --h5dir /path/to/hdf5 \
  --fon run-on.h5 \
  --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001 \
  --run-label run-001 \
  --shard-count 4 \
  --gpus 4 \
  --artifact-layout tile-rows \
  --executor local \
  --submit \
  --merge \
  --resume \
  --json
```

Local shard logs are under `_logs/run-001/`; shard artifacts are under
`_shards/run-001/`. GPU assignment preserves the inherited
`CUDA_VISIBLE_DEVICES` tokens, including GPU UUIDs, and cycles over the first
`--gpus` visible tokens. Local execution fails before launching workers when
visibility is explicitly empty or contains fewer devices than requested.

## Render or submit Slurm scripts

`--executor slurm` writes `plan.json`, a Slurm array script, and, with `--merge`,
a merge script. Add `--submit` to submit them. Supply site scheduler options
explicitly:

```bash
uv run cuphoton xray detector-artifact-distributed \
  --h5dir /path/to/hdf5 \
  --fon run-on.h5 \
  --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001 \
  --run-label run-001 \
  --shard-count 8 \
  --gpus 8 \
  --artifact-layout tile-rows \
  --executor slurm \
  --slurm-partition gpu \
  --slurm-time 01:00:00 \
  --slurm-gres gpu:1 \
  --merge \
  --json
```

Inspect the emitted scripts before adding `--submit`. With both `--submit` and
`--merge`, the launcher submits the merge as an `afterok` dependency when the
array submission returns a job ID.

The example permits eight concurrent array tasks, each requesting one GPU
on one node. The Python environment, working directory and all input/output
paths in the plan must be accessible from the compute nodes. The merge
script uses the scheduler's default resources; adjust it for the site when
submitting rendered scripts manually.

## Merge an existing plan

```bash
uv run cuphoton xray detector-artifact-merge \
  --shards-manifest \
    /path/to/artifacts/run-001/_distributed/run-001/plan.json \
  --output-dir /path/to/artifacts/run-001 \
  --json
```

The merge infers the layout from the shards and writes the corresponding
spectral files and layout metadata; it has no separate layout switch.
Alternatively, repeat `--shard-dir` for every shard directory. Strict merging
requires the manifest shard count and input count to agree; `--no-strict`
relaxes the count check. Every merge still requires matching input
fingerprints, manifest and package versions, schema and dataset names, dtype,
normalization source, and detector configuration. Shards from different input
pairs are rejected even when their array shapes match.

Artifact manifests identify input files with path digests, size and mtime,
and a full or sampled content digest. Changing an input file, normalization
cache, or detector option changes the resume identity and forces the
affected shard to run again.

## Fit diagnostics

Pass `--fit-diagnostics summary` or `--fit-diagnostics full` to retain the
same status-aware tile-row diagnostics produced by a single worker. The merge
command concatenates records in shard order and verifies the sidecar hash,
record count, status totals, ragged offsets, and common fitted time axis.
Diagnostic level and `--p2-ridge-alpha` are part of each worker's plan,
configuration hash, and resume identity. Changing either setting recomputes
the affected shards.

The default diagnostic level is `none`, and the default ridge alpha is zero.
Size regions for full diagnostics to fit the traces and reconstructions in
memory and storage.

For optional iterative fitting, pass `--fit-method iterative` and the
`--iterative-*` settings to `detector-artifact-distributed`. The plan forwards
them to every worker. Start with `--executor dry-run` to inspect the mode
count, frequency bounds and iteration budget before launching workers.
Iterative fitting uses the same CuPy worker placement as linear prediction.

An iterative artifact records its method and settings in manifest version
3. Linear-prediction artifacts retain version 2. Resume and merge checks
include the method and iterative settings; mixed methods or changed controls
require new fits. Iterative diagnostics use schema 2 with optimizer and
residual fields; linear-prediction diagnostics retain schema 1 with P1/P2
matrix diagnostics. See [Optional iterative fitting](README.md#optional-iterative-fitting)
for model assumptions and units.

Generated plans, scripts, shard directories, logs, merged arrays, and manifests
are run artifacts. Store them outside version control. A performance report
should include a redacted plan summary, code revision, environment,
GPU/driver/runtime, input identity, and the merged artifact manifest.
