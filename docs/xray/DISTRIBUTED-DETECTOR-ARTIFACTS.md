# Distributed detector artifacts

`cuphoton xray detector-artifact-distributed` divides a detector ROI into x-axis shards
and either plans or launches the same `detector-artifacts` worker command for
each shard. It supports local GPU assignment and Slurm array scripts.

## Inspect an in-memory dry run

`dry-run` prints the plan and rendered local/Slurm scripts as JSON. It does not
write `plan.json`:

```bash
uv run cuphoton xray detector-artifact-distributed \
  --h5dir /path/to/hdf5 \
  --fon run-on.h5 \
  --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001 \
  --run-label run-001 \
  --shard-count 4 \
  --gpus 4 \
  --executor dry-run \
  --json > /tmp/run-001-dry-run.json
```

Review the global ROI, tile shape, shard ranges, worker commands,
normalization, fit parameters, and concurrency.

## Persist a local plan

Use `--executor local` without `--submit` to write the plan without executing
workers:

```bash
uv run cuphoton xray detector-artifact-distributed \
  --h5dir /path/to/hdf5 \
  --fon run-on.h5 \
  --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001 \
  --run-label run-001 \
  --shard-count 4 \
  --gpus 4 \
  --executor local \
  --json
```

The plan is written to:

```text
/path/to/artifacts/run-001/_distributed/run-001/plan.json
```

`plan.json` is an operational run file, not a publishable provenance record.
It contains local input and output paths plus expanded worker commands. Keep it
with private run artifacts, or redact those fields before sharing it. Published
artifact manifests use path digests instead of the raw HDF5 and shard paths.

## Execute locally

Rerun the same plan options with `--submit`. `--merge` merges successful shards
after all workers finish. `--resume` skips a shard only when its arrays are
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
visibility is explicitly empty or contains fewer devices than requested. It
never replaces an empty visibility mask with physical GPU indices.

## Render or submit Slurm scripts

`--executor slurm` without `--submit` writes `plan.json`, a Slurm array script,
and, with `--merge`, a merge script. Supply site scheduler options explicitly:

```bash
uv run cuphoton xray detector-artifact-distributed \
  --h5dir /path/to/hdf5 \
  --fon run-on.h5 \
  --foff run-off.h5 \
  --output-dir /path/to/artifacts/run-001 \
  --run-label run-001 \
  --shard-count 8 \
  --gpus 8 \
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

## Merge an existing plan

```bash
uv run cuphoton xray detector-artifact-merge \
  --shards-manifest \
    /path/to/artifacts/run-001/_distributed/run-001/plan.json \
  --output-dir /path/to/artifacts/run-001 \
  --json
```

Alternatively, repeat `--shard-dir` for every shard directory. Strict merging
requires the manifest shard count and input count to agree; `--no-strict`
relaxes the count check. Every merge still requires matching input
fingerprints, manifest and package versions, schema and dataset names, dtype,
normalization source, and detector configuration. Shards from different input
pairs are rejected even when their array shapes match.

Artifact manifests identify input files with path digests, size and mtime, and
a full or sampled content digest. They do not publish the source HDF5 paths.
Changing an input file, normalization cache, or detector option changes the
resume identity and forces the affected shard to run again.

## Fit diagnostics

Pass `--fit-diagnostics summary` or `--fit-diagnostics full` to retain the
same status-aware tile-row diagnostics produced by a single worker. The merge
command concatenates records in shard order and verifies the sidecar hash,
record count, status totals, ragged offsets, and common fitted time axis.
Diagnostic level and `--p2-ridge-alpha` are part of each worker's plan,
configuration hash, and resume identity, so a shard from a different fit
study cannot be silently reused.

The default diagnostic level is `none`, and the default ridge alpha is zero.
Use full diagnostics only for bounded regions because traces and
reconstructions can make a detector-wide sidecar large.

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
are run artifacts. Keep them out of git. A performance report should include
a redacted plan summary, code revision, environment, GPU/driver/runtime, input
identity, and the merged artifact manifest. Do not publish raw `plan.json`.
