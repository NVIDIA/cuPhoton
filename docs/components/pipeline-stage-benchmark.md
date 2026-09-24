# Pipeline versus separate-stage benchmark

This benchmark compares one persistent GPU worker running xPOIS, xFit and
XScan with three fresh processes that exchange intermediate arrays through
files. Both treatments use the same device numerical APIs, inputs,
checkpoint and per-image candidate batches. The comparison measures the
combined cost of process lifetime, transfers and intermediate artifacts.

The separate-stage treatment uses benchmark-specific entry points. It does
not invoke the stock component CLI commands: those use different host
materialization, coefficient-solving and probability-calculation paths.

## Run the comparison

Use a CUDA 13 GPU with both CuPy and PyTorch available. From the repository
root:

```bash
uv sync --locked --extra gpu

uv run --locked --extra gpu cuphoton xscan benchmark-pipeline \
  --output /tmp/cuphoton-pipeline-forward \
  --images 4 --image-size 256 --candidates 9 --stamp-size 17 \
  --seed 2026 --device cuda:0 --warmup 1 --repeat 3 \
  --order pipeline-first
```

Each output directory must be new. The default fixture contains four
independent image pairs and nine candidates per image, so each measured
round processes four pairs and 36 candidate stamps. Candidates are supplied
positions; this benchmark does not include source detection. All images
run serially on the selected GPU.

Reverse the treatment order using the *same generated fixture*:

```bash
uv run --locked --extra gpu cuphoton xscan benchmark-pipeline \
  --output /tmp/cuphoton-pipeline-reverse \
  --config /tmp/cuphoton-pipeline-forward/input/config.json \
  --items /tmp/cuphoton-pipeline-forward/input/items.json \
  --warmup 1 --repeat 3 --order staged-first
```

`--config` and `--items` must be supplied together. They contain
`DevicePipelineConfig.to_payload()` and an ordered list of
`DevicePipelineItem.to_payload()` values, including absolute file paths and
hashes. With these options, the manifests determine the device and workload;
fixture-generation options do not replace them. Preserve the referenced
files when running the reversed comparison. Use the same CPU affinity,
thread settings and GPU, without another workload running concurrently.
`--timeout` bounds each child invocation; its default is 600 seconds.

## What each treatment runs

The pipeline initializes `DeviceWorkerContext` once, performs its warmup
rounds, then retains the model and CUDA context for measured rounds. Each
item flows through constant-kernel subtraction, stamp extraction, Gaussian
difference fitting, xFit feature conversion and model inference. Device
arrays pass to PyTorch through DLPack. The result contains compact scientific
evidence and predictions.

The separate-stage treatment starts one xPOIS child, one xFit child and one
XScan child per complete round. Each child processes all image items in
manifest order, keeping the candidate batch for each image unchanged. It
writes lossless, uncompressed NPY arrays between stages. Those files include
full subtraction images and fit residuals in addition to the compact
scientific outputs shared with the pipeline. Their transfers, hashing and
I/O are part of this treatment's cost. Its preliminary rounds prepare
caches; subsequent measured rounds still create fresh processes.

To inspect a manual round directly, invoke the command three times with
`--stage xpois`, `--stage xfit` and `--stage xscan`, in that order. Supply the
same `--config`, `--items` and `--output` directory each time. The stage
commands reject changed upstream artifacts and refuse to overwrite a stage.

Both treatments use float64 subtraction and xFit inputs, unweighted xFit
stamps, float32 triplets/features, and the same GPU sigmoid. Inference forces
AMP, TF32, compilation and cuDNN benchmarking off. An xPOIS variance plane
does not become an xFit variance plane.

## Read the timers

| Measurement | Boundary |
| --- | --- |
| Pipeline `setup_seconds` / `context_load_seconds` | Worker setup and context/model initialization, recorded separately from numerical warmup and measured batches. The external invocation also includes interpreter startup. |
| Pipeline `batch_seconds` | One ordered image batch through completed device work and per-item result JSON writes. Measured batches reuse the initialized, warmed worker. |
| Pipeline `invocation_external_seconds` | Parent-observed process duration, including imports, setup, all warmup and measured rounds, final summary and process exit. |
| Staged `batch_seconds` | Parent clock before launching xPOIS through successful XScan process exit, including all three process lifetimes and intermediate files. |
| Staged per-process `external_seconds` | Parent-observed duration of that individual child, including imports and shutdown. |

Stage summaries also record internal setup, reads, uploads, computation,
downloads and writes. Their GPU operations synchronize explicitly; the
pipeline's existing component timers are asynchronous host elapsed times,
with pending work completed at the terminal copy. Comparing those component
timers as isolated GPU kernel durations would be misleading. Use the
completed batch timers for the main workflow comparison.

The report's `fresh_stages_over_warm_pipeline_ratio` divides the median
staged batch time by the median warm pipeline batch time. It includes
repeated startup and file costs in the staged treatment. Keep setup and
whole-invocation measurements alongside that ratio when discussing a service
that may process only a few batches.

The benchmark preserves existing filesystem and CUDA caches. Fresh
processes can benefit from both. Writes close files without `fsync`, so
elapsed time does not measure durable storage completion. Fixture preparation
and the final parity audit run outside the treatment timers.

## Acceptance and artifacts

The benchmark checks every warmup and measured item against the other
treatment. Acceptance requires exact equality of all 22 compact scientific
arrays, including shapes, dtypes and NaN locations, plus matching candidate
identity/order, fit metadata, subtraction diagnostics and predictions. It
verifies retained file hashes and rechecks the original inputs. This checks
equivalence between two compositions of the same algorithms; it does not
independently establish their astronomical accuracy. Full intermediate
images are retained by the staged treatment but are outside the pipeline's
compact parity contract.

`manifest.json` records the configuration, ordered input descriptors and
runtime settings. Treatment subdirectories retain each attempted round and
child log. `parity.json` records numerical comparisons; a successful
`report.json` contains all round times and the measured medians. A failure
stops the run and leaves its logs and `failure.json` for inspection. A
successful process exit alone does not satisfy the parity check.

The default inputs are synthetic textured images with planted dipoles, a
known convolution kernel, noise and an exclusion mask around the candidates.
The fitted 15×15 kernel uses Gaussian sigmas 1.5/3/6 and degrees 2/1/0,
a constant background and no flux constraint.
A small, randomly initialized triplet model has a nonzero xFit fusion branch.
A separate CPU fit creates the canonical feature-schema artifacts before
measurement. Input arrays and model weights repeat for the same seed;
provenance paths and timestamps can change artifact hashes across newly
prepared fixtures. This fixture exercises the workflow and numerical
boundaries. It provides no trained-classifier accuracy result or claim of
representative production throughput.
