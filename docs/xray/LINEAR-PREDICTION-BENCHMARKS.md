# Running and interpreting XRay benchmarks

The XRay benchmark commands compare individual stages on synthetic traces.
Use them to investigate a workload before measuring the full detector path.
Keep generated JSON and profiling output outside the checkout, as described
in [Performance records](GPU-FIRST.md#performance-records).

## Choose a stage

| Command | Timed work | CPU reference |
| --- | --- | --- |
| `linear-prediction-benchmark` (`lpb`) | P1: Hankel construction, SVD, prediction coefficients, and roots | NumPy loop over traces |
| `linear-prediction-p2-benchmark` (`lppb`) | P2: least squares and reconstruction using modes selected before timing | NumPy loop over traces |
| `linear-prediction-savgol-benchmark` (`lpsb`) | Savitzky-Golay smoothing | One SciPy call on the two-dimensional batch |

`synthetic_trace_batch` generates rows on the host, varying frequencies,
phases, and offsets by row. These paths use float64. P2 selects its fixed
decays and frequencies from the first row before timing; it does not measure
mode selection or a complete fit for each row.

## Run a comparison

Install the CUDA 13 environment described in [Environment setup](ENVIRONMENT.md).
For example, capture one run of each stage with explicit settings:

```bash
mkdir -p /tmp/cuphoton-xray-benchmarks
uv run --locked --extra gpu cuphoton xray lpb \
  --samples 96 --traces 16 --components 8 --roots-backend eigvals \
  --repeat 3 --json > /tmp/cuphoton-xray-benchmarks/p1.json
uv run --locked --extra gpu cuphoton xray lppb \
  --samples 96 --traces 16 --components 8 --repeat 3 \
  --json > /tmp/cuphoton-xray-benchmarks/p2.json
uv run --locked --extra gpu cuphoton xray lpsb \
  --samples 96 --traces 16 --window-length 11 --polyorder 3 --repeat 3 \
  --json > /tmp/cuphoton-xray-benchmarks/savgol.json
```

For a CPU-only smoke check, omit `--extra gpu` and add `--no-gpu` to each
command. Use separate output locations when changing settings. Increase
`--traces` or `--samples` to investigate batch size and trace length; keep
the requested component count explicit in comparisons.

## Timing boundaries

CPU timings use `perf_counter` and report the minimum of `--repeat` calls.
There is no separate CPU warm-up call. GPU timings exclude the initial bulk
input upload and final result download. Each GPU path runs once before
timing, then each measured call is bracketed by
`cupy.cuda.Stream.null.synchronize()`. Internal scalar transfers and
synchronization remain in the measured interval.

The GPU serial path calls the operation for each trace; the batched path
passes all rows in one call. P1's batched comparison is available with
`--roots-backend eigvals`. The `roots` option compares serial paths only;
availability depends on the backend and library version. None of these
timers includes HDF5 loading or the complete detector workflow.

The reported minimum is a best observed time, not a median or a measure of
run-to-run variability. These commands do not retain the individual repeat
times. A performance study that needs a distribution must capture repeated
measurements explicitly; do not label the existing minimum as a median.

## Read the JSON

Check `gpu_error` and the GPU timing fields first. A missing GPU result is
not evidence of a successful comparison. With `--no-gpu`, GPU timings are
unset intentionally.

`gpu_batch_speedup` is GPU serial time divided by GPU batched time. It says
how much batching helps the GPU path. To compare with the CPU, divide
`cpu_serial_best_s` by `gpu_batched_best_s`. Despite its name,
`cpu_serial_best_s` measures one batched SciPy call for smoothing. A ratio
above one means the GPU took less time; below one means the CPU took less
time. These ratios have different baselines.

Read numerical differences alongside timings: P1 reports coefficient and
eigenvalue differences, P2 reconstruction differences, and smoothing filter
differences. Small differences in coefficients or roots alone do not prove
that mode selection and the final scientific result are unchanged. Inspect
recovered modes and reconstructions before adopting an alternative solver.

Record the source revision, command, input shape, dtype, Python/package
versions, CPU and thread settings, GPU, driver, and CUDA runtime with the
result. The benchmark JSON contains stage timings and comparison fields;
it does not capture all of this environment information automatically.

## Investigate the P1 eigenvalue solve

P1 builds companion matrices whose size grows with trace length. In
[CuPy 14.1.1](https://github.com/cupy/cupy/blob/v14.1.1/cupy/linalg/_eigenvalue.py#L181-L192),
`cupy.linalg.eigvals` loops over the matrices of a three-dimensional input
and invokes `cusolver.xgeev` once per matrix. Passing a batch therefore
does not imply one batched eigensolver operation. Profile the installed
version and workload to determine whether this stage dominates elapsed time.

The existing `subspace-benchmark` and `subspace-acceptance` commands compare
experimental matrix-pencil and ESPRIT methods with linear prediction. Their
reduced eigenproblems use the chosen subspace order. They change the estimator
and require accuracy and mode-recovery checks; they are not interchangeable
root-solving backends. Inspect their options with:

```bash
uv run --locked cuphoton xray help subspace-benchmark
uv run --locked cuphoton xray help subspace-acceptance
```
