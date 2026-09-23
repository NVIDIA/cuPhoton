# XRay

`cuphoton.xray` provides X-ray trace extraction, linear prediction, optional
iterative fitting, detector artifact generation, numerical validation, and
standalone review views. It is available through the `cuphoton xray` command
group.

XRay is GPU-first for high-throughput detector work. Supported operations fall
back to NumPy for CPU smoke and correctness runs; commands that require a GPU
report that requirement explicitly.

## Install and smoke test

```bash
# CUDA 13
uv sync --locked --extra dev --extra gpu --extra viz
uv run cuphoton xray doctor
uv run python examples/run_quickstarts.py --component xray --require-gpu

# CPU
uv sync --locked --extra dev --extra viz
uv run python examples/run_quickstarts.py --component xray --profile cpu
```

## Command groups

| Goal | Commands |
| --- | --- |
| Environment | `doctor`, `gpu-policy` |
| Input inspection | `data-probe`, `roi-candidates` |
| Trace work | `extract-trace`, `trace-smoke` |
| Linear prediction | `linear-prediction-*`, `prediction-roots-benchmark`, `model-order-sweep` |
| Detector products | `detector-mask`, `detector-artifacts`, `detector-artifact-normalize`, `detector-artifact-compare` |
| Distributed detector work | `detector-artifact-distributed`, `detector-artifact-merge` |
| Review | `report`, `validation-viz`, `workflow-viz`, `phonon-viz` |

List the complete current command surface with:

```bash
uv run cuphoton xray --help
uv run cuphoton xray help extract-trace
```

## Single-node HDF5 workflow

XRay recognizes two on/off cube layouts. Both files in a pair must use the
same schema and array shapes.

| Schema | Image cube | Delay axis | Entry counts | Normalization |
| --- | --- | --- | --- | --- |
| `cropped-cube` | `imgs` `(frame, y, x)` | `scan_var` `(frame,)` | `bin_count` `(frame,)` | `i0` and `i0_ipm3` `(frame,)`; `ROI` is also required |
| `legate-cube` | `jungfrau1M_data` `(frame, y, x)` | `binVar_bins` `(frame,)` | `nEntries` `(frame,)` | `ipm3__sum`/`ipm2__sum`, or `ipm5__sum`/`ipm4__sum`, each `(frame,)` |

Probe first, extract a small row batch, and inspect model-order behavior before
running the detector-wide GPU path:

```bash
uv run cuphoton xray data-probe \
  --h5dir /path/to/input --fon on.h5 --foff off.h5 --json

uv run cuphoton xray extract-trace \
  --h5dir /path/to/input --fon on.h5 --foff off.h5 \
  --roi-lower 0 0 --roi-dim 64 64 --row-y 0 --row-y 16 \
  --output-dir /path/to/traces --output-prefix sample --json

uv run cuphoton xray model-order-sweep \
  --trace-dir /path/to/traces --min-components 2 \
  --max-components 12 --step 2 --json

uv run cuphoton xray detector-artifacts \
  --h5dir /path/to/input --fon on.h5 --foff off.h5 \
  --output-dir /path/to/artifacts --roi-lower 0 0 --roi-dim 64 64 \
  --zero-offset-index 0 --components 6 \
  --fit-diagnostics summary --json
```

`--zero-offset-index 0` is illustrative; select a physically appropriate fit
start for the input scan. Start with a representative ROI and inspect the
manifest, fit-status array, and numerical outputs before scaling out.

For Python A/B checks, pass `batch_rows=False` to
`cuphoton.xray.detector_artifacts.build_detector_artifacts_cupy` to force
the serial row loop. The default, `True`, batches eligible rows. The manifest
records this choice, and serial runs have distinct configuration and resume
identities. Frequency, amplitude, and FFT-frequency arrays match exactly in
the batch-versus-row regression cases; `fft_all` can differ by rounding from
the batched cuFFT plan (the normalized-trace tests use `atol=1e-15`).

`--fit-diagnostics summary` writes one status-aware record for each processed
tile row, without duplicating records across the tile's x pixels. The summary
includes the scale-free residual ratio and P1/P2 conditioning. `full` also
retains the fitted time axis, traces, reconstructions, and flattened modal
arrays; use it only for bounded studies. `fit_status.npy` remains an execution
status product, and no diagnostic ratio is interpreted as a scientific
acceptance threshold by the command. P2 rank, singular values, and condition
describe the unregularized P2 design even when the recorded fit uses ridge;
they do not describe the augmented ridge system.

P2 ridge fitting is opt-in through `--p2-ridge-alpha`; its default of zero
uses the unregularized least-squares path. Treat a nonzero value as an
experiment configuration that requires independent validation, not as a
general-purpose default.

For linear prediction, detector artifact manifests use version 2 and record
the diagnostics level and ridge alpha in the resume identity. Version 1 manifests
still load, but shards written before this change no longer match the resume
identity, so a resumed linear-prediction run recomputes them once and rewrites
them as version 2. Iterative fitting uses manifest version 3.

## Optional iterative fitting

Use `iterative_fit` to fit a fixed number of damped cosine modes plus a
constant offset. Linear prediction remains the default for detector work.
Start with a small synthetic trace:

```python
import numpy as np

from cuphoton.xray.iterative_fit import IterativeFitOptions, iterative_fit

time = np.linspace(0.0, 20.0, 241)
trace = 0.2 + 2.0 * np.exp(-0.08 * time) * np.cos(2 * np.pi * 0.3 * time)
fit = iterative_fit(
    time, trace, components=1,
    options=IterativeFitOptions(max_frequency=1.0),
)
print(fit.converged, fit.status, fit.relative_residual)
print(fit.angular_frequency / (2 * np.pi))
```

The default API backend is `cpu`. Pass `backend="gpu"` for CuPy iterations;
initialization runs on the CPU and an explicitly requested GPU must be
available. Both paths return NumPy arrays. Time samples must be finite,
increasing and uniformly spaced (allowing float32 rounding of delay metadata).
Frequency bounds and detector frequency arrays use cycles per input time
unit. The API's `angular_frequency` uses radians per input time unit;
divide by `2 * np.pi` to obtain cycles per time unit, as in the example.
Decay rates use inverse time units, and phases refer to the first fitted
sample. For time measured in picoseconds, cycles per time unit are THz.

This method uses a Levenberg-Marquardt iteration with analytic derivatives.
Positive decay rates use a logarithmic parameter; an arctangent transform
keeps frequency inside the configured bounds. Automatic initial guesses
come from the trace's spectrum. The model has `4 * components + 1`
parameters; choose a small mode count relative to the available samples.
It can converge to a local minimum, especially for overlapping modes or
poorly resolved frequencies.

Frequency bounds apply to the signed internal frequency. A negative
`min_frequency` can move the lower optimization boundary away from zero.
Returned modes use nonnegative frequencies and amplitudes, with phases
adjusted to preserve the reconstructed signal.

`amplitude_l2` adds an amplitude penalty to the objective:
`0.5 * sum((reconstruction - trace)**2) + amplitude_l2 * sum(amplitude**2)`.
The default is zero. A nonzero value changes the scientific fit and must
be selected for the dataset. `chi2` is the mean squared reconstruction
residual without this penalty. A convergence flag describes the optimizer;
it does not establish that the recovered modes are physically correct.

Inspect `status` when a fit does not converge. `iteration-limit` means the
iteration budget was exhausted. `stalled` means repeated trial steps failed
to improve the objective and exhausted the damping range before reaching
the stationarity tolerance. `nonfinite` means this rejection limit was
reached with a nonfinite trial. These results retain the best finite fit
for inspection. Increasing `max_iterations` only addresses the iteration
budget; a stalled fit requires examining the trace, model and initialization.

For detector artifacts, add these options to the HDF5 example above:

```text
--fit-method iterative --components 2 --iterative-max-frequency 1.0
--iterative-max-iterations 600 --fit-diagnostics full
```

Detector fitting uses CuPy and the existing normalization and smoothing
pipeline. The iterative method returns modal center frequencies directly.
The amplitude threshold still controls the displayed signal selection.
Failed convergence counts against the detector's fit-failure budget.
Use a bounded region and inspect reconstructions and recovered modes before
running a whole detector.

The implementation follows the damped-cosine model and LM update developed
by Irina Demeshko, with fitting-workflow contributions from Malte Foerster.
It provides NumPy/CuPy numerical paths with explicit controls. Compare
representative traces against linear prediction and expected physical
modes for each experiment.

## Input and artifact boundary

Pass HDF5, trace NPZ, detector-array, and output paths explicitly. The package
does not select a data root. Keep source datasets, detector dumps, generated
reports, dashboards, and benchmark logs outside the repository.

Before a long run, use `data-probe` to confirm the HDF5 schema and on/off shape
agreement. Preserve the ROI, excluded rows, normalization fields, fit
parameters, package revision, backend/device, and GPU/driver with the result.

See [Data and artifact contracts](../data-artifacts.md#xray-hdf5-and-trace-products).

## Detailed guides

- [Environment setup](ENVIRONMENT.md)
- [GPU-first behavior](GPU-FIRST.md)
- [Distributed detector artifacts](DISTRIBUTED-DETECTOR-ARTIFACTS.md)
- [Validation visualization](VALIDATION-VIZ.md)
