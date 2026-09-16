# xRay

`cuphoton.xray` provides X-ray trace extraction, linear prediction, optional
iterative fitting, detector artifact generation, numerical validation, and
standalone review views. It is available through the `cuphoton xray` command
group.

xRay is GPU-first for high-throughput detector work. Supported operations fall
back to NumPy for CPU smoke and correctness runs; commands that require a GPU
report that requirement explicitly.

## Install and smoke test

```bash
# CUDA 13
uv sync --locked --extra dev --extra gpu --extra viz
uv run cuphoton xray doctor
uv run python examples/run_quickstarts.py --component xray --require-gpu \
  --output-dir /tmp/xray-gpu-quickstart

# CPU
uv sync --locked --extra dev --extra viz
uv run python examples/run_quickstarts.py --component xray --profile cpu \
  --output-dir /tmp/xray-cpu-quickstart
```

Choose a new output directory for each quickstart run.

## Command groups

| Goal | Commands |
| --- | --- |
| Environment | `doctor`, `gpu-policy` |
| Input inspection | `data-probe`, `roi-candidates` |
| Trace work | `extract-trace`, `trace-smoke` |
| Linear prediction | `linear-prediction-*`, `prediction-roots-benchmark`, `model-order-sweep` |
| Synthetic validation | `linear-prediction-validate` (`lpv`), `linear-prediction-refine-benchmark` (`lprb`) |
| Detector products | `detector-mask`, `detector-artifacts`, `detector-artifact-normalize`, `detector-artifact-compare` |
| Distributed detector work | `detector-artifact-distributed`, `detector-artifact-merge` |
| Review | `report`, `validation-viz`, `workflow-viz`, `phonon-viz` |

List the complete current command surface with:

```bash
uv run cuphoton xray --help
uv run cuphoton xray help extract-trace
```

## Single-node HDF5 workflow

xRay recognizes two on/off cube layouts. Both files in a pair must use the
same schema, matching array shapes and matching delay coordinates.

| Schema | Image cube | Delay axis | Entry counts | Normalization |
| --- | --- | --- | --- | --- |
| `cropped-cube` | `imgs` `(frame, y, x)` | `scan_var` `(frame,)` | `bin_count` `(frame,)` | `i0` and `i0_ipm3` `(frame,)`; `ROI` is also required |
| `legate-cube` | `jungfrau1M_data` `(frame, y, x)` | `binVar_bins` `(frame,)` | `nEntries` `(frame,)` | `ipm3__sum`/`ipm2__sum`, or `ipm5__sum`/`ipm4__sum`, each `(frame,)` |

Image sums are divided by `i0` for `cropped-cube`, or by `ipm2__sum`
(`ipm4__sum` for the alternate pair) for `legate-cube`. Delay coordinates
retain the input file's units. ROI origins use `(x, y)` and dimensions use
`(width, height)`; `--row-y` selects an absolute detector row within the ROI.

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
  --fit-diagnostics summary --artifact-layout tile-rows --json
```

`--zero-offset-index 0` is illustrative; select a physically appropriate fit
start for the input scan. The index addresses the delay axis after
`--drop-leading` (default: 1). Detector fitting also drops the last sample by
default (`--fit-trailing-drop 1`) and needs at least 16 samples in the fitted
window. Start with a representative ROI and inspect the manifest, fit-status
array, and numerical outputs before scaling out.

For each detector row, the worker fits one trace per x tile and broadcasts
the result across that tile's columns. `--tile-shape` defaults to `16 16`;
`--integrate` controls the neighboring-row integration radius (default: 3).
These settings change the spatial support and amplitude of the traces.
The extracted traces above select individual rows without that integration,
so compare them with detector fits only after matching the normalization,
smoothing, spatial integration and fitted sample window.

The example uses `--artifact-layout tile-rows` to store each spectrum once
per x tile.
The four spectral files then use the suffix `.tile-rows.npy`, with logical
shape and x boundaries in `spectral-layout.json`. This is lossless: fitted
values, masks, and pixel coordinates remain identical. A full tile with
width 16 stores one sixteenth of the spectral values. The default `dense`
layout writes the usual `.npy` arrays for direct NumPy consumers. Dense
remains the compatibility default because compact storage changes filenames
and requires the shared loader. Select the output format explicitly for each
run; pass `--artifact-layout dense` to request the original layout.

The equivalent GPU producer call from Python is:

```python
from cuphoton.xray.detector_artifacts import build_detector_artifacts_cupy

result = build_detector_artifacts_cupy(
    h5dir="/path/to/input",
    fon="on.h5",
    foff="off.h5",
    output_dir="/path/to/artifacts",
    roi_lower=(0, 0),
    roi_dim=(64, 64),
    zero_offset_index=0,
    components=6,
    fit_diagnostics="summary",
    artifact_layout="tile-rows",  # Use "dense" for pixel-shaped NPY files.
)
print(result.manifest_path)
```

Comparison, visualization, resume, and distributed merging support both
layouts. Compact shards remain compact when merged. Use the shared loader
for bounded pixel slices from either layout:

```python
from cuphoton.xray.detector_storage import load_detector_array

amplitudes = load_detector_array("/path/to/artifacts/amp_all.npy")
column = amplitudes[:, 12, :]  # Only this column is expanded.
roi = amplitudes[0:8, 8:16, :32]  # (y, x, spectral sample)
```

Pass the logical filename `amp_all.npy` even when the stored file is
`amp_all.tile-rows.npy`. Keep `spectral-layout.json` with the spectral files;
the loader uses it to reconstruct pixel positions. Indices are local to the
saved ROI. For absolute detector coordinates, subtract the manifest's
`roi_lower` origin, recorded as `(x, y)`, before indexing in `(y, x)` order.

The compact reader supports integer and slice indexing, including negative
indices and steps. Slice first to obtain NumPy arrays for downstream consumers.
Implicit conversion such as `np.asarray(amplitudes)` or `np.nanmax(amplitudes)`
raises `TypeError`; apply NumPy operations to a selected slice instead.
Request bounded slices when exporting dense pixel data; `amplitudes[:, :, :]`
expands the full logical cube and its repeated columns in memory. Compact
storage reduces spectral bytes written, but full expansion can take longer
than reading dense output. Choose dense when direct NumPy access or repeated
full-cube reads matter more than artifact size.

For Python A/B checks, pass `batch_rows=False` to
`cuphoton.xray.detector_artifacts.build_detector_artifacts_cupy` to force
the serial row loop. The default, `True`, batches eligible rows. The manifest
records this choice, and serial runs have distinct configuration and resume
identities. Batch-versus-row regression checks preserve array shapes, dtypes,
finite masks, zero padding, mode positions, fit status and FFT frequencies
exactly. Fitted frequencies, amplitudes and filtered amplitude sums can differ
by floating-point rounding, as can the batched cuFFT output. The normalized
trace fixtures use `rtol=0`, with `atol=5e-15` for fitted outputs and
`atol=1e-15` for `fft_all`; these bounds describe those fixtures rather than
arbitrary input scales.

`--fit-diagnostics summary` writes one status-aware record per detector row
within each processed tile, covering the tile's `tile_x_start` to
`tile_x_stop` x range (exclusive of the stop). The summary includes the
scale-free residual ratio and P1/P2 conditioning. `full` also
retains the fitted time axis, traces, reconstructions, and flattened modal
arrays; size the study region to fit the resulting sidecar in memory and
storage. `fit_status.npy` records execution status. Choose scientific
acceptance thresholds for the diagnostic ratios as part of the study.
P2 rank, singular values, and condition describe the unregularized P2 design,
including when the recorded fit uses ridge.

P2 ridge fitting is opt-in through `--p2-ridge-alpha`; its default of zero
uses the unregularized least-squares path. Validate a nonzero value
independently for the experiment's data and acceptance criteria.

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
parameters and requires at least that many samples; choose a small mode count
relative to the available samples.
It can converge to a local minimum, especially for overlapping modes or
poorly resolved frequencies.

`max_frequency=None` uses the sampling Nyquist frequency, `0.5 / sample_spacing`.
Explicit bounds must satisfy `-Nyquist <= min_frequency < max_frequency <= Nyquist`
with a positive upper bound. Frequency bounds apply to the signed internal
frequency. A negative `min_frequency` can move the lower optimization
boundary away from zero.
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

Detector normalization, smoothing and FFTs use CuPy. Sequential iterative
row fits run on the CPU to avoid per-iteration GPU synchronization; this
does not change the standalone API's explicit GPU option. The iterative
method returns modal center frequencies directly.
The amplitude threshold still controls the displayed signal selection.
Failed convergence counts against the detector's fit-failure budget
(`--max-fit-failures`, default: 0). A nonzero `--p2-ridge-alpha` applies only
to linear prediction and cannot be combined with iterative fitting.
Use a bounded region and inspect reconstructions and recovered modes before
running a whole detector.

The implementation follows the damped-cosine model and LM update developed
by Irina Demeshko, with fitting-workflow contributions from Malte Foerster.
It provides NumPy/CuPy numerical paths with explicit controls. Compare
representative traces against linear prediction and expected physical
modes for each experiment.

## Input and artifact boundary

Pass HDF5, trace NPZ, detector-array, and output paths explicitly to select
the data locations for each run. Keep source datasets, detector dumps, generated
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
- [Linear prediction validation](LINEAR-PREDICTION-VALIDATION.md)
- [Nonlinear refinement of linear-prediction modes](LINEAR-PREDICTION-REFINEMENT.md)
