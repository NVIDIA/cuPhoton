# Linear prediction validation on synthetic damped modes

`linear-prediction-validate` (`lpv`) checks the linear-prediction fit against
a known answer. It builds traces from a small set of damped sinusoids, adds
white Gaussian noise at chosen signal-to-noise levels, runs the fit many
times, and compares the scatter and bias of every recovered parameter with
the Cramer-Rao lower bound for the same model. It also counts how often a
true mode is missing from the fit. No external data is needed.

```bash
uv run cuphoton xray lpv --snr-db 40,30,20,10 --trials 200 --output-dir /tmp/lpv
uv run cuphoton xray lpv --distortion chirp:0.05 --snr-db 30 --output-dir /tmp/lpv-chirp
uv run cuphoton xray lpv --json
```

Keep `--output-dir` outside the checkout. It receives `summary.json` and, when
the `viz` extra is installed, `validation.html`. The HTML includes its Bokeh
resources and can be viewed offline.

## Model and fixture

The trace is

```
trace(t) = constant + sum_k amplitude_k exp(-decay_k t) cos(angular_frequency_k t + phase_k)
```

sampled at `samples` points from `t = 0` to `duration`. Time is in the input
time unit; angular frequency is in radians per time unit and decay in inverse
time units. The default fixture has two modes, `(1.25, 0.09, 2.4, 0.30)` and
`(0.45, 0.03, 0.9, -0.75)` as `(amplitude, decay, angular_frequency, phase)`,
constant `0.15`, 96 samples over 9.5 time units. With zero noise it is the
same trace as `synthetic_trace`, and the fit recovers the modes to better than
`1e-6`. `cuphoton.xray.synthetic_validation.synthetic_modes_trace` builds the
fixture; `DampedMode` describes a mode.

Noise is white Gaussian, independent per sample. A level's `snr_db` is
`20 log10(rms of the mean-removed clean trace / sigma)`. One `numpy` generator
seeded by `--seed` draws every trial in order, so a run is reproducible from
the summary's `config`.
SNR levels must be finite and produce finite, positive noise scales;
`--components` and `--trials` must be positive integers.

## Bounds

`cramer_rao_bounds` forms the Fisher information `J^T J / sigma^2` from the
numerical Jacobian of the model at the true parameters (all parameters free,
including the constant) and returns the square root of the diagonal of its
inverse per parameter. These are standard-deviation bounds; the summary
also carries the squared value as `crlb_variance`. For the single undamped
sinusoid test fixture, the angular-frequency bound matches the square root
of the closed-form variance `24 sigma^2 / (A^2 dt^2 N (N^2 - 1))`
(Kay, Estimation Theory, 1993) to within 3 percent. The numerical calculation
also frees the decay and constant.

## Mode matching and statistics

A true mode is recovered in a trial when a fitted mode has an angular
frequency within `match_tolerance` (default 0.3 rad per time unit) of it;
the assignment first maximizes the number of recovered modes, then minimizes
their total frequency error. Each fitted mode is used at most once. The
per-mode `loss_rate` is the fraction of trials in which the mode was not
recovered. A trial is successful when the estimator returned and every true
mode was recovered; a trial in which the estimator raised counts in
`estimator_errors` and as a loss of every mode. The text report includes
the error count so estimator failures can be distinguished from mode loss.

For each mode and parameter (`amplitude`, `decay`, `angular_frequency`,
`phase`) the summary gives `bias`, `std` (ddof 1) and `rmse` computed over
the recovered trials only, the bound (`crlb_std`, `crlb_variance`) and
`std_over_crlb_std`. Truth and fitted modes use nonnegative amplitudes,
folding a negative amplitude into the phase. Both phases and phase errors
are wrapped to `[-pi, pi)`; the summary records truth in this convention.
Undefined statistics are written as JSON `null`: standard deviation requires
at least two recovered trials, and bias and RMSE require at least one.

Read the decay statistics of a lightly damped mode together with its
`loss_rate`. The root filter keeps decaying roots only, so when noise pushes
the fitted decay of such a mode through zero the mode is dropped rather than
reported with a negative decay; the surviving decay estimates are truncated
at zero, and their scatter and bias are conditional on survival. The summary
labels these statistics `statistics_over: recovered_trials`. Conditional
statistics and finite-sample scatter can fall below the unconditional bound.

Every level also reports `residual_rms_over_sigma_median`, the median over
trials of the rms residual of the reconstruction divided by sigma. It is
near 1 for an accurate reconstruction at the specified noise level. A large
value can indicate model mismatch, missed modes, or estimator error; it does
not by itself distinguish these causes.

## Model mismatch

`--distortion kind:amount` replaces the clean trace with one outside the
model class:

| kind | trace |
| --- | --- |
| `chirp` | every frequency rises by the fraction `amount` across the record |
| `gaussian_envelope` | the first mode decays as a Gaussian with 1/e time `amount` |
| `baseline_drift` | a linear ramp of `amount` across the record |
| `clip` | the trace is clipped above `amount` |
| `glitch` | one sample at mid-record is offset by `amount` |

Amounts must be finite. The Gaussian 1/e time must be positive and is measured
from the first sample.

The truth used for bias and bounds stays the undistorted mode set, so the
statistics measure the estimator's response to the mismatch. A chirp or a
clipped trace can return modes with a confident, biased frequency and no
loss; the residual ratio supplies an additional reconstruction diagnostic.

## Summary schema

`summary.json` has `schema_version` 1 and five top-level keys.

- `command`: `linear-prediction-validate`.
- `config`: the model string and units, the true modes and constant, the
  sampling (`samples`, `duration`, `dt`, `start`), the noise definition and
  levels, `seed`, `trials_per_level`, `n_components` (model order), the
  mode-matching criterion and tolerance, how the statistics are computed, and
  the estimators and distortions covered.
- `runtime`: `cuphoton.core.runtime.runtime_metadata` for the CPU backend
  (package, Python, platform, NumPy versions, backend, device, dtype) plus
  `scipy_version`, `source_revision`, and `source_dirty`. A revision is
  recorded only when this module is tracked by its Git checkout;
  `source_dirty` reports tracked changes anywhere in that checkout.
  Both are `null` for an installed or untracked module, or when Git metadata
  is unavailable. Ignored and untracked run artifacts do not mark the source
  dirty.
- `results`: one entry per sweep condition (estimator by distortion by
  signal-to-noise level): `snr_db`, `signal_rms`, `noise_sigma`, `trials` with
  `attempted`, `successful`, `failed` and `estimator_errors`,
  `any_mode_lost_rate`, `residual_rms_over_sigma_median`, and `modes`, each
  with `recovered_trials`, `loss_rate`, `statistics_over` and the four
  parameter blocks described above.
- `artifacts`: filenames relative to the summary, `summary` and `figure`
  (`null` when Bokeh is not installed).

The figure shows, per estimator and per parameter, the sample standard
deviation against the bound versus signal-to-noise, and the loss rates.
The signal RMS in `config.noise` describes the first sweep; each result
records its own signal RMS because distortions can change it.

## Python entry points

`cuphoton.xray.synthetic_validation` exposes `synthetic_modes_trace`,
`distort_trace`, `cramer_rao_bounds`, `match_modes`, `validation_sweep`,
`build_summary`, `write_validation_figure` and `write_validation_run`.
`validation_sweep` accepts another estimator callable with the signature
`estimator(time, trace, n_components)` returning an object with
`angular_frequency`, `decay`, `amplitude`, `phase` and `reconstruction`
(NumPy or CuPy arrays), so the same sweep can compare estimators.
`build_summary` combines sweeps with shared truth, sampling, seed, matching,
SNR levels, trial counts and model order. It rejects incompatible settings
instead of describing later sweeps with the first sweep's configuration.
Write separate summaries for changes to those settings.
