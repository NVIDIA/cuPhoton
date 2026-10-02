# Nonlinear refinement of linear-prediction modes

These experimental, opt-in tools refine linear-prediction modes by nonlinear
least squares. `cuphoton.xray.mode_refinement` fits one trace with SciPy;
`cuphoton.xray.mode_refinement_batched` fits a batch on NumPy or CuPy. Both use
an analytic Jacobian for the model

```
trace(t) = constant + sum_k amplitude_k exp(-decay_k t) cos(angular_frequency_k t + phase_k)
```

The fits are unconstrained: decay can become negative, and frequencies have
no bounds. They are separate from the existing production
`cuphoton.xray.iterative_fit` solver, which uses positive decay and bounded
frequencies. These tools have no detector or distributed pipeline integration.

## Per-trace refinement

`linear_prediction_refined(time, trace, n_components, n_modes)` runs
`linear_prediction_numpy`, keeps the `n_modes` strongest oscillating modes as
the starting point, and refines them and the constant with
`scipy.optimize.least_squares` (Levenberg-Marquardt, `xtol` and `ftol`
1e-12). When the linear-prediction step returned fewer than `n_modes`
oscillating modes, the missing ones are seeded one at a time from the
strongest peak of the residual spectrum (amplitude from the peak height, zero
decay and phase) and refined before the next seed. `refine_modes` is the
same solver from caller-supplied starting modes.

`RefinedModes` carries the refined parameters ordered by amplitude, local
one-sigma uncertainty estimates, the reconstruction, the residual rms, the
solver status, and how many modes came from linear prediction
(`initial_mode_count`) versus the residual spectrum (`seeded_mode_count`).
The uncertainty calculation uses an SVD of the column-normalized Jacobian
and the estimated residual variance. A rank-deficient Jacobian produces
`NaN` uncertainties. Finite estimates describe the local fit under the model
and independent, equal-variance noise assumptions; they do not establish
identifiability away from that solution or account for model mismatch.

A large residual can result from model mismatch, a poor local solution, or
failure to converge. Check the solver status and the reconstruction before
interpreting fitted parameters or uncertainties.

`linear-prediction-validate --refine` runs the validation sweep a second time
with this estimator (`estimator: cpu-refined` in the summary). Compare mode
loss, bias, scatter, and residuals across noise levels and distortions.
Results depend on the fixture, initialization, and noise realization;
refinement does not guarantee recovery of every mode or attainment of the
Cramer-Rao bound.

```bash
uv run cuphoton xray lpv --refine --snr-db 40,20,10,5 --trials 200 --output-dir /tmp/lpv-refined
uv run cuphoton xray lpv --refine --distortion chirp:0.05 --snr-db 30
uv run python examples/xray_lp_modes_review.py --output /tmp/review.html
```

The example writes a Bokeh review page (requires the `viz` extra) with the
trace, the true model, the linear-prediction and refined reconstructions,
and the recovered modes against the true ones, for several noise levels and
an optional distortion.

## Batched refinement

`refine_modes_batched(xp, time, traces, theta0, n_modes)` takes traces shaped
`(B, N)` and starting parameters `(B, 4K+1)` in the same layout as the
per-trace solver (amplitude, decay, angular frequency, phase per mode, then
the constant). It normalizes the Jacobian columns and solves the damped
normal equations as a batch of small `(4K+1)` systems with `xp.linalg.solve`.
A step that lowers the cost is accepted and its damping divided by 3;
otherwise the damping is multiplied by 5 and that trace keeps its parameters.

Convergence requires the largest absolute component of the
column-normalized Jacobian's residual gradient, divided by
`max(residual norm, 1)`, to be at most `tol` (default 1e-9). This criterion
is independent of the choice of time units. A tiny damped step alone does
not establish convergence. Iteration stops when every trace has converged
or after `max_iter` (default 60). Check each trace's convergence flag;
stationarity does not guarantee a good fit or agreement with another solver.

At tight tolerances, rounding can prevent further cost reduction before the
gradient criterion is met. Such traces remain marked unconverged even when
their parameters agree closely with another fit. Choose a tolerance suited
to the required accuracy and inspect residuals as well as convergence flags.

## Benchmark

`linear-prediction-refine-benchmark` (`lprb`) builds `--traces` noisy copies
of the fixture (two modes, 96 samples, noise sigma 0.02), perturbs the true
parameters per trace to make identical starting points, and times three
paths: the SciPy solver in a Python loop over traces, the batched solver on
NumPy, and the batched solver on CuPy. Each path is timed `--repeat` times
(default 3) and the best is reported. For the batched paths the clock covers
the solver only: the arrays are on the device before the clock starts and
the device is synchronised before and after, so host-to-device transfer and
the copy of the result back are not included. One untimed CuPy call on up to
16 traces runs first to warm the device path. The command also
reports the largest parameter difference between each batched result and
the SciPy result.

```bash
uv run cuphoton xray lprb --traces 256 --repeat 3 --json
uv run cuphoton xray lprb --traces 2048 --repeat 3 --json
uv run cuphoton xray lprb --traces 8192 --repeat 3 --json
```

Use `--no-gpu` for the CPU comparison alone. These timings cover refinement
from supplied starts and exclude linear-prediction initialization, input I/O,
and detector processing. They cannot establish an end-to-end detector
speedup. Performance and parameter agreement need measurement on the target
hardware and representative traces.
