# Nonlinear refinement of linear-prediction modes

Linear prediction gives frequencies and decays without an initial guess, but
it is not a maximum-likelihood estimator and its root filter can drop a
lightly damped mode under noise. `cuphoton.xray.mode_refinement` refines the
same damped-mode model by nonlinear least squares from the linear-prediction
result; `cuphoton.xray.mode_refinement_batched` does the same for a batch of
traces at once on NumPy or CuPy. Both fit

```
trace(t) = constant + sum_k amplitude_k exp(-decay_k t) cos(angular_frequency_k t + phase_k)
```

with an analytic Jacobian. Nothing here changes the linear-prediction
commands or their defaults.

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

`RefinedModes` carries the refined parameters ordered by amplitude, one-sigma
uncertainties from the Jacobian at the solution and the residual variance,
the reconstruction, the residual rms, the solver status, and how many modes
came from linear prediction (`initial_mode_count`) versus the residual
spectrum (`seeded_mode_count`). A residual rms well above the noise means
the model class, not the fit, is wrong.

`linear-prediction-validate --refine` runs the validation sweep a second time
with this estimator (`estimator: cpu-refined` in the summary) so the two can
be compared level by level. Reference run on the default fixture, CPU,
default seed, 300 trials per level:

```bash
uv run cuphoton xray lpv --refine --trials 300 --snr-db 40,30,20,10,5
```

In that run the linear-prediction estimator loses the weak mode in 10 to 52
percent of trials depending on the level, and its frequency scatter is 1.1
to 3.8 times the bound. The refined estimator loses the weak mode in 1 of
300 trials at each of the two lowest levels of that run and in none above,
and its frequency and decay scatter are within 10 percent of the Cramer-Rao
bound at every level of that run (ratios 0.91 to 1.07; the sampling
uncertainty of a standard deviation from 300 trials is about 4 percent).
The refinement does not remove the bias a model mismatch produces, and the
residual ratio still flags that case.

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
the constant) and runs Levenberg-Marquardt for the whole batch: the Jacobian
for all traces by broadcasting, the normal equations as a batch of small
`(4K+1)` systems solved with `xp.linalg.solve`, and per-trace damping (a
step that lowers the cost is accepted and the damping divided by 3, otherwise
the damping is multiplied by 5 and the trace keeps its parameters). Iteration
stops when every trace has converged (step below `tol` times the parameter
norm, default 1e-9) or after `max_iter` (default 60). `xp` is `numpy` or
`cupy`. On convergence the result matches the per-trace SciPy solver to about
1e-8 in every parameter (table below).

The batch iterates until its slowest trace converges, so the NumPy batched
path is slower than the SciPy loop; the batched form pays off on the GPU.

## Benchmark

`linear-prediction-refine-benchmark` (`lprb`) builds `--traces` noisy copies
of the fixture (two modes, 96 samples, noise sigma 0.02), perturbs the true
parameters per trace to make identical starting points, and times three
paths: the SciPy solver in a Python loop over traces, the batched solver on
NumPy, and the batched solver on CuPy. Each path is timed `--repeat` times
(default 3) and the best is reported. For the batched paths the clock covers
the solver only: the arrays are on the device before the clock starts and
the device is synchronised before and after, so host-to-device transfer and
the copy of the result back are not included. One untimed CuPy call on 16
traces runs first to absorb compilation and allocation. The command also
reports the largest parameter difference between each batched result and
the SciPy result.

```bash
uv run cuphoton xray lprb --traces 256 --repeat 3 --json
uv run cuphoton xray lprb --traces 2048 --repeat 3 --json
uv run cuphoton xray lprb --traces 8192 --repeat 3 --json
```

### NVIDIA GeForce RTX 4050 Laptop GPU (Ada, compute capability 8.9, 6 GiB)

Host: Intel Core i9-13900H, WSL2 Ubuntu 24.04 on Windows 11, driver 596.49,
CUDA runtime 13.2, CuPy 14.1.1, Python 3.12, NumPy and SciPy from the locked
`dev` and `gpu` extras. Best of 3 repeats; the three commands above.

| traces | SciPy per trace, serial (s) | NumPy batched (s) | CuPy batched (s) | SciPy / CuPy | max abs parameter difference |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.135 | 0.258 | 0.108 | 1.3 | 8.7e-9 |
| 2048 | 1.09 | 2.49 | 0.162 | 6.7 | 1.2e-8 |
| 8192 | 4.38 | 12.7 | 0.800 | 5.5 | 1.9e-8 |

On this card the GPU path breaks even at a few hundred traces and is 5 to 7
times faster than the SciPy loop from about two thousand traces upward.
Timings at 256 traces vary by tens of percent between runs on this laptop
(a second run of the 256-trace command gave 0.178 s for CuPy); the larger
batches are stable to a few percent. In a
`cupyx.profiler.benchmark` breakdown at 256 traces on the same card, one
iteration costs about 1.5 ms in the Jacobian (launch-bound elementwise
work), 0.2 ms in the normal equations and 0.2 ms in the batched solve.
