# Linear prediction benchmarks by GPU

Measured timings of the xray linear-prediction benchmark commands on named
hardware, with the commands to reproduce them. Add a section per machine;
keep the raw JSON outside the checkout and record the source revision the
run used.

## How the commands measure

`linear-prediction-benchmark` (`lpb`), `linear-prediction-p2-benchmark`
(`lppb`) and `linear-prediction-savgol-benchmark` (`lpsb`) share one
procedure:

- The synthetic trace (`synthetic_trace`, 96 samples by default) is
  replicated `--traces` times on the host.
- CPU: the serial NumPy path over the rows, best of `--repeat` runs
  (default 3) by `perf_counter`.
- GPU: the time axis and the trace rows are copied to the device once,
  before any timing, so host-to-device and device-to-host transfers are not
  included. One untimed warm-up call of each GPU path runs first. Each
  repeat is bracketed by `cupy.cuda.Stream.null.synchronize()` and the best
  of `--repeat` runs is reported for the serial (one call per trace) and the
  batched (one call for all traces) paths.
- Numerical agreement between CPU and GPU results is reported alongside
  (`max_abs_coefficient_diff`, `max_abs_eigenvalue_diff`,
  `max_abs_reconstruction_diff`, `max_abs_filter_diff`).

`--json` prints the same fields the tables below quote.

## NVIDIA GeForce RTX 4050 Laptop GPU (Ada, compute capability 8.9, 6 GiB)

Host: Intel Core i9-13900H, WSL2 Ubuntu 24.04 on Windows 11, Linux
6.18.33.2-microsoft-standard-WSL2, driver 596.49, CUDA runtime 13.2, CuPy
14.1.1, Python 3.12, cuPhoton main 9e91835, `uv sync --locked --extra dev
--extra gpu`. WSL memory cap 5 GB, 4 processors. Best of 3 repeats, command
defaults except where stated.

```bash
uv run cuphoton xray lpb --traces 16 --json
uv run cuphoton xray lpb --traces 256 --json
uv run cuphoton xray lpb --traces 2048 --json
uv run cuphoton xray lppb --traces 16 --json
uv run cuphoton xray lpsb --traces 16 --json
```

| Benchmark | traces | CPU serial best (s) | GPU serial best (s) | GPU batched best (s) | GPU serial / GPU batched | CPU serial / GPU batched |
| --- | --- | --- | --- | --- | --- | --- |
| P1, SVD and roots (`lpb`, 6 components) | 16 | 0.0142 | 0.235 | 0.214 | 1.10 | 0.066 |
| P1, SVD and roots (`lpb`, 6 components) | 256 | 0.239 | 3.61 | 3.55 | 1.02 | 0.067 |
| P1, SVD and roots (`lpb`, 6 components) | 2048 | 1.83 | 31.4 | 29.0 | 1.08 | 0.063 |
| P2, fixed-shape fit (`lppb`, 2 modes, 8 components) | 16 | 0.00097 | 0.0527 | 0.00334 | 15.8 | 0.29 |
| Savitzky-Golay (`lpsb`, window 11, order 3) | 16 | 0.00028 | 0.0643 | 0.0039 | 16.5 | 0.072 |

CPU and GPU coefficients agree to 3e-15 and eigenvalues to 2e-13 in every P1
run; P2 reconstructions and Savitzky-Golay filters agree to 3e-15 and 7e-15.

### Where the P1 GPU time goes

Same machine and revision, `lpb --traces 256`, with `cProfile` around the
benchmark function and `cupyx.profiler.benchmark` (3 repeats after a
warm-up) on arrays of the real sizes: Hankel matrices 24 x 71 and companion
matrices 71 x 71, batch of 256.

| Stage, batch of 256 | GPU, batched 3-D call | GPU, serial Python loop | NumPy serial |
| --- | --- | --- | --- |
| SVD of the 24 x 71 Hankel matrices | 855 ms (3.34 ms per trace) | 963 ms (3.76 ms per trace) | 29 ms (0.11 ms per trace) |
| Eigenvalues of the 71 x 71 companion matrices | 2106 ms (8.23 ms per trace) | 2101 ms (8.21 ms per trace) | 171 ms (0.67 ms per trace) |

In the profile of the 13.5 s benchmark run, `cusolver.xgeev` takes 9.18 s in
1024 calls, one per matrix: `cupy.linalg.eigvals` on a 3-D array loops over
the batch and calls `xgeev` per matrix, which is why the batched time equals
the serial-loop time. `cusolverDnDgesvd` takes 0.96 s in 512 calls on the
serial path and `_gesvd_batched` 1.66 s on the batched path. At 96 samples
the per-trace linear algebra is far below the size at which these routines
are efficient, and the batched P1 path serialises at the library level, not
in cuPhoton's code.

Trace length, 32 traces, best of 2, same machine:

| samples | companion size | CPU serial per trace | GPU batched per trace | CPU / GPU batched |
| --- | --- | --- | --- | --- |
| 96 | 71 x 71 | 0.84 ms | 11.9 ms | 0.07 |
| 256 | 191 x 191 | 18.6 ms | 86.6 ms | 0.21 |
| 512 | 383 x 383 | 93.8 ms | 229 ms | 0.41 |
| 1024 | 767 x 767 | 486 ms | 718 ms | 0.68 |

The GPU closes the gap as the companion matrix grows but has not crossed
over by 1024 samples on this card. On both sides the cost is the O(m^3)
eigenproblem of a companion matrix whose order tracks the trace length
while only a handful of poles are wanted. Options, in order of effort:
route P1 to the CPU for short traces and keep the GPU for P2 and
Savitzky-Golay, which batch well; replace the companion-matrix root finding
with a matrix-pencil pole estimate (Hua and Sarkar, 1990), where the k
signal poles are the eigenvalues of a k x k matrix built from the SVD
subspace, so the large general eigenproblem and its spurious roots
disappear; or batch the small eigenproblems with a custom kernel if the
companion route must stay.
