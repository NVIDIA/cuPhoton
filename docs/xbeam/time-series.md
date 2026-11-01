# Time-series correlation

`correlate_series(source, spec)` accepts uniformly sampled real or complex
signals identified by strings. It yields cross-spectra, lag correlations,
or both. Use `ArraySource` for NumPy or CuPy vectors and `Hdf5Source` for
bounded reads from a file.

## Array API

```python
import numpy as np
from cuphoton.xbeam import ArraySource, CorrelationSpec, correlate_series

rng = np.random.default_rng(42)
a = rng.normal(size=8192).astype(np.float32)
b = np.zeros_like(a)
b[7:] = a[:-7]
source = ArraySource({"sensor-a": a, "sensor-b": b}, sample_rate_hz=1024)
spec = CorrelationSpec(
    pairs=(("sensor-a", "sensor-b"),),
    window_samples=1024,
    hop_samples=512,
    windows_per_result=4,
    output="both",
)
for result in correlate_series(source, spec):
    peak = result.lag_samples[np.argmax(abs(result.lag[0]))]
    print(result.result_index, peak)
```

Set `backend="cupy"` in `CorrelationSpec` to run transforms and products on
the GPU. Results contain host arrays. Input floating precision is preserved;
mixed signals use the wider precision and integer inputs use double
precision. Set `precision="float32"` or `"float64"` to choose explicitly.

## Numerical conventions

For each pair `(a, b)`, the cross-spectrum is
`FFT(a) * conj(FFT(b))`. Lag correlation is
`R_ab(k) = sum_n a[n] * conj(b[n+k])`: positive lag means that `b` arrives
later than `a`. Divide `result.lag_samples` by `result.sample_rate_hz` for
seconds. Peak selection returns a whole-sample delay.

`lag_mode="linear"` zero-pads to the next power of two at least `2*N-1`
and returns lags `-(N-1)` through `N-1`. All output selections use this FFT
grid. Frequencies follow standard FFT ordering, including negative
frequencies. `lag_mode="circular"` uses an N-point FFT and N centered lags,
so correlation wraps at the window boundary.

Products are raw sums averaged over the scheduled windows in each result.
Applications supply any taper, mean removal, calibration or normalization.
Spectrum-only requests skip the inverse transform and lag arrays. The radio
API has its own channelization, sideband and weighting conventions.

Optional Boolean validity vectors zero invalid samples before transforms.
Nonfinite samples marked valid are errors. Results report contributing
windows, simultaneous valid sample counts and, for lag output, valid
sample-pair counts at every lag. All-invalid pairs have `valid=False`.
Coverage counts are reported separately from products; a lag with zero
coverage has no measurement.

## HDF5 inputs and outputs

Each signal occupies a one-dimensional dataset with scalar `sample_rate_hz`
and `origin_seconds` attributes. Signals must share sample count, rate and
origin. The origin is an application-defined reference in seconds. The
adapter requires an aligned grid; applications perform any resampling or
clock correction before correlation.

The CLI accepts a JSON configuration:

```json
{
  "input": "signals.h5",
  "datasets": {"a": "/signals/a", "b": "/signals/b"},
  "validity": {"a": "/validity/a"},
  "correlation": {
    "pairs": [["a", "b"], ["a", "a"]],
    "window_samples": 1024,
    "hop_samples": 512,
    "windows_per_result": 4,
    "batch_windows": 16,
    "output": "both",
    "lag_mode": "linear"
  }
}
```

```bash
uv run cuphoton xbeam correlate-series series.json \
  --output results.h5 --backend numpy
```

Input paths are relative to the JSON file unless absolute. Signal dataset
paths and optional Boolean validity dataset paths are explicit. HDF5 reads
use bounded host slices, including for compressed or chunked inputs. GPU
execution uploads these slices from host memory; VDIF provides the separate
cuFile path.

Results are stored under `/results/<global-result-index>`, with pair
identifiers in `pairs_json`, grid metadata in attributes, and arrays named
after the `SeriesResult` fields. Products omitted by `output` are absent.
Sample bounds cover the result's windows; `window_starts` identifies the
individual windows when a hop leaves gaps. Existing output files are refused.
An interrupted write leaves the HDF5 root `complete` attribute false.

## Batches and independent shards

The default hop equals the window size. A result contains one window unless
`windows_per_result` groups several windows. Only complete windows and
complete result groups are emitted. `batch_windows` bounds working memory
independently of recording length.

HDF5 sources read one batch ahead on a background host worker by default
(`prefetch_batches=1`). Current and next batches have separate buffers.
Set `prefetch_batches=0` for synchronous reads. Read-ahead remains within
the selected shard and complete result groups. Consume the result iterator
inside the source context; closing it cancels queued work and waits for an
active read before closing the file. A blocked filesystem read must return
before cleanup can finish.

For `correlate-series`, `shard_count` and `shard_index` are keys in the JSON
`correlation` object. Set `shard_count` to the number of tasks. Each task needs
its own configuration, differing only in `shard_index`, and its own output
file. Indices run from zero to `shard_count - 1`; reusing one configuration
would repeat that shard instead of producing the other results. All tasks
read the same input. Select a local GPU through `CUDA_VISIBLE_DEVICES` and
`--backend cupy`.

`plan_series_shards(sample_count, spec)` partitions global result ordinals.
Overlapping windows read their needed samples but emit each result once.
Empty shards are valid. Combine HDF5 result groups by their global ordinal;
averaging shard outputs would change the result. Applications coordinate
file assembly and task execution.
