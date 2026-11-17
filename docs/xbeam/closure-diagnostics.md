# Radio closure diagnostics

`compute_closures` computes per-channel triangle closure phases and
four-station log closure amplitudes from host visibility records. It accepts
`VisibilityRecord`, saved `xbeam-v1` `OutputRecord`, and `SwinRecord` objects.

```python
from cuphoton.xbeam.adapters.radio.closure import compute_closures
from cuphoton.xbeam.adapters.radio.output import iter_xbeam_v1

results = compute_closures(
    iter_xbeam_v1("OUTPUT"),
    triangles=[("A", "B", "C")],
    quadrangles=[("A", "B", "C", "D")],
)
for result in results:
    print(result.key, result.kind, result.stations)
    print(result.values[result.valid])
```

Use the station identifiers recorded in the output. With `iter_swin`, pass
the channel count required by the SWIN reader and use zero-based integer
station indices in the requested triangles and quadrangles.

## Conventions

xBeam forms `Vab = A * conj(B)`. A requested triangle `(a, b, c)` returns
`arg(Vab Vbc Vca)` in radians, in `[-pi, pi]`. A requested quadrangle
`(a, b, c, d)` returns the natural logarithm
`log|Vab| + log|Vcd| - log|Vac| - log|Vbd|`. The station order specifies
the amplitude ratio; other software may use a different denominator.
Reversed baselines are conjugated automatically. Phase sums and sums of
logarithms avoid overflow from multiplying several visibility amplitudes.

Results are grouped by exact MJD, seconds of day, source, configuration,
output frequency, co-polar hand, and SWIN pulsar bin. SWIN float timestamps
are converted to `Decimal` through their string representation. Records can
arrive in any order; the function collects them in memory. For large outputs,
call it on bounded batches containing complete integration groups.

All spectra within a group must have the same channel count. The caller
must ensure that a frequency identifier denotes the same channel grid on
every baseline. The function performs no regridding or approximate time
matching. Autocorrelations and cross-hand products are ignored; scalar
closure cancellation does not apply to general cross-hand products.

Every observed co-polar cross-correlation group produces the requested
results, in first-seen group order. A missing baseline, a nonpositive or
nonfinite exposure weight, or a zero or nonfinite visibility amplitude
makes affected channels invalid: `valid` is false and `values` is NaN.
Duplicate edges, including reverse copies of the same edge, are errors.

## Interpretation

Nonzero scalar station gains cancel in these closures when the visibilities
refer to matching channels and simultaneous samples. Positive xBeam weights
represent exposure and are used only to identify usable records; they are
not inverse noise variances. No uncertainty estimate, noise debiasing, time
averaging, or flux calibration is applied.

Closure agreement is useful alongside full visibility and metadata checks.
It cannot expose every station-based error, verify UVW coordinates, or
remove polarization leakage or baseline-dependent corruption. Closure of
averaged visibilities also need not equal an average of individual closures
when station gains vary within the averaging interval.

See the [public M87 comparison](public-m87-closures.md) for a reproducible
check against eht-imaging on released observational visibilities.
