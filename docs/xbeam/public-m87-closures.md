# Compare closures on public M87 observations

The [comparison example](../../examples/xbeam/compare_public_m87_closures.py)
checks xBeam's scalar closure diagnostics against
[eht-imaging 1.3.2](https://github.com/achael/eht-imaging/tree/v1.3.2), using
public EHT observations of M87. It compares closure arithmetic on calibrated
visibilities.

## Run the comparison

From a development checkout containing the closure diagnostics:

```bash
uv sync --locked --extra dev
uv run --with 'ehtim==1.3.2' python \
  examples/xbeam/compare_public_m87_closures.py \
  /tmp/m87-closures/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits \
  --download > /tmp/m87-closures-result.json
```

The download is optional. To use a file already downloaded from the release,
pass its path without `--download`. The example verifies SHA-256 before
reading it and refuses a different file. The data and results stay outside
the repository. `ehtim` is an optional reference dependency for this example.
The comparison runs on CPU and does not use its optional NFFT package. The separately installed
reference package is [GPLv3 licensed](https://github.com/achael/eht-imaging/blob/v1.3.2/LICENSE.txt);
no eht-imaging source or data is bundled with this example.

The JSON result reports closure counts, maximum errors, the input URL and
checksum, and package versions. A missing closure or an error above `1e-12`
raises an exception and exits unsuccessfully. Phase errors use circular
angular distance in radians; log-amplitude errors are dimensionless.

For this file, expect 7,447 visibility records across 781 integration times,
8,895 closure phases, and 19,617 log closure amplitudes. The maximal closure
sets contain algebraic redundancies; these counts are not counts of
independent measurements. Missing station combinations remain unavailable
and are counted separately. The example requires identical available
closure keys in both implementations.

Optional offline adapter tests use a synthetic observation and require no
network or observational data:

```bash
uv run --extra dev --with 'ehtim==1.3.2' pytest \
  tests/xbeam/test_public_m87_comparison.py
```

## Data provenance

The input is the April 11, 2017 low-band M87 observation from the EHT
Collaboration's [2019-D01-01 calibrated data release](https://doi.org/10.25739/g85n-f134).
Its public [release README](https://github.com/eventhorizontelescope/2019-D01-01/blob/27ae76aacb9067a83388395c3826012b97aac371/README.md)
describes EHT-HOPS fringe fitting, a priori calibration and network
calibration. The data have already been averaged over 10 seconds and all 32
intermediate frequencies. The release is dedicated to the public domain
under the Open Data Commons PDDL; cite the EHT Collaboration and
[M87 Paper III](https://doi.org/10.3847/2041-8213/ab0c57) when using it.

- File: `SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits`
- [Public download](https://data.cyverse.org/dav-anon/iplant/home/shared/commons_repo/curated/EHTC_FirstM87Results_Apr2019/uvfits/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits)
- Size: 650,880 bytes
- SHA-256: `697af2bb3bbf732115108ffefacd3e59e307f38fe685c3c4579146b0bd661298`

The release contains estimated Stokes I only. Its UVFITS RR and LL products
both contain that estimate; RL and LR are zero with zero statistical weight.
The example loads Stokes I and does not interpret those products as measured
polarization. The upstream loader may warn about zero-weight cross-hands;
the comparison checks that the loaded Stokes I inputs are finite and valid.

## Matching conventions and limits

Both paths receive the same `eht-imaging` UVFITS load, with no further
averaging, SNR cut, calibration or amplitude debiasing. This isolates the
closure calculations from file parsing and calibration. It does not
independently validate the UVFITS loader.

For triangle `(a, b, c)`, both compute `arg(Vab Vbc Vca)`. For quadrangle
`(a, b, c, d)`, xBeam computes `log|Vab Vcd / (Vac Vbd)|`.
The reference uses `log|V12 V34 / (V14 V23)|`, so the example swaps its last
two station labels when requesting the corresponding xBeam quadrangle.
Timestamp conversion is identical on both paths, without tolerance-based
matching.

The adapter represents each averaged Stokes I visibility as one channel
with a unit positive validity weight. It does not map statistical
uncertainty into xBeam's exposure weight or perform a flux-unit conversion.
Closure diagnostics use no UVW coordinates. This exercise therefore does
not qualify raw-recording correlation, visibility calibration, UVW metadata,
uncertainty estimates, polarization leakage correction, or imaging.

The [2026 M87 closure-invariant study](https://doi.org/10.1038/s41586-026-11241-9)
motivates checking calibration-insensitive observables. This example
compares conventional scalar closure phases and amplitudes; it does not
reproduce that study's learned reconstruction method or inference results.
