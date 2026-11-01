# VDIF sidecars

Sidecars index frame headers without storing voltage payloads. Build them with:

```console
cuphoton xbeam build-vdif-sidecar RECORDING-*.vdif \
  --index-dir SIDECARS --backend cufile --devices 0,1,2,3
```

For this command, `--backend` selects the I/O path and defaults to `cufile`.
Use `--backend posix` without GDS. The correlation commands use `--backend`
for NumPy/CuPy computation and `--vdif-io` for I/O (default `posix`).
Existing valid sidecars are reused. Add
`--force` to rebuild; it does not relax source validation.

Pass `--index-dir SIDECARS` to `correlate` as well. Without `--index-dir`,
`build-vdif-sidecar` writes each sidecar beside its recording, and `correlate`
looks for it there. Correlation does not build missing sidecars.

## Validation

Malformed headers and inconsistent formats fail before publication. A VDIF
invalid-data bit is accepted only on a structurally valid, consistent header.
The exact recorded-filler signature, including its recorded thread tag and
optional recorder status trailer, is marked invalid and excluded from thread
discovery. Other byte patterns undergo ordinary validation.

Sidecars require source provenance: source size, modification/change times,
and sampled frame fingerprints. Copying, restoring, or modifying a recording
can require rebuilding its index. Keep sources and indexes immutable during
correlation. These checks detect ordinary filesystem-visible changes; they
are not a cryptographic verification of the complete recording.

Buffered-reader replay requires schema-v4, including physical timestamp runs.
Ordinary profiles accept older supported schemas with valid provenance.
Indexes missing provenance or containing positive rejected-header/format
counters must be rebuilt. A still-malformed source will fail the rebuild.

## Lookup and duplicate basenames

The primary name is `<recording-basename>.vdix`. Ownership hints distinguish a
stale index from one belonging to another recording. When the basename is
already owned by a different source, a suffix derived from source metadata
and sampled bytes distinguishes the new index. That suffix is independent of
the source's absolute path. Ownership hints do not replace provenance checks.

Legacy path-hashed names remain readable. A stale or corrupt index reports an
error. A legacy basename without ownership hints may be ambiguous; use separate
index directories or explicitly relocate that index before building the
colliding recording.

A single logical thread can alias a different sole physical thread only when
every source has a complete thread inventory. Bounded header probes do not
establish that inventory. Ambiguous multi-thread input fails; validated aliases
are recorded in the ingest manifest.

Without a sidecar, correlation requires independently verified ordered input
and the explicit `--assume-ordered` assertion. Selected frames are still
validated and `--max-read-bytes` remains enforced.
