# Compatibility profiles

Select numerical behavior with `--profile`:

| Profile | Behavior |
| --- | --- |
| `analytic-v1` | Normalized JSON jobs with fractional sample validity |
| `interop-real-usb-v1` | Supported real-sampled upper-sideband text metadata, with integer-second execution boundaries |
| `interop-real-usb-buffered-window-replay-v1` | Input-derived buffered-reader source selection, timing, admission, and validity |

Replay has additional [input requirements](buffered-window-compatibility.md).
Unsupported profiles and feature combinations fail explicitly.

## Input and timing

Text telescope labels are not interpreted as VDIF two-byte station identifiers;
VDIF station consistency is enforced across referenced threads. All declared
delay/UVW model segments are retained and selected at their epochs.

Without `TARGET FREQ`, a text baseline group uses datastream A's recorded or
zoom output frequency as the target. Separate frequency identifiers are
accepted only when their spectral grids agree. An explicit target remains
authoritative.

An explicitly empty file list omits that unavailable feed and its products,
with a warning. Surviving station and stream identifiers are preserved.
A named missing file or malformed recording remains an error.

LO phase uses time measured from the original job start, including when
executing windows, shards, or internal tiles. Fringe order 2 rejects nonzero
cubic or higher delay/clock coefficients; trailing zero coefficients are
accepted.

## Invalid samples and weights

Invalid samples are zero-filled without discarding other samples in the same
FFT. Station block weight is the valid sample fraction. Self-products use
that fraction; cross-products use the product of the two station fractions.
Cross-product weighting is not a sample-by-sample intersection when both
stations are partially invalid.

The ordinary profile represents absent timestamps as missing samples without
an extra replay interval. Malformed headers and inconsistent formats are
errors rather than timed invalid samples. The exact recorded-filler sentinel
is recognized as invalid and excluded from thread discovery.

CPU and GPU execution consume the same compiled replay facts. Compiled plans
must cover the requested scope. Explicit plan normalization and reported
exposure remain distinct quantities.
