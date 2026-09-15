# xRep (xReproject)

`cuphoton.xrep` reprojects two-dimensional images onto celestial WCS grids.
It can derive a north-up grid or use an existing image as the destination. It supports bilinear and Lanczos-3 interpolation, optional mask
propagation, relative-area scaling, and shared-grid stacks. Its CLI group is
`cuphoton xrep`.

## Install and smoke test

```bash
# CUDA 13
uv sync --locked --extra dev --extra gpu
uv run python examples/run_quickstarts.py --component xrep --require-gpu

# CPU
uv sync --locked --extra dev --extra torch
uv run python examples/run_quickstarts.py \
  --component xrep --profile cpu
```

Automatic selection prefers CuPy, then CUDA PyTorch, then CPU. The resolved
backend and timings are written to `summary.json`.

## Inspect and reproject one image

The FITS input must contain a 2D image and a valid celestial WCS in the selected
HDU:

```bash
uv run cuphoton xrep inspect-image --input /path/to/image.fits

uv run cuphoton xrep reproject-image \
  --input /path/to/image.fits \
  --backend auto \
  --interpolation lanczos3 \
  --write-fits \
  --output-dir /path/to/runs
```

If a target grid is not supplied, the workflow derives one from the source
WCS. Provide a reference sky position and pixel scale for controlled
cross-image comparisons. Validate the output bounding box, WCS alignment,
flux behavior, and mask footprint.

## Reproject a stack

```bash
uv run cuphoton xrep reproject-stack \
  --inputs /path/to/a.fits,/path/to/b.fits,/path/to/c.fits \
  --backend auto \
  --output-dir /path/to/runs
```

Stack members are mapped to one grid and bounding box. The run writes a stacked
NumPy array, optional mask stack, and a summary of the shared grid and each
member.

### Use an existing image as the destination

`reproject-stack --target-wcs` uses a selected FITS image HDU's full celestial
WCS and exact `(height, width)`. `--target-hdu` selects its zero-based HDU index;
omitting it selects the first 2D image. The output origin is `(0, 0)` on that
image, so input footprints never enlarge the destination.

```bash
uv run cuphoton xrep reproject-stack \
  --inputs a.fits,b.fits \
  --target-wcs destination.fits --target-hdu 1 \
  --backend cpu --mapping-grid-step 1 --write-fits \
  --output-dir runs
```

The target retains rotation, reference-pixel shifts, and SIP distortion.
Lookup-table distortions are unsupported. `--grid-crval-ra`,
`--grid-crval-dec`, and `--pixel-scale-arcsec` conflict with `--target-wcs`.
Interpolation and relative-area scaling use their existing settings; pixels
outside each source footprint retain the normal NaN fill and invalid mask.
`--mapping-grid-step 1` evaluates the WCS mapping at every pixel; larger values
retain the existing coarse-grid approximation.

The summary records the selected target HDU, dimensions, and serialized WCS
header. FITS output retains that WCS, including SIP coefficients. Read the
celestial WCS from a stack cube with `WCS(header, naxis=2)`. Each input remains
a separate stack member.

Python callers can pass `Grid.from_wcs(target_wcs)` and
`output_bbox=BBox(0, 0, width, height)` to `build_stack_spec_from_fits`.


## Compare implementations

```bash
uv run cuphoton xrep compare-backends \
  --input /path/to/image.fits \
  --backends cpu,cupy \
  --output-dir /path/to/runs

uv run cuphoton xrep benchmark-backend-variants \
  --input /path/to/image.fits \
  --variants cupy-elementwise,cupy-raw \
  --mask-cases none,mask \
  --output-dir /path/to/runs
```

Parity checks compare images and masks; benchmark runs keep host and device
timings separate. Use the same grid, interpolation, dtype, and area-scaling
policy when comparing results.

## Python API

The curated API exposes grid and bounding-box types, reprojection
specifications, mapping preparation, FITS helpers, single-array reprojections,
masked image reprojections, and stack reprojections:

```python
from cuphoton.xrep import BBox, ReprojectionSpec, reproject_array

spec = ReprojectionSpec(mapping=mapping, output_bbox=BBox(0, 0, 256, 256))
result = reproject_array(image, spec, backend="cpu")
```

Use one prepared geometry to reproject an image, variance, and integer mask
together:

```python
from cuphoton.xrep import prepare_reprojection, reproject_masked_array

prepared = prepare_reprojection(spec, source_shape=image.shape)
result = reproject_masked_array(
    image,
    variance,
    mask,
    spec,
    backend="cpu",
    prepared=prepared,
)
```

Masked reprojections use squared normalized interpolation weights for variance,
square the relative-area Jacobian, and preserve mask neighborhoods by exact
bitwise OR over every nonzero contributor in the selected interpolation
kernel. The returned variance is explicitly a diagonal approximation:
interpolation-induced covariance is not represented. Pass
`variance_fill_value` to `reproject_masked_array` to set an out-of-footprint
variance sentinel independently of the image `fill_value`; it defaults to
NaN. Finite variance values must be nonnegative; NaN and positive infinity
remain available as no-data representations.

The `reproject_array` API ORs the nonzero bilinear mask neighborhood even when
the image interpolation is Lanczos-3. Use `reproject_masked_array` when the
mask must cover every nonzero contributor in the selected image kernel.

See [Data and artifact contracts](../data-artifacts.md#xrep-fits-and-arrays)
for persisted run files and coordinate conventions.
