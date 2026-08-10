# Glossary

**Artifact**
: A persisted numeric, tabular, configuration, metric, or review product from
  a workflow run. Generated artifacts are not source files.

**Backend**
: The numerical implementation selected for an operation, such as CPU NumPy,
  PyTorch, CuPy, Numba-CUDA, or cuTile.

**Difference image**
: An image formed by subtracting a PSF- and background-matched reference from
  a target/search image.

**Exposure**
: One detector image and its associated variance, mask, PSF, timing, and
  metadata.

**Fit mask**
: A boolean array selecting pixels used to estimate a model. In cuPhoton fit
  masks, `True` means selected.

**HSC**
: Hyper Suprime-Cam. Some data adapters use array layouts derived from local
  HSC products; they do not require an online service.

**LSST**
: The Legacy Survey of Space and Time conducted by Vera C. Rubin Observatory.
  cuPhoton consumes local products and does not install the survey pipeline
  stack.

**OIS**
: Optimal image subtraction: estimate a convolution kernel and differential
  background so two images can be compared or subtracted.

**PSF**
: Point-spread function, the response of an imaging system to a point source.

**Reference/template image**
: The image convolved or otherwise matched to the target/search image before
  subtraction.

**ROI**
: Region of interest. Detector CLIs generally express its origin as `(x, y)`
  and dimensions as `(width, height)`; NumPy arrays remain `(y, x)`.

**Run directory**
: A self-contained workflow output containing a summary, effective
  configuration, and numeric or review artifacts.

**WCS**
: World Coordinate System metadata that maps image pixels to sky coordinates.

**XRay**
: The cuPhoton component for X-ray trace and detector artifact analysis.
