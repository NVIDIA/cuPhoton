# Glossary

**Artifact**
: A persisted numeric, tabular, configuration, metric, or review product from
  a workflow run. Generated artifacts are not source files.

**Backend**
: The numerical implementation selected for an operation, such as CPU NumPy,
  PyTorch, CuPy, Numba-CUDA, or cuTile.

**Candidate**
: A detection or object selected for further investigation. A stable candidate
  ID connects its image cutouts, fitted measurements, scores, and review records.

**Difference image**
: An image formed by subtracting a PSF- and background-matched reference from
  a target/search image.

**Dipole**
: A positive lobe beside a negative lobe in a difference image. A positional
  mismatch or source motion can produce this pattern.

**Exposure**
: One detector image and its associated variance, mask, WCS, PSF, timing, and
  metadata.

**Fit mask**
: A boolean array selecting pixels used to estimate a model. In cuPhoton fit
  masks, `True` means selected.

**FITS**
: Flexible Image Transport System, an astronomy file format for images,
  tables, and their metadata. A file can contain several HDUs.

**HDU**
: Header/data unit, one part of a FITS file containing a header and associated
  data. Separate HDUs can hold image pixels, variance, masks, or tables.

**HSC**
: Hyper Suprime-Cam. Some data adapters use array layouts derived from local
  HSC products; they do not require an online service.

**Kernel**
: In image matching, an array of convolution weights: each output pixel is
  a weighted sum of neighboring input pixels. In GPU programming, a kernel is
  a function launched on the device.

**LSST**
: The Legacy Survey of Space and Time conducted by Vera C. Rubin Observatory.
  cuPhoton consumes local products and does not install the survey pipeline
  stack.

**OIS**
: Optimal image subtraction: estimate a convolution kernel and differential
  background so two images can be compared or subtracted.

**PSF**
: Point-spread function, the response of an imaging system to a point source.

**Pump/probe**
: An experiment in which one pulse perturbs a sample and another measures its
  response after a controlled delay. Repeating at different delays produces a
  time-dependent trace.

**Real/bogus**
: A classification of plausible astronomical detections versus artifacts under
  a dataset's labeling policy. It does not identify an object's astrophysical
  type.

**Reference/template image**
: The image convolved or otherwise matched to the target/search image before
  subtraction.

**Reprojection**
: Resampling an image under a coordinate mapping so its output pixels refer to
  a chosen sky grid. Combining exposures and matching their blur are separate
  operations.

**ROI**
: Region of interest. Detector CLIs generally express its origin as `(x, y)`
  and dimensions as `(width, height)`; NumPy arrays remain `(y, x)`.

**Run directory**
: A self-contained workflow output containing a summary, effective
  configuration, and numeric or review artifacts.

**Stamp**
: A small image cutout around a candidate or another selected location. xFit
  and XScan operate on batches of stamps.

**Trace**
: A one-dimensional signal across time or experimental delay. XRay extracts
  traces from selected detector regions across a delay scan.

**Variance**
: Uncertainty squared, expressed in squared image units. A variance plane
  describes the noise level at each pixel and can determine its fit weight.

**WCS**
: World Coordinate System metadata that maps image pixels to sky coordinates.

**XRay**
: The cuPhoton component for X-ray trace and detector artifact analysis.
