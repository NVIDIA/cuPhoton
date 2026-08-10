# Scientific references

cuPhoton implements reference workflows inspired by established methods. Read
the original literature and validate assumptions against the target instrument
before drawing scientific conclusions.

## Image subtraction and PSF matching

- C. Alard and R. H. Lupton, “A Method for Optimal Image Subtraction,”
  *The Astrophysical Journal*, 503:325-331, 1998.
  [doi:10.1086/305984](https://doi.org/10.1086/305984)
- C. Alard, “Image subtraction using a space-varying kernel,”
  *Astronomy & Astrophysics Supplement Series*, 144:363-370, 2000.
  [doi:10.1051/aas:2000214](https://doi.org/10.1051/aas:2000214)

XPOIS uses Gaussian-times-polynomial bases, optional polynomial
backgrounds, fit masks, and weighted least squares in this family of methods.

## Transient classification

- A. Inada et al., “Transformer-based Neural Network for Transient Detection
  without Image Subtraction,” *The Astronomical Journal*, 2026.
  [doi:10.3847/1538-3881/ae38d8](https://doi.org/10.3847/1538-3881/ae38d8)
- T. Acero-Cuellar et al., “What's the Difference? The Potential for
  Convolutional Neural Networks for Transient Detection without Template
  Subtraction,” *The Astronomical Journal*, 2023.
  [doi:10.3847/1538-3881/ace9d8](https://doi.org/10.3847/1538-3881/ace9d8)

XScan's pair and triplet paths reproduce the image-channel comparison in
this line of work, with a transformer-family model. Its data assumptions must
be evaluated on the intended domain, with splits that prevent leakage across
related samples.

## FITS and WCS

- E. W. Greisen and M. R. Calabretta, “Representations of world coordinates in
  FITS,” *Astronomy & Astrophysics*, 395:1061-1075, 2002.
  [doi:10.1051/0004-6361:20021326](https://doi.org/10.1051/0004-6361:20021326)
- M. R. Calabretta and E. W. Greisen, “Representations of celestial
  coordinates in FITS,” *Astronomy & Astrophysics*, 395:1077-1122, 2002.
  [doi:10.1051/0004-6361:20021327](https://doi.org/10.1051/0004-6361:20021327)

xRep relies on Astropy's FITS/WCS implementation and uses north-up TAN
grids for its shared-grid workflows.

## Software references

The exact versions of Astropy, NumPy, SciPy, PyTorch, CuPy, Numba-CUDA, and
other dependencies used for a result are recorded by `uv.lock` or the
downstream environment. Cite those projects when their methods materially
support a publication, and cite cuPhoton as described in
[`CITATION.md`](../CITATION.md).
