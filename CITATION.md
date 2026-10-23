# Citation

If cuPhoton contributes to a publication, cite the version you used and
record the exact commit in the methods or software-availability section.
For a development checkout, include the full version reported by
`cuphoton --version` and the output of `git rev-parse HEAD`.

Suggested citation for the 0.1.3 release:

> NVIDIA Corporation. *cuPhoton: GPU-accelerated astronomy and imaging
> reference workflows*. Version 0.1.3, 2026.
> https://github.com/NVIDIA/cuPhoton

The repository also contains machine-readable citation metadata in
[`CITATION.cff`](CITATION.cff). GitHub can export that record in common
bibliography formats.

The current citation metadata does not include an archived DOI. If the
release you use publishes one, prefer its DOI while retaining the exact
version or commit used.

## xImage

If you use xImage reconstruction in a publication, also cite the ImageMM
method paper:

> Yashil Sukurdeep, Tamás Budavári, Andrew J. Connolly, and Fausto Navarro.
> “ImageMM: Joint Multi-frame Image Restoration and Super-resolution.”
> *The Astronomical Journal*, 170(4), article 233, 2025.
> [doi:10.3847/1538-3881/adfb72](https://doi.org/10.3847/1538-3881/adfb72)

Cite the cuPhoton version and commit used for the implementation; the
ImageMM citation credits the method. See the
[xImage component guide](docs/components/ximage.md) for supported behavior
and validation limits.
