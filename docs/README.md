# cuPhoton documentation

cuPhoton is organized as a set of reference workflows. Use the README's
[workflow chooser](../README.md#choose-a-workflow) to match your inputs to a
component and see what it produces. The [component overview](../README.md#components)
explains the scientific operations.

For a view across components, read
[how the science components fit together](architecture.md#how-the-science-components-fit-together)
and [what an exposure contains](architecture.md#what-an-exposure-contains).
Then run a synthetic quickstart and read the contract for the component you
intend to adapt.

## First run

- [Getting started](getting-started.md): prerequisites and installation
  profiles.
- [Quickstarts](quickstarts.md): data-independent CPU and GPU smoke runs.
- Imaging walkthrough:
  [notebook](../examples/imaging-pipeline/run_imaging_pipeline.ipynb) and
  [script](../examples/imaging-pipeline/run_imaging_pipeline.py) with setup for
  synthetic FITS loading, alignment, subtraction, dipole fitting, and plotting;
  requires a CUDA 13-capable NVIDIA GPU and XDR's native extension.
- [Command-line index](cli.md): the umbrella executable and command groups.

## Workflow guides

- [Core](components/core.md): shared CLI and configuration behavior.
- [xDataReader](components/xdr.md): GPU-native FITS loading with GDS and
  nvCOMP.
- [xFit](components/xfit.md): batched nonlinear least-squares dipole fitting.
- [XPOIS](components/xpois.md): kernel fitting and image subtraction.
- [XScan](components/xscan.md): transient datasets, classification,
  evaluation, and review.
- [xRep (xReproject)](components/xrep.md): FITS/WCS reprojection.
- [XRay](xray/README.md): X-ray trace and detector analysis.

## Adapting and validating

- [Architecture](architecture.md): package boundaries and execution model.
- [Dragon transport and coordination](dragon.md): transport selection, queue
  placement, and timing comparisons.
- [Adapting the workflows](adapting-workflows.md): a practical extension
  process.
- [Data and artifact contracts](data-artifacts.md): shapes, files, metadata,
  and provenance.
- [Troubleshooting](troubleshooting.md): environment, CUDA, data, and CLI
  failures.
- [Glossary](glossary.md): project and astronomy terminology.
- [Scientific references](references.md): methods represented in the code.

Project-wide policies are in [Contributing](../CONTRIBUTING.md),
[Governance](../GOVERNANCE.md), [Support](../SUPPORT.md), and
[Security](../SECURITY.md).
