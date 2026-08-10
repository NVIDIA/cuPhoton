# cuPhoton documentation

cuPhoton is organized as a set of reference workflows. Start with a synthetic
run, then read the contract for the component you intend to adapt.

## First run

- [Getting started](getting-started.md): prerequisites and installation
  profiles.
- [Quickstarts](quickstarts.md): data-independent CPU and GPU smoke runs.
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
