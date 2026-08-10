# Governance

cuPhoton is maintained by the NVIDIA project maintainers listed in
[`MAINTAINERS.md`](MAINTAINERS.md). Those maintainers also own repository-area
review and approval decisions.

## Decision making

Technical decisions are made through issue discussion and pull request review.
Maintainers weigh scientific correctness, reproducibility, performance,
dependency licensing, maintenance cost, and fit with the reference-workflow
scope. The maintainers have final responsibility for accepting changes and
cutting releases.

Substantial proposals should describe:

- the research workflow and users affected;
- the input, output, and metadata contracts;
- the expected CPU and GPU behavior;
- correctness and performance validation;
- dependency and packaging effects; and
- how another institution could adapt or reproduce the result.

## Triage

Issues and pull requests should name the affected namespace or repository
area: `cuphoton.core`, `cuphoton.xdr`, `cuphoton.xfit`, `cuphoton.xpois`,
`cuphoton.xscan`, `cuphoton.xrep`, `cuphoton.xray`, documentation, or
packaging. Maintainers may close reports that are security-sensitive, out of
scope, unreproducible, or inactive.

## Releases

Releases are tagged from `main` after the relevant correctness, packaging,
documentation, dependency, and public-content checks pass. Because this is a
reference-workflow project, a minor release may revise an interface when the
scientific or operational benefit justifies it. Such changes are documented in
the release notes.
