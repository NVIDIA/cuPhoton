# Documentation developer tools

This repository-only project builds and publishes cuPhoton's versioned user
guide. It is installed explicitly and reuses the product CLI framework.
It does not publish software releases or upload packages.

## Install and check

From a complete cuPhoton checkout:

```sh
uv sync --project dev --locked
uv run --project dev --locked cuphoton dev --help
```

For a pip environment, install both editable projects explicitly:

```sh
python -m pip install -e . -e dev
```

The separate `dev/.venv` keeps root `uv sync` independent. Normal product
installs, root editable installs, root sync and the existing product `dev`
extra do not install `cuphoton-dev` or advertise `cuphoton dev`. Uninstalling
`cuphoton-dev` removes that route from help. `python -m cuphoton_dev` provides
the same commands in an explicitly installed environment.

Run the developer checks separately:

```sh
uv lock --project dev --check
uv run --project dev --locked ruff check dev scripts/ci
uv run --project dev --locked ruff format --check dev scripts/ci
uv run --project dev --locked mypy --config-file dev/pyproject.toml dev/src/cuphoton_dev
uv run --project dev --locked pytest -c dev/pyproject.toml dev/tests
```

CI writes developer coverage separately from product coverage. The changes
classifier includes core CLI code, package entrypoints, dependency locks,
packaging exclusions, shared configuration, workflows and guide inputs. It
compares PRs to their actual target merge base, vetted PR copies to that PR's
target, and ordinary pushes to their before/after commits. Uncertain ranges
run the developer tests. Existing product CI requirements remain in force.

When root dependencies change, update both locks and check both projects:

```sh
uv lock
uv lock --project dev
uv lock --check
uv lock --project dev --check
```

Beautiful Soup and Soup Sieve are MIT-licensed developer-only dependencies.
Keep dependency/license review and `dev/uv.lock` synchronized when updating
them; they are not product dependencies or product payloads.

Keep new dependency paths in `scripts/ci/dev_changes.py`. Changes to guide
markup, install wording, source URLs or search output also require review of
transformation fixtures and real rendered output. See [AGENTS.md](AGENTS.md)
for instructions for agents working in this project.

## Build and inspect

Make `uv` and `git` available on `PATH`. The command reuses an installed Quarto
matching the target's renderer pin. Otherwise, on Linux amd64 and arm64 it
downloads the target renderer from the official release, verifies its digest
against the main controller's trusted table and extracts it into the temporary
render workspace. It does not replace the host's Quarto installation. Unknown
renderer versions fail with instructions to update the table.

When upgrading Quarto, add its official release digests to
`dev/src/cuphoton_dev/renderers.py` and retain all historical entries so missed
release publications can still be backfilled. The command creates a complete
isolated source checkout and uses the target's locked Python dependencies.
CPU examples execute from that checkout. The output includes build provenance
and hashes; authored source remains unchanged.

```sh
uv run --project dev --locked cuphoton dev docs-build --ref main --output /tmp/cuphoton-main
uv run --project dev --locked cuphoton dev docs-build --ref v0.2.0 --output /tmp/cuphoton-v0.2.0
uv run --project dev --locked cuphoton dev docs-check /tmp/cuphoton-v0.2.0 --json
uv run --project dev --locked cuphoton dev docs-status --json
```

A draft release can be built and checked locally. Publication additionally
requires a published, non-prerelease GitHub Release whose tag is a full final
version such as `v0.2.0`. Older tags without the guide are reported as
unavailable; they are not populated from newer content.

## Site layout and ownership

The site at <https://nvidia.github.io/cuPhoton/> contains:

| Path | Meaning |
| --- | --- |
| `main/` | Moving development guide |
| `v0.2.0/`, later full versions | Retained release snapshots |
| `stable/` | Highest documented, published final release |
| Root and existing chapter paths | Redirects to stable, or main before the first eligible snapshot |
| `versions.json` | Generated shared catalog |

Every snapshot reads the shared catalog. Version selection preserves the
chapter and a valid anchor where available, falling back to the target home
page. Root and stable redirects preserve query strings and fragments. Search,
assets and canonical URLs remain within the selected version.

Main owns rendering orchestration, conversion and publication. Release tags
own the source code, guide and dependency lock used to render their snapshot.
Authored documentation continues to use main URLs; HTML postprocessing converts
release links, install examples, labels, search data and canonical URLs.
The `gh-pages` branch retains complete validated output. Main publications
replace only `main/`; release publications retain existing release bytes.
Catalogs and redirects are generated, not hand-maintained.
The publisher records this generated output with GitHub Actions bot commits.
These archive commits are exempt from source-contribution identity, signature
and DCO requirements; each snapshot records the source and controller commit
SHAs. Source changes continue to follow the contributor signing rules.

## Publish, bootstrap and recover

Authenticate `gh` with access to dispatch repository Actions. For a normal
publication, the CLI dispatches `docs.yml` explicitly on main and reports the
exact run URL. It waits for completion and returns a failing exit status when
the workflow fails. Use `--json` for machine-readable results or `--no-wait`
to return after dispatch. Failed runs and wait timeouts retain the known
request and workflow fields in JSON output while returning a failing exit
status. An accepted dispatch whose run is not yet discoverable reports its
request ID with null run fields; use that ID before attempting a new dispatch.
If submission fails before confirmation, `dispatch_unknown` preserves the
request ID without claiming acceptance; inspect that request before retrying.

```sh
uv run --project dev --locked cuphoton dev docs-publish --ref main --json
uv run --project dev --locked cuphoton dev docs-publish --ref v0.2.0 --json
uv run --project dev --locked cuphoton dev docs-publish --missing --json
```

The existing `v0.2.0` tag predates the release hook. Its once-off bootstrap is
the `--ref v0.2.0` command above, after the GitHub Release becomes eligible.
Keep its tag unchanged. Before eligibility, finish the local render and check
and leave this command ready; documentation rollout does not authorize release
publication.

The workflow serializes archive updates through Pages deployment. It re-reads
retained state under the lock, rejects a moved tag or stale main rollback,
validates the complete site and publishes through native Pages actions. A
failed render leaves the live site and catalog intact. If deployment fails
after retaining the assembled site, retry the workflow or repeat the same CLI
command. Matching release snapshots are reused, so duplicate requests are safe.
Use `--missing` for deliberate reconciliation of eligible missing releases.

If validation rejects a retained snapshot or tag identity, inspect the failing
run and provenance before retrying. Do not repair this by moving a release tag,
copying current-main chapters, or editing generated catalog entries. A change
to an immutable snapshot requires a separate reviewed correction procedure.

After successful deployment, verify the live root, `main/`, `stable/`, release
paths and existing chapter bookmarks. Check the shared catalog, search,
light/dark themes, desktop/mobile navigation and keyboard version selection.
Compare served files to retained hashes and confirm older release hashes
survive the next publication. A successful build alone is not deployment
verification.

## Future release hooks

Before cutting a release, verify that its branch contains
`.github/workflows/release-docs.yml`. New release branches inherit it from main.
For the existing `0.2.x` branch, add only this hook before its next tag; no
developer package or documentation-content backport is needed.

The hook responds to `published` and `released`, including promotion of a
prerelease to a final Release. It accepts only the canonical repository and a
published final semantic-version tag, then dispatches `docs.yml` with
`ref: main` and the string `tag` input. The receiving publisher independently
checks eligibility and the exact tag. Keep that workflow filename and input
compatible with hooks already stored in historical tags.

Publishing a GitHub Release with `GITHUB_TOKEN` does not trigger the ordinary
downstream release workflow. A programmatic release publisher must explicitly
dispatch documentation after it publishes the final Release:

```sh
gh workflow run docs.yml --repo NVIDIA/cuPhoton --ref main --field tag=v0.2.1
```

Give that integration `actions: write` and check the resulting documentation
run. The existing package-upload workflow uploads packages and does not
publish a GitHub Release, so it is not the documentation trigger. Prereleases,
draft releases and package uploads alone must not promote stable docs.
