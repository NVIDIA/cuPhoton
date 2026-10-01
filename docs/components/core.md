# Core

`cuphoton.core` owns the command-line, context, path, logging, invariant,
shared FITS reading, and Dragon/MPI execution framework. Component algorithms,
datasets, scientific validation, and output formatting remain in their owning
`cuphoton.*` namespaces.

## What it provides

- the fixed `xdr`, `xfit`, `xpois`, `xscan`, `xrep`, and `xray` component
  registry;
- root, group, and command help through one `cuphoton` entry point;
- class-based command discovery and invariant-backed option validation;
- consistent version, error, and logging behavior;
- side-effect-free resolution of component XDG config, state, data, run, and
  log paths;
- shared host/device FITS image reading with explicit reader provenance; and
- persistent Dragon/MPI worker lifecycle and result validation.

The public surface is pinned by the CLI contract tests. Five groups also
support a component-level `version` command; xDataReader is the exception.

Workflow-specific YAML `--config` options belong to each component.

Shared executor rounds measure `batch_wall_sec` through receipt of all worker
completions. Coordinator artifact audits and scientific output merging follow
that interval; `finalization_sec` measures the component's merge separately.
Inputs retained by an adapter are loaded during worker setup. A round summary
records a completed pass, while the root terminal summary is published after
worker cleanup and the executor's lifecycle checks. MPI process finalization
and launcher exit occur after the executor returns.

## Shared FITS reading

`cuphoton.core.fits_io` inspects two-dimensional FITS image HDUs and reads
selected planes to NumPy or CuPy arrays. It supplies the FITS reader used by
xPois, xRep, xFit, xScan, and the combined imaging pipeline.

`inspect_fits_images` and `inspect_fits_image` inspect image headers without
decoding pixels. `read_fits_images` accepts HDU indices, a reader policy, and
an optional bounded `(y, x)` section. Results preserve HDU order and integer
mask values, use native byte order, and include headers and read provenance.
`FitsReadResult.metadata()` records the requested and actual readers, fallback
reason, array location, shapes, dtypes, and decoded byte count.

The low-level API defaults to `reader="astropy"` and host arrays. A workflow
can request `auto` to choose xDR for supported images when its native extension
and GPU dependencies are available, or `xdr` to require that path. Unsupported
scaling, null values, compression, and uncompressed image sections use Astropy
under `auto`; explicit `xdr` rejects them before payload reading. Read errors
after reader selection propagate to the caller. Choosing xDR does not establish
that the storage transfer used GPUDirect Storage.

See the [FITS input contract](../data-artifacts.md) and
[xDR installation requirements](xdr.md#native-extension-availability).

## Public facade

Import shared CLI infrastructure from `cuphoton.core.cli`. Its stable facade
exports:

- `ComponentSpec`, `COMPONENTS`, `COMPONENT_REGISTRY`, and `get_component`;
- `ApplicationContext`;
- `CLI`, `CommandLine`, `Command`, `CommandError`, and
  `InvariantAwareCommand`; and
- scalar, set, path, CSV, sequence, pair, and positional invariant classes.

Repeated values and pairs preserve both declaration order and duplicates.
Variable positionals remain ordered. Component command discovery uses Python
introspection and admits only concrete, public command classes defined
directly in that component's `commands` module. Imported, private, and
abstract classes are excluded; duplicate command names or aliases are errors.

`build_component_cli` and `run_component` also accept an external
`ComponentSpec` directly. This builds and runs the external component while
preserving the public root CLI's fixed registry.

## Common CLI behavior

List groups, then inspect a group or command:

```bash
uv run cuphoton --help
uv run cuphoton xrep --help
uv run cuphoton xrep help reproject-image
```

The equivalent module invocation is `uv run python -m cuphoton`, followed by
the component and command names.

Run-producing commands normally use the component directory beneath
`$XDG_STATE_HOME/cuphoton` when no explicit output root is supplied. Runs and
logs are separated into `runs/` and `logs/`. Pass an explicit output path in
automation to keep artifact locations consistent across environments.

## Extending a component CLI

Define concrete public command classes in the component's `commands` module
and derive their options from Core invariants. Generic parsing, path, logging,
or invariant behavior belongs in Core. Domain-specific options, algorithms,
and scientific validation belong in the component.

A new command should have a unique name and aliases, useful help, tests for
success and invalid input, and a documented artifact contract.

See [Architecture](../architecture.md) and
[Adapting the workflows](../adapting-workflows.md).
