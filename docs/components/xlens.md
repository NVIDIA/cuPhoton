# cuPhoton xLens

`cuphoton.xlens` collects scientific measurements, deterministic findings and
evidence references. It provides a dashboard, JSON API and standalone HTML
export. Applications own input formats, scientific rules, units, thresholds
and detailed views. Optional model investigation reads selected scalar evidence
and writes separate advisory reports. The optional NOOA backend can choose
evidence queries and run isolated Python during an investigation.

## Run the invented-measurement example

From a current development checkout, install the base package:

```bash
uv sync --locked
uv run cuphoton xlens quickstart --output-dir xlens-demo
uv run cuphoton xlens serve --state xlens-demo/state --port 8765
```

Open `http://127.0.0.1:8765` or the generated `xlens-demo/report.html`.

The eight invented samples include a level change and a missing measurement.
Their irregular observation times are independent of their sequence numbers.
The dashboard supports sample/category filtering, named measurement histories,
scientific findings, provenance and separate model reports. `/api/state` supplies JSON;
`/health` supplies collector/storage status. The served dashboard refreshes its saved state automatically. An offline
export is a fixed snapshot.

For an installed wheel, use `cuphoton` directly in place of `uv run cuphoton`.
Python 3.12–3.14 is supported. Monitoring and export need no inference service,
key, GPU, Node/npm, browser library or model framework. The Python artifacts
include the HTML/CSS/JavaScript resources; these are authored with cuPhoton and
contain no bundled third-party libraries or fonts.

## Serve an application site

An installed application can provide a collection of monitors and its own
landing page through a local `xlens-site.toml`:

```toml
[site]
application = "my_monitoring_app.site:serve"
prefix = "/monitoring/"

[collection]
title = "Observation review"
```

Point the shared command at the directory containing that file, or at the
manifest itself:

```sh
cuphoton xlens serve --site ./observations --host 0.0.0.0 --port 8765
```

The application package must be installed in the same environment as
cuPhoton. The manifest names trusted application code; loading it invokes
`serve(*, root, config, host, port, prefix)`. `root` is the manifest directory,
`config` is the complete parsed TOML, and `prefix` is a validated absolute URL
path ending in `/`. `--prefix` overrides the configured mount. The application
owns routing, discovery, configuration of its monitors and server shutdown.
Reading saved measurements and serving prepared domain views requires no GPU.

Use exactly one of `--site` and `--state`. A single saved monitor also supports
`--prefix /monitoring/run/`. Mount URLs redirect to their trailing-slash form
so the dashboard's relative API requests stay inside their mount. Live pages
load domain views as samples are selected; offline exports include all visible
samples. A collection server can reuse `Dashboard.render(ui=...)` and
`Dashboard.display_sample(sample_id, ui=...)`, calling `Monitor.refresh()`
before reading independently updated state. The optional `ui` fields
`home_url`, `home_label` and `description` supply navigation and explanatory
text for the host application. Pass the same `ui` configuration to both
`render` and `display_sample` so navigation remains available after polling.
The JSON and health routes also live beneath the mount, for example
`/monitoring/run/api/state` and `/monitoring/run/health`.

## Connect an application

```python
from cuphoton.xlens import Dashboard, Finding, Measurement, Monitor, Sample

monitor = Monitor("observation-state", run_id="example-night")
monitor.publish(Sample(
    id="exposure-17/source-4", sequence=8,
    observed_start=104.5, observed_end=109.2, time_unit="s",
    measurements={"flux": Measurement(82.1, "count/s", coverage=0.95)},
    findings=(Finding("inspect", "Measured flux exceeds the example rule."),),
))
Dashboard(monitor, title="Example observations").export("observations.html")
```

A `Sample` identifies one application subject, such as a window, exposure,
candidate or reconstruction iteration. Multiple subjects may share a sequence.
Sequence controls display order; physical/observation times and their unit/origin
come from the adapter. xLens records processing time in UTC separately. Metadata
can describe a solver step, configuration or provenance without redefining time.

`Measurement` has a scalar value, unit, status, optional coverage fraction and
note. Status is `valid`, `missing`, `invalid` or `unavailable`. Missing and
unavailable values are null. Nonfinite floating values become null with invalid
status. Coverage does not normalize a measurement. Applications decide whether
coverage supports a scientific conclusion. `Finding` categories are `scientific`,
`data_quality` and `processing`; severity is independent of category.

Use `monitor.ingest(item, analyzer, source_id="stable-input-id")` to invoke an
ordinary callable returning a sample. `monitor.replay(items, analyzer)` consumes
an iterable in its supplied order. The application controls input readiness,
discovery, acquisition or polling. Per-item analysis errors are recorded and
later items continue. Source enumeration failure marks the collector failed.
Diagnostics retain exception classes and application-provided source IDs;
application logs can retain additional details under their own disclosure policy.

Large arrays stay in application artifacts. `Artifact(name, path, sha256,
media_type)` records a relative reference and digest. Applications can use
`cuphoton.core.artifacts.file_sha256` to calculate it and verify it when reading.
xLens records references without opening them. A sample is limited to 64 KiB,
including at most 16 KiB of metadata and 128 measurements.

Pass a trusted `render_sample(sample) -> str` callable to `Dashboard` for domain
views. It returns application-authored HTML, including bounded inline SVG or
PNG images. Package any application assets in its own wheel. Each view is
limited to 2 MiB, and the shell displays the latest 256 samples. This ordinary
Python callback owns plotting choices; no frontend registry or schema is needed.
Escape external text before inserting it into HTML. Model narratives are always
rendered as text. Offline exports retain the same domain views and reports.
The generic CLI export renders the generic view; an application's export command
can supply its own renderer.

## Revisions, history and recovery

Use one writer process per state directory. A new `Monitor` refuses a nonempty
directory. `Monitor.load(path)` explicitly recovers a run and marks unfinished
collection/investigation interrupted; invoke recovery after the prior writer
has stopped. `Monitor.load(path, recover=False)` opens a viewer without mutation.
The HTTP server refreshes its detached state from disk on each request.

Publishing an exact current duplicate is a no-op. Changed evidence requires an
increasing revision for the same sample ID. Reusing an old revision for different
content is an error. Older records remain in history; reports referring to a
superseded revision become stale, including reports that finish after a revision.
Recompute application aggregates and milestones from `monitor.samples` after
revisions or late arrivals. Invalid replacement input should publish a new
invalid sample when the application can still identify its subject; a generic
ingestion failure records the failure without inventing replacement measurements.

State snapshots are detached and written by atomic file replacement. Storage
errors are raised, and the writer exposes a storage-error flag while retaining
the previous state. A separate disk viewer can only see the last durable
snapshot. Limits are 10,000 subjects, 50,000 revisions and 1,000 reports per run;
start another run when reached. The latest 100 ingestion errors are retained.
This store is intended for small local monitoring sessions. It has no multiwriter
coordination, distributed execution or power-loss durability guarantee.

## Configure optional investigation

Monitoring and export work with the base package. Select an inference profile
only when you want model-generated reports:

| Backend | Installation | Execution |
| --- | --- | --- |
| `basic` (default) | `cuphoton[xlens-inference]` | One bounded scalar snapshot and response; Python 3.12–3.14 |
| `nooa` | `cuphoton[xlens-nooa]` | Iterative evidence queries and isolated Python; Linux x86-64/AArch64, Python 3.12/3.13 |

NOOA is installed from its upstream Python distribution only when its extra is
selected. The base, GPU and Dragon profiles do not select it. From a checkout:

```bash
uv sync --locked --python 3.13 --extra xlens-nooa
```

The extra pins NOOA 0.0.12 and includes the HTTP client. Install your Linux
distribution's `bubblewrap` and libseccomp runtime packages separately
(`bubblewrap` and `libseccomp2` on Ubuntu), and run as a non-root user.
Check that the host permits the required namespaces:

```bash
bwrap --unshare-all --unshare-user --disable-userns --die-with-parent \
  --ro-bind / / true
```

The command should exit successfully without output. On Ubuntu 24.04, AppArmor
restricts unprivileged user namespaces by default. Errors such as
`bwrap: setting up uid map: Permission denied` or
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` can mean that
bubblewrap needs an AppArmor profile allowing `userns`. An administrator can
install the following profile as `/etc/apparmor.d/bwrap` and load it with
`sudo apparmor_parser --replace /etc/apparmor.d/bwrap`, then rerun the check:

```text
abi <abi/4.0>,
profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
}
```

This profile grants the namespace permission to `/usr/bin/bwrap`. See the
[Ubuntu namespace guidance](https://documentation.ubuntu.com/release-notes/24.04/#unprivileged-user-namespace-restrictions)
and [AppArmor documentation](https://gitlab.com/apparmor/apparmor/-/wikis/unprivileged_userns_restriction)
for site policy and administrator alternatives. The CLI and live session check
the interpreter, package version, namespace support and syscall filter before
enabling investigation.
Python 3.14 supports the basic backend; it cannot run the NOOA worker.

For a basic investigation, supply a TOML file with the exact model identifier
accepted by your endpoint:

```toml
[inference]
backend = "basic"
api = "chat_completions"
base_url = "http://127.0.0.1:8000/v1"
model = "institution-model"
auth = "none"
timeout_seconds = 120
max_tokens = 4096
token_parameter = "max_tokens"
response_format = "text"
max_samples = 64
max_context_bytes = 262144
max_response_bytes = 65536
max_concurrent = 1
```

After installing and checking NOOA, use this table for iterative investigation.
The shared settings above also apply to this backend:

```toml
[inference]
backend = "nooa"
api = "chat_completions"
base_url = "http://127.0.0.1:8000/v1"
model = "institution-model"
auth = "none"
max_iterations = 8
max_tool_calls = 32
max_evidence_bytes = 8388608
```

For bearer authentication, set `auth = "bearer"` and
`api_key_env = "CUPHOTON_XLENS_API_KEY"`, then inject the secret through your
institution's secret-management procedure. The config contains the variable
name, never the value. No-auth mode requires omitting `api_key_env` and ignores
ambient credential variables. No endpoint, model, credential, model registry or
tracing destination is selected implicitly. Bearer authentication requires HTTPS
for non-loopback endpoints; plain HTTP is accepted for loopback endpoints.
HTTPS uses normal Python certificate verification, including
`SSL_CERT_FILE`/`SSL_CERT_DIR`; xLens configuration cannot disable verification.
Ambient proxies and HTTP redirects are disabled.

Save the configuration as `inference.toml`, then ask about saved measurements:

```bash
cuphoton xlens ask --state xlens-demo/state --config inference.toml \
  --sample sample-4 --sample sample-6 \
  --question "Which measurements need inspection, and why?"
```

`ask` reads one saved-state snapshot, investigates an isolated temporary copy,
and prints the report as JSON. It does not save reports or recover state in the
source run, so it can inspect a run while its application writer remains active.
Use the live session's question box to retain a report in that run.

Applications can await
`Investigator(config).investigate(monitor, sample_ids, question=..., instructions=...)`.
Instructions supply domain terminology and interpretation limits. The request
for the basic backend contains the selected samples' measurements, findings,
observation times, units and revisions. Run-scoped questions include the run's
current samples within the configured sample and byte limits.

NOOA starts with the selected identities and can inspect the bounded scalar
catalog through `status`, `get_sample`, `metric_series`, `compare_samples` and
`recent_findings`. Applications can expose additional diagnostics with a
`domain_views` name/description mapping and a trusted
`domain_query(sample_id, query_name) -> dict` callback. The callback must perform
bounded reads promptly, verify artifact provenance and return JSON: at most
16 KiB per view and 64 KiB total. It runs in the application process and should
not launch long-running work.

The parent brokers queries against a detached evidence snapshot. Metadata and
artifact paths are excluded; full artifacts, evaluation truth and previous model
reports are never added automatically. Applications choose the contents of
measurements, findings, diagnostics, instructions and questions, and must exclude
evaluation truth from those inputs. Selected and queried revisions, plus final
citations, determine report staleness.
Reports remain separate from deterministic results. Every citation must match
an available sample revision, measurement name, value and unit. Reference
validation does not establish the correctness of scientific interpretation.

### Choose the model API

Both backends use the exact configured endpoint and model, with nonstreaming
requests. Select the API your endpoint supports:

| `api` | Request path | Token limit |
| --- | --- | --- |
| `chat_completions` (default) | `<base_url>/chat/completions` | `max_tokens` or `max_completion_tokens`, selected by `token_parameter` |
| `responses` | `<base_url>/responses` | `max_tokens` is sent as `max_output_tokens` |

NOOA uses function tools for `execute_python` and `return_result`; the selected
model must support those calls through the chosen API. The basic backend
requests JSON in ordinary text; `response_format="json_object"` selects JSON
mode when supported. Neither backend requires model enumeration. Unsupported
endpoint/model/capability combinations fail visibly without changing the
configured destination, API or model.

Responses requests use `store=false`. During a NOOA investigation, the client
passes returned reasoning items, including encrypted reasoning content, back to
the endpoint along with tool results so it can continue the same investigation.
Those items are not added to xLens reports. The inference service receives the
requested evidence and applies its own data-handling policy; `store=false` is
not a statement about every form of provider retention.

An institution's [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
can supply the endpoint when configured for the selected model and required
chat/tool capabilities. xLens does not deploy a model server. The
[Chat Completions API](https://developers.openai.com/api/reference/resources/chat)
and [Responses API](https://developers.openai.com/api/reference/resources/responses)
describe the wire formats; compatibility depends on the actual server and model.

### Optional GPT-6 Rosalind configuration

OpenAI lists `gpt-6-rosalind` for organizations approved through its trusted-access
program for authorized internal research. See the
[October 2026 API announcement](https://developers.openai.com/api/docs/changelog#october-2026)
for availability and access guidance. Select it explicitly if your API
organization has access and the endpoint supports the required function tools:

```toml
[inference]
backend = "nooa"
api = "responses"
base_url = "https://api.openai.com/v1"
model = "gpt-6-rosalind"
auth = "bearer"
api_key_env = "CUPHOTON_XLENS_API_KEY"
timeout_seconds = 120
max_tokens = 4096
max_context_bytes = 262144
max_response_bytes = 262144
max_iterations = 8
max_tool_calls = 32
```

Provide the credential through the named environment variable, then use the
same `cuphoton xlens ask` or live-session commands. This is an opt-in
configuration example: cuPhoton does not provide API access, select Rosalind
automatically or distribute model weights. A live Rosalind tool-call round trip
has not been qualified for this integration. Test a small synthetic
investigation with your endpoint before relying on it for scientific work.

### NOOA execution boundary

NOOA runs its CodeAct loop and generated Python in a separate Linux process.
Bubblewrap isolates filesystem, process and network namespaces. The worker sees
read-only Python runtime, declared NOOA dependencies and system libraries, plus
a 16 MiB temporary workspace. It starts without ambient environment variables,
host home directories, monitor storage or application files. Only the parent
HTTP client holds model credentials and endpoint settings. The adapter suppresses
NOOA's automatic tracing initialization and unused LiteLLM journal flush, and
the worker has no network access.
The worker has an empty, read-only `/proc`: process command lines and mount
tables are unavailable, including the host paths used to construct its mounts.
Installed interpreter and dependency files remain trusted runtime inputs.

The parent uses system libseccomp to build a syscall deny list that bubblewrap
installs before starting Python. It blocks selected process-inspection,
mount-management and kernel-management calls, including mount metadata queries.
This adds a layer to the namespace and resource restrictions; it is not a
complete syscall allow list. NOOA is unavailable if the filter cannot be built
or installed. The basic backend does not load libseccomp.

The parent enforces model-turn, tool/read, byte, sample, concurrency and
total-time limits. The worker has per-process address-space, CPU-time and
file-descriptor limits, and cannot create child processes or threads. Its root
and device filesystems are read-only; temporary writes share the bounded
workspace. Cancellation terminates the isolated session and its
descendants; the remote model server may continue work it already accepted.
These controls do not provide an aggregate host-resource quota or qualify the
viewer as a public service for arbitrary untrusted code.

The worker can analyze the evidence exposed to it. It does not automatically
acquire archive access, application files, GPU access or job-submission tools.
The basic backend performs no code or tool execution.

## Live monitoring and browser questions

A `MonitorSession` owns polling, background investigations and browser requests
in the same process as the monitor. Supply the application's ordinary polling
callback:

```python
from cuphoton.xlens import InferenceConfig, MonitorSession, serve

session = MonitorSession(
    monitor,
    config=InferenceConfig.from_toml("inference.toml"),
    poll=poll_ready_inputs,  # Zero-argument callback; ingest ready inputs.
    poll_interval=1.0,
    auto_investigate=True,
    instructions="Describe measured changes and suggest inspections.",
)
serve(Dashboard(monitor), session=session)
```

Automatic investigation covers each sample's current revision with a warning
or error finding and no automatic report, including samples saved before the
session started.
Questions can target selected samples or the run, with progress and cancellation
in the browser. Polling and deterministic analysis continue while a model
request is running. Unavailable inference and bounded admission are visible
states; they do not turn incomplete input into valid measurements. Replaying an
unchanged revision does not create another automatic investigation.

For the generic saved-state viewer, use `cuphoton xlens serve --state ...`
with `--inference-config inference.toml` and, when desired,
`--auto-investigate`. Applications supply the polling callback through
`MonitorSession`; the generic viewer does not discover application input files.
The configured server owns the state writer; stop any previous writer before
starting it. A static export retains plots and reports but cannot start
investigations.

Shared investigation access requires an independent browser token, supplied
with `access_token_env="XLENS_BROWSER_TOKEN"` or `--access-token-env` and at least
24 characters. Enter that token in the browser's password field; it stays in
page memory and is not embedded in the HTML export or bootstrap. It is distinct
from the model API credential, which never goes to the browser. The token
protects asking and cancelling questions only. Anyone who can reach an allowed
host name can read the dashboard, samples, and reports without that token.
Use institutional
TLS and access controls when sharing the service. Shared deployments run the
lightweight HTTP viewer behind their reverse proxy and service manager;
cuPhoton does not configure TLS, process supervision or public access policy.

The built-in server rejects unknown `Host` authorities on every route and
checks browser `Origin` against its trusted authorities. Loopback binds accept
`127.0.0.1`, `localhost`, and `[::1]` with the actual listening port. For a
proxy or LAN hostname, pass `allowed_hosts=["monitor.example"]` to
`serve`/`make_server`, or repeat `--allowed-host monitor.example` with
`serve --state`. Include the public port when it appears in the browser URL;
omit it for a default-port URL. Site applications configure their own server's
allowlist. These checks do not replace deployment access controls.
