# cuPhoton xMon

`cuphoton.xmon` collects scientific measurements, deterministic findings and
evidence references. It provides a dashboard, JSON API and standalone HTML
export. Applications own input formats, scientific rules, units, thresholds
and detailed views. Optional model investigation reads selected scalar evidence
and writes separate advisory reports. The optional NOOA backend can choose
read-only evidence queries and execute Python in an isolated worker.

## Run the invented-measurement example

From a current development checkout, install the base package:

```bash
uv sync --locked
uv run cuphoton xmon quickstart --output-dir xmon-demo
uv run cuphoton xmon serve --state xmon-demo/state --port 8765
```

Open `http://127.0.0.1:8765` or the generated `xmon-demo/report.html`.
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

## Connect an application

```python
from cuphoton.xmon import Dashboard, Finding, Measurement, Monitor, Sample

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
come from the adapter. xMon records processing time in UTC separately. Metadata
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
xMon records references without opening them. A sample is limited to 64 KiB,
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

Two optional profiles use the same explicit OpenAI-compatible Chat Completions
endpoint configuration:

| Profile | Installation | Execution |
| --- | --- | --- |
| Basic | `cuphoton[xmon-inference]` | One bounded scalar snapshot and response; Python 3.12–3.14 |
| NOOA | `cuphoton[xmon-nooa]` | Iterative evidence queries and isolated Python; Linux, Python 3.12/3.13 |

For NOOA, install your distribution's `bubblewrap` package and enable its
supported unprivileged user-namespace configuration. From a checkout use
`uv sync --locked --extra xmon-nooa`. The extra pins NOOA 0.0.10. The backend
fails visibly when the required interpreter, package or isolation is unavailable;
it does not fall back to running generated code in the monitor process.
The deterministic package and basic profile continue to support Python 3.14.

Supply a TOML file with the exact model identifier accepted by your endpoint:

```toml
[inference]
backend = "nooa"
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
max_iterations = 8
max_tool_calls = 32
max_evidence_bytes = 8388608
```

Use `backend = "basic"` for the single-request profile. For bearer
authentication, set `auth = "bearer"` and
`api_key_env = "CUPHOTON_XMON_API_KEY"`, then inject the secret through your
institution's secret-management procedure. The config contains the variable
name, never the value. No-auth mode requires omitting `api_key_env` and ignores
ambient credential variables. No endpoint, model, credential, model registry or
tracing destination is selected implicitly. TLS uses normal Python trust,
including `SSL_CERT_FILE`/`SSL_CERT_DIR`; xMon configuration cannot disable it.
Ambient proxies and HTTP redirects are disabled.

```bash
cuphoton xmon ask --state xmon-demo/state --config inference.toml \
  --sample sample-4 --sample sample-6 \
  --question "Which measurements need inspection, and why?"
```

Applications can await
`Investigator(config).investigate(monitor, sample_ids, question=..., instructions=...)`.
Instructions supply domain terminology and interpretation limits. NOOA starts
with the selected identities and may inspect the run's bounded scalar catalog
through `status`, `get_sample`, `metric_series`, `compare_samples` and
`recent_findings`. Series retain observation times, units and revisions.
Application diagnostics use an optional `domain_views` name/description mapping
and trusted `domain_query(sample_id, query_name) -> dict` callback. The callback
must perform bounded reads promptly, verify its artifact provenance, exclude
evaluation truth, and return JSON: at most 16 KiB per view and 64 KiB total.

The parent process brokers these queries against a detached evidence snapshot.
Metadata and artifact paths are excluded; full artifacts, evaluation truth and
previous model reports are never added automatically. Applications choose the
contents of measurements, findings, diagnostics, instructions and questions.
Selected and queried revisions, plus final citations, determine report staleness.
Reports remain separate from deterministic results. Every citation must match
an available sample revision, measurement name, value and unit. Reference
validation does not establish the correctness of scientific interpretation.

### Model and execution boundary

Each model turn sends a nonstreaming `POST <base_url>/chat/completions` request
with the exact configured `model`, `messages`, `stream=false`, and token-limit
field. Choose `max_completion_tokens` when required by the endpoint. NOOA also
sends function tools for `execute_python` and `return_result`; its selected model
must support that Chat Completions tool format. The basic profile requests JSON
in ordinary text; `response_format="json_object"` requests optional JSON mode for
that profile. Neither profile requires a JSON-schema structured-output API or
model enumeration. Unsupported endpoint/model/capability combinations fail
visibly without switching the configured destination or model.

An institution's [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
can supply the endpoint when configured for the selected model and required
chat/tool capabilities. xMon does not deploy a model server. The
[Chat Completions API](https://developers.openai.com/api/reference/resources/chat)
describes the wire format; compatibility depends on the actual server and model.

NOOA runs its real CodeAct iteration loop and generated Python in a separate
Linux process. Bubblewrap isolates its filesystem, process and network
namespaces. The worker sees read-only Python runtime and declared NOOA
dependency mounts, plus a 16 MiB temporary workspace. Unrelated installed application packages and the
base interpreter's site-packages are hidden. Only installed `METADATA` files
are exposed for distribution metadata; local install URLs and plugin entry
points are excluded. Shared third-party namespace roots remain trusted runtime
input. The worker starts without ambient environment variables,
site customization, host home directories, monitor storage or application files.
Only the parent HTTP client holds model credentials and endpoint settings.
Installed interpreter and dependency files are trusted runtime inputs. The
version-pinned adapter suppresses NOOA's automatic tracing initialization, and
the worker network namespace independently prevents external connections.

The parent enforces model-turn, tool/read, byte and total-time limits. The
worker also has a 2 GiB address-space cap, CPU-time limit, descriptor limit and
no core dumps. Cancellation terminates the isolated session and descendants;
the remote model server may continue work it already accepted. Progress records
identify actions and evidence reads without retaining hidden reasoning.
These bounds support local scientific investigation; they do not constitute a
production hosting or arbitrary untrusted-code service guarantee. The configured
inference service receives requested evidence and controls its own retention.

## Live monitoring and browser questions

A `MonitorSession` owns polling, background investigations and browser requests
in the same process as the monitor. Supply the application's ordinary polling
callback and optional diagnostic catalog:

```python
from cuphoton.xmon import InferenceConfig, MonitorSession, serve

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

Automatic investigation selects newly published warning/error findings.
Questions can target selected samples or the run, with progress and cancellation
in the browser. Polling and deterministic analysis continue while a model
request is running. Unavailable inference and bounded admission are visible
states; they do not turn incomplete input into valid measurements. Replaying an
unchanged revision does not create another automatic investigation.

For the generic saved-state viewer, use `cuphoton xmon serve --state ...`
with `--inference-config inference.toml` and, when desired,
`--auto-investigate`. Applications supply polling and domain callbacks through
`MonitorSession`; the generic viewer does not discover application input files.
The configured server owns the state writer; stop any previous writer before
starting it. A static export retains plots and reports but cannot start
investigations.

Shared interactive access requires an independent browser access token, supplied
with `access_token_env="XMON_BROWSER_TOKEN"` or `--access-token-env` and at least
24 characters. Enter that token in the browser's password field; it stays in
page memory and is not embedded in the HTML export or bootstrap. It is distinct
from the model API credential, which never goes to the browser. Use institutional
TLS and access controls when sharing the service. The lightweight server is a
local inspection tool rather than a production hosting platform.
