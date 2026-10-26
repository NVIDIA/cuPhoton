# cuPhoton xMon

`cuphoton.xmon` collects scientific measurements, deterministic findings and
evidence references. It provides a dashboard, JSON API and standalone HTML
export. Applications own input formats, scientific rules, units, thresholds
and detailed views. Optional model investigation reads selected scalar evidence
and writes separate advisory reports.

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
The dashboard supports sample/category filtering, scalar history, scientific
findings, provenance and separate model reports. `/api/state` supplies JSON;
`/health` supplies collector/storage status. Reload the served page to read
the latest saved snapshot. An offline export is a fixed snapshot.

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

Install `cuphoton[xmon-inference]` from your selected source or artifact. From
a checkout use `uv sync --locked --extra xmon-inference`. HTTPX is the only new
direct optional dependency. Supply a TOML file with an explicit endpoint and
the exact model ID that endpoint accepts:

```toml
[inference]
base_url = "http://127.0.0.1:8000/v1"
model = "institution-model"
auth = "none"
timeout_seconds = 60
max_tokens = 2048
token_parameter = "max_tokens"
response_format = "text"
max_samples = 64
max_context_bytes = 65536
max_response_bytes = 65536
max_concurrent = 1
```

For authentication, set `auth = "bearer"` and
`api_key_env = "CUPHOTON_XMON_API_KEY"`, then inject that secret into the process
environment using your institution's secret-management procedure. The config
contains the variable name, never the value. No-auth mode requires omitting
`api_key_env` and ignores all ambient credential variables. No endpoint, model,
credential or tracing destination is selected implicitly. Client TLS validation
uses normal Python trust, including `SSL_CERT_FILE`/`SSL_CERT_DIR`; it cannot be
disabled through xMon configuration. Ambient proxies and HTTP redirects are
disabled so they cannot silently change the configured destination.

```bash
cuphoton xmon ask --state xmon-demo/state --config inference.toml \
  --sample sample-4 --sample sample-6 \
  --question "Which measurements need inspection, and why?"
```

Python applications can await
`Investigator(config).investigate(monitor, sample_ids, question=..., instructions=...)`.
Instructions should supply domain terminology and interpretation limits.
The request includes the selected IDs/revisions, observation times, measurements
and deterministic findings. It excludes metadata, artifact contents/paths,
domain views, previous reports and any separately held evaluation truth.
Applications are responsible for the contents of their measurements, findings,
source IDs, instructions and questions. Credentials never enter saved state,
dashboard configuration or provenance. The browser is a read-only viewer;
investigation runs through Python or the CLI with server-side credentials.

### Wire contract and capability limits

xMon makes one nonstreaming `POST <base_url>/chat/completions` request with
`model`, text `messages`, `stream=false`, and the configured token-limit field.
The baseline requests JSON in ordinary text and validates it locally. Select
`max_completion_tokens` for endpoints requiring that field; IDs are passed
unchanged and model enumeration is unnecessary. `response_format="json_object"`
explicitly requests JSON mode when supported. It does not require a JSON-schema
structured-output API. Unsupported chat/model/output combinations fail visibly;
xMon never switches endpoints, models or output modes automatically.

Streaming, generated-code execution and model-selected tool calls are not part
of this adapter. It has a narrower evidence-only authority: the model receives
one bounded snapshot and returns narrative plus citations. It cannot browse
files or execute Python. Calls have byte/token/sample limits, bounded concurrent
admission, and a total client deadline. Cancellation closes the client request;
the endpoint may continue work it already accepted. Reports record completed,
failed, busy, timed-out, cancelled, interrupted or stale states separately from
measured results. A citation must match an available sample revision, measurement
name, scalar value and unit. Validation of those references does not establish
the correctness of the narrative's scientific interpretation.

An institution's existing [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
can provide this interface when its selected model has an appropriate chat
template. Point the configuration at its `/v1` base and supplied model ID; use
the bearer configuration if the administrator enables API-key authentication.
xMon does not deploy a model server. See the
[Chat Completions API](https://developers.openai.com/api/reference/resources/chat)
for the wire format. Compatibility depends on the server and model: a local
client fixture establishes the requested HTTP contract, not universal server
compatibility.

The adapter creates no traces or telemetry and has no external fallback. The
configured inference service receives the selected evidence and controls its
downstream processing and retention. Shared dashboard deployment needs access
control and TLS supplied by the institution; the lightweight read-only HTTP
server is a local inspection tool, not a production hosting platform.
