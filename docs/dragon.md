# Dragon transport and coordination

Dragon places and manages cuPhoton workers; the selected numerical backend
runs each image pair inside its worker. Transport selection belongs to the
Dragon launcher. It does not change cuPhoton's numerical backend or fit options.

## Execution models

The [XPOIS batch executor](components/xpois.md#launch-with-dragon)
assigns a complete shard to each worker at launch. Workers process their
assigned items and send one terminal result to the coordinator in single-pass
mode. Repeated rounds and component commands retain workers as follows:

| Path | Coordination and retained state |
|---|---|
| Single-pass XPOIS | One assigned shard and one terminal result per worker; no per-round command queue. |
| [Repeated XPOIS rounds](components/xpois.md#repeat-a-batch-in-persistent-workers) | `--warmup-rounds` and `--measure-rounds` reuse workers and CUDA contexts. Each worker has a command queue on its own host and repeats its assigned shard. |
| [Shared executors](../src/cuphoton/core/executors.py) and [component commands](components/xscan.md#distributed-inference) | xFit, XScan and `xscan run-pipeline` use the shared Dragon/MPI lifecycle. Dragon uses consumer-local command queues for both a single pass and repeated rounds; component workers retain their input/model context. |

### Persistent behavior

Persistence does not remove all per-item work. XPOIS still reads, transfers and
writes ordinary scientific artifacts each round. Standalone xFit retains its
loaded host input; XScan retains metadata, its model and loader across tasks.
Distributed XScan defaults to `--num-workers 0`. Explicitly requested loader
processes start before READY and persist across tasks and rounds. The combined
pipeline keeps intermediate arrays inside each GPU worker while it
processes an image pair. Each GPU handles complete items; these paths do not
split one image across GPUs.

### Launch descriptors

Dragon workers load immutable, hash-checked launch descriptors from the shared
filesystem. Launch arguments carry descriptor references rather than the full
manifest, with a 96 KiB size guard. This bounds process-launch payloads; queue
placement addresses a separate coordination cost.

## Select the application and overlay transports

Dragon has separate application and infrastructure overlay transports.
`-t tcp -o tcp` selects its Python TCP transport for both. With a Dragon 0.14.2
installation that includes native HSTA, select native HSTA for application
traffic and retain Python TCP for the overlay:

```bash
DRAGON_HSTA_FORCE_BACKEND=tcp \
  .venv/bin/dragon -m -N 2 -w slurm -t hsta -o tcp \
  examples/xpois/dragon_batch.py \
  --backend cupy \
  --manifest /shared/manifests/fixed-32.yaml \
  --output-dir /shared/results/xpois-hsta \
  --name fixed-32-hsta16 \
  --max-workers 16 \
  --worker-timeout-sec 3600
```

As in the XPOIS launch example, these paths must be accessible to the allocated
nodes. The manifest needs enough image pairs for the selected workers. Adapt
the node and worker counts to the allocation.

`DRAGON_HSTA_FORCE_BACKEND=tcp` selects HSTA's native TCP implementation. This
configuration uses TCP rather than UCX or RDMA. Inspect the launcher and
transport logs to confirm HSTA started on each node and exited successfully;
record the Dragon version, arguments and environment with the run. INFO logs
can confirm native HSTA startup without providing network byte counters.

HSTA TCP is a useful candidate for workloads with frequent small messages.
Qualify it on the target installation with representative numerical work,
output validation and cleanup checks. The Python TCP launch in the XPOIS guide
remains an explicit alternative. Transport availability and the fastest queue
layout depend on the Dragon build and workload.

## Place command queues with their consumers

The repeated XPOIS and shared Dragon executors use a READY/command/result
protocol. Each worker sends READY after initialization, then waits on its own
command queue. The coordinator waits for all READY messages before releasing
a round. Custom coordinators with the same protocol need to consider queue
placement.

In Dragon 0.14.2's Python TCP transport, remote receives, sends and polls share
an executor. A receive waiting on an empty remote queue can occupy a thread
until a message arrives or its timeout expires. If command queues are all on
the coordinator's node, enough idle remote workers can consume the threads
needed to deliver the remaining READY messages. Both sides then wait.

For this pattern with Python TCP, place each command queue on its consumer's
host. Keep the result queue on its consumer, the coordinator:

```python
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node
from dragon.native.queue import Queue

# worker_node_id comes from the allocation's discovered Dragon node IDs.
worker_host = Node(worker_node_id).hostname
worker_policy = Policy(
    placement=Policy.Placement.HOST_NAME,
    host_name=worker_host,
)
command_queue = Queue(maxsize=1, policy=worker_policy)
```

Use the hostname reported by Dragon's node discovery for both the queue and
worker placement. This makes the worker's idle receive local and the
coordinator's command send remote. It removes the central population of idle
receive waits while retaining command traffic and result collection.

`DRAGON_TRANSPORT_TCP_MAX_THREADS` can increase the Python TCP executor ceiling.
It allowed the centralized layout to make progress in testing, but preserved
its idle remote receives and higher control latency. Native HSTA has a different
progress implementation; the Python TCP thread setting still applies to a
Python TCP overlay. Consumer-local queues are not universally faster under HSTA.

The repeated-round and shared executors use consumer-local
command queues. The [single-pass XPOIS path](../src/cuphoton/xpois/dragon.py)
uses only a result queue and needs no command-queue relocation. Neither path
automatically changes the transport or raises the Python TCP thread ceiling;
those remain launcher settings.

The shared Dragon executor publishes terminal success only after worker
shutdown and lifecycle audits. Per-round success does not establish a
successful complete run. Silent worker exits and shutdown failures fail the
run. After a reported round failure, healthy peers can finish and retain their
artifacts. Check the root terminal summary before consuming a benchmark result.

## Evidence and timing boundaries

A Dragon 0.14.2 control experiment used two nodes, with the coordinator on one
and 64 CPU worker processes on the other. Each worker returned a 1024-byte
payload per round; no GPU work or timed receipt-file writes occurred. Each
successful invocation had 12 rounds. Two invocations per configuration reversed
the queue-layout order. The table gives the median of rounds 2–12 for each
invocation; the first round was retained separately.

| Application transport | Command queue host | Python TCP ceiling | Later-round medians |
|---|---|---:|---:|
| Python TCP | Coordinator | 1024 | 47.235 / 45.307 ms |
| Python TCP | Consumer | 1024 | 33.604 / 28.123 ms |
| Native HSTA TCP | Coordinator | 32 for overlay | 2.371 / 2.545 ms |
| Native HSTA TCP | Consumer | 32 for overlay | 3.027 / 3.093 ms |

A separate invocation with coordinator queues and a 32-thread Python TCP
ceiling stalled after 32 of 64 READY messages. Consumer queues completed all
12 rounds at the same ceiling. HSTA's first rounds took 5.6–6.3 ms, compared
with the 2.4–3.1 ms later medians above. These are whole control-round timings,
not network-only measurements or predicted application speedups.

A separate persistent-worker imaging harness compared MPI and both queue
layouts under Python TCP and HSTA TCP on eight GB200 GPUs. It processed
16 image-pair occurrences per round from two base pairs, for three rounds per
configuration. All 240 measured scientific outputs agreed. Batch medians
ranged from 7.515 to 7.719 seconds: the large control-only improvement did not
produce a comparable whole-pipeline gain. These results do not qualify a
512-GPU speedup or establish the best layout at larger node counts. This was
a separate harness experiment, with one invocation per configuration.

Later two-node/eight-GPU product checks covered XPOIS, standalone xFit and
XScan, and the combined pipeline under Dragon and MPI. They established
numerical parity, persistent identities and cleanup for the tested revisions.
They predate later lifecycle, loader and input-ownership fixes; they do not
qualify the final implementations in #50/#54/#56.

### 256-GPU follow-up

A later instrumented harness compared MPI and Dragon on 256 GPUs across
64 nodes. Fixed batches contained 512 image-pair occurrences (two per GPU);
weak batches contained 4,096 (16 per GPU). All Dragon treatments used the
1024-thread Python TCP ceiling for the applicable application or overlay
transport. The table shows the range of
batch duration minus longest worker duration across all three rounds of each
accepted Dragon invocation, including the first command round:

| Dragon treatment | Fixed batch remainder | Weak batch remainder |
|---|---:|---:|
| Python TCP, coordinator queues | 209.590–357.407 ms | 361.694–462.434 ms |
| Python TCP, consumer queues | 24.325–32.729 ms | 39.762–45.234 ms |
| Native HSTA TCP, coordinator queues | 8.126–8.629 ms | 7.863–9.451 ms |

The fixed-batch consumer results combine two accepted launches; each other
table cell comes from one. The complete MPI/Dragon campaign retained 10 accepted
launches, with 30 measured rounds and 58,368 validated image-pair outputs,
alongside two startup failures and four unrun launches. Consumer-local queues
and HSTA reduced the observed coordination remainder, which also includes
dispatch, serialization and collection. It is not a network-only timer.

The coordinator-queue control also lacked the historical multi-second spike.
Allocation, scheduler segmentation and logging differed from the earlier
campaign, so the spike's cause remains unresolved. There was no 512-GPU
retest. These measurements exercised the retained harness, not the latest
product PR revisions; they do not establish large-scale performance or launch
reliability for those revisions.

### Compare timing boundaries

For a new comparison, hold the corpus, item count, numerical configuration,
worker CPU budget and runtime versions constant. Rotate run order and retain
the first command round as well as subsequent rounds. Record separately:

- Launcher-to-exit wall time, including startup, warmup and shutdown.
- Time from harness or executor entry until all workers report READY.
- Whole-batch time from release through collection of worker completions.
- Worker execution time, with the treatment of output and receipt writes.
- Coordinator artifact audits, scientific finalization and shutdown.

### Executor timing fields

In persistent-round reports, `batch_wall_sec` stops at completion collection;
coordinator artifact audits and scientific finalization follow it. The shared
executor records component merging separately as `finalization_sec`. Keep
readiness and shutdown outside this batch interval, and measure external
launcher-to-exit time independently. MPI finalization and process exit are
outside the executor's reported coordinator time.

Requested warmup rounds exercise real dispatch, work and collection. Their
outputs and timings remain in the report. Only measured rounds enter timing
statistics; failed warmup rounds, measured rounds or cleanup invalidate the
aggregate. A numerical warmup performed before READY in a separate harness
does not necessarily warm the first command round. Match these policies before
comparing results.

### Interpreting comparisons

Subtracting the longest worker duration from the whole-batch duration combines
start skew, synchronization, receipt writes, serialization and collection. It
cannot isolate network time. Validate complete output identities and scientific
results, worker exits and runtime cleanup alongside timings. The control probe
also observed an external Dragon launcher exit of zero after an application
failure; inspect the application's terminal result as well as the launcher.

[The pipeline/stage benchmark](components/pipeline-stage-benchmark.md)
compares a resident device pipeline with fresh processes and intermediate
files on one GPU. It measures workflow reuse and process/file costs, retaining
raw timings and separately reporting the cost of additional hash verification.
It does not compare Dragon transports or reproduce the stock component CLI
chain.

Free-threaded Python does not free an executor thread blocked in a native
receive: that receive already releases the GIL. A separate serialization or
dispatch profile is needed to establish whether Python execution is a remaining
bottleneck. The experiments above used conventional CPython 3.12; they do not
qualify a free-threaded Dragon runtime.
