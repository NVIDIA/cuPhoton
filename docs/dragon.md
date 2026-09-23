# Dragon transport and coordination

Dragon places and manages cuPhoton workers; the selected numerical backend
runs each image pair inside its worker. Transport selection belongs to the
Dragon launcher. It does not change cuPhoton's numerical backend or fit options.

The [XPOIS batch executor](components/xpois.md#launch-with-dragon) assigns a
complete shard to each worker at launch. Workers process their assigned items
and send one terminal result to the coordinator. This avoids a command exchange
for every image pair.

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

## Place queues for a custom command loop

A custom coordinator that repeatedly releases work can have a different
communication pattern from cuPhoton's static shard executor. Suppose each worker
sends READY and then waits on its own command queue, while the coordinator waits
for all READY messages before sending commands.

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

The [existing XPOIS executor](../src/cuphoton/xpois/dragon.py) has one result
queue and passes each shard directly to its worker. It does not need this
command-queue change. For new long-lived services, send useful chunks of work
between coordination points and keep intermediate arrays inside their GPU
worker. If central fan-out remains expensive, a node coordinator can dispatch
to local GPU workers and aggregate their completions while preserving all item
identities and failures.

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
512-GPU speedup or establish the best layout at larger node counts.

For a new comparison, hold the corpus, item count, numerical configuration,
worker CPU budget and runtime versions constant. Rotate run order and retain
the first command round as well as subsequent rounds. Record separately:

- Launcher-to-exit wall time, including startup, warmup and shutdown.
- Time from harness entry until all workers are ready.
- Whole-batch time from release through collection.
- Worker execution time, with the treatment of output and receipt writes.

Subtracting the longest worker duration from the whole-batch duration combines
start skew, synchronization, receipt writes, serialization and collection. It
cannot isolate network time. Validate complete output identities and scientific
results, worker exits and runtime cleanup alongside timings. The control probe
also observed an external Dragon launcher exit of zero after an application
failure; inspect the application's terminal result as well as the launcher.

Free-threaded Python does not free an executor thread blocked in a native
receive: that receive already releases the GIL. A separate serialization or
dispatch profile is needed to establish whether Python execution is a remaining
bottleneck. The experiments above used conventional CPython 3.12; they do not
qualify a free-threaded Dragon runtime.
