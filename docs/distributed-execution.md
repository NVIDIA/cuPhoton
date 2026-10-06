# Run a pipeline across GPU nodes

The xPois, xFit and xScan pipeline assigns complete image pairs to GPU
workers. Each worker loads its model once and processes its assigned pairs
through subtraction, dipole fitting and classification. MPI and Dragon
provide alternative launchers for the same workload. The
[distributed architecture](distributed.md) explains work assignment, process
ownership and data movement.

Three inputs determine a run:

| Input | Controls |
| --- | --- |
| `pipeline.json` | Model, numerical settings, image files and candidate coordinates |
| Slurm allocation or launcher host file | Machines available to the run |
| Launcher and cuPhoton options | Process placement, GPU worker count and repeated rounds |

`mpiexec -n 8` requests eight processes. The host file or scheduler allocation
determines where they run. With Dragon, `--nodes 2` selects two nodes and
cuPhoton's `--max-workers 8` caps the total GPU workers. The examples below
use two nodes with four GPUs each and at least eight image-pair items.
They use the default placement of one worker per GPU.

## Prepare the environment and workload

Use Linux nodes with CUDA 13-compatible drivers and a shared filesystem.
Every participating node must see the same checkout, Python environment,
inputs and writable output path at the same absolute locations. Use matching
CPU architectures and runtime versions across nodes. For SSH launches, set
up noninteractive SSH between the participating hosts and make their runtime
network interfaces mutually reachable.

From a checkout on that shared filesystem, install the GPU dependencies and
both executor extras. Python 3.12 supports both runtimes:

```bash
uv sync --locked --python 3.12 --extra gpu --extra mpi --extra dragon
export CUPHOTON_CHECKOUT="$(pwd -P)"
export CUPHOTON_ENV="$CUPHOTON_CHECKOUT/.venv"
source "$CUPHOTON_ENV/bin/activate"
export CUPHOTON_RUN_ROOT=/shared/cuphoton-demo
mkdir -p "$CUPHOTON_RUN_ROOT"
```

Activation also puts Dragon's backend/helper executables on `PATH`.
Replace `/shared/cuphoton-demo` with your shared scratch directory. The `mpi`
extra installs mpi4py; load the site's compatible Open MPI installation
separately, including its remote executable/library paths. Verify the binding
with `"$CUPHOTON_ENV/bin/python" -c 'from mpi4py import MPI;
print(MPI.Get_library_version())'`. See
[installation profiles](getting-started.md#optional-runtimes) and
[Open MPI's SSH setup](https://docs.open-mpi.org/en/v5.0.7/launching-apps/ssh.html).

Generate a small workload on the CPU before requesting GPUs:

```bash
"$CUPHOTON_ENV/bin/python" examples/distributed-pipeline/prepare_example.py \
  --output "$CUPHOTON_RUN_ROOT/input" --images 8
```

The [preparation script](../examples/distributed-pipeline/prepare_example.py)
reuses the [pipeline benchmark fixture](components/pipeline-stage-benchmark.md).
It writes eight distinct 256 by 256 image pairs, nine candidates per pair,
variance/mask planes, a feature schema, an untrained model and `pipeline.json`.
Its random model exercises execution; its predictions are not an astronomical
classification result. This small workload is a launch smoke test, not a
representative scaling benchmark. The input directory must be new. Generated
paths are absolute: prepare it at its final shared location.

### Supply your own pipeline.json

The complete [pipeline JSON template](../examples/distributed-pipeline/pipeline.example.json)
shows one NPY image pair with a 63 by 63 stamp configuration. Replace every
`<SHA256 ...>` placeholder with the corresponding file's lowercase SHA-256,
and supply actual paths, array shapes/dtypes and candidate coordinates. Hash
`checkpoint_dir/checkpoint.pt` for `checkpoint_sha256`. For eight MPI ranks,
provide at least eight items with distinct `item_id` values; multiple
candidates within a pair still form one item. Use `-n 1` for a one-item run.

Paths in the manifest may be relative to the manifest's directory. Reference
and target images must already share a pixel grid and compatible photometric
units. The checkpoint must be a triplet fusion model with its exact training
feature schema and matching stamp size. The pipeline uses unmasked,
unweighted Gaussian difference fits for xFit. Optional item `variance` and
`fit_mask` describe xPois inputs. FITS descriptors additionally specify the
HDU and reader policy; see the
[pipeline input API](components/xscan.md#persistent-xpois-xfit-and-xscan-pipeline).

The raw JSON schema requires all fields, including null optional values and
the inference policy. The Python `DevicePipelineConfig` and
`DevicePipelineItem` constructors supply defaults; their `to_payload()`
methods produce the full JSON structure. The linked input API shows this
construction for your own files. Keep `device` as `cuda:0`: each worker sees
its assigned GPU under that local ordinal after binding.

## Launch inside a Slurm allocation

Request an interactive allocation on your site's GPU partition. Adapt the
account, partition, CPU, memory and GPU resource options to the cluster:

```bash
salloc --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --exclusive \
  --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --time=00:20:00
```

Run the following commands from the allocated shell, where `SLURM_JOB_ID`
is set. Inspect the selected hosts with
`scontrol show hostnames "$SLURM_JOB_NODELIST"`. The allocation supplies host
discovery; a separate host file is unnecessary.

### Open MPI

This example assumes two four-GPU nodes, with all local GPUs `0,1,2,3`
allocated on each node. `salloc` may leave GPU visibility unset in its shell;
the command below preserves an existing mask and otherwise uses that list.
For other layouts, supply the site's per-node GPU binding instead of this
uniform mask. Preserve scheduler restrictions.
The rank wrapper expects the full per-node list in local-rank order, rather
than a list already narrowed to a single GPU by a per-task launcher.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
mpiexec -n 8 --map-by ppr:4:node --bind-to none \
  -x CUDA_VISIBLE_DEVICES \
  "$CUPHOTON_ENV/bin/cuphoton-openmpi-rank-exec" -- \
  "$CUPHOTON_ENV/bin/cuphoton" xscan run-pipeline --executor mpi \
  --manifest "$CUPHOTON_RUN_ROOT/input/pipeline.json" \
  --output-dir "$CUPHOTON_RUN_ROOT/runs" --name slurm-mpi \
  --warmup-rounds 1 --measure-rounds 3
```

`ppr:4:node` places four ranks on each node. The wrapper binds each local
rank to a different GPU before Python or MPI initializes CUDA. It is specific
to Open MPI; another MPI implementation needs its own launcher or scheduler
GPU binding. See [Open MPI with Slurm](https://docs.open-mpi.org/en/v5.0.7/launching-apps/slurm.html).

### Dragon

Start one Dragon launcher in the allocated shell. Dragon discovers the
allocated nodes, and cuPhoton discovers their GPUs and places its workers:

```bash
"$CUPHOTON_ENV/bin/dragon" --wlm slurm --nodes 2 -t tcp -o tcp \
  "$CUPHOTON_ENV/bin/cuphoton" xscan run-pipeline --executor dragon \
  --manifest "$CUPHOTON_RUN_ROOT/input/pipeline.json" \
  --output-dir "$CUPHOTON_RUN_ROOT/runs" --name slurm-dragon \
  --max-workers 8 --warmup-rounds 1 --measure-rounds 3
```

`--max-workers` caps the total process count. With the default
`--workers-per-gpu 1`, Dragon also limits workers by available GPUs and item
count. This example selects TCP for both Dragon transports. Use the
site's routable interface selection when needed; see
[Dragon transport and performance](dragon-performance.md).

For multiple workers per GPU, see [Dragon GPU sharing and MPS](dragon.md#share-a-gpu-between-workers)
and the [GPU sharing commands](components/xscan.md#share-a-gpu-between-image-pairs).
An explicit MPS connection requires an existing service on each worker host.
The launcher or administrator manages that service separately.

## Launch on named hosts through SSH

Use dedicated hosts available to you without a scheduler, with the same
shared paths and all four GPUs available on each node. Dragon 0.14.2's SSH
launcher does not forward `CUDA_VISIBLE_DEVICES` to its remote backends;
a mask in the launching shell alone does not restrict their GPU discovery.
Copy and edit the representative host files:

```bash
cp examples/distributed-pipeline/mpi-hosts.example.txt \
  "$CUPHOTON_RUN_ROOT/mpi-hosts.txt"
cp examples/distributed-pipeline/dragon-hosts.example.txt \
  "$CUPHOTON_RUN_ROOT/dragon-hosts.txt"
```

The [Open MPI host file](../examples/distributed-pipeline/mpi-hosts.example.txt)
contains process slots. For this example, four slots correspond to four GPU
workers on each host:

```text
gpu-node-01 slots=4
gpu-node-02 slots=4
```

The [Dragon host file](../examples/distributed-pipeline/dragon-hosts.example.txt)
contains only hostnames, one per line, with no comments or blank lines:

```text
gpu-node-01
gpu-node-02
```

Replace these illustrative names with your hosts. For Open MPI, use the same
command as in the Slurm section, adding
`--hostfile "$CUPHOTON_RUN_ROOT/mpi-hosts.txt"` immediately after `mpiexec`
and choosing a fresh run name such as `ssh-mpi`. The visibility list must
describe available GPUs on every listed host.

For Dragon, select its SSH launcher and hostname file:

```bash
"$CUPHOTON_ENV/bin/dragon" --wlm ssh --nodes 2 \
  --hostfile "$CUPHOTON_RUN_ROOT/dragon-hosts.txt" -t tcp -o tcp \
  "$CUPHOTON_ENV/bin/cuphoton" xscan run-pipeline --executor dragon \
  --manifest "$CUPHOTON_RUN_ROOT/input/pipeline.json" \
  --output-dir "$CUPHOTON_RUN_ROOT/runs" --name ssh-dragon \
  --max-workers 8 --warmup-rounds 1 --measure-rounds 3
```

### Inspect or save Dragon's network configuration

Dragon can discover network addresses at launch or read a generated network
configuration. For example, inside a Slurm allocation, run this in a new
directory on the shared filesystem:

```bash
"$CUPHOTON_ENV/bin/dragon-network-config" --wlm slurm --output-to-json
```

The generated `slurm.json` describes that allocation's host identities and
network addresses. Pass its absolute path with the Dragon launcher option
`--network-config /shared/path/to/slurm.json`. For SSH discovery, use
`--wlm ssh --hostfile /shared/path/to/dragon-hosts.txt --output-to-json`,
which writes `ssh.json`. Regenerate after the selected hosts or their network
configuration change. Use the site's `--network-prefix` when hosts have
multiple interfaces and the default selection is unsuitable. This option
is an interface-name regular expression, such as `'^ib0$'`, rather than an IP
subnet. Apply the same selection to discovery and launch. The generated
file is separate from the cuPhoton workload manifest. See
[Dragon's multi-node guide](https://dragonhpc.github.io/dragon/doc/_build/html/uses/multinode.html).

## Inspect the run

Each command creates `runs/<name>/summary.json` and retains warmup and
measured outputs under `rounds/warmup-*` and `rounds/measure-*`. Inspect the
summary and worker records for actual host/GPU placement, successful item
completion and cleanup. MPI rejects more ranks than image-pair items;
unexpected sharing or a physical GPU identity that disagrees with the
requested placement also fails validation.

Choose a new `--name` for another attempt. Readiness, batch execution,
validation and cleanup have separate timing fields. Batch timing includes
dispatch, input loading, numerical work and worker output publication;
coordinator audits follow that timer. The
[pipeline guide](components/xscan.md#persistent-xpois-xfit-and-xscan-pipeline)
describes the output and timing contract. Preserve setup and whole-invocation
costs when comparing configurations, and release the Slurm allocation after
the launchers have exited and the workers have stopped.
