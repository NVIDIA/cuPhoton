# cuPhoton

cuPhoton provides GPU-accelerated reference workflows for astronomical imaging,
signal correlation and X-ray trace analysis.

## Installation

Choose the extras for your workflow:

```bash
# GPU FITS I/O
python -m pip install 'cuphoton[io]==0.2.1'
# GPU imaging and fitting
python -m pip install 'cuphoton[gpu]==0.2.1'
# GPU workflows with visualization and catalogs
python -m pip install 'cuphoton[gpu,viz,catalog]==0.2.1'
# Scene reconstruction
python -m pip install 'cuphoton[ximage]==0.2.1'
```

Linux wheels support CPython 3.12–3.14 on x86-64 and ARM64. GPU execution
requires a CUDA 13-compatible NVIDIA driver. Release wheels include the native
xDR, xFit, imaging and structured xRay extensions.

For development from a source checkout, see the
[source installation guide](https://nvidia.github.io/cuPhoton/install.html).

## Documentation and source

- [cuPhoton User Guide](https://nvidia.github.io/cuPhoton/)
- [v0.2.1 source](https://github.com/NVIDIA/cuPhoton/tree/v0.2.1)
- [v0.2.1 release notes](https://github.com/NVIDIA/cuPhoton/releases/tag/v0.2.1)
