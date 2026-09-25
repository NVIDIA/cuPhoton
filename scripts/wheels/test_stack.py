# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed single-node wheel checks for optional compute runtimes.

Run with python -I, using mpiexec or dragon for the corresponding mode.
CPU runtime checks execute xPois in real workers; --backend cupy exercises
the GPU-only product executors. Run test_installed.py --mode gpu separately
to qualify native XDR, CFITSIO, KvikIO, and nvCOMP on the same wheel.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import traceback
from importlib import metadata
from pathlib import Path

RUNTIMES = {
    "cutile": ("cuda-tile", "cuda.tile"),
    "mpi": ("mpi4py", "mpi4py.MPI"),
    "dragon": ("dragonhpc", "dragon"),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_install(report):
    require(sys.flags.isolated, "Run this script with python -I")
    distribution = metadata.distribution("cuphoton")
    require(
        "Root-Is-Purelib: false" in (distribution.read_text("WHEEL") or ""),
        "Expected a native wheel",
    )
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(
        not direct.get("dir_info", {}).get("editable", False),
        "Editable installations cannot qualify a wheel",
    )
    recorded = {
        Path(distribution.locate_file(item)).resolve()
        for item in distribution.files or ()
    }
    for name in ("cuphoton", "cuphoton.xpois"):
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path in recorded, f"{name} is outside wheel RECORD: {path}")
    report["versions"] = {"cuphoton": distribution.version}
    report["checks"].append("installed_wheel_origin")


def check_imports(names, report):
    require(bool(names), "--expect must select at least one runtime")
    for name in names:
        require(name in RUNTIMES, f"Unknown runtime: {name}")
        distribution, module = RUNTIMES[name]
        importlib.import_module(module)
        report["versions"][distribution] = metadata.version(distribution)
        report["checks"].append(f"import_{module}")


def fixture(worker_id=0):
    import numpy as np
    from scipy.signal import fftconvolve

    from cuphoton.xpois import (
        GaussianBasisComponent,
        build_gaussian_polynomial_basis,
    )

    reference = np.random.default_rng(913 + worker_id).normal(size=(48, 53))
    components = [GaussianBasisComponent(sigma=1.5, degree=0)]
    basis, _ = build_gaussian_polynomial_basis((9, 9), components)
    scale = 1.7 + worker_id / 10
    target = fftconvolve(reference, scale * basis[0], mode="same") + 0.2
    return reference, target, components, scale


def solve(backend, worker_id=0):
    import numpy as np

    from cuphoton.xpois import solve_constant_kernel

    reference, target, components, scale = fixture(worker_id)
    result = solve_constant_kernel(
        reference,
        target,
        components,
        kernel_shape=(9, 9),
        variance=np.ones_like(target),
        background_degree=0,
        backend=backend,
    )
    require(result.backend == backend, "Requested backend was not used")
    np.testing.assert_allclose(result.kernel_coefficients, [scale], atol=1e-8)
    np.testing.assert_allclose(
        result.background_coefficients, [0.2], atol=1e-8
    )
    np.testing.assert_allclose(result.residual[result.fit_mask], 0, atol=1e-8)
    return {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "backend": result.backend,
        "scale": float(result.kernel_coefficients[0]),
        "fit_pixel_count": result.fit_pixel_count,
    }


def check_workers(results, count):
    import numpy as np

    require(len(results) == count, "Missing worker results")
    ordered = sorted(results, key=lambda item: item["worker_id"])
    require(
        [item["worker_id"] for item in ordered] == list(range(count)),
        "Missing or duplicate worker identities",
    )
    require(
        len({(item["hostname"], item["pid"]) for item in ordered}) == count,
        "Workers did not execute in distinct processes",
    )
    np.testing.assert_allclose(
        [item["scale"] for item in ordered],
        [1.7 + worker / 10 for worker in range(count)],
        atol=1e-8,
    )


def check_gpu(report):
    import cupy as cp
    import torch

    from cuphoton.xscan.training import cupy_to_torch, predict_tensors

    require(cp.cuda.runtime.getDeviceCount() > 0, "A CUDA GPU is required")
    require(torch.cuda.is_available(), "Torch CUDA is required")
    for distribution in ("cupy-cuda13x", "cuda-tile", "numba-cuda", "torch"):
        report["versions"][distribution] = metadata.version(distribution)
    report["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for backend in ("cpu", "cupy", "numba-cuda", "cutile"):
        report[backend] = solve(backend)
        report["checks"].append(f"xpois_{backend}_known_solution")

    device = torch.device("cuda:0")
    values = (
        cp.arange(2 * 2 * 9 * 9, dtype=cp.float32).reshape(2, 2, 9, 9) / 100
    )
    view = cupy_to_torch(values, device=device)
    require(view.tensor.data_ptr() == values.data.ptr, "DLPack copied data")
    linear = torch.nn.Linear(2 * 9 * 9, 1).to(device)
    with torch.no_grad():
        linear.weight.fill_(1 / (2 * 9 * 9))
        linear.bias.zero_()
    model = torch.nn.Sequential(
        torch.nn.Flatten(), linear, torch.nn.Flatten(0, 1)
    )
    result = predict_tensors(model=model, images=view.tensor, device=device)
    torch.testing.assert_close(
        result["logits"], view.tensor.mean(dim=(1, 2, 3))
    )
    torch.testing.assert_close(
        result["probabilities"], result["logits"].sigmoid()
    )
    torch.cuda.synchronize()
    report["checks"].append("xscan_cupy_dlpack_torch_cuda_inference")


def create_manifest(directory, count):
    import numpy as np

    pairs = []
    for worker_id in range(count):
        reference, target, _, _ = fixture(worker_id)
        pair = {"id": f"pair-{worker_id}"}
        for name, data in (("reference", reference), ("target", target)):
            path = directory / f"{name}-{worker_id}.npy"
            np.save(path, data)
            pair[name] = str(path)
        pairs.append(pair)
    path = directory / "pairs.json"
    path.write_text(
        json.dumps(
            {"schema": "cuphoton.xpois.image-pairs/v1", "pairs": pairs}
        )
    )
    return path


def check_product_result(result, workers, report):
    import numpy as np

    from cuphoton.xpois import build_gaussian_polynomial_basis

    require(result is not None, "Coordinator did not return a result")
    report["executor_summary"] = result.summary
    require(result.status == "success", "Product executor failed")
    summary = result.summary
    if summary["executor"] == "dragon":
        require(len(summary["placements"]) == workers, "Too few GPU workers")
    for worker_id in range(workers):
        artifacts = (
            result.run_dir / "items" / f"pair-{worker_id}" / "artifacts"
        )
        residual = np.load(artifacts / "residual.npy")
        np.testing.assert_allclose(residual[4:-4, 4:-4], 0, atol=1e-8)
        _, _, components, scale = fixture(worker_id)
        basis, _ = build_gaussian_polynomial_basis((9, 9), components)
        np.testing.assert_allclose(
            np.load(artifacts / "kernel.npy"), scale * basis[0], atol=1e-8
        )
    report["checks"].append(f"xpois_{summary['executor']}_gpu_executor")


def options():
    from cuphoton.xpois.batch import BatchFitOptions

    return BatchFitOptions(
        kernel_shape=(9, 9),
        basis_sigmas=(1.5,),
        basis_degrees=(0,),
        backend="cupy",
    )


def check_mpi(args, report):
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    require(
        comm.size == args.workers, "MPI world size differs from --workers"
    )
    report["versions"]["mpi4py"] = metadata.version("mpi4py")
    report["mpi_library"] = MPI.Get_library_version()
    if args.backend == "cpu":
        result = solve("cpu", comm.rank)
        total = comm.allreduce(result["scale"], op=MPI.SUM)
        np.testing.assert_allclose(
            total,
            sum(1.7 + rank / 10 for rank in range(comm.size)),
            atol=1e-8,
        )
        results = comm.gather(result, root=0)
        if comm.rank == 0:
            check_workers(results, args.workers)
            report["workers"] = results
            report["checks"].append("mpi_launched_xpois_cpu_collectives")
    else:
        from cuphoton.xpois.mpi import run_mpi_image_pair_batch

        directory = comm.bcast(
            tempfile.mkdtemp(prefix="cuphoton-stack-")
            if comm.rank == 0
            else None,
            root=0,
        )
        root = Path(directory)
        if comm.rank == 0:
            create_manifest(root, args.workers)
        comm.Barrier()
        result = run_mpi_image_pair_batch(
            manifest_path=root / "pairs.json",
            output_root=root / "output",
            run_id="acceptance",
            aggregation_mode="mpi",
            rank_timeout_sec=None,
            attempt_id=None,
            options=options(),
            rank_setup_timeout_sec=120,
        )
        if comm.rank == 0:
            check_product_result(result, args.workers, report)
            shutil.rmtree(root)
    return comm.rank == 0


def check_dragon(args, report):
    from dragon.native.process import ProcessTemplate
    from dragon.native.process_group import ProcessGroup

    report["versions"]["dragonhpc"] = metadata.version("dragonhpc")
    with tempfile.TemporaryDirectory(prefix="cuphoton-stack-") as directory:
        root = Path(directory)
        if args.backend == "cupy":
            from cuphoton.xpois.dragon import run_dragon_image_pair_batch

            result = run_dragon_image_pair_batch(
                manifest_path=create_manifest(root, args.workers),
                output_root=root / "output",
                run_id="acceptance",
                max_workers=args.workers,
                result_timeout_sec=30,
                worker_timeout_sec=120,
                options=options(),
            )
            check_product_result(result, args.workers, report)
            return
        group = ProcessGroup(restart=False)
        try:
            for worker_id in range(args.workers):
                group.add_process(
                    nproc=1,
                    template=ProcessTemplate(
                        target=sys.executable,
                        args=(
                            "-I",
                            str(Path(__file__).resolve()),
                            "--mode",
                            "worker",
                            "--worker-id",
                            str(worker_id),
                            "--report",
                            str(root / f"worker-{worker_id}.json"),
                        ),
                        cwd=directory,
                    ),
                )
            group.init()
            group.start()
            group.join(timeout=120)
            statuses = list(group.inactive_puids)
            require(
                len(statuses) == args.workers, "Missing Dragon exit status"
            )
            require(
                all(code == 0 for _, code in statuses), "Dragon worker failed"
            )
            results = [
                json.loads((root / f"worker-{worker_id}.json").read_text())[
                    "result"
                ]
                for worker_id in range(args.workers)
            ]
            check_workers(results, args.workers)
            report["workers"] = results
            report["checks"].append("dragon_launched_xpois_cpu_workers")
        finally:
            group.close(patience=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("imports", "gpu", "mpi", "dragon", "worker"),
        required=True,
    )
    parser.add_argument("--backend", choices=("cpu", "cupy"), default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--expect", default="")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "schema": "cuphoton.wheel-stack-acceptance/v1",
        "mode": args.mode,
        "backend": args.backend if args.mode in {"mpi", "dragon"} else None,
        "requested_workers": (
            args.workers if args.mode in {"mpi", "dragon"} else None
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "checks": [],
        "status": "failed",
    }
    writer = True
    try:
        require(args.workers > 0, "--workers must be positive")
        check_install(report)
        if args.mode == "imports":
            check_imports(
                args.expect.split(",") if args.expect else [], report
            )
        elif args.mode == "gpu":
            check_gpu(report)
        elif args.mode == "mpi":
            writer = check_mpi(args, report)
        elif args.mode == "dragon":
            check_dragon(args, report)
        else:
            report["result"] = solve("cpu", args.worker_id)
        report["status"] = "passed"
        return 0
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        # Rank-specific failure receipts avoid races if a collective fails.
        if args.mode == "mpi" and "mpi4py.MPI" in sys.modules:
            from mpi4py import MPI

            if report["status"] != "passed":
                args.report = args.report.with_name(
                    f"{args.report.stem}.rank-{MPI.COMM_WORLD.rank}.json"
                )
            else:
                writer = MPI.COMM_WORLD.rank == 0
        if writer:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
