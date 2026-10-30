# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check a native wheel from an isolated Python outside the source tree."""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import gc
import gzip
import hashlib
import importlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

GPU_DISTRIBUTIONS = (
    "cupy-cuda13x",
    "cuda-pathfinder",
    "kvikio-cu13",
    "libkvikio-cu13",
    "nvidia-libnvcomp-cu13",
    "nvidia-nvcomp-cu13",
    "cuda-toolkit",
    "nvidia-cuda-runtime",
    "nvidia-cublas",
    "nvidia-cuda-nvrtc",
    "nvidia-cufile",
    "nvidia-nvjitlink",
    "numba",
    "numba-cuda",
    "torch",
)
HDU_INDICES = (5, 1, 3, 2, 4)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def passed(report, name):
    report["checks"].append(name)
    print(f"PASS {name}", file=sys.stderr, flush=True)


def module_path(distribution, name):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    installed = {
        Path(distribution.locate_file(item)).resolve()
        for item in distribution.files or ()
    }
    require(path in installed, f"{name} is outside wheel RECORD: {path}")
    return path


def loaded_native_libraries():
    prefixes = (
        "libcudart",
        "libcublas",
        "libcufile",
        "libcfitsio",
        "libnvcomp",
        "libkvikio",
    )
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(prefixes):
            paths.add(fields[5])
    return sorted(paths)


def check_install(mode, components, report):
    require(sys.flags.isolated, "Run this script with python -I")
    distribution = metadata.distribution("cuphoton")
    wheel = distribution.read_text("WHEEL") or ""
    require("Root-Is-Purelib: false" in wheel, "Expected a native wheel")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(
        not direct_url.get("dir_info", {}).get("editable", False),
        "Editable installations cannot qualify a wheel",
    )
    report["wheel_metadata"] = wheel
    report["versions"] = {"cuphoton": distribution.version}
    report["paths"] = {
        name: str(module_path(distribution, name))
        for name in ("cuphoton", *(f"cuphoton.{name}" for name in components))
    }
    for name in GPU_DISTRIBUTIONS:
        try:
            report["versions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        require(mode != "base", f"Base environment includes {name}")
    if "xfit" in components:
        check_xfit_cpu(report)
    if mode == "base":
        require(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in (
                    "cupy",
                    "kvikio",
                    "nvidia.nvcomp",
                    "cuda.pathfinder",
                    "cuphoton.xfit._native_ext",
                )
            ),
            "Base imports loaded an optional GPU package",
        )
    for argv in (["--version"], *([name, "--help"] for name in components)):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "from cuphoton.core.cli import main; "
                f"raise SystemExit(main({argv!r}))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        require(bool(result.stdout.strip()), f"Empty CLI output for {argv}")
        if argv == ["--version"]:
            require(
                result.stdout.strip() == distribution.version,
                "CLI and installed metadata versions differ",
            )
    passed(report, "installed_origin_and_base_imports")
    return distribution


def xfit_fixture(dtype, mode="difference"):
    import numpy as np

    from cuphoton.xfit import GaussianDipoleModel

    truth = np.asarray(
        [
            [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
        ],
        dtype=dtype,
    )
    model = GaussianDipoleModel((11, 15), dtype=dtype)
    initial = truth + np.asarray(
        [0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1], dtype=dtype
    )
    images = model.evaluate(truth, mode=mode)
    return (
        truth,
        images,
        {
            "model": model,
            "initial": initial,
            "mode": mode,
            "variance": np.full(images.shape, 0.5, dtype=dtype),
        },
    )


def check_xfit_cpu(report):
    import numpy as np

    from cuphoton.xfit import fit_dipoles

    truth, images, options = xfit_fixture(np.float64)
    result = fit_dipoles(images, backend="numpy", **options)
    require(result.converged.all(), "NumPy xFit did not converge")
    np.testing.assert_allclose(result.parameters, truth, rtol=1e-7, atol=1e-7)
    passed(report, "xfit_numpy_gaussian_fit")


def check_xfit_native(distribution, report):
    from cuphoton.xfit._native import load_extension

    load_extension()
    extension = module_path(distribution, "cuphoton.xfit._native_ext")
    report["paths"]["xfit_native_extension"] = str(extension)
    report["xfit_extension_sha256"] = hashlib.sha256(
        extension.read_bytes()
    ).hexdigest()
    report["loaded_native_libraries"] = loaded_native_libraries()
    passed(report, "xfit_native_extension_load")


def check_xfit_gpu(report):
    import cupy as cp
    import numpy as np

    from cuphoton.xfit import fit_dipoles
    from cuphoton.xfit._native import release_workspace

    require(cp.cuda.runtime.getDeviceCount() > 0, "A CUDA GPU is required")
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report["gpu"] = {
        "name": properties["name"].decode(),
        "compute_capability": [properties["major"], properties["minor"]],
    }
    report["cupy_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
    report["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
    for dtype in (np.float32, np.float64):
        tolerance = 3e-5 if dtype is np.float32 else 5e-12
        for mode in ("difference", "split"):
            _, images, options = xfit_fixture(dtype, mode)
            expected = fit_dipoles(images, backend="cupy", **options)
            actual = fit_dipoles(images, backend="native", **options)
            require(actual.backend == "native", "xFit fell back from native")
            require(actual.converged.all(), "Native xFit did not converge")
            require(
                actual.uncertainty_valid.all(),
                "Native xFit covariance is invalid",
            )
            for name in ("status", "evaluations", "converged"):
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(expected, name)
                )
            for name in (
                "parameters",
                "residuals",
                "chi_square",
                "covariance",
                "standard_errors",
            ):
                np.testing.assert_allclose(
                    getattr(actual, name),
                    getattr(expected, name),
                    rtol=tolerance,
                    atol=tolerance,
                    err_msg=f"native xFit {name}",
                )
            passed(
                report, f"xfit_native_{np.dtype(dtype).name}_{mode}_parity"
            )
    release_workspace()


def check_xray(distribution, report, *, gpu=False):
    """Exercise installed native roots, DD refinement and packaged CUDA."""
    import numpy as np

    from cuphoton.xray import _structured_native as native

    extension = module_path(distribution, "cuphoton.xray._structured_ext")
    report["paths"]["xray_extension"] = str(extension)
    kernel = "cuphoton/xray/src/structured_roots.cu"
    require(
        kernel in {str(path) for path in distribution.files or ()},
        "Structured CUDA source is absent from wheel RECORD",
    )
    require(
        Path(distribution.locate_file(kernel)).is_file(),
        "Structured CUDA source is absent from the installation",
    )
    polynomial = np.zeros(598)
    polynomial[0], polynomial[-1] = 1, -2
    rows = polynomial[None, :]
    roots, stats, _ = native.solve_roots(rows)
    require(stats[0, :3].tolist() == [0, 597, 0], "xRay CPU roots failed")
    require(np.isfinite(roots).all(), "xRay CPU roots are nonfinite")
    expected = 2 ** (1 / 597)
    np.testing.assert_allclose(np.abs(roots), expected, rtol=0, atol=2e-11)
    index = int(np.argmin(np.abs(roots[0] - expected)))
    seed = roots[0, index]
    separation = np.array([np.min(np.abs(np.delete(roots[0], index) - seed))])
    refined, _, status = native.refine_selected(
        polynomial, np.array([[seed.real, seed.imag]]), separation
    )
    require(status[0, 0] == 0, "xRay DD refinement failed")
    np.testing.assert_allclose(refined, [[expected, 0]], rtol=0, atol=2e-15)
    passed(report, "installed_xray_cpu_roots_and_dd_refinement")
    if gpu:
        import cupy as cp

        from cuphoton.xray._structured_gpu import StructuredRootsGPU

        solver = StructuredRootsGPU(cp)
        try:
            gpu_roots, gpu_stats, _ = solver.solve(rows)
        finally:
            solver.close()
        np.testing.assert_array_equal(gpu_stats[:, :3], stats[:, :3])
        remaining = list(roots[0])
        for root in gpu_roots[0]:
            nearest = min(
                range(len(remaining)), key=lambda i: abs(root - remaining[i])
            )
            require(
                abs(root - remaining.pop(nearest)) <= 2e-11,
                "Installed xRay GPU roots differ from CPU",
            )
        passed(report, "installed_xray_gpu_roots")


def create_fits(directory):
    import numpy as np
    from astropy.io import fits

    paths = []
    grid = np.arange(17 * 23, dtype=np.int32).reshape(17, 23) - 200
    for index in range(6):
        integers = grid + index * 1000
        floats = integers.astype(np.float32) / 4
        hdus = [
            fits.PrimaryHDU(),
            fits.ImageHDU(integers.astype(np.int16)),
        ]
        for data in (integers, floats):
            for compression in ("GZIP_1", "GZIP_2"):
                hdus.append(
                    fits.CompImageHDU(
                        data=data,
                        compression_type=compression,
                        tile_shape=(6, 7),
                        quantize_level=0,
                    )
                )
        path = directory / f"image-{index}.fits"
        fits.HDUList(hdus).writeto(path)
        paths.append(path)
    return [paths[index] for index in (4, 0, 5, 2, 1, 3)]


def reference_images(paths, hdu_indices=HDU_INDICES, section=None):
    import numpy as np
    from astropy.io import fits

    outputs = [[] for _ in hdu_indices]
    for path in paths:
        with fits.open(path, memmap=False) as hdus:
            for output, hdu_index in zip(outputs, hdu_indices, strict=True):
                data = hdus[hdu_index].data
                if section is not None:
                    data = data[section]
                output.append(data.astype(data.dtype.newbyteorder("=")))
    return tuple(np.stack(output) for output in outputs)


def check_native(distribution, paths, report):
    from cuphoton.xdr.nvcomp_batch import (
        cpp_helper_available,
        get_native_batch_builder,
        get_native_plan_files,
    )

    require(cpp_helper_available(), "Native extension cannot be loaded")
    get_native_batch_builder(required=True)
    planner = get_native_plan_files(required=True)
    extension = module_path(distribution, "cuphoton.xdr._nvcomp_batch_ext")
    report["paths"]["native_extension"] = str(extension)
    report["extension_sha256"] = hashlib.sha256(
        extension.read_bytes()
    ).hexdigest()
    report["loaded_native_libraries"] = loaded_native_libraries()
    plans = planner(
        [str(path) for path in paths],
        list(range(len(paths))),
        list(HDU_INDICES),
        2,
        None,
    )
    require(len(plans) == len(paths), "Native planner lost files")
    require(
        [plan[1] for plan in plans] == list(range(len(paths))),
        "Native planner changed file order",
    )
    passed(report, "native_extension_builder_and_cfitsio_planner")


def check_images(actual, expected):
    import cupy as cp
    import numpy as np

    require(len(actual) == len(expected), "Output HDU count differs")
    for output, reference in zip(actual, expected, strict=True):
        require(isinstance(output, cp.ndarray), "Output is not on the GPU")
        require(output.shape == reference.shape, "Output shape differs")
        require(output.dtype == reference.dtype, "Output dtype differs")
        np.testing.assert_array_equal(cp.asnumpy(output), reference)


def check_gpu(paths, report):
    import cupy as cp
    import kvikio.defaults
    import numpy as np

    from cuphoton.xdr import batch_to_device, batch_to_device_stream
    from cuphoton.xdr.nvcomp_batch import (
        gpu_gzip_decompress_batch,
        native_device_pool_stats,
    )

    require(cp.cuda.runtime.getDeviceCount() > 0, "A CUDA GPU is required")
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report["gpu"] = {
        "name": properties["name"].decode(),
        "compute_capability": [properties["major"], properties["minor"]],
    }
    require(
        kvikio.defaults.is_compat_mode_preferred(),
        "Compatibility I/O was not enabled",
    )
    report["io_mode"] = "kvikio_compat"
    report["cupy_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
    report["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
    runtime_path = next(
        path
        for path in report["loaded_native_libraries"]
        if Path(path).name.startswith("libcudart")
    )
    runtime = ctypes.CDLL(runtime_path)
    runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
    version = ctypes.c_int()
    require(
        runtime.cudaRuntimeGetVersion(ctypes.byref(version)) == 0,
        "Loaded CUDA runtime version query failed",
    )
    report["loaded_cuda_runtime_version"] = version.value
    raw = bytes(range(256)) * 4
    compressed = gzip.compress(raw, mtime=0)
    encoded = cp.asarray(np.frombuffer(compressed, dtype=np.uint8))
    decoded, offsets = gpu_gzip_decompress_batch(
        encoded, [0], [len(compressed)], [len(raw)], use_cpp_helper=True
    )
    np.testing.assert_array_equal(
        cp.asnumpy(decoded), np.frombuffer(raw, "u1")
    )
    np.testing.assert_array_equal(offsets, [0])
    passed(report, "forced_cpp_gzip_decompression")
    expected = reference_images(paths)
    options = dict(
        hdu_indices=HDU_INDICES,
        native_batcher=True,
        native_read_threads=2,
        native_plan_threads=2,
        decode_batch_files=2,
        batch_queue_depth=1,
        prefetch_depth=2,
    )
    retained = batch_to_device(paths, **options)
    check_images(retained, expected)
    passed(report, "native_image_gzip1_gzip2_order_and_dtype")

    stream = cp.cuda.Stream(non_blocking=True)
    supplied = tuple(
        cp.empty(array.shape, dtype=array.dtype) for array in expected
    )
    actual = batch_to_device_stream(
        paths, out=supplied, stream=stream, **options
    )
    require(
        all(
            output.data.ptr == buffer.data.ptr
            for output, buffer in zip(actual, supplied, strict=True)
        ),
        "Caller-provided output buffers were replaced",
    )
    check_images(actual, expected)
    gc.collect()
    check_images(retained, expected)
    passed(report, "nondefault_stream_outputs_and_retained_arrays")

    section = (slice(2, 16), slice(3, 22))
    roi_options = dict(options, hdu_indices=(5, 2, 3, 4), section=section)
    roi_expected = reference_images(
        paths, roi_options["hdu_indices"], section
    )
    check_images(batch_to_device_stream(paths, **roi_options), roi_expected)
    passed(report, "compressed_edge_tiles_and_roi")

    fallback = dict(options, native_batcher=False)
    check_images(batch_to_device_stream(paths, **fallback), expected)
    passed(report, "python_prefetch_with_native_planning")

    device_id = cp.cuda.Device().id

    def load_concurrently():
        with cp.cuda.Device(device_id):
            own_stream = cp.cuda.Stream(non_blocking=True)
            arrays = batch_to_device_stream(
                paths, stream=own_stream, **options
            )
            check_images(arrays, expected)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(load_concurrently) for _ in range(2)]
        for future in futures:
            future.result()
    check_images(retained, expected)
    passed(report, "two_concurrent_native_readers")
    del actual, supplied, retained
    cp.cuda.runtime.deviceSynchronize()
    gc.collect()
    stats = native_device_pool_stats(device_id)
    require(stats is not None, "Native device pool statistics unavailable")
    require(stats["checked_out_bytes"] == 0, "Native device buffers leaked")
    report["native_device_pool"] = stats
    passed(report, "native_buffer_release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("base", "native", "gpu"), required=True
    )
    parser.add_argument(
        "--components",
        choices=("xdr", "xfit", "xray"),
        nargs="+",
        default=["xdr", "xfit", "xray"],
        help="Native components to qualify (default: xdr xfit xray)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    require(args.timeout > 0, "--timeout must be positive")
    output = args.output.resolve() if args.output else None
    report = {
        "mode": args.mode,
        "components": args.components,
        "python": sys.version,
        "architecture": platform.machine(),
        "checks": [],
        "ok": False,
    }
    faulthandler.dump_traceback_later(args.timeout, exit=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    try:
        if args.mode == "gpu" and "xdr" in args.components:
            os.environ["KVIKIO_COMPAT_MODE"] = "ON"
            os.environ["KVIKIO_NTHREADS"] = "2"
        with tempfile.TemporaryDirectory(prefix="cuphoton-wheel-") as scratch:
            os.chdir(scratch)
            if args.mode == "gpu":
                os.environ["CUPY_CACHE_DIR"] = str(
                    Path(scratch) / "cupy-cache"
                )
            distribution = check_install(args.mode, args.components, report)
            if "xray" in args.components:
                check_xray(distribution, report, gpu=args.mode == "gpu")
            if args.mode != "base":
                if "xfit" in args.components:
                    check_xfit_native(distribution, report)
                if "xdr" in args.components:
                    paths = create_fits(Path(scratch))
                    report["versions"]["astropy"] = metadata.version(
                        "astropy"
                    )
                    check_native(distribution, paths, report)
                    if args.mode == "gpu":
                        check_gpu(paths, report)
                if args.mode == "gpu" and "xfit" in args.components:
                    check_xfit_gpu(report)
            report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
