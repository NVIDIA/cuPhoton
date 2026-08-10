# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-in / device-out nvCOMP batched DEFLATE decompression.

Accepts a single contiguous device buffer holding N concatenated
gzip-compressed tile payloads plus per-tile (offset, length) tables, and
produces a contiguous device buffer holding the decompressed tiles plus the
out-offsets table.

No host roundtrip on the compressed-data path. The only host cost is parsing
the per-tile gzip headers, which requires a small peek of the first ~16 bytes
of each tile — done via a single batched D2H (`N * 16` bytes).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

_DEFLATE_CODEC = None
_CPP_EXT = None
_CPP_EXT_PROBED = False
_CPP_EXT_IMPORT_ERROR: str | None = None
_CPP_EXT_LIBS_PRELOADED = False
_FALLBACK_WARNED = False
_NVCOMP_LIB_ENV = "CUPHOTON_XDR_NVCOMP_LIB_DIR"
_KVIKIO_LIB_ENV = "CUPHOTON_XDR_KVIKIO_LIB_DIR"
_RAPIDS_LOGGER_LIB_ENV = "CUPHOTON_XDR_RAPIDS_LOGGER_LIB_DIR"


def _load_shared_library(path: Path | str) -> None:
    ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))


def _package_dir(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    if locations:
        return Path(next(iter(locations)))
    if spec.origin:
        return Path(spec.origin).parent
    return None


def _library_in_dir(lib_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = lib_dir / name
        if path.is_file():
            return path
    return None


def _find_library_path(
    package_base: Path,
    names: tuple[str, ...],
    *,
    env_var: str,
    description: str,
) -> Path:
    env_value = os.environ.get(env_var)
    if env_value:
        lib_dir = Path(env_value)
        path = _library_in_dir(lib_dir, names)
        if path is not None:
            return path
        raise ImportError(
            f"{env_var}={lib_dir} does not contain {description}; "
            f"expected one of: {', '.join(names)}"
        )

    for child in ("lib64", "lib"):
        path = _library_in_dir(package_base / child, names)
        if path is not None:
            return path

    for name in names:
        matches = sorted(package_base.rglob(name))
        if matches:
            return matches[0]

    raise ImportError(
        f"Could not find {description} under {package_base}. "
        "Searched lib64, lib, and recursive matches for: "
        f"{', '.join(names)}. "
        f"Set {env_var} to the directory containing the library."
    )


def _candidate_cuda_homes():
    seen = set()
    for value in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
    ):
        if value:
            path = Path(value)
            if path not in seen:
                seen.add(path)
                yield path
    default = Path("/usr/local/cuda")
    if default not in seen:
        yield default


def _preload_cudart() -> None:
    candidates: list[Path | str] = []
    for cuda_home in _candidate_cuda_homes():
        for lib_name in ("lib64", "lib"):
            lib_dir = cuda_home / lib_name
            candidates.extend(
                [
                    lib_dir / "libcudart.so.13",
                    lib_dir / "libcudart.so",
                ]
            )

    cuda_runtime_base = _package_dir("nvidia.cuda_runtime")
    if cuda_runtime_base is not None:
        runtime_lib = cuda_runtime_base / "lib"
        candidates.extend(
            [
                runtime_lib / "libcudart.so.13",
                runtime_lib / "libcudart.so",
            ]
        )

    candidates.append("libcudart.so.13")
    candidates.append("libcudart.so")

    last_error: OSError | None = None
    for candidate in candidates:
        if isinstance(candidate, Path) and not candidate.exists():
            continue
        try:
            _load_shared_library(candidate)
            return
        except OSError as exc:
            last_error = exc

    detail = f" Last loader error was: {last_error}" if last_error else ""
    raise ImportError(
        "Could not preload libcudart for "
        "cuphoton.xdr._nvcomp_batch_ext. "
        "Set CUDA_HOME to a CUDA toolkit root containing libcudart." + detail
    )


def _preload_gpu_package_libraries() -> None:
    nvcomp_base = _package_dir("nvidia.libnvcomp")
    rapids_logger_base = _package_dir("rapids_logger")
    kvikio_base = _package_dir("libkvikio")
    if (
        nvcomp_base is None
        or rapids_logger_base is None
        or kvikio_base is None
    ):
        raise ImportError(
            "Could not find libkvikio, rapids-logger, or "
            "nvidia-libnvcomp Python packages required by "
            "cuphoton.xdr._nvcomp_batch_ext."
        )

    libraries = [
        _find_library_path(
            nvcomp_base,
            ("libnvcomp.so.5", "libnvcomp.so"),
            env_var=_NVCOMP_LIB_ENV,
            description="nvCOMP library",
        ),
        _find_library_path(
            rapids_logger_base,
            ("librapids_logger.so",),
            env_var=_RAPIDS_LOGGER_LIB_ENV,
            description="RAPIDS logger library",
        ),
        _find_library_path(
            kvikio_base,
            ("libkvikio.so",),
            env_var=_KVIKIO_LIB_ENV,
            description="KvikIO library",
        ),
    ]
    for library in libraries:
        if not library.exists():
            raise ImportError(
                f"Required GPU shared library is missing: {library}"
            )
        _load_shared_library(library)


def _preload_cpp_ext_libraries() -> None:
    global _CPP_EXT_LIBS_PRELOADED
    if _CPP_EXT_LIBS_PRELOADED:
        return
    _preload_cudart()
    _preload_gpu_package_libraries()
    _CPP_EXT_LIBS_PRELOADED = True


def _get_codec():
    """Return a cached nvcomp deflate+RAW codec. Creating a codec allocates a
    CUDA stream (~3-4 ms); cache it."""
    global _DEFLATE_CODEC
    if _DEFLATE_CODEC is None:
        import nvidia.nvcomp as nvcomp

        _DEFLATE_CODEC = nvcomp.Codec(
            algorithm="deflate",
            bitstream_kind=nvcomp.BitstreamKind.RAW,
        )
    return _DEFLATE_CODEC


def _try_get_cpp_ext():
    """Return the `_nvcomp_batch_ext` module or None if the shared lib is
    missing (e.g. extension never built on this checkout). Cached.

    If the import fails, the error message is saved so the first-fallback
    warning can report exactly why (missing .so vs undefined symbol vs
    version mismatch etc.).
    """
    global _CPP_EXT, _CPP_EXT_PROBED, _CPP_EXT_IMPORT_ERROR
    if not _CPP_EXT_PROBED:
        _CPP_EXT_PROBED = True
        try:
            _preload_cpp_ext_libraries()
            from . import _nvcomp_batch_ext as ext

            _CPP_EXT = ext
        except (ImportError, OSError) as e:
            _CPP_EXT = None
            _CPP_EXT_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    return _CPP_EXT


def cpp_helper_available() -> bool:
    """Return whether `_nvcomp_batch_ext` is importable."""
    return _try_get_cpp_ext() is not None


def get_native_batch_builder(required: bool = False):
    """Return the native KvikIO batch-builder class, or None.

    The native builder is compiled into `_nvcomp_batch_ext` alongside the
    nvCOMP helper. Older builds of the extension may import successfully but
    not expose `NativeBatchBuilder`; callers that require it should pass
    ``required=True``.
    """
    ext = _try_get_cpp_ext()
    cls = (
        getattr(ext, "NativeBatchBuilder", None) if ext is not None else None
    )
    if cls is not None:
        return cls
    if required:
        msg = (
            "native_batcher=True but "
            "`_nvcomp_batch_ext.NativeBatchBuilder` is "
            "not importable. Build it from a source checkout with "
            "`bash src/cuphoton/xdr/src/build.sh` (see "
            "docs/components/xdr.md); the native builder requires KvikIO."
        )
        if _CPP_EXT_IMPORT_ERROR:
            msg += f" Import error was: {_CPP_EXT_IMPORT_ERROR}"
        elif ext is not None:
            msg += " The loaded extension appears to be stale."
        raise RuntimeError(msg)
    return None


def get_native_batch_reader(required: bool = False):
    """Backward-compatible alias for `get_native_batch_builder`."""
    return get_native_batch_builder(required=required)


def get_native_plan_files(required: bool = False):
    """Return the native CFITSIO planner function, or None if unavailable."""
    ext = _try_get_cpp_ext()
    fn = getattr(ext, "plan_native_files", None) if ext is not None else None
    if fn is not None:
        return fn
    if required:
        msg = (
            "native_batcher=True but "
            "`_nvcomp_batch_ext.plan_native_files` is "
            "not importable. Build it from a source checkout with "
            "`bash src/cuphoton/xdr/src/build.sh` (see "
            "docs/components/xdr.md); the native planner requires CFITSIO."
        )
        if _CPP_EXT_IMPORT_ERROR:
            msg += f" Import error was: {_CPP_EXT_IMPORT_ERROR}"
        elif ext is not None:
            msg += " The loaded extension appears to be stale."
        raise RuntimeError(msg)
    return None


def native_batch_builder_available() -> bool:
    """Return whether the native KvikIO stream batch builder is available."""
    return get_native_batch_builder(required=False) is not None


def native_plan_files_available() -> bool:
    """Return whether the native CFITSIO planner is available."""
    return get_native_plan_files(required=False) is not None


def native_batch_reader_available() -> bool:
    """Return whether the native KvikIO stream batch reader is available."""
    return native_batch_builder_available()


def native_pinned_pool_stats(device_id: int = -1) -> dict | None:
    """Return legacy native pinned host buffer pool stats."""
    ext = _try_get_cpp_ext()
    fn = (
        getattr(ext, "native_pinned_pool_stats", None)
        if ext is not None
        else None
    )
    if fn is None:
        return None
    return dict(fn(int(device_id)))


def clear_native_pinned_pool(device_id: int = -1) -> None:
    """Free idle legacy native pinned host buffers."""
    ext = _try_get_cpp_ext()
    fn = (
        getattr(ext, "clear_native_pinned_pool", None)
        if ext is not None
        else None
    )
    if fn is not None:
        fn(int(device_id))


def native_device_pool_stats(device_id: int = -1) -> dict | None:
    """Return native device scratch/staging buffer pool stats."""
    ext = _try_get_cpp_ext()
    fn = (
        getattr(ext, "native_device_pool_stats", None)
        if ext is not None
        else None
    )
    if fn is None:
        return None
    return dict(fn(int(device_id)))


def clear_native_device_pool(device_id: int = -1) -> None:
    """Free idle native device scratch/staging buffers."""
    ext = _try_get_cpp_ext()
    fn = (
        getattr(ext, "clear_native_device_pool", None)
        if ext is not None
        else None
    )
    if fn is not None:
        fn(int(device_id))


def _native_device_empty_uint8(nbytes: int, *, ext=None):
    """Allocate a uint8 CuPy array from the native device pool."""
    import cupy as cp

    nbytes = int(nbytes)
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    if nbytes == 0:
        return cp.empty(0, dtype=cp.uint8)

    if ext is None:
        ext = _try_get_cpp_ext()
    fn = (
        getattr(ext, "acquire_native_device_buffer", None)
        if ext is not None
        else None
    )
    if fn is None:
        return cp.empty(nbytes, dtype=cp.uint8)

    device_id = int(cp.cuda.Device().id)
    owner, ptr, size = fn(nbytes, device_id)
    mem = cp.cuda.UnownedMemory(int(ptr), int(size), owner)
    return cp.ndarray(
        (nbytes,),
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(mem, 0),
    )


def _native_device_empty(shape, dtype, *, ext=None):
    """Allocate a typed CuPy array from the native device pool."""
    import cupy as cp

    dtype = cp.dtype(dtype)
    shape = tuple(int(v) for v in shape)
    nitems = int(np.prod(shape, dtype=np.int64))
    buf = _native_device_empty_uint8(nitems * dtype.itemsize, ext=ext)
    return buf.view(dtype).reshape(shape)


def _warn_python_fallback_once() -> None:
    """Warn once when `gpu_gzip_decompress_batch` uses Python fallback.

    The fallback is ~6× slower than the C++ helper — without this warning a
    missing build silently regresses performance (as happened on the H100 box,
    2026-04-23). To suppress, set
    `CUPHOTON_XDR_SILENCE_CPP_FALLBACK=1` or pass
    `use_cpp_helper=False` to the function.
    """
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True

    import os

    if os.environ.get("CUPHOTON_XDR_SILENCE_CPP_FALLBACK", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    import warnings

    msg = (
        "xdr: batched deflate decompression is running on the "
        "Python `nvcomp.as_array` fallback path, which is ~6x slower "
        "than the "
        "optional C++ helper (`_nvcomp_batch_ext`). "
        "To enable the fast path, build the native extension: "
        "`bash src/cuphoton/xdr/src/build.sh`."
    )
    if _CPP_EXT_IMPORT_ERROR:
        msg += f"\nImport error was: {_CPP_EXT_IMPORT_ERROR}"
    msg += "\nSilence this warning by setting "
    msg += "CUPHOTON_XDR_SILENCE_CPP_FALLBACK=1 or passing "
    msg += "use_cpp_helper=False."
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _parse_gzip_header_len(head: bytes) -> int:
    """Return the length of a gzip RFC 1952 header given its first ~16 bytes.

    Raises ValueError if the magic bytes are wrong. Handles FEXTRA / FNAME /
    FCOMMENT / FHCRC flags. Most FITS producers (astropy, cfitsio/fpack) emit
    a minimal 10-byte header (FLG=0), but we parse for correctness.
    """
    if len(head) < 10 or head[0] != 0x1F or head[1] != 0x8B:
        raise ValueError("not a gzip stream (wrong magic)")
    flg = head[3]
    pos = 10

    def require(count: int) -> None:
        if pos + count > len(head):
            raise ValueError("truncated gzip header")

    def skip_c_string() -> int:
        nonlocal pos
        while True:
            require(1)
            value = head[pos]
            pos += 1
            if value == 0:
                return pos

    if flg & 0x04:  # FEXTRA
        require(2)
        extra_len = struct.unpack_from("<H", head, pos)[0]
        pos += 2
        require(extra_len)
        pos += extra_len
    if flg & 0x08:  # FNAME (null-terminated)
        skip_c_string()
    if flg & 0x10:  # FCOMMENT
        skip_c_string()
    if flg & 0x02:  # FHCRC
        require(2)
        pos += 2
    return pos


def _compute_header_sizes_from_device(
    d_concat, rel_offsets: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Parse the gzip header length for each tile.

    Does a single D2H of the first 16 bytes of each tile (N_tiles × 16 bytes),
    then parses each header on host. Fast for any realistic tile count.
    """
    rel_offsets = np.asarray(rel_offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    n = rel_offsets.size
    peek_len = 16
    if lengths.size != n:
        raise ValueError("gzip tile offsets and lengths must have same size")
    buffer_size = int(d_concat.size)
    invalid_probe = (
        (lengths < peek_len)
        | (rel_offsets < 0)
        | (rel_offsets > buffer_size - peek_len)
    )
    if np.any(invalid_probe):
        raise ValueError(
            "gzip tile header probe exceeds tile or device buffer"
        )

    import cupy as cp

    # Gather first `peek_len` bytes of each tile into one contiguous host buf.
    # Indexing by a CuPy index array is a single kernel launch + D2H.
    idx = np.repeat(rel_offsets.astype(np.int64), peek_len) + np.tile(
        np.arange(peek_len, dtype=np.int64), n
    )
    peeks_d = d_concat[cp.asarray(idx)]
    peeks = cp.asnumpy(peeks_d).reshape(n, peek_len).tobytes()

    header_sizes = np.empty(n, dtype=np.int64)
    for i in range(n):
        header_sizes[i] = _parse_gzip_header_len(
            peeks[i * peek_len : (i + 1) * peek_len]
        )
    return header_sizes


def gpu_gzip_decompress_batch(
    d_concat,
    rel_offsets: Sequence[int],
    lengths: Sequence[int],
    uncompressed_sizes: Sequence[int],
    *,
    gzip_wrapped: bool = True,
    use_cpp_helper: str | bool = "auto",
    use_native_pool: bool = False,
    keepalive: list | None = None,
):
    """Batch-decompress gzip tiles already resident on the device.

    Parameters
    ----------
    d_concat : cupy.ndarray (uint8)
        Contiguous device buffer with concatenated compressed tile bytes.
    rel_offsets, lengths, uncompressed_sizes
        Per-tile metadata arrays.
    gzip_wrapped : bool
        If True (default), each tile is an RFC 1952 gzip stream; the
        variable header + 8-byte trailer are stripped before nvcomp.
    use_cpp_helper : "auto" | True | False
        "auto" (default) → use the C++ pybind11 helper when importable, else
        fall back to the Python loop. True forces the C++ path (raises if
        the extension is missing). False forces the Python path.
    use_native_pool : bool
        If True and the rebuilt C++ helper is available, allocate the decoded
        tile buffer and native nvCOMP scratch from the native device pool.
    keepalive : list or None
        Optional owner list that keeps pooled scratch alive until the caller's
        stream event says the decode/scatter work has completed.

    Returns
    -------
    d_out : cupy.ndarray (uint8)
        Contiguous device buffer with decompressed tile bytes concatenated.
    out_offsets : numpy.ndarray (int64)
        Start offset of each decompressed tile inside ``d_out``.
    """
    import cupy as cp

    rel_offsets = np.asarray(rel_offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    uncompressed_sizes = np.asarray(uncompressed_sizes, dtype=np.int64)
    n = rel_offsets.size

    if gzip_wrapped:
        header_sizes = _compute_header_sizes_from_device(
            d_concat, rel_offsets, lengths
        )
        trailer_sizes = np.full(n, 8, dtype=np.int64)
    else:
        header_sizes = np.zeros(n, dtype=np.int64)
        trailer_sizes = np.zeros(n, dtype=np.int64)

    deflate_offsets = rel_offsets + header_sizes
    deflate_lengths = lengths - header_sizes - trailer_sizes
    if np.any(deflate_lengths < 0):
        raise ValueError(
            "negative DEFLATE payload length — malformed gzip tile?"
        )

    # Pick the backend.
    if use_cpp_helper is True:
        ext = _try_get_cpp_ext()
        if ext is None:
            raise RuntimeError(
                "use_cpp_helper=True but `_nvcomp_batch_ext` is not "
                "importable. "
                "Run `bash src/cuphoton/xdr/src/build.sh` first. "
                f"Import error: {_CPP_EXT_IMPORT_ERROR}"
            )
    elif use_cpp_helper is False:
        ext = None
    else:
        ext = _try_get_cpp_ext()
        if ext is None:
            _warn_python_fallback_once()

    # Allocate one concatenated output buffer and slice it per tile.
    out_offsets = np.concatenate(
        ([0], np.cumsum(uncompressed_sizes)[:-1])
    ).astype(np.int64)
    total = int(uncompressed_sizes.sum())
    d_out = (
        _native_device_empty_uint8(total, ext=ext)
        if use_native_pool and ext is not None
        else cp.empty(total, dtype=cp.uint8)
    )

    if ext is not None:
        # C++ path: device pointers + length arrays, no per-tile Python loop.
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        d_concat_ptr = int(d_concat.data.ptr)
        d_out_ptr = int(d_out.data.ptr)
        pooled_fn = getattr(ext, "batch_deflate_decompress_pooled", None)
        if (
            use_native_pool
            and pooled_fn is not None
            and keepalive is not None
        ):
            scratch_owner = pooled_fn(
                d_concat_ptr,
                deflate_offsets,
                deflate_lengths,
                d_out_ptr,
                out_offsets,
                uncompressed_sizes,
                stream_ptr,
            )
            keepalive.append(scratch_owner)
            return d_out, out_offsets

        ext.batch_deflate_decompress(
            d_concat_ptr,
            deflate_offsets,
            deflate_lengths,
            d_out_ptr,
            out_offsets,
            uncompressed_sizes,
            stream_ptr,
        )
        return d_out, out_offsets

    # Python fallback: original path, retained for compile-less checkouts.
    import nvidia.nvcomp as nvcomp

    src_arrays = [
        nvcomp.as_array(
            d_concat[
                deflate_offsets[i] : deflate_offsets[i] + deflate_lengths[i]
            ]
        )
        for i in range(n)
    ]
    out_arrays = [
        nvcomp.as_array(
            d_out[out_offsets[i] : out_offsets[i] + uncompressed_sizes[i]]
        )
        for i in range(n)
    ]
    codec = _get_codec()
    codec.decode(src_arrays, out=out_arrays)
    return d_out, out_offsets
