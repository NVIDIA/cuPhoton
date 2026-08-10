# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Class-based CLI commands for XPOIS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from numpy.linalg import LinAlgError

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    FloatInvariant,
    InvariantAwareCommand,
    NonNegativeIntegerInvariant,
    PositiveIntegerInvariant,
    SetInvariant,
    StringInvariant,
)

from .data import inspect_hsc_data_tree
from .ois import (
    EXPLICIT_BACKENDS,
    SUPPORTED_BACKENDS,
    GaussianBasisComponent,
)
from .workflows import (
    benchmark_constant_kernel_backends,
    evaluate_subtraction_run,
    rebuild_interactive_review,
    run_constant_kernel_fit,
)

T = TypeVar("T")

_KNOWN_COMMAND_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    ImportError,
    IndexError,
    LinAlgError,
    NotImplementedError,
    OSError,
    RuntimeError,
    ValueError,
)


def _emit_json_payload(payload: Any, *, out: Callable[[str], None]) -> None:
    out(json.dumps(payload, indent=2, default=str))


class PathSpecInvariant(StringInvariant):
    _type_desc = "path"
    _minlen = 1
    _maxlen = 4096


class ImagePathInvariant(PathSpecInvariant):
    expected = "a path to a FITS or NPY image"


class CsvSpecInvariant(StringInvariant):
    expected = "a comma-separated list of numeric values"


class BackendListInvariant(StringInvariant):
    expected = "a comma-separated backend list"
    _type_desc = "backend list"
    _minlen = 1
    _maxlen = 256


class XPOISCommand(InvariantAwareCommand):
    def _call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except _KNOWN_COMMAND_EXCEPTIONS as exc:
            raise CommandError(str(exc)) from exc

    def _path(self, value: str | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    def _emit_json(self, payload: Any) -> None:
        _emit_json_payload(payload, out=self._out)

    def _csv_backends(self, value: str) -> list[str]:
        parts = [
            item.strip().lower() for item in value.split(",") if item.strip()
        ]
        if not parts:
            raise CommandError("expected at least one backend")
        unknown = sorted(set(parts) - set(EXPLICIT_BACKENDS))
        if unknown:
            raise CommandError(
                "unsupported backend(s): " + ", ".join(unknown)
            )
        return parts


class DataInspectCommand(XPOISCommand):
    """Inspect the shared HSC data tree used by XPOIS."""

    base = None

    class BaseArg(PathSpecInvariant):
        _arg = "--base"
        _help = "Optional override for the shared HSC data root."
        _mandatory = False
        _default = None

    def run(self) -> None:
        summary = self._call(inspect_hsc_data_tree, self._path(self.base))
        self._emit_json(summary)


class _KernelSolveCommand(XPOISCommand):
    reference = None
    target = None
    reference_hdu = None
    target_hdu = None
    variance = None
    variance_hdu = None
    reference_mask = None
    target_mask = None
    reference_mask_hdu = None
    target_mask_hdu = None
    mask_policy = None
    fit_mask = None
    auto_stamp_mask = None
    auto_stamp_size = None
    auto_stamp_count = None
    auto_peak_percentile = None
    crop_y0 = None
    crop_x0 = None
    crop_height = None
    crop_width = None
    output_dir = None
    run_name = None
    kernel_height = None
    kernel_width = None
    basis_sigmas = None
    basis_degrees = None
    background_degree = None
    flux_conserve = None

    class ReferenceArg(ImagePathInvariant):
        _arg = "--reference"
        _help = "Reference image path (.fits or .npy)."
        _mandatory = True

    class TargetArg(ImagePathInvariant):
        _arg = "--target"
        _help = "Target image path (.fits or .npy)."
        _mandatory = True

    class ReferenceHduArg(NonNegativeIntegerInvariant):
        _arg = "--reference-hdu"
        _help = "Optional explicit HDU index for the reference FITS file."
        _mandatory = False
        _default = None

    class TargetHduArg(NonNegativeIntegerInvariant):
        _arg = "--target-hdu"
        _help = "Optional explicit HDU index for the target FITS file."
        _mandatory = False
        _default = None

    class VarianceArg(PathSpecInvariant):
        _arg = "--variance"
        _help = "Optional variance image path (.fits or .npy)."
        _mandatory = False
        _default = None

    class VarianceHduArg(NonNegativeIntegerInvariant):
        _arg = "--variance-hdu"
        _help = "Optional explicit HDU index for the variance FITS file."
        _mandatory = False
        _default = None

    class ReferenceMaskArg(PathSpecInvariant):
        _arg = "--reference-mask"
        _help = (
            "Optional reference mask path (.fits or .npy). "
            "Defaults to --reference when mask-policy is enabled on FITS "
            "input."
        )
        _mandatory = False
        _default = None

    class TargetMaskArg(PathSpecInvariant):
        _arg = "--target-mask"
        _help = (
            "Optional target mask path (.fits or .npy). "
            "Defaults to --target when mask-policy is enabled on FITS input."
        )
        _mandatory = False
        _default = None

    class ReferenceMaskHduArg(NonNegativeIntegerInvariant):
        _arg = "--reference-mask-hdu"
        _help = "Optional explicit HDU index for the reference FITS mask."
        _mandatory = False
        _default = None

    class TargetMaskHduArg(NonNegativeIntegerInvariant):
        _arg = "--target-mask-hdu"
        _help = "Optional explicit HDU index for the target FITS mask."
        _mandatory = False
        _default = None

    class MaskPolicyArg(SetInvariant):
        _arg = "--mask-policy"
        _help = (
            "Built-in mask policy to apply before fitting. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "none"
        _set = {"none", "strict", "hsc-masklite", "masklite"}

    class FitMaskArg(PathSpecInvariant):
        _arg = "--fit-mask"
        _help = "Optional boolean .npy mask selecting fit pixels."
        _mandatory = False
        _default = None

    class AutoStampMaskArg(BoolInvariant):
        _arg = "--auto-stamp-mask"
        _help = "Auto-build a compact-source stamp fit mask."
        _mandatory = False
        _default = False

    class AutoStampSizeArg(PositiveIntegerInvariant):
        _arg = "--auto-stamp-size"
        _help = "Odd compact-source stamp size. [default: %default]"
        _mandatory = False
        _default = 31

    class AutoStampCountArg(PositiveIntegerInvariant):
        _arg = "--auto-stamp-count"
        _help = "Maximum number of compact-source stamps. [default: %default]"
        _mandatory = False
        _default = 5

    class AutoPeakPercentileArg(FloatInvariant):
        _arg = "--auto-peak-percentile"
        _help = (
            "Compact-source peak percentile threshold. [default: %default]"
        )
        _mandatory = False
        _default = 99.5
        _min = 0.0
        _max = 100.0

    class CropY0Arg(NonNegativeIntegerInvariant):
        _arg = "--crop-y0"
        _help = "Optional top row for a rectangular crop."
        _mandatory = False
        _default = None

    class CropX0Arg(NonNegativeIntegerInvariant):
        _arg = "--crop-x0"
        _help = "Optional left column for a rectangular crop."
        _mandatory = False
        _default = None

    class CropHeightArg(PositiveIntegerInvariant):
        _arg = "--crop-height"
        _help = "Optional crop height."
        _mandatory = False
        _default = None

    class CropWidthArg(PositiveIntegerInvariant):
        _arg = "--crop-width"
        _help = "Optional crop width."
        _mandatory = False
        _default = None

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = (
            "Output root directory. [default: "
            "~/.local/state/cuphoton/xpois/runs or XDG_STATE_HOME]"
        )
        _mandatory = False
        _default = None

    class RunNameArg(StringInvariant):
        _arg = "--name"
        _help = "Optional run directory name override."
        _mandatory = False
        _default = None
        _minlen = 0

    class KernelHeightArg(PositiveIntegerInvariant):
        _arg = "--kernel-height"
        _help = "Odd kernel height. [default: %default]"
        _mandatory = False
        _default = 15

    class KernelWidthArg(PositiveIntegerInvariant):
        _arg = "--kernel-width"
        _help = "Odd kernel width. [default: %default]"
        _mandatory = False
        _default = 15

    class BasisSigmasArg(CsvSpecInvariant):
        _arg = "--basis-sigmas"
        _help = "Comma-separated Gaussian sigmas. [default: %default]"
        _mandatory = False
        _default = "1.5,3.0,6.0"

    class BasisDegreesArg(CsvSpecInvariant):
        _arg = "--basis-degrees"
        _help = (
            "Comma-separated polynomial degrees. The default is a "
            "conservative real-data basis. [default: %default]"
        )
        _mandatory = False
        _default = "2,1,0"

    class BackgroundDegreeArg(NonNegativeIntegerInvariant):
        _arg = "--background-degree"
        _help = "Polynomial background degree. [default: %default]"
        _mandatory = False
        _default = 0

    class FluxConserveArg(BoolInvariant):
        _arg = "--flux-conserve"
        _help = "Use a flux-conserving difference-basis rewrite."
        _mandatory = False
        _default = False

    def _components(self) -> list[GaussianBasisComponent]:
        try:
            sigmas = [
                float(item)
                for item in self.basis_sigmas.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise CommandError(
                "--basis-sigmas must contain only numeric values"
            ) from exc
        try:
            degrees = [
                int(item)
                for item in self.basis_degrees.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise CommandError(
                "--basis-degrees must contain only integer values"
            ) from exc
        if not sigmas or not degrees:
            raise CommandError(
                "--basis-sigmas and --basis-degrees must provide "
                "at least one basis term"
            )
        if len(sigmas) != len(degrees):
            raise CommandError(
                "--basis-sigmas and --basis-degrees must have the same length"
            )
        return [
            GaussianBasisComponent(sigma=sigma, degree=degree)
            for sigma, degree in zip(sigmas, degrees)
        ]

    def _output_root(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir).expanduser()
        return self.context.runs_dir


class _FitCommand(_KernelSolveCommand):
    backend = None

    class BackendArg(SetInvariant):
        _arg = "--backend"
        _help = (
            "Fit backend. Auto tries cupy, then numba-cuda, then cpu. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "auto"
        _set = set(SUPPORTED_BACKENDS)


class FitKernelCommand(_FitCommand):
    """Fit a constant kernel and persist reviewable subtraction artifacts.

    This is the main source-backed OIS workflow entrypoint. It supports:

    - raw FITS or `.npy` image inputs
    - optional variance images / variance HDUs
    - optional input-mask policies and explicit mask HDUs
    - optional rectangular workflow crops
    - explicit fit masks or auto-selected compact-source stamp masks

    Successful runs persist numeric artifacts and, when Bokeh is installed,
    an interactive HTML review under the run-local `artifacts/` directory.
    """

    def run(self) -> None:
        result = self._call(
            run_constant_kernel_fit,
            reference_path=Path(self.reference).expanduser(),
            target_path=Path(self.target).expanduser(),
            output_root=self._output_root(),
            name=self.run_name or None,
            reference_hdu=self.reference_hdu,
            target_hdu=self.target_hdu,
            kernel_shape=(self.kernel_height, self.kernel_width),
            components=self._components(),
            variance_path=self._path(self.variance),
            variance_hdu=self.variance_hdu,
            reference_mask_path=self._path(self.reference_mask),
            target_mask_path=self._path(self.target_mask),
            reference_mask_hdu=self.reference_mask_hdu,
            target_mask_hdu=self.target_mask_hdu,
            mask_policy=self.mask_policy,
            crop_y0=self.crop_y0,
            crop_x0=self.crop_x0,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            fit_mask_path=self._path(self.fit_mask),
            auto_stamp_mask=bool(self.auto_stamp_mask),
            auto_stamp_size=self.auto_stamp_size,
            auto_stamp_count=self.auto_stamp_count,
            auto_peak_percentile=self.auto_peak_percentile,
            background_degree=self.background_degree,
            flux_conserve=bool(self.flux_conserve),
            backend=self.backend,
            workflow_name="fit_kernel",
            run_prefix="fit-kernel",
        )
        self._emit_json(result.summary)


class SubtractCommand(_FitCommand):
    """Run the same constant-kernel workflow under subtraction naming.

    This currently shares the same underlying solve path as `fit-kernel`, but
    emits `workflow="subtract"` and uses the `subtract-*` run-prefix naming
    convention for users who think in subtraction-oriented rather than
    kernel-fitting terminology.
    """

    def run(self) -> None:
        result = self._call(
            run_constant_kernel_fit,
            reference_path=Path(self.reference).expanduser(),
            target_path=Path(self.target).expanduser(),
            output_root=self._output_root(),
            name=self.run_name or None,
            reference_hdu=self.reference_hdu,
            target_hdu=self.target_hdu,
            kernel_shape=(self.kernel_height, self.kernel_width),
            components=self._components(),
            variance_path=self._path(self.variance),
            variance_hdu=self.variance_hdu,
            reference_mask_path=self._path(self.reference_mask),
            target_mask_path=self._path(self.target_mask),
            reference_mask_hdu=self.reference_mask_hdu,
            target_mask_hdu=self.target_mask_hdu,
            mask_policy=self.mask_policy,
            crop_y0=self.crop_y0,
            crop_x0=self.crop_x0,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            fit_mask_path=self._path(self.fit_mask),
            auto_stamp_mask=bool(self.auto_stamp_mask),
            auto_stamp_size=self.auto_stamp_size,
            auto_stamp_count=self.auto_stamp_count,
            auto_peak_percentile=self.auto_peak_percentile,
            background_degree=self.background_degree,
            flux_conserve=bool(self.flux_conserve),
            backend=self.backend,
            workflow_name="subtract",
            run_prefix="subtract",
        )
        self._emit_json(result.summary)


class BenchmarkBackendsCommand(_KernelSolveCommand):
    """Benchmark constant-kernel CPU/CuPy backends and parity.

    This command isolates the constant-kernel solve/application path. It loads
    the input arrays once, repeats `solve_constant_kernel` for each backend,
    and persists timing plus numerical comparison artifacts under the run
    directory.
    """

    backends = None
    reference_backend = None
    repeats = None
    warmup = None
    atol = None
    rtol = None

    class BackendsArg(BackendListInvariant):
        _arg = "--backends"
        _help = "Comma-separated backends to compare. [default: %default]"
        _mandatory = False
        _default = "cpu,cupy"

    class ReferenceBackendArg(SetInvariant):
        _arg = "--reference-backend"
        _help = "Backend used as numerical reference. [default: %default]"
        _mandatory = False
        _default = "cpu"
        _set = set(EXPLICIT_BACKENDS)

    class RepeatsArg(PositiveIntegerInvariant):
        _arg = "--repeats"
        _help = "Timed repeats per backend. [default: %default]"
        _mandatory = False
        _default = 3

    class WarmupArg(NonNegativeIntegerInvariant):
        _arg = "--warmup"
        _help = "Warmup runs per backend. [default: %default]"
        _mandatory = False
        _default = 1

    class AtolArg(FloatInvariant):
        _arg = "--atol"
        _help = "Absolute tolerance for parity checks. [default: %default]"
        _mandatory = False
        _default = 1e-8
        _min = 0.0

    class RtolArg(FloatInvariant):
        _arg = "--rtol"
        _help = "Relative tolerance for parity checks. [default: %default]"
        _mandatory = False
        _default = 1e-9
        _min = 0.0

    def run(self) -> None:
        result = self._call(
            benchmark_constant_kernel_backends,
            reference_path=Path(self.reference).expanduser(),
            target_path=Path(self.target).expanduser(),
            output_root=self._output_root(),
            name=self.run_name or None,
            reference_hdu=self.reference_hdu,
            target_hdu=self.target_hdu,
            kernel_shape=(self.kernel_height, self.kernel_width),
            components=self._components(),
            variance_path=self._path(self.variance),
            variance_hdu=self.variance_hdu,
            fit_mask_path=self._path(self.fit_mask),
            crop_y0=self.crop_y0,
            crop_x0=self.crop_x0,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            background_degree=self.background_degree,
            flux_conserve=bool(self.flux_conserve),
            backends=self._csv_backends(self.backends),
            reference_backend=self.reference_backend,
            repeats=self.repeats,
            warmup=self.warmup,
            atol=self.atol,
            rtol=self.rtol,
        )
        self._emit_json(result.summary)


class EvaluateSubtractionCommand(XPOISCommand):
    """Compute saved-run residual summary metrics from persisted artifacts.

    Evaluation reloads the saved residual and effective fit mask from the run
    directory so the reported fit-region and whole-image residual metrics are
    reproducible even after the run is moved.
    """

    run_dir = None

    class RunDirArg(PathSpecInvariant):
        _arg = "--run-dir"
        _help = "Subtraction run directory to evaluate."
        _mandatory = True

    def run(self) -> None:
        result = self._call(
            evaluate_subtraction_run,
            Path(self.run_dir).expanduser(),
        )
        self._emit_json(result)


class ReviewBokehCommand(XPOISCommand):
    """Regenerate the interactive Bokeh review for an existing run.

    This command does not refit the kernel. It rebuilds the interactive review
    artifact from the saved subtraction outputs and returns the regenerated
    `review_bokeh.html` path as JSON.
    """

    run_dir = None

    class RunDirArg(PathSpecInvariant):
        _arg = "--run-dir"
        _help = "Subtraction run directory to rebuild for interactive review."
        _mandatory = True

    def run(self) -> None:
        result = self._call(
            rebuild_interactive_review,
            Path(self.run_dir).expanduser(),
        )
        self._emit_json(result)
