# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Class-based CLI commands for xRep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    ExistingPathInvariant,
    FloatInvariant,
    InvariantAwareCommand,
    NonNegativeIntegerInvariant,
    PositiveIntegerInvariant,
    SetInvariant,
    StringInvariant,
)

from .workflows import (
    benchmark_backend_variants_reproject_image,
    benchmark_reproject_image,
    compare_backends_reproject_image,
    inspect_image,
    run_reproject_image,
    run_reproject_stack,
)

T = TypeVar("T")

_KNOWN_COMMAND_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    ImportError,
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


class ExistingPathSpecInvariant(ExistingPathInvariant):
    _type_desc = "existing path"


class CsvPathInvariant(StringInvariant):
    expected = "a comma-separated list of paths"
    _type_desc = "csv path list"
    _minlen = 1
    _maxlen = 16384


class BackendInvariant(SetInvariant):
    _set = {"cpu", "torch", "cupy"}


class AutoBackendInvariant(SetInvariant):
    _set = {"auto", "cpu", "torch", "cupy"}


class BackendListInvariant(StringInvariant):
    expected = "a comma-separated backend list"
    _type_desc = "backend list"
    _minlen = 1
    _maxlen = 256


class BackendVariantListInvariant(StringInvariant):
    expected = "a comma-separated backend variant list"
    _type_desc = "backend variant list"
    _minlen = 1
    _maxlen = 512


class MaskCaseListInvariant(StringInvariant):
    expected = "a comma-separated mask case list"
    _type_desc = "mask case list"
    _minlen = 1
    _maxlen = 64


class BackendVariantInvariant(SetInvariant):
    _set = {"cpu", "torch", "cupy", "cupy-elementwise", "cupy-raw"}


class InterpolationInvariant(SetInvariant):
    _set = {"bilinear", "lanczos3"}


class XRepCommand(InvariantAwareCommand):
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

    def _csv_paths(self, value: str) -> list[Path]:
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise CommandError("expected at least one path")
        return [Path(item).expanduser() for item in parts]

    def _csv_backends(self, value: str) -> list[str]:
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise CommandError("expected at least one backend")
        unknown = sorted(set(parts) - BackendInvariant._set)
        if unknown:
            raise CommandError(
                "unsupported backend(s): " + ", ".join(unknown)
            )
        return parts

    def _csv_backend_variants(self, value: str) -> list[str]:
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise CommandError("expected at least one backend variant")
        unknown = sorted(set(parts) - BackendVariantInvariant._set)
        if unknown:
            raise CommandError(
                "unsupported backend variant(s): " + ", ".join(unknown)
            )
        return parts

    def _csv_mask_cases(self, value: str) -> list[str]:
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise CommandError("expected at least one mask case")
        unknown = sorted(set(parts) - {"none", "mask"})
        if unknown:
            raise CommandError(
                "unsupported mask case(s): " + ", ".join(unknown)
            )
        return parts


class _SharedReprojectionCommand(XRepCommand):
    input = None
    hdu = None
    backend = None
    interpolation = None
    grid_crval_ra = None
    grid_crval_dec = None
    pixel_scale_arcsec = None
    mapping_grid_step = None
    disable_area_scaling = None
    output_dir = None
    name = None
    write_fits = None
    repeats = None
    warmup = None

    class HduArg(NonNegativeIntegerInvariant):
        _arg = "--hdu"
        _help = "Optional explicit FITS HDU index."
        _mandatory = False
        _default = None

    class BackendArg(AutoBackendInvariant):
        _arg = "--backend"
        _help = "Interpolation backend. Default: auto (cupy > torch > cpu)."
        _mandatory = False
        _default = None

    class InterpolationArg(InterpolationInvariant):
        _arg = "--interpolation"
        _help = "Interpolation kernel. [default: %default]"
        _mandatory = False
        _default = "lanczos3"

    class GridCrvalRaArg(FloatInvariant):
        _arg = "--grid-crval-ra"
        _help = "Optional explicit grid reference RA in degrees."
        _mandatory = False
        _default = None

    class GridCrvalDecArg(FloatInvariant):
        _arg = "--grid-crval-dec"
        _help = "Optional explicit grid reference Dec in degrees."
        _mandatory = False
        _default = None

    class PixelScaleArcsecArg(FloatInvariant):
        _arg = "--pixel-scale-arcsec"
        _help = "Optional explicit grid pixel scale in arcsec/pixel."
        _mandatory = False
        _default = None
        _min = 0.0

    class MappingGridStepArg(PositiveIntegerInvariant):
        _arg = "--mapping-grid-step"
        _help = "Mapping coarse-grid spacing in pixels. [default: %default]"
        _mandatory = False
        _default = 100

    class DisableAreaScalingArg(BoolInvariant):
        _arg = "--disable-area-scaling"
        _help = "Disable relative-area flux scaling."
        _mandatory = False
        _default = False

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = (
            "Output root directory. [default: "
            "~/.local/state/cuphoton/xrep/runs or XDG_STATE_HOME]"
        )
        _mandatory = False
        _default = None

    class NameArg(StringInvariant):
        _arg = "--name"
        _help = "Optional explicit run directory name."
        _mandatory = False
        _default = None
        _minlen = 1
        _maxlen = 128

    class WriteFitsArg(BoolInvariant):
        _arg = "--write-fits"
        _help = "Also save FITS output artifacts."
        _mandatory = False
        _default = False

    class RepeatsArg(PositiveIntegerInvariant):
        _arg = "--repeats"
        _help = "Number of measured iterations. [default: %default]"
        _mandatory = False
        _default = 5

    class WarmupArg(NonNegativeIntegerInvariant):
        _arg = "--warmup"
        _help = "Number of warmup iterations. [default: %default]"
        _mandatory = False
        _default = 1


class InspectImageCommand(_SharedReprojectionCommand):
    """Inspect a FITS image and report its default reprojection grid."""

    input = None

    class InputArg(ExistingPathSpecInvariant):
        _arg = "--input"
        _help = "Input FITS image path."
        _mandatory = True

    def run(self) -> None:
        payload = self._call(
            inspect_image,
            Path(self.input).expanduser(),
            hdu=self.hdu,
            grid_crval_ra=self.grid_crval_ra,
            grid_crval_dec=self.grid_crval_dec,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
        )
        self._emit_json(payload)


class ReprojectImageCommand(_SharedReprojectionCommand):
    """Reproject one FITS image onto one shared grid."""

    input = None
    mask = None
    mask_hdu = None

    class InputArg(ExistingPathSpecInvariant):
        _arg = "--input"
        _help = "Input FITS image path."
        _mandatory = True

    class MaskArg(ExistingPathSpecInvariant):
        _arg = "--mask"
        _help = "Optional input FITS mask path."
        _mandatory = False
        _default = None

    class MaskHduArg(NonNegativeIntegerInvariant):
        _arg = "--mask-hdu"
        _help = "Optional explicit FITS HDU index for --mask."
        _mandatory = False
        _default = None

    def run(self) -> None:
        result = self._call(
            run_reproject_image,
            input_path=Path(self.input).expanduser(),
            output_root=self._path(self.output_dir),
            name=self.name,
            hdu=self.hdu,
            mask_path=self._path(self.mask),
            mask_hdu=self.mask_hdu,
            backend=self.backend,
            interpolation=self.interpolation,
            grid_crval_ra=self.grid_crval_ra,
            grid_crval_dec=self.grid_crval_dec,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            mapping_grid_step=self.mapping_grid_step,
            area_scaling=not self.disable_area_scaling,
            write_fits=self.write_fits,
        )
        self._emit_json(result.summary)


class ReprojectStackCommand(_SharedReprojectionCommand):
    """Reproject multiple FITS inputs onto one shared grid."""

    inputs = None

    class InputsArg(CsvPathInvariant):
        _arg = "--inputs"
        _help = "Comma-separated FITS image paths."
        _mandatory = True

    def run(self) -> None:
        result = self._call(
            run_reproject_stack,
            input_paths=self._csv_paths(self.inputs),
            output_root=self._path(self.output_dir),
            name=self.name,
            hdu=self.hdu,
            backend=self.backend,
            interpolation=self.interpolation,
            grid_crval_ra=self.grid_crval_ra,
            grid_crval_dec=self.grid_crval_dec,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            mapping_grid_step=self.mapping_grid_step,
            area_scaling=not self.disable_area_scaling,
            write_fits=self.write_fits,
        )
        self._emit_json(result.summary)


class BenchmarkReprojectImageCommand(_SharedReprojectionCommand):
    """Benchmark one FITS reprojection and report split timing summaries."""

    input = None
    mask = None
    mask_hdu = None

    class InputArg(ExistingPathSpecInvariant):
        _arg = "--input"
        _help = "Input FITS image path."
        _mandatory = True

    class MaskArg(ExistingPathSpecInvariant):
        _arg = "--mask"
        _help = "Optional input FITS mask path."
        _mandatory = False
        _default = None

    class MaskHduArg(NonNegativeIntegerInvariant):
        _arg = "--mask-hdu"
        _help = "Optional explicit FITS HDU index for --mask."
        _mandatory = False
        _default = None

    def run(self) -> None:
        result = self._call(
            benchmark_reproject_image,
            input_path=Path(self.input).expanduser(),
            output_root=self._path(self.output_dir),
            name=self.name,
            hdu=self.hdu,
            mask_path=self._path(self.mask),
            mask_hdu=self.mask_hdu,
            backend=self.backend,
            interpolation=self.interpolation,
            grid_crval_ra=self.grid_crval_ra,
            grid_crval_dec=self.grid_crval_dec,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            mapping_grid_step=self.mapping_grid_step,
            area_scaling=not self.disable_area_scaling,
            write_fits=self.write_fits,
            repeats=self.repeats,
            warmup=self.warmup,
        )
        self._emit_json(result.summary)


class BenchmarkBackendVariantsCommand(_SharedReprojectionCommand):
    """Benchmark cached-geometry backend variants and parity."""

    input = None
    mask = None
    mask_hdu = None
    variants = None
    reference_variant = None
    mask_cases = None
    atol = None
    rtol = None

    class InputArg(ExistingPathSpecInvariant):
        _arg = "--input"
        _help = "Input FITS image path."
        _mandatory = True

    class MaskArg(ExistingPathSpecInvariant):
        _arg = "--mask"
        _help = "Optional input FITS mask path."
        _mandatory = False
        _default = None

    class MaskHduArg(NonNegativeIntegerInvariant):
        _arg = "--mask-hdu"
        _help = "Optional explicit FITS HDU index for --mask."
        _mandatory = False
        _default = None

    class VariantsArg(BackendVariantListInvariant):
        _arg = "--variants"
        _help = "Comma-separated backend variants. [default: %default]"
        _mandatory = False
        _default = "cupy-elementwise,cupy-raw"

    class ReferenceVariantArg(BackendVariantInvariant):
        _arg = "--reference-variant"
        _help = "Variant used as the numerical reference. [default: %default]"
        _mandatory = False
        _default = "cupy-elementwise"

    class MaskCasesArg(MaskCaseListInvariant):
        _arg = "--mask-cases"
        _help = "Comma-separated mask cases: none,mask. [default: %default]"
        _mandatory = False
        _default = "none,mask"

    class AtolArg(FloatInvariant):
        _arg = "--atol"
        _help = "Absolute tolerance for image parity. [default: %default]"
        _mandatory = False
        _default = 1e-6
        _min = 0.0

    class RtolArg(FloatInvariant):
        _arg = "--rtol"
        _help = "Relative tolerance for image parity. [default: %default]"
        _mandatory = False
        _default = 1e-10
        _min = 0.0

    def run(self) -> None:
        result = self._call(
            benchmark_backend_variants_reproject_image,
            input_path=Path(self.input).expanduser(),
            output_root=self._path(self.output_dir),
            name=self.name,
            hdu=self.hdu,
            mask_path=self._path(self.mask),
            mask_hdu=self.mask_hdu,
            variants=self._csv_backend_variants(self.variants),
            reference_variant=self.reference_variant,
            mask_cases=self._csv_mask_cases(self.mask_cases),
            interpolation=self.interpolation,
            grid_crval_ra=self.grid_crval_ra,
            grid_crval_dec=self.grid_crval_dec,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            mapping_grid_step=self.mapping_grid_step,
            area_scaling=not self.disable_area_scaling,
            write_fits=self.write_fits,
            repeats=self.repeats,
            warmup=self.warmup,
            atol=self.atol,
            rtol=self.rtol,
        )
        self._emit_json(result.summary)


class CompareBackendsCommand(_SharedReprojectionCommand):
    """Run one FITS reprojection across backends and report parity metrics."""

    input = None
    mask = None
    mask_hdu = None
    backends = None
    reference_backend = None
    atol = None
    rtol = None

    class InputArg(ExistingPathSpecInvariant):
        _arg = "--input"
        _help = "Input FITS image path."
        _mandatory = True

    class MaskArg(ExistingPathSpecInvariant):
        _arg = "--mask"
        _help = "Optional input FITS mask path."
        _mandatory = False
        _default = None

    class MaskHduArg(NonNegativeIntegerInvariant):
        _arg = "--mask-hdu"
        _help = "Optional explicit FITS HDU index for --mask."
        _mandatory = False
        _default = None

    class BackendsArg(BackendListInvariant):
        _arg = "--backends"
        _help = "Comma-separated backends to compare. [default: %default]"
        _mandatory = False
        _default = "cpu,cupy"

    class ReferenceBackendArg(BackendInvariant):
        _arg = "--reference-backend"
        _help = "Backend used as the numerical reference. [default: %default]"
        _mandatory = False
        _default = "cpu"

    class AtolArg(FloatInvariant):
        _arg = "--atol"
        _help = "Absolute tolerance for image parity. [default: %default]"
        _mandatory = False
        _default = 1e-6
        _min = 0.0

    class RtolArg(FloatInvariant):
        _arg = "--rtol"
        _help = "Relative tolerance for image parity. [default: %default]"
        _mandatory = False
        _default = 1e-10
        _min = 0.0

    def run(self) -> None:
        result = self._call(
            compare_backends_reproject_image,
            input_path=Path(self.input).expanduser(),
            output_root=self._path(self.output_dir),
            name=self.name,
            hdu=self.hdu,
            mask_path=self._path(self.mask),
            mask_hdu=self.mask_hdu,
            backends=self._csv_backends(self.backends),
            reference_backend=self.reference_backend,
            interpolation=self.interpolation,
            grid_crval_ra=self.grid_crval_ra,
            grid_crval_dec=self.grid_crval_dec,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            mapping_grid_step=self.mapping_grid_step,
            area_scaling=not self.disable_area_scaling,
            write_fits=self.write_fits,
            repeats=self.repeats,
            warmup=self.warmup,
            atol=self.atol,
            rtol=self.rtol,
        )
        self._emit_json(result.summary)
