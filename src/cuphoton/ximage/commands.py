# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Class-based CLI commands for xImage."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    CSVStringInvariant,
    FloatInvariant,
    InvariantAwareCommand,
    InvariantError,
    NonNegativeIntegerInvariant,
    PathInvariant,
    PositiveIntegerInvariant,
    SetInvariant,
    StringInvariant,
)

from .data import (
    ARTIFACTS,
    load_ximage_bundle,
    summarize_bundle,
    validate_bundle,
)
from .data.rubin import (
    build_rubin_manifest,
    build_rubin_store,
    probe_fits_products,
    summarize_rubin_coadd_semantics,
)
from .data.rubin_catalog import (
    load_visit_file,
    parse_visit_text,
    validate_rubin_catalog_export,
)
from .data.rubin_image import RUBIN_IMAGE_FORMATS
from .tools.store import (
    benchmark_store,
    resolve_artifacts,
)
from .workflows import run_psf_solve, run_reconstruction

T = TypeVar("T")
SUPPORTED_WORKFLOWS = {
    "psf-adam",
    "psf-mm",
    "reconstruct-l2",
    "reconstruct-rho",
}
SUPPORTED_BACKENDS = {"auto", "torch"}
SUPPORTED_DEVICES = {"cpu", "cuda"}

_KNOWN_COMMAND_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    ImportError,
    NotImplementedError,
    RuntimeError,
    ValueError,
)


def _emit_json_payload(payload: Any, *, out: Callable[[str], None]) -> None:
    out(json.dumps(payload, indent=2, default=str))


def benchmark_devices(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy wrapper so catalog-only commands avoid image imports."""

    from .benchmark import benchmark_devices as _benchmark_devices

    return _benchmark_devices(*args, **kwargs)


def build_rubin_coadd_reference_catalog(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lazy wrapper around evaluation helpers."""

    from .evaluation import (
        build_rubin_coadd_reference_catalog as _build_reference,
    )

    return _build_reference(*args, **kwargs)


def evaluate_reconstruction_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy wrapper around evaluation helpers."""

    from .evaluation import evaluate_reconstruction_run as _evaluate

    return _evaluate(*args, **kwargs)


class PathSpecInvariant(PathInvariant):
    """xImage path argument based on the shared core invariant."""


class ArtifactSpecInvariant(StringInvariant):
    """Comma-separated artifact list or ``all``."""

    expected = "'all' or a comma-separated list of valid artifact keys"

    @classmethod
    def validate(cls, value: Any) -> str | None:
        text = super().validate(value)
        if text is None:
            return None
        try:
            artifacts = resolve_artifacts(text)
        except ValueError as exc:
            raise InvariantError(str(exc)) from exc
        if not artifacts:
            raise InvariantError("must contain at least one artifact key")
        return text


class FiniteFloatInvariant(FloatInvariant):
    """Finite floating-point value."""

    @classmethod
    def validate(cls, value: Any) -> float | None:
        converted = super().validate(value)
        if converted is not None and not math.isfinite(converted):
            raise InvariantError("must be finite")
        return converted


class PositiveFloatInvariant(FiniteFloatInvariant):
    """Finite floating-point value greater than zero."""

    @classmethod
    def validate(cls, value: Any) -> float | None:
        converted = super().validate(value)
        if converted is not None and converted <= 0.0:
            raise InvariantError("must be greater than zero")
        return converted


class UnitIntervalFloatInvariant(FiniteFloatInvariant):
    """Finite floating-point value in the interval ``(0, 1]``."""

    @classmethod
    def validate(cls, value: Any) -> float | None:
        converted = super().validate(value)
        if converted is not None and not 0.0 < converted <= 1.0:
            raise InvariantError("must be in the interval (0, 1]")
        return converted


class FormatInvariant(SetInvariant):
    """Storage format choice."""

    _set = {"npy", "hdf5", "zarr"}


class BackendInvariant(SetInvariant):
    """Workflow backend choice."""

    _set = SUPPORTED_BACKENDS


class DeviceInvariant(SetInvariant):
    """Workflow execution device choice."""

    _set = {"auto", "cpu", "cuda"}


class CSVChoiceSetInvariant(CSVStringInvariant):
    """Expose the shared multiple-choice validation as one CSV option."""

    # The core parser renders SetInvariant subclasses as argparse choices,
    # which would reject the comma-separated token before validation.
    _set: set[str] = set()

    @classmethod
    def validate(cls, value: Any) -> str | None:
        text = super().validate(value)
        if text is not None:
            items = [item.strip() for item in text.split(",") if item.strip()]
            if any(item not in cls._set for item in items):
                choices = ", ".join(sorted(cls._set))
                raise InvariantError(f"must be a subset of: {choices}")
        return text


class DeviceSetInvariant(CSVChoiceSetInvariant):
    """Comma-separated execution device choices."""

    _set = SUPPORTED_DEVICES


class WorkflowInvariant(SetInvariant):
    """Benchmarkable workflow selection."""

    _set = SUPPORTED_WORKFLOWS


class ArtifactKeyInvariant(SetInvariant):
    """Single artifact key choice."""

    _set = set(ARTIFACTS)


class RubinProductSetInvariant(CSVChoiceSetInvariant):
    """Comma-separated Rubin product choices."""

    _set = {
        "visit_image",
        "difference_image",
        "template_coadd",
        "deep_coadd",
    }


class RubinCatalogProductSetInvariant(CSVChoiceSetInvariant):
    """Comma-separated Rubin catalog product choices."""

    _set = {
        "dia-forced-source",
        "dia-object",
        "dia-source",
        "forced-source",
        "object",
        "source",
    }


class RubinImageFormatInvariant(SetInvariant):
    """Explicit normalized Rubin image backend choice."""

    _set = set(RUBIN_IMAGE_FORMATS)


class XImageCommand(InvariantAwareCommand):
    """Base command with shared JSON emission and error translation."""

    _shortname_ = None
    _log_level_ = False

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


class RubinProbeCommand(XImageCommand):
    """Probe Rubin/LSST FITS products for xImage compatibility."""

    input_path: str | None = None
    max_files: int = 32
    read_sample: bool = False
    sample_section: str | None = None

    class InputPathArg(PathSpecInvariant):
        _arg = "--input"
        _help = "FITS file or directory tree to probe."
        _mandatory = True

    class MaxFilesArg(PositiveIntegerInvariant):
        _arg = "--max-files"
        _help = "Maximum number of FITS files to inspect. [default: %default]"
        _mandatory = False
        _default = 32

    class ReadSampleArg(BoolInvariant):
        _arg = "--read-sample"
        _help = "Read a small 2D ROI from image HDUs and report statistics."
        _mandatory = False
        _default = False

    class SampleSectionArg(StringInvariant):
        _arg = "--sample-section"
        _help = (
            "Explicit ROI as y0:y1,x0:x1 for --read-sample. "
            "Defaults to a centered 31x31 section."
        )
        _mandatory = False
        _default = None

    def run(self) -> None:
        assert self.input_path is not None
        result = self._call(
            probe_fits_products,
            self._path(self.input_path),
            max_files=int(self.max_files) if self.max_files else None,
            read_sample=bool(self.read_sample),
            sample_section=self.sample_section,
        )
        self._emit_json(result)


class RubinBuildManifestCommand(XImageCommand):
    """Build a deterministic Rubin/LSST FITS manifest."""

    input_path: str | None = None
    output_path: str | None = None
    inventory: str | None = None
    products: str = "visit_image,difference_image"
    sample_per_product: int = 0
    group_prefix: str = "local"
    skip_metadata: bool = False

    class InputPathArg(PathSpecInvariant):
        _arg = "--input"
        _help = "FITS file or directory tree to enumerate."
        _mandatory = True

    class OutputPathArg(PathSpecInvariant):
        _arg = "--output"
        _help = "Destination manifest JSON path."
        _mandatory = False
        _default = None

    class InventoryArg(PathSpecInvariant):
        _arg = "--inventory"
        _help = (
            "Optional TSV inventory with path or size/path rows. "
            "When set, --input is treated as the source root."
        )
        _mandatory = False
        _default = None

    class ProductsArg(RubinProductSetInvariant):
        _arg = "--products"
        _help = "Comma-separated Rubin products. [default: %default]"
        _mandatory = False
        _default = "visit_image,difference_image"

    class SamplePerProductArg(NonNegativeIntegerInvariant):
        _arg = "--sample-per-product"
        _help = (
            "Maximum files per product; 0 means all matches. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 0

    class GroupPrefixArg(StringInvariant):
        _arg = "--group-prefix"
        _help = "Manifest group prefix. [default: %default]"
        _mandatory = False
        _default = "local"

    class SkipMetadataArg(BoolInvariant):
        _arg = "--skip-metadata"
        _help = "Skip FITS metadata inspection and only enumerate paths."
        _mandatory = False
        _default = False

    def run(self) -> None:
        assert self.input_path is not None
        if isinstance(self.products, str):
            products = tuple(
                token.strip()
                for token in self.products.split(",")
                if token.strip()
            )
        else:
            products = tuple(self.products)
        result = self._call(
            build_rubin_manifest,
            self._path(self.input_path),
            output_path=self._path(self.output_path),
            inventory_path=self._path(self.inventory),
            products=products,
            sample_per_product=int(self.sample_per_product),
            inspect_metadata=not bool(self.skip_metadata),
            group_prefix=self.group_prefix,
        )
        self._emit_json(result)


class RubinBuildStoreCommand(XImageCommand):
    """Build an xImage NPY store from Rubin/LSST FITS products."""

    manifest: str | None = None
    input_root: str | None = None
    output_path: str | None = None
    group: str | None = None
    product: str = "visit_image"
    max_files: int = 0
    roi: str | None = None
    mask_policy: str = "rubin-bad-planes"
    psf_size: int = 25
    align_wcs: bool = False
    reference_index: int = 0
    overwrite: bool = False

    class ManifestArg(PathSpecInvariant):
        _arg = "--manifest"
        _help = "Rubin local-batch manifest JSON."
        _mandatory = True

    class InputRootArg(PathSpecInvariant):
        _arg = "--input-root"
        _help = "Directory containing FITS paths relative to the manifest."
        _mandatory = True

    class OutputPathArg(PathSpecInvariant):
        _arg = "--output"
        _help = "Destination xImage-compatible NPY store directory."
        _mandatory = True

    class GroupArg(StringInvariant):
        _arg = "--group"
        _help = "Manifest group to convert."
        _mandatory = True

    class ProductArg(SetInvariant):
        _arg = "--product"
        _help = "Rubin product to stack. [default: %default]"
        _mandatory = False
        _default = "visit_image"
        _set = {
            "visit_image",
            "difference_image",
            "template_coadd",
            "deep_coadd",
        }

    class MaxFilesArg(NonNegativeIntegerInvariant):
        _arg = "--max-files"
        _help = (
            "Maximum files to stack; 0 means all matches. [default: %default]"
        )
        _mandatory = False
        _default = 0

    class RoiArg(StringInvariant):
        _arg = "--roi"
        _help = "ROI as y0:y1,x0:x1. Defaults to centered 512x512 sections."
        _mandatory = False
        _default = None

    class MaskPolicyArg(SetInvariant):
        _arg = "--mask-policy"
        _help = "Rubin MASK conversion policy. [default: %default]"
        _mandatory = False
        _default = "rubin-bad-planes"
        _set = {"rubin-bad-planes", "nonzero-bad", "none-good"}

    class PsfSizeArg(PositiveIntegerInvariant):
        _arg = "--psf-size"
        _help = (
            "Odd PSF kernel size for summary-Gaussian or delta kernels. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 25

    class AlignWcsArg(BoolInvariant):
        _arg = "--align-wcs"
        _help = (
            "Resample each FITS exposure onto the reference IMAGE WCS "
            "before stacking."
        )
        _mandatory = False
        _default = False

    class ReferenceIndexArg(NonNegativeIntegerInvariant):
        _arg = "--reference-index"
        _help = (
            "Selected manifest row to use as the WCS reference for "
            "--align-wcs. [default: %default]"
        )
        _mandatory = False
        _default = 0

    class OverwriteArg(BoolInvariant):
        _arg = "--overwrite"
        _help = "Replace an existing output store."
        _mandatory = False
        _default = False

    def run(self) -> None:
        assert self.manifest is not None
        assert self.input_root is not None
        assert self.output_path is not None
        assert self.group is not None
        max_files = int(self.max_files) if self.max_files else 0
        result = self._call(
            build_rubin_store,
            manifest_path=self._path(self.manifest),
            input_root=self._path(self.input_root),
            output_dir=self._path(self.output_path),
            group=self.group,
            product=self.product,
            max_files=max_files or None,
            roi=self.roi,
            mask_policy=self.mask_policy,
            psf_size=int(self.psf_size),
            align_wcs=bool(self.align_wcs),
            reference_index=int(self.reference_index),
            overwrite=bool(self.overwrite),
        )
        self._emit_json(result)


class RubinBuildReferenceCatalogCommand(XImageCommand):
    """Detect Rubin coadd sources on the reconstruction pixel grid."""

    coadd_fits: str | None = None
    store: str | None = None
    output_csv: str | None = None
    output_summary: str | None = None
    threshold: float = 5.0
    minarea: int = 5
    overwrite: bool = False

    class CoaddFitsArg(PathSpecInvariant):
        _arg = "--coadd-fits"
        _help = "Pipeline coadd FITS to detect sources from."
        _mandatory = True

    class StoreArg(PathSpecInvariant):
        _arg = "--store"
        _help = "WCS-aligned Rubin xImage store directory."
        _mandatory = True

    class OutputCsvArg(PathSpecInvariant):
        _arg = "--output-csv"
        _help = "Reference catalog CSV output path."
        _mandatory = True

    class OutputSummaryArg(PathSpecInvariant):
        _arg = "--output-summary"
        _help = "Optional JSON summary output path."
        _mandatory = False
        _default = None

    class ThresholdArg(PositiveFloatInvariant):
        _arg = "--threshold"
        _help = "Absolute source-detection threshold. [default: %default]"
        _mandatory = False
        _default = "5.0"

    class MinareaArg(PositiveIntegerInvariant):
        _arg = "--minarea"
        _help = "Minimum connected source pixels. [default: %default]"
        _mandatory = False
        _default = 5

    class OverwriteArg(BoolInvariant):
        _arg = "--overwrite"
        _help = "Replace existing catalog outputs."
        _mandatory = False
        _default = False

    def run(self) -> None:
        assert self.coadd_fits is not None
        assert self.store is not None
        assert self.output_csv is not None
        result = self._call(
            build_rubin_coadd_reference_catalog,
            coadd_fits_path=self._path(self.coadd_fits),
            store_dir=self._path(self.store),
            output_csv=self._path(self.output_csv),
            output_summary=self._path(self.output_summary),
            threshold=float(self.threshold),
            minarea=int(self.minarea),
            overwrite=bool(self.overwrite),
        )
        self._emit_json(result)


class RubinValidateCatalogsCommand(XImageCommand):
    """Validate Rubin/LSST catalog CSV exports before evaluation."""

    input_dir: str | None = None
    visits: str | None = None
    visits_file: str | None = None
    products: str = "object,source,forced-source"
    band: str = "r"

    class InputDirArg(PathSpecInvariant):
        _arg = "--input-dir"
        _help = "Directory containing local Rubin catalog CSV files."
        _mandatory = True

    class VisitsArg(StringInvariant):
        _arg = "--visits"
        _help = "Optional comma-separated expected visit IDs."
        _mandatory = False
        _default = None

    class VisitsFileArg(PathSpecInvariant):
        _arg = "--visits-file"
        _help = "Optional file with one expected visit ID per line."
        _mandatory = False
        _default = None

    class ProductsArg(RubinCatalogProductSetInvariant):
        _arg = "--products"
        _help = "Comma-separated catalog products. [default: %default]"
        _mandatory = False
        _default = "object,source,forced-source"

    class BandArg(StringInvariant):
        _arg = "--band"
        _help = "Band-specific columns to require. [default: %default]"
        _mandatory = False
        _default = "r"

    def run(self) -> None:
        assert self.input_dir is not None
        product_names = (
            token.strip()
            for token in str(self.products).split(",")
            if token.strip()
        )
        visits = self._call(parse_visit_text, self.visits) + self._call(
            load_visit_file, self._path(self.visits_file)
        )
        result = self._call(
            validate_rubin_catalog_export,
            input_dir=self._path(self.input_dir),
            products=tuple(product_names),
            visits=visits,
            band=self.band,
        )
        self._emit_json(result)


class RubinCoaddSemanticsCommand(XImageCommand):
    """Report Rubin coadd metadata and reconstruction limitations."""

    coadd_fits: str | None = None
    image_hdu: str = "IMAGE"
    rubin_image_format: str = "legacy-fits"
    output_json: str | None = None
    require_coadd_psf: bool = False
    require_coadd_inputs: bool = False
    require_visit_info: bool = False
    require_ok: bool = False

    class CoaddFitsArg(PathSpecInvariant):
        _arg = "--coadd-fits"
        _help = "Rubin pipeline coadd FITS to inspect."
        _mandatory = True

    class ImageHduArg(StringInvariant):
        _arg = "--image-hdu"
        _help = "FITS image HDU to inspect. [default: %default]"
        _mandatory = False
        _default = "IMAGE"

    class RubinImageFormatArg(RubinImageFormatInvariant):
        _arg = "--rubin-image-format"
        _help = (
            "Explicit Rubin parser backend: legacy-fits, "
            "lsst-images-archive, or lsst-images-cutdown. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "legacy-fits"

    class OutputJsonArg(PathSpecInvariant):
        _arg = "--output-json"
        _help = "Optional destination for the coadd semantics JSON."
        _mandatory = False
        _default = None

    class RequireCoaddPsfArg(BoolInvariant):
        _arg = "--require-coadd-psf"
        _help = "Require coadd PSF metadata."
        _mandatory = False
        _default = False

    class RequireCoaddInputsArg(BoolInvariant):
        _arg = "--require-coadd-inputs"
        _help = "Require a CoaddInputs HDU."
        _mandatory = False
        _default = False

    class RequireVisitInfoArg(BoolInvariant):
        _arg = "--require-visit-info"
        _help = "Require a VisitInfo HDU."
        _mandatory = False
        _default = False

    class RequireOkArg(BoolInvariant):
        _arg = "--require-ok"
        _help = (
            "Exit with an error unless archive completeness and requested "
            "metadata requirements are met. Requires lsst-images-archive."
        )
        _mandatory = False
        _default = False

    def run(self) -> None:
        assert self.coadd_fits is not None
        result = self._call(
            summarize_rubin_coadd_semantics,
            self._path(self.coadd_fits),
            image_hdu=self.image_hdu,
            rubin_image_format=self.rubin_image_format,
            output_path=self._path(self.output_json),
            require_coadd_psf=bool(self.require_coadd_psf),
            require_coadd_inputs=bool(self.require_coadd_inputs),
            require_visit_info=bool(self.require_visit_info),
        )
        self._emit_json(result)
        if self.require_ok and not result.get("requirements_met"):
            raise CommandError(
                "Rubin coadd metadata requirements were not met"
            )


class BenchmarkDevicesCommand(XImageCommand):
    """Benchmark one workflow config on CPU and CUDA devices."""

    config: str | None = None
    workflow: str | None = None
    devices: str = "cpu,cuda"
    repeats: int = 3
    warmup: int = 1
    output_dir: str | None = None
    cuda_visible_devices: str | None = None
    evaluate: bool = False
    coadd_method: str = "median"
    pipeline_coadd_fits: str | None = None
    catalog_parquet: str | None = None
    reference_catalog_csv: str | None = None
    catalog_band: str = "i"
    catalog_match_radius_px: float = 3.0
    background_min_fraction: float = 0.5
    photometry_threshold: float = 0.1
    photometry_minarea: int = 5

    class ConfigArg(PathSpecInvariant):
        _arg = "--config"
        _help = "Workflow YAML configuration file to benchmark."
        _mandatory = True

    class WorkflowArg(WorkflowInvariant):
        _arg = "--workflow"
        _help = "Packaged workflow to benchmark."
        _mandatory = True

    class DevicesArg(DeviceSetInvariant):
        _arg = "--devices"
        _help = "Comma-separated devices to compare. [default: %default]"
        _mandatory = False
        _default = "cpu,cuda"

    class RepeatsArg(PositiveIntegerInvariant):
        _arg = "--repeats"
        _help = "Measured repetitions per device. [default: %default]"
        _mandatory = False
        _default = 3

    class WarmupArg(NonNegativeIntegerInvariant):
        _arg = "--warmup"
        _help = "Warmup runs per device before timing. [default: %default]"
        _mandatory = False
        _default = 1

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Optional output root to keep benchmark run artifacts."
        _mandatory = False
        _default = None

    class CudaVisibleDevicesArg(StringInvariant):
        _arg = "--cuda-visible-devices"
        _help = (
            "Optional CUDA_VISIBLE_DEVICES override applied before "
            "Torch imports."
        )
        _mandatory = False
        _default = None

    class EvaluateArg(BoolInvariant):
        _arg = "--evaluate"
        _help = "Also evaluate reconstruction quality for measured runs."
        _mandatory = False
        _default = False

    class CoaddMethodArg(SetInvariant):
        _arg = "--coadd-method"
        _help = "Synthetic coadd method for evaluation. [default: %default]"
        _mandatory = False
        _default = "median"
        _set = {"median", "mean"}

    class PipelineCoaddFitsArg(PathSpecInvariant):
        _arg = "--pipeline-coadd-fits"
        _help = "Optional pipeline-coadd FITS override for evaluation."
        _mandatory = False
        _default = None

    class CatalogParquetArg(PathSpecInvariant):
        _arg = "--catalog-parquet"
        _help = "Optional LSST object-table parquet override."
        _mandatory = False
        _default = None

    class ReferenceCatalogCsvArg(PathSpecInvariant):
        _arg = "--reference-catalog-csv"
        _help = (
            "Optional reference catalog CSV, such as the output from "
            "rubin-build-reference-catalog or a local Rubin catalog "
            "with sky coordinates. A local catalog directory "
            "selects object.csv when present."
        )
        _mandatory = False
        _default = None

    class CatalogBandArg(StringInvariant):
        _arg = "--catalog-band"
        _help = (
            "Catalog band prefix to use for reference fluxes. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "i"

    class CatalogMatchRadiusPxArg(PositiveFloatInvariant):
        _arg = "--catalog-match-radius-px"
        _help = (
            "Maximum detection-to-catalog match radius in pixels. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "3.0"

    class BackgroundMinFractionArg(UnitIntervalFloatInvariant):
        _arg = "--background-min-fraction"
        _help = (
            "Minimum mean mask value for a pixel to count as "
            "background during evaluation. [default: %default]"
        )
        _mandatory = False
        _default = "0.5"

    class PhotometryThresholdArg(PositiveFloatInvariant):
        _arg = "--photometry-threshold"
        _help = (
            "Absolute source-detection threshold for photometry. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "0.1"

    class PhotometryMinareaArg(PositiveIntegerInvariant):
        _arg = "--photometry-minarea"
        _help = "Minimum connected source pixels. [default: %default]"
        _mandatory = False
        _default = 5

    def run(self) -> None:
        assert self.config is not None
        assert self.workflow is not None
        if isinstance(self.devices, str):
            devices = [
                token.strip()
                for token in self.devices.split(",")
                if token.strip()
            ]
        else:
            devices = sorted(self.devices)
        background_min_fraction = (
            float(self.background_min_fraction)
            if self.background_min_fraction is not None
            else 0.5
        )
        photometry_minarea = (
            int(self.photometry_minarea)
            if self.photometry_minarea is not None
            else 5
        )
        photometry_threshold = (
            float(self.photometry_threshold)
            if self.photometry_threshold is not None
            else 0.1
        )
        catalog_match_radius_px = (
            float(self.catalog_match_radius_px)
            if self.catalog_match_radius_px is not None
            else 3.0
        )
        result = self._call(
            benchmark_devices,
            workflow=self.workflow,
            config_path=Path(self.config).expanduser(),
            devices=devices,
            repeats=self.repeats,
            warmup=self.warmup,
            output_root=(
                None
                if self.output_dir is None
                else Path(self.output_dir).expanduser()
            ),
            cuda_visible_devices=self.cuda_visible_devices,
            evaluate=bool(self.evaluate),
            coadd_method=self.coadd_method,
            pipeline_coadd_fits=self._path(self.pipeline_coadd_fits),
            catalog_parquet=self._path(self.catalog_parquet),
            reference_catalog_csv=self._path(self.reference_catalog_csv),
            catalog_band=self.catalog_band,
            catalog_match_radius_px=catalog_match_radius_px,
            background_min_fraction=background_min_fraction,
            photometry_threshold=photometry_threshold,
            photometry_minarea=photometry_minarea,
        )
        self._emit_json(result)


class DataInspectCommand(XImageCommand):
    """Inspect arrays in an NPY, HDF5, or Zarr reconstruction bundle."""

    store: str | None = None
    artifacts: str = "all"

    class StoreArg(PathSpecInvariant):
        _arg = "--store"
        _help = "Reconstruction bundle directory or container to inspect."
        _mandatory = True

    class ArtifactsArg(ArtifactSpecInvariant):
        _arg = "--artifacts"
        _help = (
            "Artifact selection as 'all' or a comma-separated list, "
            "such as 'images,psfs'. [default: %default]"
        )
        _mandatory = False
        _default = "all"

    def run(self) -> None:
        assert self.store is not None
        with self._call(
            load_ximage_bundle,
            self._path(self.store),
            artifacts=self.artifacts,
        ) as bundle:
            summary = summarize_bundle(bundle)
        self._emit_json(summary)


class DataValidateCommand(XImageCommand):
    """Validate a complete resolved xImage reconstruction bundle."""

    base: str = "data"

    class BaseArg(PathSpecInvariant):
        _arg = "--base"
        _help = (
            "Base data directory or explicit store path to validate. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "data"

    def run(self) -> None:
        with self._call(load_ximage_bundle, self._path(self.base)) as bundle:
            issues = validate_bundle(bundle)
            if issues:
                details = "\n".join(f"- {issue}" for issue in issues)
                raise CommandError(
                    f"xImage bundle validation failed:\n{details}"
                )
            summary = summarize_bundle(bundle)
        self._emit_json(summary)


class DataBenchmarkCommand(XImageCommand):
    """Benchmark spatial ROI reads from a xImage reconstruction bundle."""

    store: str | None = None
    format: str | None = None
    artifact: str = "images"
    roi_height: int = 512
    roi_width: int = 512
    repeats: int = 5

    class StoreArg(PathSpecInvariant):
        _arg = "--store"
        _help = "Path to the reconstruction bundle to benchmark."
        _mandatory = True

    class FormatArg(FormatInvariant):
        _arg = "--format"
        _help = "Expected storage format for the benchmark target."
        _mandatory = True

    class ArtifactArg(ArtifactKeyInvariant):
        _arg = "--artifact"
        _help = "Artifact key to benchmark. [default: %default]"
        _mandatory = False
        _default = "images"

    class RoiHeightArg(PositiveIntegerInvariant):
        _arg = "--roi-height"
        _help = "ROI height in pixels. [default: %default]"
        _mandatory = False
        _default = 512

    class RoiWidthArg(PositiveIntegerInvariant):
        _arg = "--roi-width"
        _help = "ROI width in pixels. [default: %default]"
        _mandatory = False
        _default = 512

    class RepeatsArg(PositiveIntegerInvariant):
        _arg = "--repeats"
        _help = "Number of timing repetitions. [default: %default]"
        _mandatory = False
        _default = 5

    def run(self) -> None:
        assert self.store is not None
        assert self.format is not None
        result = self._call(
            benchmark_store,
            store=self._path(self.store),
            fmt=self.format,
            artifact=ARTIFACTS[self.artifact],
            roi_height=self.roi_height,
            roi_width=self.roi_width,
            repeats=self.repeats,
        )
        self._emit_json(result)


class _WorkflowCommand(XImageCommand):
    """Shared options for packaged workflow commands."""

    config: str | None = None
    backend: str | None = None
    device: str | None = None
    output_dir: str | None = None

    class ConfigArg(PathSpecInvariant):
        _arg = "--config"
        _help = "Workflow YAML configuration file."
        _mandatory = True

    class BackendArg(BackendInvariant):
        _arg = "--backend"
        _help = "Backend override for this invocation."
        _mandatory = False
        _default = None

    class DeviceArg(DeviceInvariant):
        _arg = "--device"
        _help = "Execution device override for this invocation."
        _mandatory = False
        _default = None

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "Override the workflow output root directory."
        _mandatory = False
        _default = None

    def _config_path(self) -> Path:
        assert self.config is not None
        return Path(self.config).expanduser()

    def _output_root(self) -> Path | None:
        return (
            None
            if self.output_dir is None
            else Path(self.output_dir).expanduser()
        )

    def _default_output_root(self) -> Path | None:
        return self.context.runs_dir if self.context is not None else None


class PsfSolveAdamCommand(_WorkflowCommand):
    """Run the packaged h-PSF solve workflow using Adam optimization."""

    def run(self) -> None:
        result = self._call(
            run_psf_solve,
            self._config_path(),
            backend_override=self.backend,
            device_override=self.device,
            output_root=self._output_root(),
            default_output_root=self._default_output_root(),
            method_override="adam",
        )
        self._emit_json(result.summary)


class PsfSolveMmCommand(_WorkflowCommand):
    """Run the packaged h-PSF solve workflow using multiplicative updates."""

    def run(self) -> None:
        result = self._call(
            run_psf_solve,
            self._config_path(),
            backend_override=self.backend,
            device_override=self.device,
            output_root=self._output_root(),
            default_output_root=self._default_output_root(),
            method_override="mm",
        )
        self._emit_json(result.summary)


class ReconstructL2Command(_WorkflowCommand):
    """Run packaged L2 image reconstruction and emit the run summary."""

    def run(self) -> None:
        result = self._call(
            run_reconstruction,
            self._config_path(),
            backend_override=self.backend,
            device_override=self.device,
            output_root=self._output_root(),
            default_output_root=self._default_output_root(),
            algorithm_override="l2",
        )
        self._emit_json(result.summary)


class ReconstructRhoCommand(_WorkflowCommand):
    """Run packaged rho-loss image reconstruction and emit the run summary."""

    def run(self) -> None:
        result = self._call(
            run_reconstruction,
            self._config_path(),
            backend_override=self.backend,
            device_override=self.device,
            output_root=self._output_root(),
            default_output_root=self._default_output_root(),
            algorithm_override="rho",
        )
        self._emit_json(result.summary)


class EvaluateReconstructionCommand(XImageCommand):
    """Evaluate a packaged reconstruction run against source data."""

    run_dir: str | None = None
    base: str | None = None
    coadd_method: str = "median"
    pipeline_coadd_fits: str | None = None
    catalog_parquet: str | None = None
    reference_catalog_csv: str | None = None
    catalog_band: str = "i"
    catalog_match_radius_px: float = 3.0
    background_min_fraction: float = 0.5
    photometry_threshold: float = 0.1
    photometry_minarea: int = 5

    class RunDirArg(PathSpecInvariant):
        _arg = "--run-dir"
        _help = "Packaged reconstruction run directory to evaluate."
        _mandatory = True

    class BaseArg(PathSpecInvariant):
        _arg = "--base"
        _help = "Optional data-base override instead of the run config value."
        _mandatory = False
        _default = None

    class CoaddMethodArg(SetInvariant):
        _arg = "--coadd-method"
        _help = "Synthetic coadd method. [default: %default]"
        _mandatory = False
        _default = "median"
        _set = {"median", "mean"}

    class PipelineCoaddFitsArg(PathSpecInvariant):
        _arg = "--pipeline-coadd-fits"
        _help = "Optional pipeline-coadd FITS override."
        _mandatory = False
        _default = None

    class CatalogParquetArg(PathSpecInvariant):
        _arg = "--catalog-parquet"
        _help = "Optional LSST object-table parquet override."
        _mandatory = False
        _default = None

    class ReferenceCatalogCsvArg(PathSpecInvariant):
        _arg = "--reference-catalog-csv"
        _help = (
            "Optional reference catalog CSV, such as the output from "
            "rubin-build-reference-catalog or a local Rubin catalog "
            "with sky coordinates. A local catalog directory "
            "selects object.csv when present."
        )
        _mandatory = False
        _default = None

    class CatalogBandArg(StringInvariant):
        _arg = "--catalog-band"
        _help = (
            "Catalog band prefix to use for reference fluxes. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "i"

    class CatalogMatchRadiusPxArg(PositiveFloatInvariant):
        _arg = "--catalog-match-radius-px"
        _help = (
            "Maximum detection-to-catalog match radius in pixels. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "3.0"

    class BackgroundMinFractionArg(UnitIntervalFloatInvariant):
        _arg = "--background-min-fraction"
        _help = (
            "Minimum mean mask value for a pixel to count as background. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "0.5"

    class PhotometryThresholdArg(PositiveFloatInvariant):
        _arg = "--photometry-threshold"
        _help = (
            "Absolute source-detection threshold for photometry. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "0.1"

    class PhotometryMinareaArg(PositiveIntegerInvariant):
        _arg = "--photometry-minarea"
        _help = "Minimum connected source pixels. [default: %default]"
        _mandatory = False
        _default = 5

    def run(self) -> None:
        assert self.run_dir is not None
        background_min_fraction = (
            float(self.background_min_fraction)
            if self.background_min_fraction is not None
            else 0.5
        )
        photometry_minarea = (
            int(self.photometry_minarea)
            if self.photometry_minarea is not None
            else 5
        )
        photometry_threshold = (
            float(self.photometry_threshold)
            if self.photometry_threshold is not None
            else 0.1
        )
        catalog_match_radius_px = (
            float(self.catalog_match_radius_px)
            if self.catalog_match_radius_px is not None
            else 3.0
        )
        result = self._call(
            evaluate_reconstruction_run,
            self._path(self.run_dir),
            base_override=self._path(self.base),
            coadd_method=self.coadd_method,
            pipeline_coadd_fits=self._path(self.pipeline_coadd_fits),
            catalog_parquet=self._path(self.catalog_parquet),
            reference_catalog_csv=self._path(self.reference_catalog_csv),
            catalog_band=self.catalog_band,
            catalog_match_radius_px=catalog_match_radius_px,
            background_min_fraction=background_min_fraction,
            photometry_threshold=photometry_threshold,
            photometry_minarea=photometry_minarea,
            write_artifacts=True,
        )
        self._emit_json(result)
