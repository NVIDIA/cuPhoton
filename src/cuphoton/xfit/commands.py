# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Invariant-aware command-line interface for xFit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import numpy as np

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    ExistingPathInvariant,
    FloatInvariant,
    InvariantAwareCommand,
    PositiveIntegerInvariant,
    SetInvariant,
    StringInvariant,
)

from ._types import (
    BACKEND_REQUESTS,
    COMPUTE_DTYPES,
    FIT_MODES,
    MODEL_NAMES,
    STAMP_EVALUATIONS,
)
from .io import (
    XFitDataset,
    inspect_xfit_dataset,
    load_xfit_dataset,
    write_fit_artifacts,
)
from .models import StampDipoleModel

T = TypeVar("T")
P = ParamSpec("P")

_KNOWN_COMMAND_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class PathSpecInvariant(StringInvariant):
    _type_desc = "path"
    _minlen = 1
    _maxlen = 4096


class ExistingNPZInvariant(ExistingPathInvariant):
    _type_desc = "existing .npz path"


class ModelInvariant(SetInvariant):
    _set = MODEL_NAMES


class ModeInvariant(SetInvariant):
    _set = FIT_MODES


class BackendInvariant(SetInvariant):
    _set = BACKEND_REQUESTS


class ComputeDtypeInvariant(SetInvariant):
    _set = COMPUTE_DTYPES


class StampEvaluationInvariant(SetInvariant):
    _set = STAMP_EVALUATIONS


class FiniteFloatInvariant(FloatInvariant):
    @classmethod
    def validate(cls, value: Any) -> float | None:
        converted = super().validate(value)
        if converted is not None and not np.isfinite(converted):
            raise ValueError("must be finite")
        return converted


class PositiveFloatInvariant(FiniteFloatInvariant):
    @classmethod
    def validate(cls, value: Any) -> float | None:
        converted = super().validate(value)
        if converted is not None and converted <= 0:
            raise ValueError("must be greater than zero")
        return converted


class NonNegativeFloatInvariant(FiniteFloatInvariant):
    _min = 0.0


class XFitCommand(InvariantAwareCommand):
    """Shared xFit command behavior."""

    def _call(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        try:
            return func(*args, **kwargs)
        except _KNOWN_COMMAND_EXCEPTIONS as exc:
            raise CommandError(str(exc)) from exc

    def _emit_json(self, payload: object) -> None:
        self._out(
            json.dumps(
                payload,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )


class DataInspectCommand(XFitCommand):
    """Inspect a pickle-free NPZ of numeric or Unicode arrays."""

    input = None

    class InputArg(ExistingNPZInvariant):
        _arg = "--input"
        _help = "Input .npz containing candidate_id and images arrays."
        _mandatory = True

    def run(self) -> None:
        dataset = self._call(load_xfit_dataset, self.input)
        self._emit_json(inspect_xfit_dataset(dataset))


class _ValidatedDatasetCommand(XFitCommand):
    input = None
    model = None
    mode = None

    class InputArg(ExistingNPZInvariant):
        _arg = "--input"
        _help = "Input .npz containing candidate_id and images arrays."
        _mandatory = True

    class ModelArg(ModelInvariant):
        _arg = "--model"
        _help = "Dipole model: gaussian or stamp."
        _mandatory = True

    class ModeArg(ModeInvariant):
        _arg = "--mode"
        _help = "Image layout: difference or split. [default: %default]"
        _mandatory = False
        _default = "difference"

    def _load_dataset(self) -> XFitDataset:
        return self._call(
            load_xfit_dataset,
            self.input,
            model=self.model,
            mode=self.mode,
        )


class DataValidateCommand(_ValidatedDatasetCommand):
    """Validate an xFit NPZ input for a selected model and image mode."""

    def run(self) -> None:
        dataset = self._load_dataset()
        payload = inspect_xfit_dataset(dataset)
        payload.update(
            {
                "model": self.model,
                "valid": True,
            }
        )
        self._emit_json(payload)


class FitDipolesCommand(_ValidatedDatasetCommand):
    """Fit a batch of astronomical dipoles and persist safe artifacts."""

    output_dir = None
    backend = None
    compute_dtype = None
    stamp_evaluation = None
    stamp_scale = None
    f_tol = None
    x_tol = None
    g_tol = None
    max_evaluations = None
    use_finite_difference = None

    class OutputDirArg(PathSpecInvariant):
        _arg = "--output-dir"
        _help = "New directory for the xFit run artifacts."
        _mandatory = True

    class BackendArg(BackendInvariant):
        _arg = "--backend"
        _help = "Array backend: auto, numpy, or cupy. [default: %default]"
        _mandatory = False
        _default = "auto"

    class ComputeDtypeArg(ComputeDtypeInvariant):
        _arg = "--compute-dtype"
        _help = (
            "Solver dtype: input, float32, or float64. [default: %default]"
        )
        _mandatory = False
        _default = "input"

    class StampEvaluationArg(StampEvaluationInvariant):
        _arg = "--stamp-evaluation"
        _help = (
            "Sampled-stamp evaluation: bilinear, bilinear-vignetted, or "
            "finite-volume. [default: %default]"
        )
        _mandatory = False
        _default = "bilinear"

    class StampScaleArg(PositiveFloatInvariant):
        _arg = "--stamp-scale"
        _help = "Sampled-stamp coordinate scale. [default: %default]"
        _mandatory = False
        _default = 1.0

    class FTolArg(NonNegativeFloatInvariant):
        _arg = "--f-tol"
        _help = "Optional relative residual tolerance."
        _mandatory = False
        _default = None

    class XTolArg(NonNegativeFloatInvariant):
        _arg = "--x-tol"
        _help = "Optional absolute step-norm tolerance."
        _mandatory = False
        _default = None

    class GTolArg(NonNegativeFloatInvariant):
        _arg = "--g-tol"
        _help = "Optional gradient tolerance."
        _mandatory = False
        _default = None

    class MaxEvaluationsArg(PositiveIntegerInvariant):
        _arg = "--max-evaluations"
        _help = "Optional maximum residual evaluations per candidate."
        _mandatory = False
        _default = None

    class UseFiniteDifferenceArg(BoolInvariant):
        _arg = "--use-finite-difference"
        _help = "Use finite differences instead of an analytic Jacobian."
        _mandatory = False
        _default = False

    def _model_specification(
        self, dataset: XFitDataset
    ) -> Literal["gaussian"] | StampDipoleModel:
        if self.model == "gaussian":
            return "gaussian"

        basis = np.asarray(dataset.stamp_basis)
        if basis.ndim == 3:
            basis = basis[0]
        return StampDipoleModel(
            basis,
            image_shape=tuple(dataset.images.shape[-2:]),
            evaluation=self.stamp_evaluation,
            scale=self.stamp_scale,
        )

    def run(self) -> None:
        from . import LMConfig, fit_dipoles

        dataset = self._load_dataset()
        fit_images = (
            dataset.images
            if self.compute_dtype == "input"
            else dataset.images.astype(self.compute_dtype, copy=False)
        )
        output_dir = Path(self.output_dir).expanduser().resolve()
        if output_dir.exists():
            raise CommandError(
                f"output directory already exists: {output_dir}"
            )
        resolved_finite_difference = (
            self.use_finite_difference or self.model == "stamp"
        )
        config = LMConfig(
            f_tol=self.f_tol,
            x_tol=self.x_tol,
            g_tol=self.g_tol,
            max_evaluations=self.max_evaluations,
            use_finite_difference=resolved_finite_difference,
        )
        result = self._call(
            fit_dipoles,
            fit_images,
            model=self._model_specification(dataset),
            initial=dataset.initial,
            mask=dataset.mask,
            variance=dataset.variance,
            mode=self.mode,
            backend=self.backend,
            config=config,
        )
        dtype = np.dtype(result.dtype)
        default_tolerance = float(np.sqrt(np.finfo(dtype).eps))
        parameter_count = len(result.parameter_names)
        effective_config = {
            "schema_version": 1,
            "command": "fit-dipoles",
            "input": str(dataset.path),
            "output_dir": str(output_dir.resolve()),
            "model": self.model,
            "mode": self.mode,
            "backend": self.backend,
            "compute_dtype": {
                "requested": self.compute_dtype,
                "resolved": result.dtype,
            },
            "stamp": (
                {
                    "evaluation": self.stamp_evaluation,
                    "scale": self.stamp_scale,
                    "basis": "input:stamp_basis",
                }
                if self.model == "stamp"
                else None
            ),
            "solver": {
                "f_tol": (
                    default_tolerance if self.f_tol is None else self.f_tol
                ),
                "x_tol": (
                    default_tolerance if self.x_tol is None else self.x_tol
                ),
                "g_tol": (
                    default_tolerance if self.g_tol is None else self.g_tol
                ),
                "max_evaluations": (
                    config.resolved_max_evaluations(parameter_count)
                ),
                "initial_damping": config.initial_damping,
                "damping_increase": config.damping_increase,
                "damping_decrease": config.damping_decrease,
                "finite_difference_step": (
                    default_tolerance
                    if config.finite_difference_step is None
                    else config.finite_difference_step
                ),
                "use_finite_difference": config.use_finite_difference,
            },
        }
        summary = self._call(
            write_fit_artifacts,
            output_dir,
            dataset=dataset,
            result=result,
            effective_config=effective_config,
        )
        self._emit_json(summary)


__all__ = [
    "DataInspectCommand",
    "DataValidateCommand",
    "FitDipolesCommand",
]
