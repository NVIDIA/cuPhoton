# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Invariant-backed commands for the XRay CLI surface."""

from __future__ import annotations

import json
import re
from dataclasses import asdict as dataclass_asdict
from pathlib import Path

from cuphoton.core.cli import (
    BoolInvariant,
    CommandError,
    FloatInvariant,
    IntegerInvariant,
    InvariantAwareCommand,
    NonNegativeIntegerInvariant,
    PairInvariant,
    PathValueInvariant,
    SequenceInvariant,
    SetInvariant,
    StringInvariant,
    VariablePositionalInvariant,
)

from .doctor import (
    collect_doctor_report,
    format_doctor_json,
    format_doctor_text,
)
from .gpu import gpu_first_policy

_KNOWN_COMMAND_EXCEPTIONS = (
    FileExistsError,
    FileNotFoundError,
    ImportError,
    NotImplementedError,
    OSError,
    RuntimeError,
    ValueError,
)


class _XRayCommand(InvariantAwareCommand):
    """Run one XRay domain handler with invariant-backed options."""

    _log_level_ = False
    _quiet_ = False
    _verbose_ = False
    _handler_name_: str

    def run(self) -> None:
        handler = globals()[self._handler_name_]
        try:
            returncode = int(handler(self) or 0)
        except _KNOWN_COMMAND_EXCEPTIONS as exc:
            raise CommandError(str(exc)) from exc
        if returncode:
            raise SystemExit(returncode)


class DoctorCommand(_XRayCommand):
    _description_ = "Check local GPU, site, and report dependencies."
    _shortname_ = None
    _handler_name_ = "_doctor"

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionProfileSummaryCommand(_XRayCommand):
    _description_ = "Summarize linear-prediction profile logs."
    _shortname_ = "lpps"
    _handler_name_ = "_linear_prediction_profile_summary"

    logs = None

    class LogsArg(VariablePositionalInvariant):
        _help = "One or more logs containing linearpred_profile output."
        _required = True
        _nargs = "+"
        _item_type = Path
        _metavar = "logs"

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False

    steady_state_skip_profile_lines = None

    class SteadyStateSkipProfileLinesArg(NonNegativeIntegerInvariant):
        _arg = "--steady-state-skip-profile-lines"
        _help = (
            "Also report profile counters after skipping the first N "
            "raw linearpred_profile rows from each log."
        )
        _required = False
        _default = None
        _metavar = "N"


class ReportCommand(_XRayCommand):
    _description_ = "Build an HTML report from analysis artifacts."
    _shortname_ = None
    _handler_name_ = "_report"

    input = None

    class InputArg(PathValueInvariant):
        _arg = "-i/--input"
        _help = "Run output directory containing analysis artifacts."
        _required = True

    output = None

    class OutputArg(PathValueInvariant):
        _arg = "-o/--output"
        _help = "Directory where report files should be written."
        _required = True

    run_number = None

    class RunNumberArg(IntegerInvariant):
        _arg = "-n/--run"
        _help = "Run number used in artifact filenames."
        _required = True
        _metavar = "RUN"


class ValidationVizCommand(_XRayCommand):
    _description_ = (
        "Build a Bokeh HTML dashboard for human output validation."
    )
    _shortname_ = "vv"
    _handler_name_ = "_validation_viz"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a multi-row review surface."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    profile_log = []

    class ProfileLogArg(SequenceInvariant):
        _arg = "--profile-log"
        _help = "Log with linearpred_profile output. Repeatable."
        _required = False
        _item_type = Path
        _default = []

    output = None

    class OutputArg(PathValueInvariant):
        _arg = "--output"
        _help = "Output standalone HTML path."
        _required = True

    title = "XRay Validation Review"

    class TitleArg(StringInvariant):
        _arg = "--title"
        _help = "Dashboard title."
        _required = False
        _default = "XRay Validation Review"

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = (
            "Linear-prediction component count for reconstruction overlays."
        )
        _required = False
        _default = 30

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _help = "CPU roots backend used for reconstruction overlays."
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    max_traces = 16

    class MaxTracesArg(IntegerInvariant):
        _arg = "--max-traces"
        _help = "Maximum number of trace NPZ files to include."
        _required = False
        _default = 16

    no_fit = False

    class NoFitArg(BoolInvariant):
        _arg = "--no-fit"
        _help = "Skip CPU fit overlays and render trace/profile views only."
        _required = False


class PhononVizCommand(_XRayCommand):
    _description_ = "Build a Bokeh phonon-dispersion review surface."
    _shortname_ = "pv"
    _handler_name_ = "_phonon_viz"

    workflow_bundle = None

    class WorkflowBundleArg(PathValueInvariant):
        _arg = "--workflow-bundle"
        _help = "Workflow-viz NPZ bundle to render as a dispersion view."
        _required = False

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a row-slice phonon proxy."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    detector_artifact_dir = None

    class DetectorArtifactDirArg(PathValueInvariant):
        _arg = "--detector-artifact-dir"
        _help = (
            "Directory containing freq_all.npy, amp_all.npy, "
            "fft_all.npy, and fft_freq_all.npy from a dumped "
            "detector-wide analysis run."
        )
        _required = False

    x_value = None

    class XValueArg(IntegerInvariant):
        _arg = "--x-value"
        _help = "Detector x column for detector artifact mode."
        _required = False

    y_start = None

    class YStartArg(IntegerInvariant):
        _arg = "--y-start"
        _help = "First detector row for detector artifact mode."
        _required = False

    y_end = None

    class YEndArg(IntegerInvariant):
        _arg = "--y-end"
        _help = "Exclusive last detector row for detector artifact mode."
        _required = False

    output = None

    class OutputArg(PathValueInvariant):
        _arg = "--output"
        _help = "Output standalone HTML path."
        _required = True

    title = "XRay Phonon Dispersion"

    class TitleArg(StringInvariant):
        _arg = "--title"
        _help = "Dashboard title."
        _required = False
        _default = "XRay Phonon Dispersion"

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Linear-prediction component count for trace proxy mode."
        _required = False
        _default = 30

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _help = "Root solver backend for trace proxy mode."
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    max_traces = 256

    class MaxTracesArg(IntegerInvariant):
        _arg = "--max-traces"
        _help = (
            "Maximum number of trace NPZ files to fit in trace proxy mode."
        )
        _required = False
        _default = 256

    max_points = 60000

    class MaxPointsArg(IntegerInvariant):
        _arg = "--max-points"
        _help = "Maximum detector mode points to render from dumped arrays."
        _required = False
        _default = 60000

    amp_threshold = None

    class AmpThresholdArg(FloatInvariant):
        _arg = "--amp-threshold"
        _help = (
            "Detector artifact amplitude threshold for "
            "filtered-frequency phonon dispersion. Uses amp_all > "
            "threshold and amp_all < 1e6."
        )
        _required = False


class WorkflowVizCommand(_XRayCommand):
    _description_ = "Build a linked XRay workflow visualization workbench."
    _shortname_ = "wv"
    _handler_name_ = "_workflow_viz"

    bundle = None

    class BundleArg(PathValueInvariant):
        _arg = "--bundle"
        _help = "Existing workflow-viz NPZ bundle to render."
        _required = False

    bundle_output = None

    class BundleOutputArg(PathValueInvariant):
        _arg = "--bundle-output"
        _help = "Optional workflow-viz NPZ bundle path to write."
        _required = False

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a row-slice workflow proxy."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    detector_artifact_dir = None

    class DetectorArtifactDirArg(PathValueInvariant):
        _arg = "--detector-artifact-dir"
        _help = (
            "Optional directory containing detector-wide freq_all.npy "
            "and amp_all.npy for trace proxy context."
        )
        _required = False

    phonon_detector_artifact_dir = None

    class PhononDetectorArtifactDirArg(PathValueInvariant):
        _arg = "--phonon-detector-artifact-dir"
        _help = (
            "Optional detector artifact directory for the filtered "
            "phonon panel. Defaults to --detector-artifact-dir."
        )
        _required = False

    x_value = None

    class XValueArg(IntegerInvariant):
        _arg = "--x-value"
        _help = "Detector x column for detector artifact context."
        _required = False

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = "Directory containing on/off HDF5 cubes."
        _required = False

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename."
        _required = False

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename."
        _required = False

    roi_lower = None

    class RoiLowerArg(PairInvariant):
        _arg = "--roi-lower"
        _help = "ROI origin pixel. Defaults to 0 0 for HDF5 input."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("X", "Y")

    roi_dim = None

    class RoiDimArg(PairInvariant):
        _arg = "--roi-dim"
        _help = "ROI dimensions. Defaults to the HDF5 image remainder."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _help = (
            "Detector row range to exclude, as start:end. Repeat or "
            "comma-separate to provide multiple ranges."
        )
        _required = False
        _default = []

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading HDF5 frames to drop."
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _help = "HDF5 frame chunk size for ROI reductions."
        _required = False
        _default = 16

    no_reference_shift = False

    class NoReferenceShiftArg(BoolInvariant):
        _arg = "--no-reference-shift"
        _help = "Disable the reference off-signal shift used for ratios."
        _required = False

    output = None

    class OutputArg(PathValueInvariant):
        _arg = "--output"
        _help = "Output standalone HTML path."
        _required = True

    title = "XRay Workflow Workbench"

    class TitleArg(StringInvariant):
        _arg = "--title"
        _help = "Dashboard title."
        _required = False
        _default = "XRay Workflow Workbench"

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Linear-prediction component count for fit overlays."
        _required = False
        _default = 30

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _help = "Root solver backend for fit overlays."
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    max_traces = 48

    class MaxTracesArg(IntegerInvariant):
        _arg = "--max-traces"
        _help = "Maximum row traces to include in the workbench."
        _required = False
        _default = 48

    max_points = 60000

    class MaxPointsArg(IntegerInvariant):
        _arg = "--max-points"
        _help = "Maximum detector artifact points to render."
        _required = False
        _default = 60000

    phonon_amp_threshold = None

    class PhononAmpThresholdArg(FloatInvariant):
        _arg = "--phonon-amp-threshold"
        _help = (
            "Amplitude threshold for embedding detector "
            "filtered-frequency phonon dispersion when "
            "--detector-artifact-dir is provided."
        )
        _required = False


class LinearPredictionSmokeCommand(_XRayCommand):
    _description_ = (
        "Compare CPU and GPU linear prediction on a synthetic trace."
    )
    _shortname_ = "lps"
    _handler_name_ = "_linear_prediction_smoke"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components to fit."
        _required = False
        _default = 8

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _help = "Root-solving backend for the companion polynomial."
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the GPU comparison."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionBenchmarkCommand(_XRayCommand):
    _description_ = "Benchmark serial and batched linear-prediction P1."
    _shortname_ = "lpb"
    _handler_name_ = "_linear_prediction_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of synthetic detector-row traces to benchmark."
        _required = False
        _default = 16

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components to fit."
        _required = False
        _default = 8

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _help = "Root-solving backend for serial P1 comparisons."
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy serial and batched comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionP2BenchmarkCommand(_XRayCommand):
    _description_ = "Benchmark fixed-shape linear-prediction P2 work."
    _shortname_ = "lppb"
    _handler_name_ = "_linear_prediction_p2_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of detector-row traces to benchmark."
        _required = False
        _default = 16

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select modes."
        _required = False
        _default = 8

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy serial and batched comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionSavgolBenchmarkCommand(_XRayCommand):
    _description_ = "Benchmark fixed Savitzky-Golay smoothing work."
    _shortname_ = "lpsb"
    _handler_name_ = "_linear_prediction_savgol_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of detector-row traces to benchmark."
        _required = False
        _default = 16

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order."
        _required = False
        _default = 3

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy serial and batched comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionFixedStagesBenchmarkCommand(_XRayCommand):
    _description_ = "Benchmark grouped Savitzky-Golay and P2 work."
    _shortname_ = "lpfsb"
    _handler_name_ = "_linear_prediction_fixed_stages_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of detector-row traces to benchmark."
        _required = False
        _default = 16

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select fixed P2 modes."
        _required = False
        _default = 8

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order."
        _required = False
        _default = 3

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy serial and batched comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionVariableP2BenchmarkCommand(_XRayCommand):
    _description_ = "Benchmark P2 batching with per-row fitted modes."
    _shortname_ = "lpvpb"
    _handler_name_ = "_linear_prediction_variable_p2_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of detector-row traces to benchmark."
        _required = False
        _default = 16

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select row modes."
        _required = False
        _default = 8

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length before fitting modes."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order before fitting modes."
        _required = False
        _default = 3

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    batched_solver = "pinv"

    class BatchedSolverArg(SetInvariant):
        _arg = "--batched-solver"
        _help = "Batched P2 solver prototype to run."
        _required = False
        _set = {"grouped-normal", "grouped-qr", "pinv", "grouped-pinv"}
        _default = "pinv"
        _metavar = "{pinv,grouped-pinv,grouped-normal,grouped-qr}"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a real-trace batch."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy serial and batched comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionVariableStagesBenchmarkCommand(_XRayCommand):
    _description_ = (
        "Benchmark batched Savitzky-Golay and P2 with per-row modes."
    )
    _shortname_ = "lpvsb"
    _handler_name_ = "_linear_prediction_variable_stages_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of detector-row traces to benchmark."
        _required = False
        _default = 16

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select row modes."
        _required = False
        _default = 8

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length before fitting modes."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order before fitting modes."
        _required = False
        _default = 3

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    batched_solver = "grouped-pinv"

    class BatchedSolverArg(SetInvariant):
        _arg = "--batched-solver"
        _help = "Batched P2 solver prototype to run after batched filtering."
        _required = False
        _set = {"grouped-normal", "grouped-qr", "pinv", "grouped-pinv"}
        _default = "grouped-pinv"
        _metavar = "{pinv,grouped-pinv,grouped-normal,grouped-qr}"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a real-trace batch."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy serial and batched comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionVariableArtifactsBenchmarkCommand(_XRayCommand):
    _description_ = "Benchmark full P2 artifact parity with per-row modes."
    _shortname_ = "lpvab"
    _handler_name_ = "_linear_prediction_variable_artifacts_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of detector-row traces to benchmark."
        _required = False
        _default = 16

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select row modes."
        _required = False
        _default = 8

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length before fitting modes."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order before fitting modes."
        _required = False
        _default = 3

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    batched_solver = "grouped-pinv"

    class BatchedSolverArg(SetInvariant):
        _arg = "--batched-solver"
        _help = "Batched P2 artifact solver prototype."
        _required = False
        _set = {"pinv", "grouped-pinv"}
        _default = "grouped-pinv"
        _metavar = "{pinv,grouped-pinv}"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a real-trace batch."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy serial and batched comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionRuntimeBridgeBenchmarkCommand(_XRayCommand):
    _description_ = (
        "Compare post-P1 serial, per-tile batched, and "
        "multi-tile grouped legacy-row work."
    )
    _shortname_ = "lprbb"
    _handler_name_ = "_linear_prediction_runtime_bridge_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 96

    tiles = 4

    class TilesArg(IntegerInvariant):
        _arg = "--tiles"
        _help = "Number of synthetic runtime tiles to benchmark."
        _required = False
        _default = 4

    rows_per_tile = 16

    class RowsPerTileArg(IntegerInvariant):
        _arg = "--rows-per-tile"
        _help = "Number of detector-row traces per runtime tile."
        _required = False
        _default = 16

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select row modes."
        _required = False
        _default = 8

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length before fitting modes."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order before fitting modes."
        _required = False
        _default = 3

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    batched_solver = "grouped-pinv"

    class BatchedSolverArg(SetInvariant):
        _arg = "--batched-solver"
        _help = "Batched P2 artifact solver prototype."
        _required = False
        _set = {"pinv", "grouped-pinv"}
        _default = "grouped-pinv"
        _metavar = "{pinv,grouped-pinv}"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a real-trace batch."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy runtime bridge comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionVariableArtifactsAcceptanceCommand(_XRayCommand):
    _description_ = (
        "Gate full P2 artifact parity for variable-mode "
        "solvers across trace NPZ files."
    )
    _shortname_ = "lpvaa"
    _handler_name_ = "_linear_prediction_variable_artifacts_acceptance"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a real-trace acceptance set."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select row modes."
        _required = False
        _default = 30

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length before fitting modes."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order before fitting modes."
        _required = False
        _default = 3

    batched_solver = None

    class BatchedSolverArg(SetInvariant):
        _arg = "--batched-solver"
        _help = (
            "Batched P2 artifact solver prototype. Repeat to compare solvers."
        )
        _required = False
        _set = {"pinv", "grouped-pinv"}
        _action = "append"
        _default = None
        _metavar = "{pinv,grouped-pinv}"

    max_filter_diff_ratio = 1e-09

    class MaxFilterDiffRatioArg(FloatInvariant):
        _arg = "--max-filter-diff-ratio"
        _help = "Maximum filter delta divided by trace range."
        _required = False
        _default = 1e-09

    max_time_component_diff_ratio = 1e-09

    class MaxTimeComponentDiffRatioArg(FloatInvariant):
        _arg = "--max-time-component-diff-ratio"
        _help = "Maximum time-component delta divided by trace range."
        _required = False
        _default = 1e-09

    max_reconstruction_diff_ratio = 1e-09

    class MaxReconstructionDiffRatioArg(FloatInvariant):
        _arg = "--max-reconstruction-diff-ratio"
        _help = "Maximum reconstruction delta divided by trace range."
        _required = False
        _default = 1e-09

    max_coefficient_diff = 1e-09

    class MaxCoefficientDiffArg(FloatInvariant):
        _arg = "--max-coefficient-diff"
        _help = "Maximum absolute P2 coefficient delta."
        _required = False
        _default = 1e-09

    max_amplitude_diff = 1e-09

    class MaxAmplitudeDiffArg(FloatInvariant):
        _arg = "--max-amplitude-diff"
        _help = "Maximum absolute amplitude delta."
        _required = False
        _default = 1e-09

    max_phase_diff = 1e-09

    class MaxPhaseDiffArg(FloatInvariant):
        _arg = "--max-phase-diff"
        _help = "Maximum absolute phase delta."
        _required = False
        _default = 1e-09

    max_frequency_center_diff = 1e-09

    class MaxFrequencyCenterDiffArg(FloatInvariant):
        _arg = "--max-frequency-center-diff"
        _help = "Maximum absolute frequency-center delta."
        _required = False
        _default = 1e-09

    max_spectrum_component_diff = 1e-08

    class MaxSpectrumComponentDiffArg(FloatInvariant):
        _arg = "--max-spectrum-component-diff"
        _help = "Maximum absolute spectral-component delta."
        _required = False
        _default = 1e-08

    max_spectrum_total_diff = 1e-08

    class MaxSpectrumTotalDiffArg(FloatInvariant):
        _arg = "--max-spectrum-total-diff"
        _help = "Maximum absolute spectral-total delta."
        _required = False
        _default = 1e-08

    max_chi2_diff = 1e-09

    class MaxChi2DiffArg(FloatInvariant):
        _arg = "--max-chi2-diff"
        _help = "Maximum absolute chi2 delta."
        _required = False
        _default = 1e-09

    min_gpu_speedup = 1.0

    class MinGpuSpeedupArg(FloatInvariant):
        _arg = "--min-gpu-speedup"
        _help = "Minimum speedup versus serial CuPy artifact work."
        _required = False
        _default = 1.0

    repeat = 1

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 1

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionVariableStagesAcceptanceCommand(_XRayCommand):
    _description_ = (
        "Gate combined Savitzky-Golay and variable-mode P2 "
        "solvers across trace NPZ files."
    )
    _shortname_ = "lpvsa"
    _handler_name_ = "_linear_prediction_variable_stages_acceptance"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a real-trace acceptance set."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select row modes."
        _required = False
        _default = 30

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length before fitting modes."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order before fitting modes."
        _required = False
        _default = 3

    batched_solver = None

    class BatchedSolverArg(SetInvariant):
        _arg = "--batched-solver"
        _help = (
            "Batched P2 solver prototype to run. Repeat to compare solvers."
        )
        _required = False
        _set = {"grouped-normal", "grouped-qr", "pinv", "grouped-pinv"}
        _action = "append"
        _default = None
        _metavar = "{pinv,grouped-pinv,grouped-normal,grouped-qr}"

    max_filter_diff_ratio = 1e-09

    class MaxFilterDiffRatioArg(FloatInvariant):
        _arg = "--max-filter-diff-ratio"
        _help = "Maximum filter delta divided by trace range."
        _required = False
        _default = 1e-09

    max_reconstruction_diff_ratio = 1e-09

    class MaxReconstructionDiffRatioArg(FloatInvariant):
        _arg = "--max-reconstruction-diff-ratio"
        _help = "Maximum reconstruction delta divided by trace range."
        _required = False
        _default = 1e-09

    min_gpu_speedup = 1.0

    class MinGpuSpeedupArg(FloatInvariant):
        _arg = "--min-gpu-speedup"
        _help = "Minimum speedup versus serial CuPy fixed-stage work."
        _required = False
        _default = 1.0

    repeat = 1

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 1

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class LinearPredictionVariableP2AcceptanceCommand(_XRayCommand):
    _description_ = "Gate variable-mode P2 solvers across trace NPZ files."
    _shortname_ = "lpvpa"
    _handler_name_ = "_linear_prediction_variable_p2_acceptance"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a real-trace acceptance set."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to select row modes."
        _required = False
        _default = 30

    window_length = 11

    class WindowLengthArg(IntegerInvariant):
        _arg = "--window-length"
        _help = "Savitzky-Golay odd window length before fitting modes."
        _required = False
        _default = 11

    polyorder = 3

    class PolyorderArg(IntegerInvariant):
        _arg = "--polyorder"
        _help = "Savitzky-Golay polynomial order before fitting modes."
        _required = False
        _default = 3

    batched_solver = None

    class BatchedSolverArg(SetInvariant):
        _arg = "--batched-solver"
        _help = (
            "Batched P2 solver prototype to run. Repeat to compare solvers."
        )
        _required = False
        _set = {"grouped-normal", "grouped-qr", "pinv", "grouped-pinv"}
        _action = "append"
        _default = None
        _metavar = "{pinv,grouped-pinv,grouped-normal,grouped-qr}"

    max_diff_ratio = 1e-09

    class MaxDiffRatioArg(FloatInvariant):
        _arg = "--max-diff-ratio"
        _help = "Maximum reconstruction delta divided by trace range."
        _required = False
        _default = 1e-09

    min_gpu_speedup = 1.0

    class MinGpuSpeedupArg(FloatInvariant):
        _arg = "--min-gpu-speedup"
        _help = "Minimum speedup versus serial CuPy P2."
        _required = False
        _default = 1.0

    repeat = 1

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 1

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip the CuPy comparisons."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class PredictionRootsBenchmarkCommand(_XRayCommand):
    _description_ = "Benchmark root-solving backends for linear prediction."
    _shortname_ = "prb"
    _handler_name_ = "_prediction_roots_benchmark"

    samples = 800

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = "Number of synthetic trace samples."
        _required = False
        _default = 800

    traces = 16

    class TracesArg(IntegerInvariant):
        _arg = "--traces"
        _help = "Number of coefficient vectors to benchmark."
        _required = False
        _default = 16

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Number of SVD components used to derive coefficients."
        _required = False
        _default = 30

    repeat = 3

    class RepeatArg(IntegerInvariant):
        _arg = "--repeat"
        _help = "Number of timing repeats; best time is reported."
        _required = False
        _default = 3

    backend = None

    class BackendArg(SetInvariant):
        _arg = "--backend"
        _help = "Backend to run. Repeat to choose multiple backends."
        _required = False
        _set = {
            "cupy-eigvals-batched",
            "cupy-eigvals-serial",
            "cupy-roots-serial",
            "numpy-eigvals",
            "numpy-roots",
        }
        _action = "append"
        _default = None
        _metavar = (
            "{numpy-eigvals,numpy-roots,cupy-eigvals-serial,"
            "cupy-eigvals-batched,cupy-roots-serial}"
        )

    no_gpu = False

    class NoGpuArg(BoolInvariant):
        _arg = "--no-gpu"
        _help = "Skip CuPy backends."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class ModelOrderSweepCommand(_XRayCommand):
    _description_ = "Sweep linear-prediction component counts for one trace."
    _shortname_ = "mos"
    _handler_name_ = "_model_order_sweep"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = (
            "Number of synthetic trace samples when HDF5 input is omitted."
        )
        _required = False
        _default = 96

    component_values = None

    class ComponentValuesArg(SequenceInvariant):
        _arg = "--component"
        _help = (
            "Requested component count. Repeat to provide an explicit "
            "sweep; otherwise "
            "--min-components/--max-components/--step are used."
        )
        _required = False
        _item_type = int
        _metavar = "COMPONENT"

    min_components = 2

    class MinComponentsArg(IntegerInvariant):
        _arg = "--min-components"
        _help = "Smallest requested component count for generated sweeps."
        _required = False
        _default = 2

    max_components = 16

    class MaxComponentsArg(IntegerInvariant):
        _arg = "--max-components"
        _help = "Largest requested component count for generated sweeps."
        _required = False
        _default = 16

    step = 2

    class StepArg(IntegerInvariant):
        _arg = "--step"
        _help = "Component-count step for generated sweeps."
        _required = False
        _default = 2

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _help = "Root-solving backend for the companion polynomial."
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    relative_tolerance = 0.01

    class RelativeToleranceArg(FloatInvariant):
        _arg = "--relative-tolerance"
        _help = (
            "Choose the smallest model with RMS residual within this "
            "relative tolerance of the best observed residual."
        )
        _required = False
        _default = 0.01

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat to "
            "sweep a batch of extracted real-data traces with a "
            "shared time axis."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = (
            "Directory containing the HDF5 inputs for real-data trace mode."
        )
        _required = False

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = False

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = False

    roi_lower = None

    class RoiLowerArg(PairInvariant):
        _arg = "--roi-lower"
        _help = "ROI lower bound for HDF5 trace mode."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("X", "Y")

    roi_dim = None

    class RoiDimArg(PairInvariant):
        _arg = "--roi-dim"
        _help = "ROI dimensions for HDF5 trace mode."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    row_y = None

    class RowYArg(IntegerInvariant):
        _arg = "--row-y"
        _help = "Absolute detector row to extract inside the ROI."
        _required = False

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _help = (
            "Detector row range to exclude, as start:end. Repeat or "
            "comma-separate to provide multiple ranges."
        )
        _required = False
        _default = []

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading delay bins to drop in HDF5 mode."
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _help = "Number of frames to read per HDF5 chunk in HDF5 mode."
        _required = False
        _default = 16

    no_reference_shift = False

    class NoReferenceShiftArg(BoolInvariant):
        _arg = "--no-reference-shift"
        _help = "Skip the positive offset used by the reference analysis."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class SubspaceBenchmarkCommand(_XRayCommand):
    _description_ = "Compare NumPy matrix-pencil and ESPRIT prototypes."
    _shortname_ = "sb"
    _handler_name_ = "_subspace_benchmark"

    samples = 96

    class SamplesArg(IntegerInvariant):
        _arg = "--samples"
        _help = (
            "Number of synthetic trace samples when HDF5 input is omitted."
        )
        _required = False
        _default = 96

    model_order = 5

    class ModelOrderArg(IntegerInvariant):
        _arg = "--model-order"
        _help = "Subspace model order used by matrix-pencil/ESPRIT."
        _required = False
        _default = 5

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Current LPSVD component count used as the baseline."
        _required = False
        _default = 8

    method = None

    class MethodArg(SetInvariant):
        _arg = "--method"
        _help = "Subspace method to run. Repeat to choose multiple methods."
        _required = False
        _set = {"matrix-pencil", "esprit"}
        _action = "append"
        _default = None
        _metavar = "{matrix-pencil,esprit}"

    pencil_rows = None

    class PencilRowsArg(IntegerInvariant):
        _arg = "--pencil-rows"
        _help = "Rows in the Hankel pencil; defaults to half the samples."
        _required = False

    svd_backend = None

    class SvdBackendArg(SetInvariant):
        _arg = "--svd-backend"
        _help = (
            "Subspace SVD backend to run. Repeat to compare full, "
            "randomized truncated, and partial Lanczos SVD."
        )
        _required = False
        _set = {"full", "randomized", "partial"}
        _action = "append"
        _default = None
        _metavar = "{full,randomized,partial}"

    randomized_oversamples = 8

    class RandomizedOversamplesArg(IntegerInvariant):
        _arg = "--randomized-oversamples"
        _help = "Oversampled columns for randomized truncated SVD."
        _required = False
        _default = 8

    power_iterations = 1

    class PowerIterationsArg(IntegerInvariant):
        _arg = "--power-iterations"
        _help = "Power iterations for randomized truncated SVD."
        _required = False
        _default = 1

    random_seed = 0

    class RandomSeedArg(IntegerInvariant):
        _arg = "--random-seed"
        _help = "Deterministic seed for randomized truncated SVD."
        _required = False
        _default = 0

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat to "
            "compare a batch of extracted real-data traces."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = (
            "Directory containing the HDF5 inputs for real-data trace mode."
        )
        _required = False

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = False

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = False

    roi_lower = None

    class RoiLowerArg(PairInvariant):
        _arg = "--roi-lower"
        _help = "ROI lower bound for HDF5 trace mode."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("X", "Y")

    roi_dim = None

    class RoiDimArg(PairInvariant):
        _arg = "--roi-dim"
        _help = "ROI dimensions for HDF5 trace mode."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    row_y = None

    class RowYArg(IntegerInvariant):
        _arg = "--row-y"
        _help = "Absolute detector row to extract inside the ROI."
        _required = False

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _help = (
            "Detector row range to exclude, as start:end. Repeat or "
            "comma-separate to provide multiple ranges."
        )
        _required = False
        _default = []

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading delay bins to drop in HDF5 mode."
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _help = "Number of frames to read per HDF5 chunk in HDF5 mode."
        _required = False
        _default = 16

    no_reference_shift = False

    class NoReferenceShiftArg(BoolInvariant):
        _arg = "--no-reference-shift"
        _help = "Skip the positive offset used by the reference analysis."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class SubspaceAcceptanceCommand(_XRayCommand):
    _description_ = (
        "Gate subspace prototypes across extracted trace NPZ files."
    )
    _shortname_ = "sa"
    _handler_name_ = "_subspace_acceptance"

    trace_npz = None

    class TraceNpzArg(SequenceInvariant):
        _arg = "--trace-npz"
        _help = (
            "NPZ file containing 1D time and trace arrays. Repeat for "
            "a multi-trace acceptance set."
        )
        _required = False
        _item_type = Path

    trace_dir = None

    class TraceDirArg(PathValueInvariant):
        _arg = "--trace-dir"
        _help = "Directory containing extracted trace NPZ files."
        _required = False

    model_order = 5

    class ModelOrderArg(IntegerInvariant):
        _arg = "--model-order"
        _help = "Subspace model order used by matrix-pencil/ESPRIT."
        _required = False
        _default = 5

    components = 8

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Current LPSVD component count used as the baseline."
        _required = False
        _default = 8

    method = None

    class MethodArg(SetInvariant):
        _arg = "--method"
        _help = "Subspace method to run. Repeat to choose multiple methods."
        _required = False
        _set = {"matrix-pencil", "esprit"}
        _action = "append"
        _default = None
        _metavar = "{matrix-pencil,esprit}"

    pencil_rows = None

    class PencilRowsArg(IntegerInvariant):
        _arg = "--pencil-rows"
        _help = "Rows in the Hankel pencil; defaults to half the samples."
        _required = False

    svd_backend = None

    class SvdBackendArg(SetInvariant):
        _arg = "--svd-backend"
        _help = (
            "Subspace SVD backend to run. Repeat to compare full, "
            "randomized truncated, and partial Lanczos SVD."
        )
        _required = False
        _set = {"full", "randomized", "partial"}
        _action = "append"
        _default = None
        _metavar = "{full,randomized,partial}"

    randomized_oversamples = 8

    class RandomizedOversamplesArg(IntegerInvariant):
        _arg = "--randomized-oversamples"
        _help = "Oversampled columns for randomized truncated SVD."
        _required = False
        _default = 8

    power_iterations = 1

    class PowerIterationsArg(IntegerInvariant):
        _arg = "--power-iterations"
        _help = "Power iterations for randomized truncated SVD."
        _required = False
        _default = 1

    random_seed = 0

    class RandomSeedArg(IntegerInvariant):
        _arg = "--random-seed"
        _help = "Deterministic seed for randomized truncated SVD."
        _required = False
        _default = 0

    max_rms_ratio = 1.05

    class MaxRmsRatioArg(FloatInvariant):
        _arg = "--max-rms-ratio"
        _help = (
            "Maximum method RMS residual divided by the current LPSVD "
            "baseline RMS residual."
        )
        _required = False
        _default = 1.05

    max_diff_ratio = 0.05

    class MaxDiffRatioArg(FloatInvariant):
        _arg = "--max-diff-ratio"
        _help = (
            "Maximum absolute reconstruction delta divided by the "
            "trace range."
        )
        _required = False
        _default = 0.05

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class DataProbeCommand(_XRayCommand):
    _description_ = (
        "Inspect an on/off HDF5 cube pair before launching analysis."
    )
    _shortname_ = "dp"
    _handler_name_ = "_data_probe"

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = "Directory containing the HDF5 inputs."
        _required = True

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = True

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = True

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class TraceSmokeCommand(_XRayCommand):
    _description_ = (
        "Reduce an on/off HDF5 pair and run the zero-offset check."
    )
    _shortname_ = "ts"
    _handler_name_ = "_trace_smoke"

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = "Directory containing the HDF5 inputs."
        _required = True

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = True

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = True

    zero_offset = 0.0

    class ZeroOffsetArg(FloatInvariant):
        _arg = "--zero-offset"
        _help = "Delay value used as the target zero offset."
        _required = False
        _default = 0.0

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading delay bins to drop."
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _help = "Number of frames to read per HDF5 chunk."
        _required = False
        _default = 16

    no_reference_shift = False

    class NoReferenceShiftArg(BoolInvariant):
        _arg = "--no-reference-shift"
        _help = "Skip the positive offset used by the reference analysis."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class ExtractTraceCommand(_XRayCommand):
    _description_ = "Write an HDF5 on/off ROI or detector-row trace as NPZ."
    _shortname_ = "et"
    _handler_name_ = "_extract_trace"

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = "Directory containing the HDF5 inputs."
        _required = True

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = True

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = True

    output = None

    class OutputArg(PathValueInvariant):
        _arg = "--output"
        _help = "Output NPZ path for a single extracted trace."
        _required = False

    output_dir = None

    class OutputDirArg(PathValueInvariant):
        _arg = "--output-dir"
        _help = "Output directory for repeated --row-y trace extraction."
        _required = False

    output_prefix = "trace"

    class OutputPrefixArg(StringInvariant):
        _arg = "--output-prefix"
        _help = "Filename prefix for --output-dir trace artifacts."
        _required = False
        _default = "trace"

    roi_lower = None

    class RoiLowerArg(PairInvariant):
        _arg = "--roi-lower"
        _help = "ROI lower bound for HDF5 trace mode."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("X", "Y")

    roi_dim = None

    class RoiDimArg(PairInvariant):
        _arg = "--roi-dim"
        _help = "ROI dimensions for HDF5 trace mode."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    row_y = None

    class RowYArg(SequenceInvariant):
        _arg = "--row-y"
        _help = (
            "Absolute detector row to extract inside the ROI. Repeat "
            "with --output-dir to write a trace batch."
        )
        _required = False
        _item_type = int

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _help = (
            "Detector row range to exclude, as start:end. Repeat or "
            "comma-separate to provide multiple ranges."
        )
        _required = False
        _default = []

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading delay bins to drop."
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _help = "Number of frames to read per HDF5 chunk."
        _required = False
        _default = 16

    no_reference_shift = False

    class NoReferenceShiftArg(BoolInvariant):
        _arg = "--no-reference-shift"
        _help = "Skip the positive offset used by the reference analysis."
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class DetectorArtifactsCommand(_XRayCommand):
    _description_ = (
        "Write CuPy detector artifact arrays from on/off HDF5 cubes."
    )
    _shortname_ = "da"
    _handler_name_ = "_detector_artifacts"

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = "Directory containing the HDF5 inputs."
        _required = True

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = True

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = True

    output_dir = None

    class OutputDirArg(PathValueInvariant):
        _arg = "--output-dir"
        _help = (
            "Directory where detector artifact .npy files will be written."
        )
        _required = True

    roi_lower = (0, 0)

    class RoiLowerArg(PairInvariant):
        _arg = "--roi-lower"
        _help = "ROI lower bound. Defaults to the detector origin."
        _required = False
        _action = None
        _item_type = int
        _default = (0, 0)
        _metavar = ("X", "Y")

    roi_dim = None

    class RoiDimArg(PairInvariant):
        _arg = "--roi-dim"
        _help = (
            "ROI dimensions. Omit for the image remainder; 0 means remainder."
        )
        _required = False
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    tile_shape = (16, 16)

    class TileShapeArg(PairInvariant):
        _arg = "--tile-shape"
        _help = "Detector tile width and height for row-wise fitting."
        _required = False
        _action = None
        _item_type = int
        _default = (16, 16)
        _metavar = ("WIDTH", "HEIGHT")

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _help = (
            "Detector row range to exclude, as start:end. Repeat or "
            "comma-separate to provide multiple ranges."
        )
        _required = False
        _default = []

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading HDF5 frames to drop."
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _help = "HDF5 frame chunk size for full-detector reductions."
        _required = False
        _default = 16

    zero_offset = 0.0

    class ZeroOffsetArg(FloatInvariant):
        _arg = "--zero-offset"
        _help = "Delay value used to find the reference fit start index."
        _required = False
        _default = 0.0

    zero_offset_index = None

    class ZeroOffsetIndexArg(IntegerInvariant):
        _arg = "--zero-offset-index"
        _help = "Manual fit start index, bypassing zero-offset detection."
        _required = False

    fit_trailing_drop = 1

    class FitTrailingDropArg(IntegerInvariant):
        _arg = "--fit-trailing-drop"
        _help = "Number of trailing samples to drop from the LPF fit."
        _required = False
        _default = 1

    integrate = 3

    class IntegrateArg(IntegerInvariant):
        _arg = "--integrate"
        _help = "Reference row-integration pixel radius."
        _required = False
        _default = 3

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _help = "Linear-prediction component count."
        _required = False
        _default = 30

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _help = "CuPy linear-prediction roots backend."
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    savgol_window = 5

    class SavgolWindowArg(IntegerInvariant):
        _arg = "--savgol-window"
        _help = "Savitzky-Golay smoothing window used before fitting."
        _required = False
        _default = 5

    savgol_polyorder = 3

    class SavgolPolyorderArg(IntegerInvariant):
        _arg = "--savgol-polyorder"
        _help = "Savitzky-Golay polynomial order used before fitting."
        _required = False
        _default = 3

    amp_threshold = 1.6

    class AmpThresholdArg(FloatInvariant):
        _arg = "--amp-threshold"
        _help = "Amplitude threshold for amp_all_sum_filtered.npy."
        _required = False
        _default = 1.6

    max_fit_failures = 0

    class MaxFitFailuresArg(IntegerInvariant):
        _arg = "--max-fit-failures"
        _help = "Maximum row-fit failures allowed before the command fails."
        _required = False
        _default = 0

    hdf5_reader = "h5py"

    class Hdf5ReaderArg(SetInvariant):
        _arg = "--hdf5-reader"
        _help = (
            "HDF5 block reader mode. hdf5-ts-funcwrap is the beta "
            "multithreaded HDF5 target when the runtime is built from "
            "qkoziol/hdf5 ts_funcwrap_1."
        )
        _required = False
        _set = {"hdf5-ts-funcwrap", "h5py", "h5py-threaded"}
        _default = "h5py"
        _metavar = "{h5py,h5py-threaded,hdf5-ts-funcwrap}"

    hdf5_reader_workers = 2

    class Hdf5ReaderWorkersArg(IntegerInvariant):
        _arg = "--hdf5-reader-workers"
        _help = "Worker count for threaded HDF5 reader modes."
        _required = False
        _default = 2

    max_tiles = None

    class MaxTilesArg(IntegerInvariant):
        _arg = "--max-tiles"
        _help = "Diagnostic limit on the number of ROI tiles to process."
        _required = False

    normalization_cache = None

    class NormalizationCacheArg(PathValueInvariant):
        _arg = "--normalization-cache"
        _help = (
            "Directory or normalization.json/normalization.npz file "
            "from detector-artifact-normalize. Skips the "
            "full-detector normalization scan."
        )
        _required = False

    shard_index = None

    class ShardIndexArg(IntegerInvariant):
        _arg = "--shard-index"
        _help = "Distributed worker shard index for manifest provenance."
        _required = False

    shard_count = None

    class ShardCountArg(IntegerInvariant):
        _arg = "--shard-count"
        _help = "Distributed worker shard count for manifest provenance."
        _required = False

    global_roi_lower = None

    class GlobalRoiLowerArg(PairInvariant):
        _arg = "--global-roi-lower"
        _help = "Global distributed-run ROI lower bound."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("X", "Y")

    global_roi_dim = None

    class GlobalRoiDimArg(PairInvariant):
        _arg = "--global-roi-dim"
        _help = "Global distributed-run ROI dimensions."
        _required = False
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class DetectorArtifactNormalizeCommand(_XRayCommand):
    _description_ = (
        "Cache full-detector normalization for sharded artifact runs."
    )
    _shortname_ = "dan"
    _handler_name_ = "_detector_artifact_normalize"

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = "Directory containing the HDF5 inputs."
        _required = True

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = True

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = True

    output_dir = None

    class OutputDirArg(PathValueInvariant):
        _arg = "--output-dir"
        _help = "Directory where normalization.json/.npz will be written."
        _required = True

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading HDF5 frames to drop."
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _help = "HDF5 frame chunk size for full-detector reductions."
        _required = False
        _default = 16

    zero_offset = 0.0

    class ZeroOffsetArg(FloatInvariant):
        _arg = "--zero-offset"
        _help = "Delay value used to find the reference fit start index."
        _required = False
        _default = 0.0

    zero_offset_index = None

    class ZeroOffsetIndexArg(IntegerInvariant):
        _arg = "--zero-offset-index"
        _help = "Manual fit start index, bypassing zero-offset detection."
        _required = False

    hdf5_reader = "h5py"

    class Hdf5ReaderArg(SetInvariant):
        _arg = "--hdf5-reader"
        _help = "HDF5 block reader mode."
        _required = False
        _set = {"hdf5-ts-funcwrap", "h5py", "h5py-threaded"}
        _default = "h5py"
        _metavar = "{h5py,h5py-threaded,hdf5-ts-funcwrap}"

    hdf5_reader_workers = 2

    class Hdf5ReaderWorkersArg(IntegerInvariant):
        _arg = "--hdf5-reader-workers"
        _help = "Worker count for threaded HDF5 reader modes."
        _required = False
        _default = 2

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class DetectorArtifactDistributedCommand(_XRayCommand):
    _description_ = "Plan or launch x-sharded detector artifact runs."
    _shortname_ = "dad"
    _handler_name_ = "_detector_artifact_distributed"

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _required = True

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _required = True

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _required = True

    output_dir = None

    class OutputDirArg(PathValueInvariant):
        _arg = "--output-dir"
        _required = True

    roi_lower = (0, 0)

    class RoiLowerArg(PairInvariant):
        _arg = "--roi-lower"
        _required = False
        _action = None
        _item_type = int
        _default = (0, 0)
        _metavar = ("X", "Y")

    roi_dim = None

    class RoiDimArg(PairInvariant):
        _arg = "--roi-dim"
        _required = False
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    tile_shape = (16, 16)

    class TileShapeArg(PairInvariant):
        _arg = "--tile-shape"
        _required = False
        _action = None
        _item_type = int
        _default = (16, 16)
        _metavar = ("WIDTH", "HEIGHT")

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _required = False
        _default = []

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _required = False
        _default = 1

    chunk_frames = 16

    class ChunkFramesArg(IntegerInvariant):
        _arg = "--chunk-frames"
        _required = False
        _default = 16

    zero_offset = 0.0

    class ZeroOffsetArg(FloatInvariant):
        _arg = "--zero-offset"
        _required = False
        _default = 0.0

    zero_offset_index = None

    class ZeroOffsetIndexArg(IntegerInvariant):
        _arg = "--zero-offset-index"
        _required = False

    fit_trailing_drop = 1

    class FitTrailingDropArg(IntegerInvariant):
        _arg = "--fit-trailing-drop"
        _required = False
        _default = 1

    integrate = 3

    class IntegrateArg(IntegerInvariant):
        _arg = "--integrate"
        _required = False
        _default = 3

    components = 30

    class ComponentsArg(IntegerInvariant):
        _arg = "--components"
        _required = False
        _default = 30

    roots_backend = "eigvals"

    class RootsBackendArg(SetInvariant):
        _arg = "--roots-backend"
        _required = False
        _set = {"eigvals", "roots"}
        _default = "eigvals"
        _metavar = "{eigvals,roots}"

    savgol_window = 5

    class SavgolWindowArg(IntegerInvariant):
        _arg = "--savgol-window"
        _required = False
        _default = 5

    savgol_polyorder = 3

    class SavgolPolyorderArg(IntegerInvariant):
        _arg = "--savgol-polyorder"
        _required = False
        _default = 3

    amp_threshold = 1.6

    class AmpThresholdArg(FloatInvariant):
        _arg = "--amp-threshold"
        _required = False
        _default = 1.6

    max_fit_failures = 0

    class MaxFitFailuresArg(IntegerInvariant):
        _arg = "--max-fit-failures"
        _required = False
        _default = 0

    hdf5_reader = "h5py"

    class Hdf5ReaderArg(SetInvariant):
        _arg = "--hdf5-reader"
        _required = False
        _set = {"hdf5-ts-funcwrap", "h5py", "h5py-threaded"}
        _default = "h5py"
        _metavar = "{h5py,h5py-threaded,hdf5-ts-funcwrap}"

    hdf5_reader_workers = 2

    class Hdf5ReaderWorkersArg(IntegerInvariant):
        _arg = "--hdf5-reader-workers"
        _required = False
        _default = 2

    max_tiles = None

    class MaxTilesArg(IntegerInvariant):
        _arg = "--max-tiles"
        _required = False

    normalization_cache = None

    class NormalizationCacheArg(PathValueInvariant):
        _arg = "--normalization-cache"
        _required = False

    executor = "dry-run"

    class ExecutorArg(SetInvariant):
        _arg = "--executor"
        _required = False
        _set = {"local", "slurm", "dry-run"}
        _default = "dry-run"
        _metavar = "{dry-run,local,slurm}"

    shard_count = None

    class ShardCountArg(IntegerInvariant):
        _arg = "--shard-count"
        _required = False

    shard_width = None

    class ShardWidthArg(IntegerInvariant):
        _arg = "--shard-width"
        _required = False

    gpus = 1

    class GpusArg(IntegerInvariant):
        _arg = "--gpus"
        _required = False
        _default = 1

    gpus_per_node = None

    class GpusPerNodeArg(IntegerInvariant):
        _arg = "--gpus-per-node"
        _required = False

    nodes = None

    class NodesArg(IntegerInvariant):
        _arg = "--nodes"
        _required = False

    run_label = None

    class RunLabelArg(StringInvariant):
        _arg = "--run-label"
        _required = False

    submit = False

    class SubmitArg(BoolInvariant):
        _arg = "--submit"
        _required = False

    merge = False

    class MergeArg(BoolInvariant):
        _arg = "--merge"
        _required = False

    resume = False

    class ResumeArg(BoolInvariant):
        _arg = "--resume"
        _required = False

    slurm_partition = None

    class SlurmPartitionArg(StringInvariant):
        _arg = "--slurm-partition"
        _required = False

    slurm_account = None

    class SlurmAccountArg(StringInvariant):
        _arg = "--slurm-account"
        _required = False

    slurm_time = None

    class SlurmTimeArg(StringInvariant):
        _arg = "--slurm-time"
        _required = False

    slurm_constraint = None

    class SlurmConstraintArg(StringInvariant):
        _arg = "--slurm-constraint"
        _required = False

    slurm_qos = None

    class SlurmQosArg(StringInvariant):
        _arg = "--slurm-qos"
        _required = False

    slurm_gres = None

    class SlurmGresArg(StringInvariant):
        _arg = "--slurm-gres"
        _required = False

    slurm_cpus_per_task = None

    class SlurmCpusPerTaskArg(StringInvariant):
        _arg = "--slurm-cpus-per-task"
        _required = False

    slurm_job_name = None

    class SlurmJobNameArg(StringInvariant):
        _arg = "--slurm-job-name"
        _required = False

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _required = False


class DetectorArtifactMergeCommand(_XRayCommand):
    _description_ = "Merge x-sharded detector artifact directories."
    _shortname_ = "dam"
    _handler_name_ = "_detector_artifact_merge"

    shards_manifest = None

    class ShardsManifestArg(PathValueInvariant):
        _arg = "--shards-manifest"
        _help = "Distributed plan JSON containing shard output directories."
        _required = False

    shard_dir = []

    class ShardDirArg(SequenceInvariant):
        _arg = "--shard-dir"
        _help = "Shard artifact directory. Repeat for all shards."
        _required = False
        _item_type = Path
        _default = []

    output_dir = None

    class OutputDirArg(PathValueInvariant):
        _arg = "--output-dir"
        _help = "Merged detector artifact output directory."
        _required = True

    no_strict = False

    class NoStrictArg(BoolInvariant):
        _arg = "--no-strict"
        _help = (
            "Do not require the manifest shard count to match input count."
        )
        _required = False

    chunk_rows = 16

    class ChunkRowsArg(IntegerInvariant):
        _arg = "--chunk-rows"
        _required = False
        _default = 16

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _required = False


class DetectorArtifactCompareCommand(_XRayCommand):
    _description_ = (
        "Compare detector artifact arrays against a reference dump."
    )
    _shortname_ = "dac"
    _handler_name_ = "_detector_artifact_compare"

    reference_dir = None

    class ReferenceDirArg(PathValueInvariant):
        _arg = "--reference-dir"
        _help = (
            "Reference artifact directory containing detector .npy arrays."
        )
        _required = True

    candidate_dir = None

    class CandidateDirArg(PathValueInvariant):
        _arg = "--candidate-dir"
        _help = (
            "Candidate artifact directory containing detector .npy arrays."
        )
        _required = True

    candidate_origin = None

    class CandidateOriginArg(PairInvariant):
        _arg = "--candidate-origin"
        _help = (
            "Candidate ROI origin in the reference detector. Defaults "
            "to candidate manifest roi_lower or 0 0."
        )
        _required = False
        _action = None
        _item_type = int
        _metavar = ("X", "Y")

    amp_threshold = 1.6

    class AmpThresholdArg(FloatInvariant):
        _arg = "--amp-threshold"
        _help = "Amplitude threshold for filtered-mode agreement checks."
        _required = False
        _default = 1.6

    chunk_rows = 16

    class ChunkRowsArg(IntegerInvariant):
        _arg = "--chunk-rows"
        _help = "Rows per memory-mapped comparison block."
        _required = False
        _default = 16

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class RoiCandidatesCommand(_XRayCommand):
    _description_ = (
        "Score ROI tiles from an on/off HDF5 pair before GPU analysis."
    )
    _shortname_ = "rc"
    _handler_name_ = "_roi_candidates"

    h5dir = None

    class H5dirArg(PathValueInvariant):
        _arg = "--h5dir"
        _help = "Directory containing the HDF5 inputs."
        _required = True

    fon = None

    class FonArg(StringInvariant):
        _arg = "--fon"
        _help = "Laser-on HDF5 filename or absolute path."
        _required = True

    foff = None

    class FoffArg(StringInvariant):
        _arg = "--foff"
        _help = "Laser-off HDF5 filename or absolute path."
        _required = True

    tile_width = 16

    class TileWidthArg(IntegerInvariant):
        _arg = "--tile-width"
        _help = "ROI tile width in detector pixels."
        _required = False
        _default = 16

    tile_height = 16

    class TileHeightArg(IntegerInvariant):
        _arg = "--tile-height"
        _help = "ROI tile height in detector pixels."
        _required = False
        _default = 16

    stride_x = 128

    class StrideXArg(IntegerInvariant):
        _arg = "--stride-x"
        _help = "Horizontal scan stride in detector pixels."
        _required = False
        _default = 128

    stride_y = 128

    class StrideYArg(IntegerInvariant):
        _arg = "--stride-y"
        _help = "Vertical scan stride in detector pixels."
        _required = False
        _default = 128

    drop_leading = 1

    class DropLeadingArg(IntegerInvariant):
        _arg = "--drop-leading"
        _help = "Number of leading delay bins to drop."
        _required = False
        _default = 1

    limit = 10

    class LimitArg(IntegerInvariant):
        _arg = "--limit"
        _help = "Number of candidates to print."
        _required = False
        _default = 10

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _help = (
            "Detector row range to exclude, as start:end. Repeat or "
            "comma-separate to provide multiple ranges."
        )
        _required = False
        _default = []

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class DetectorMaskCommand(_XRayCommand):
    _description_ = "Summarize detector row exclusions for a tiled ROI."
    _shortname_ = "dm"
    _handler_name_ = "_detector_mask"

    roi_lower = None

    class RoiLowerArg(PairInvariant):
        _arg = "--roi-lower"
        _help = "ROI lower bound in detector pixels."
        _required = True
        _action = None
        _item_type = int
        _metavar = ("X", "Y")

    roi_dim = None

    class RoiDimArg(PairInvariant):
        _arg = "--roi-dim"
        _help = "ROI dimensions in detector pixels."
        _required = True
        _action = None
        _item_type = int
        _metavar = ("WIDTH", "HEIGHT")

    tile_width = 16

    class TileWidthArg(IntegerInvariant):
        _arg = "--tile-width"
        _help = "ROI tile width in detector pixels."
        _required = False
        _default = 16

    tile_height = 16

    class TileHeightArg(IntegerInvariant):
        _arg = "--tile-height"
        _help = "ROI tile height in detector pixels."
        _required = False
        _default = 16

    exclude_y = []

    class ExcludeYArg(SequenceInvariant):
        _arg = "--exclude-y"
        _help = (
            "Detector row range to exclude, as start:end. Repeat or "
            "comma-separate to provide multiple ranges."
        )
        _required = False
        _default = []

    json = False

    class JsonArg(BoolInvariant):
        _arg = "--json"
        _help = "Emit machine-readable JSON."
        _required = False


class GpuPolicyCommand(_XRayCommand):
    _description_ = "Print the current GPU-first policy."
    _shortname_ = "gp"
    _handler_name_ = "_gpu_policy"


def _doctor(args):
    report = collect_doctor_report()
    if args.json:
        print(format_doctor_json(report))
    else:
        print(format_doctor_text(report))
    return 0


def _linear_prediction_profile_summary(args):
    from .profile_summary import (
        summarize_linear_prediction_profile_files,
        summarize_linear_prediction_steady_state_profile_files,
    )

    summary = summarize_linear_prediction_profile_files(args.logs)
    payload = dataclass_asdict(summary)
    payload["sources"] = [str(path) for path in args.logs]
    if args.steady_state_skip_profile_lines is not None:
        steady_state = summarize_linear_prediction_steady_state_profile_files(
            args.logs,
            skip_profile_lines=args.steady_state_skip_profile_lines,
        )
        payload["steady_state"] = dataclass_asdict(steady_state)
        payload["steady_state"]["skip_profile_lines_per_source"] = (
            args.steady_state_skip_profile_lines
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"sources={len(args.logs)}")
        print(f"profile_lines={payload['profile_lines']}")
        print(f"fit_lines={payload['fit_lines']}")
        print(f"roi_task_lines={payload['roi_task_lines']}")
        print(
            f"processing_tiles={_optional_value(payload['processing_tiles'])}"
        )
        print(f"raw_fits={payload['raw_fits']}")
        print(f"failures={payload['failures']}")
        print(f"skipped_fits={payload['skipped_fits']}")
        print(f"filtered_fits={payload['filtered_fits']}")
        print(
            "tile_processing_seconds="
            f"{_optional_value(payload['tile_processing_seconds'])}"
        )
        print(
            "command_real_seconds="
            f"{_optional_value(payload['command_real_seconds'])}"
        )
        if payload["amp_thresholds"]:
            thresholds = ",".join(
                f"{threshold:g}" for threshold in payload["amp_thresholds"]
            )
            print(f"amp_thresholds={thresholds}")
        for stage in payload["stages"]:
            print(
                f"{stage['stage']}_seconds={stage['seconds']:.6f} "
                f"count={stage['count']} mean={stage['mean_s']:.6f}"
            )
        if "steady_state" in payload:
            steady_state = payload["steady_state"]
            print(
                "steady_state_skip_profile_lines_per_source="
                f"{steady_state['skip_profile_lines_per_source']}"
            )
            print(
                f"steady_state_profile_lines={steady_state['profile_lines']}"
            )
            print(
                "steady_state_skipped_profile_lines="
                f"{steady_state['skipped_profile_lines']}"
            )
            print(
                "steady_state_included_profile_lines="
                f"{steady_state['included_profile_lines']}"
            )
            for stage in steady_state["stages"]:
                print(
                    f"steady_state_{stage['stage']}_seconds="
                    f"{stage['seconds']:.6f} count={stage['count']} "
                    f"mean={stage['mean_s']:.6f}"
                )
    return 0


def _optional_value(value):
    return "-" if value is None else f"{value:g}"


def _report(args):
    from .report import build_report

    result = build_report(args.input, args.output, args.run_number)
    print(f"report={result.report_html}")
    print(f"copied_plots={len(result.copied_plots)}")
    if result.missing_optional_plots:
        print(
            "missing_optional_plots="
            + ",".join(result.missing_optional_plots)
        )
    return 0


def _validation_viz(args):
    from .validation_viz import build_validation_viz

    result = build_validation_viz(
        output=args.output,
        trace_paths=tuple(args.trace_npz or ()),
        trace_dir=args.trace_dir,
        profile_logs=tuple(args.profile_log or ()),
        title=args.title,
        components=args.components,
        roots_backend=args.roots_backend,
        max_traces=args.max_traces,
        fit=not args.no_fit,
    )
    print(f"validation_viz={result.html_path}")
    print(f"traces={result.trace_count}")
    print(f"fits={result.fit_count}")
    print(f"profile_logs={result.profile_log_count}")
    return 0


def _phonon_viz(args):
    from .phonon_viz import build_phonon_viz

    result = build_phonon_viz(
        output=args.output,
        workflow_bundle=args.workflow_bundle,
        trace_paths=tuple(args.trace_npz or ()),
        trace_dir=args.trace_dir,
        detector_artifact_dir=args.detector_artifact_dir,
        x_value=args.x_value,
        y_start=args.y_start,
        y_end=args.y_end,
        title=args.title,
        components=args.components,
        roots_backend=args.roots_backend,
        max_traces=args.max_traces,
        max_points=args.max_points,
        amp_threshold=args.amp_threshold,
    )
    print(f"phonon_viz={result.html_path}")
    print(f"source={result.source_kind}")
    print(f"rows={result.trace_count}")
    print(f"modes={result.mode_count}")
    return 0


def _workflow_viz(args):
    from .detector_mask import parse_y_ranges
    from .workflow_viz import build_workflow_viz

    result = build_workflow_viz(
        output=args.output,
        bundle=args.bundle,
        bundle_output=args.bundle_output,
        trace_paths=tuple(args.trace_npz or ()),
        trace_dir=args.trace_dir,
        h5dir=args.h5dir,
        fon=args.fon,
        foff=args.foff,
        roi_lower=tuple(args.roi_lower) if args.roi_lower else None,
        roi_dim=tuple(args.roi_dim) if args.roi_dim else None,
        detector_artifact_dir=args.detector_artifact_dir,
        phonon_detector_artifact_dir=args.phonon_detector_artifact_dir,
        x_value=args.x_value,
        exclude_y=parse_y_ranges(args.exclude_y),
        drop_leading=args.drop_leading,
        chunk_frames=args.chunk_frames,
        reference_shift=not args.no_reference_shift,
        title=args.title,
        components=args.components,
        roots_backend=args.roots_backend,
        max_traces=args.max_traces,
        max_points=args.max_points,
        phonon_amp_threshold=args.phonon_amp_threshold,
    )
    print(f"workflow_viz={result.html_path}")
    if result.bundle_path is not None:
        print(f"bundle={result.bundle_path}")
    print(f"source={result.source_kind}")
    print(f"rows={result.row_count}")
    print(f"modes={result.mode_count}")
    return 0


def _data_probe(args):
    from .hdf5 import probe_hdf5_pair

    result = probe_hdf5_pair(h5dir=args.h5dir, fon=args.fon, foff=args.foff)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_file_probe("on", result.on)
        _print_file_probe("off", result.off)
    return 0


def _trace_smoke(args):
    from .hdf5 import load_hdf5_pair_trace
    from .zero_offset import find_value_drop_position

    trace = load_hdf5_pair_trace(
        h5dir=args.h5dir,
        fon=args.fon,
        foff=args.foff,
        drop_leading=args.drop_leading,
        chunk_frames=args.chunk_frames,
        reference_shift=not args.no_reference_shift,
    )
    drop = find_value_drop_position(
        trace.delay,
        trace.ratio_minus_one,
        zero_offset=args.zero_offset,
    )
    payload = {
        "trace": trace.summary(),
        "zero_offset": dataclass_asdict(drop),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload["trace"]
        print(f"samples={summary['samples']}")
        print(f"schema={summary['on']['schema']}")
        print(
            "normalization="
            f"{summary['on']['normalization_dataset']}/"
            f"{summary['off']['normalization_dataset']}"
        )
        print(f"delay_range={summary['delay_min']}:{summary['delay_max']}")
        print(
            "ratio_range="
            f"{summary['ratio_min']}:{summary['ratio_mean']}:"
            f"{summary['ratio_max']}"
        )
        print(f"zero_offset_status={drop.status}")
        print(f"zero_offset_index={drop.selected_index}")
        if drop.selected_time is not None:
            print(f"zero_offset_time={drop.selected_time}")
    return 0


def _extract_trace(args):
    row_values = tuple(args.row_y or ())
    if args.output is not None and args.output_dir is not None:
        raise ValueError("provide only one of --output or --output-dir")
    if args.output is None and args.output_dir is None:
        raise ValueError("provide --output or --output-dir")
    if args.output_dir is not None:
        if not row_values:
            raise ValueError("--output-dir requires at least one --row-y")
        if len(set(row_values)) != len(row_values):
            raise ValueError("--output-dir requires distinct --row-y values")
        output_prefix = Path(args.output_prefix)
        if output_prefix.name != args.output_prefix:
            raise ValueError("--output-prefix must be a filename prefix")
        summaries = []
        for batch_index, row_y in enumerate(row_values):
            output_name = f"{args.output_prefix}-y{row_y}.npz"
            output = Path(args.output_dir) / output_name
            summaries.append(
                _extract_trace_to_npz(
                    args,
                    output,
                    row_y=row_y,
                    batch_index=batch_index,
                    batch_count=len(row_values),
                )
            )
        payload = {
            "kind": "trace-npz-batch",
            "trace_count": len(summaries),
            "artifacts": tuple(summary["artifact"] for summary in summaries),
            "traces": tuple(summaries),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"trace_count={payload['trace_count']}")
            for artifact in payload["artifacts"]:
                print(f"artifact={artifact}")
        return 0

    if len(row_values) > 1:
        raise ValueError("multiple --row-y values require --output-dir")
    row_y = row_values[0] if row_values else None
    summary = _extract_trace_to_npz(args, Path(args.output), row_y=row_y)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"artifact={summary['artifact']}")
        print(f"samples={summary['samples']}")
        print(f"source={summary['kind']}")
        if "roi_lower" in summary:
            print(f"roi_lower={summary['roi_lower']}")
            print(f"roi_dim={summary['roi_dim']}")
            print(f"row_y={summary['row_y']}")
        print(f"delay_range={summary['delay_min']}:{summary['delay_max']}")
        print(
            "ratio_range="
            f"{summary['ratio_min']}:{summary['ratio_mean']}:"
            f"{summary['ratio_max']}"
        )
    return 0


def _extract_trace_to_npz(
    args,
    output: Path,
    *,
    row_y: int | None,
    batch_index: int | None = None,
    batch_count: int | None = None,
):
    import numpy as np

    from .detector_mask import format_y_ranges, parse_y_ranges
    from .hdf5 import load_hdf5_pair_roi_trace, load_hdf5_pair_trace

    if output.suffix != ".npz":
        raise ValueError("--output must end with .npz")

    exclude_y = parse_y_ranges(args.exclude_y)
    has_roi = args.roi_lower is not None or args.roi_dim is not None
    has_row_selection = row_y is not None or bool(exclude_y)
    if has_row_selection and not has_roi:
        raise ValueError(
            "--roi-lower and --roi-dim are required with row/mask selection"
        )
    if has_roi:
        if args.roi_lower is None or args.roi_dim is None:
            raise ValueError(
                "--roi-lower and --roi-dim must be provided together"
            )
        trace = load_hdf5_pair_roi_trace(
            h5dir=args.h5dir,
            fon=args.fon,
            foff=args.foff,
            roi_x=args.roi_lower[0],
            roi_y=args.roi_lower[1],
            roi_width=args.roi_dim[0],
            roi_height=args.roi_dim[1],
            row_y=row_y,
            exclude_y=exclude_y,
            drop_leading=args.drop_leading,
            chunk_frames=args.chunk_frames,
            reference_shift=not args.no_reference_shift,
        )
        source = {
            "kind": "hdf5-roi",
            "roi_lower": list(args.roi_lower),
            "roi_dim": list(args.roi_dim),
            "row_y": row_y,
            "exclude_y": format_y_ranges(exclude_y),
        }
    else:
        trace = load_hdf5_pair_trace(
            h5dir=args.h5dir,
            fon=args.fon,
            foff=args.foff,
            drop_leading=args.drop_leading,
            chunk_frames=args.chunk_frames,
            reference_shift=not args.no_reference_shift,
        )
        source = {"kind": "hdf5-pair"}

    summary = {
        "artifact": str(output),
        "h5dir": str(args.h5dir),
        "fon": args.fon,
        "foff": args.foff,
        "drop_leading": args.drop_leading,
        "chunk_frames": args.chunk_frames,
        "reference_shift": not args.no_reference_shift,
        **source,
        **trace.summary(),
    }
    if batch_index is not None:
        summary["batch_index"] = batch_index
    if batch_count is not None:
        summary["batch_count"] = batch_count

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        time=trace.delay,
        trace=trace.ratio_minus_one,
        summary=json.dumps(summary, sort_keys=True),
    )
    return summary


def _detector_artifacts(args):
    from .detector_artifacts import build_detector_artifacts_cupy
    from .detector_mask import parse_y_ranges

    result = build_detector_artifacts_cupy(
        h5dir=args.h5dir,
        fon=args.fon,
        foff=args.foff,
        output_dir=args.output_dir,
        roi_lower=tuple(args.roi_lower),
        roi_dim=tuple(args.roi_dim) if args.roi_dim is not None else None,
        tile_shape=tuple(args.tile_shape),
        exclude_y=parse_y_ranges(args.exclude_y),
        drop_leading=args.drop_leading,
        chunk_frames=args.chunk_frames,
        zero_offset=args.zero_offset,
        zero_offset_index=args.zero_offset_index,
        fit_trailing_drop=args.fit_trailing_drop,
        integrate_pixels=args.integrate,
        components=args.components,
        roots_backend=args.roots_backend,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        amp_threshold=args.amp_threshold,
        max_fit_failures=args.max_fit_failures,
        hdf5_reader=args.hdf5_reader,
        hdf5_reader_workers=args.hdf5_reader_workers,
        max_tiles=args.max_tiles,
        normalization_cache=args.normalization_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        global_roi_lower=(
            tuple(args.global_roi_lower)
            if args.global_roi_lower is not None
            else None
        ),
        global_roi_dim=(
            tuple(args.global_roi_dim)
            if args.global_roi_dim is not None
            else None
        ),
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"detector_artifacts={result.output_dir}")
        print(f"manifest={result.manifest_path}")
        print(f"shape={result.shape}")
        print(f"roi_lower={result.roi_lower}")
        print(f"roi_dim={result.roi_dim}")
        print(f"zero_offset_index={result.zero_offset_index}")
        print(f"zero_offset_status={result.zero_offset_status}")
        print(f"processed_tiles={result.processed_tiles}")
        print(f"raw_fits={result.raw_fits}")
        print(f"failures={result.failures}")
        print(f"skipped_fits={result.skipped_fits}")
        print(f"filtered_fits={result.filtered_fits}")
        print(f"elapsed_s={result.elapsed_s:.6f}")
    return 0


def _detector_artifact_normalize(args):
    from .detector_artifacts import write_detector_artifact_normalization

    result = write_detector_artifact_normalization(
        h5dir=args.h5dir,
        fon=args.fon,
        foff=args.foff,
        output_dir=args.output_dir,
        drop_leading=args.drop_leading,
        chunk_frames=args.chunk_frames,
        zero_offset=args.zero_offset,
        zero_offset_index=args.zero_offset_index,
        hdf5_reader=args.hdf5_reader,
        hdf5_reader_workers=args.hdf5_reader_workers,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"normalization={result.output_dir}")
        print(f"manifest={result.manifest_path}")
        print(f"cache={result.cache_path}")
        print(f"detector_shape={result.detector_shape}")
        print(f"zero_offset_index={result.zero_offset_index}")
        print(f"zero_offset_status={result.zero_offset_status}")
        print(f"elapsed_s={result.elapsed_s:.6f}")
    return 0


def _detector_artifact_distributed(args):
    from .detector_distributed import (
        build_detector_artifact_distributed_plan,
        run_detector_artifact_distributed,
    )

    detector_options = {
        "tile_shape": tuple(args.tile_shape),
        "exclude_y": tuple(args.exclude_y or ()),
        "drop_leading": args.drop_leading,
        "chunk_frames": args.chunk_frames,
        "zero_offset": args.zero_offset,
        "zero_offset_index": args.zero_offset_index,
        "fit_trailing_drop": args.fit_trailing_drop,
        "integrate": args.integrate,
        "components": args.components,
        "roots_backend": args.roots_backend,
        "savgol_window": args.savgol_window,
        "savgol_polyorder": args.savgol_polyorder,
        "amp_threshold": args.amp_threshold,
        "max_fit_failures": args.max_fit_failures,
        "hdf5_reader": args.hdf5_reader,
        "hdf5_reader_workers": args.hdf5_reader_workers,
        "max_tiles": args.max_tiles,
    }
    plan = build_detector_artifact_distributed_plan(
        h5dir=args.h5dir,
        fon=args.fon,
        foff=args.foff,
        output_dir=args.output_dir,
        roi_lower=tuple(args.roi_lower),
        roi_dim=tuple(args.roi_dim) if args.roi_dim is not None else None,
        tile_shape=tuple(args.tile_shape),
        shard_count=args.shard_count,
        shard_width=args.shard_width,
        gpus=args.gpus,
        gpus_per_node=args.gpus_per_node,
        nodes=args.nodes,
        run_label=args.run_label,
        detector_options=detector_options,
        normalization_cache=args.normalization_cache,
    )
    result = run_detector_artifact_distributed(
        plan=plan,
        executor=args.executor,
        submit=args.submit,
        merge=args.merge,
        resume=args.resume,
        slurm_options={
            "partition": args.slurm_partition,
            "account": args.slurm_account,
            "time": args.slurm_time,
            "constraint": args.slurm_constraint,
            "qos": args.slurm_qos,
            "gres": args.slurm_gres,
            "cpus_per_task": args.slurm_cpus_per_task,
            "job_name": args.slurm_job_name,
        },
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"executor={result.executor}")
        print(f"output_dir={result.output_dir}")
        if result.plan_path is not None:
            print(f"plan={result.plan_path}")
        print(f"shards={result.shard_count}")
        print(f"submitted={result.submitted}")
        print(f"merged={result.merged}")
        if "submit_command" in payload:
            print(f"submit_command={payload['submit_command']}")
    return int(result.returncode or 0)


def _detector_artifact_merge(args):
    from .detector_artifacts import merge_detector_artifact_shards

    shard_dirs = tuple(args.shard_dir or ())
    if args.shards_manifest is not None:
        manifest = json.loads(
            args.shards_manifest.read_text(encoding="utf-8")
        )
        if manifest.get("kind") != "xray-detector-artifact-distributed-plan":
            raise ValueError(
                "--shards-manifest must be a distributed detector plan"
            )
        shard_dirs = tuple(
            Path(item["output_dir"]) for item in manifest["shards"]
        )
    if not shard_dirs:
        raise ValueError(
            "provide --shards-manifest or at least one --shard-dir"
        )
    result = merge_detector_artifact_shards(
        shard_dirs=shard_dirs,
        output_dir=args.output_dir,
        strict=not args.no_strict,
        chunk_rows=args.chunk_rows,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"detector_artifacts={result.output_dir}")
        print(f"manifest={result.manifest_path}")
        print(f"shards={result.shard_count}")
        print(f"shape={result.shape}")
        print(f"roi_lower={result.roi_lower}")
        print(f"roi_dim={result.roi_dim}")
        print(f"elapsed_s={result.elapsed_s:.6f}")
    return 0


def _detector_artifact_compare(args):
    from .detector_artifacts import compare_detector_artifacts

    payload = compare_detector_artifacts(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        amp_threshold=args.amp_threshold,
        candidate_origin=(
            tuple(args.candidate_origin)
            if args.candidate_origin is not None
            else None
        ),
        chunk_rows=args.chunk_rows,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"reference_dir={payload['reference_dir']}")
        print(f"candidate_dir={payload['candidate_dir']}")
        print(f"candidate_origin={payload['candidate_origin']}")
        print(f"comparable={payload['comparable']}")
        for name, stats in payload["arrays"].items():
            if not stats.get("present"):
                print(f"{name}=missing")
                continue
            print(
                f"{name}=max_abs_diff:{stats.get('max_abs_diff')} "
                f"rms_diff:{stats.get('rms_diff')} "
                f"finite_mismatch_count:"
                f"{stats.get('finite_mismatch_count')}"
            )
        filtered = payload["filtered_modes"]
        if filtered.get("present"):
            print(
                "filtered_modes="
                f"mask_agreement_ratio:"
                f"{filtered.get('mask_agreement_ratio')} "
                f"reference_filtered_points:"
                f"{filtered.get('reference_filtered_points')} "
                f"candidate_filtered_points:"
                f"{filtered.get('candidate_filtered_points')} "
                f"intersection_filtered_points:"
                f"{filtered.get('intersection_filtered_points')}"
            )
    return 0


def _roi_candidates(args):
    from .detector_mask import parse_y_ranges
    from .hdf5 import scan_roi_candidates

    exclude_y = parse_y_ranges(args.exclude_y)
    candidates = scan_roi_candidates(
        h5dir=args.h5dir,
        fon=args.fon,
        foff=args.foff,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        stride_x=args.stride_x,
        stride_y=args.stride_y,
        drop_leading=args.drop_leading,
        max_candidates=args.limit,
        exclude_y=exclude_y,
    )
    payload = [candidate.to_dict() for candidate in candidates]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for candidate in candidates:
            print(
                f"x={candidate.x} y={candidate.y} "
                f"size={candidate.width}x{candidate.height} "
                f"usable_rows={candidate.usable_rows} "
                f"score={candidate.score:.6g} "
                f"row_peak_max={candidate.row_peak_max:.6g} "
                f"row_std_mean={candidate.row_std_mean:.6g}"
            )
    return 0


def _detector_mask(args):
    from .detector_mask import parse_y_ranges, summarize_tiled_roi

    exclude_y = parse_y_ranges(args.exclude_y)
    summary = summarize_tiled_roi(
        roi_x=args.roi_lower[0],
        roi_y=args.roi_lower[1],
        roi_width=args.roi_dim[0],
        roi_height=args.roi_dim[1],
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        exclude_y=exclude_y,
    )
    payload = summary.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "roi="
            f"[{summary.roi_x}:{summary.roi_x + summary.roi_width},"
            f"{summary.roi_y}:{summary.roi_y + summary.roi_height}]"
        )
        print(f"tile_size={summary.tile_width}x{summary.tile_height}")
        print(
            "exclude_y=" + ",".join(str(item) for item in summary.exclude_y)
        )
        print(f"total_tiles={summary.total_tiles}")
        print(f"active_tiles={summary.active_tiles}")
        print(f"fully_excluded_tiles={summary.fully_excluded_tiles}")
        print(f"partially_excluded_tiles={summary.partially_excluded_tiles}")
        print(f"total_row_fits={summary.total_row_fits}")
        print(f"active_row_fits={summary.active_row_fits}")
        print(f"skipped_row_fits={summary.skipped_row_fits}")
    return 0


def _print_file_probe(label, probe):
    print(f"{label}_path={probe.path}")
    print(f"{label}_size_bytes={probe.size_bytes}")
    print(f"{label}_schema={probe.schema}")
    print(f"{label}_ipm_pairs={','.join(probe.ipm_pairs) or '-'}")
    print(f"{label}_keys={','.join(probe.keys)}")
    for dataset in probe.datasets:
        shape = "x".join(str(dim) for dim in dataset.shape)
        chunks = (
            "-"
            if dataset.chunks is None
            else "x".join(str(dim) for dim in dataset.chunks)
        )
        print(
            f"{label}_dataset={dataset.name} "
            f"shape={shape} dtype={dataset.dtype} chunks={chunks}"
        )


def _linear_prediction_smoke(args):
    from .linear_prediction import (
        LinearPredictionComparison,
        linear_prediction_cupy,
        linear_prediction_numpy,
        synthetic_trace,
    )

    gpu_error = None
    time, trace = synthetic_trace(args.samples)
    cpu = linear_prediction_numpy(
        time,
        trace,
        args.components,
        roots_backend=args.roots_backend,
    )
    if args.no_gpu:
        gpu = None
    else:
        try:
            gpu = linear_prediction_cupy(
                time,
                trace,
                args.components,
                roots_backend=args.roots_backend,
            )
        except (AttributeError, NotImplementedError, RuntimeError) as exc:
            gpu_error = str(exc)
            gpu = None

    if gpu is None:
        comparison = LinearPredictionComparison(
            cpu=cpu,
            gpu=None,
            max_abs_reconstruction_diff=None,
            rms_reconstruction_diff=None,
        )
    else:
        import numpy as np

        diff = cpu.reconstruction - gpu.reconstruction
        comparison = LinearPredictionComparison(
            cpu=cpu,
            gpu=gpu,
            max_abs_reconstruction_diff=float(np.max(np.abs(diff))),
            rms_reconstruction_diff=float(np.sqrt(np.mean(diff**2))),
        )

    payload = _linear_prediction_payload(
        comparison,
        samples=args.samples,
        components=args.components,
        gpu_error=gpu_error,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"components={payload['components']}")
        print(f"cpu_elapsed_s={payload['cpu']['elapsed_s']:.6g}")
        print(f"cpu_chi2={payload['cpu']['chi2']:.6g}")
        print(f"cpu_modes={payload['cpu']['modes']}")
        if payload["gpu"] is None:
            print("gpu_status=skipped")
        elif payload["gpu"].get("error"):
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu']['error']}")
        else:
            print(f"gpu_elapsed_s={payload['gpu']['elapsed_s']:.6g}")
            print(f"gpu_chi2={payload['gpu']['chi2']:.6g}")
            print(f"gpu_modes={payload['gpu']['modes']}")
            print(
                "max_abs_reconstruction_diff="
                f"{payload['max_abs_reconstruction_diff']:.6g}"
            )
            print(
                "rms_reconstruction_diff="
                f"{payload['rms_reconstruction_diff']:.6g}"
            )
    return 0


def _linear_prediction_benchmark(args):
    from .linear_prediction import benchmark_linear_prediction_p1_batch

    result = benchmark_linear_prediction_p1_batch(
        samples=args.samples,
        traces=args.traces,
        n_components=args.components,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
        roots_backend=args.roots_backend,
    )
    payload = dataclass_asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"repeat={payload['repeat']}")
        print(f"cpu_serial_best_s={payload['cpu_serial_best_s']:.6g}")
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_batched_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(f"gpu_serial_best_s={payload['gpu_serial_best_s']:.6g}")
            print(f"gpu_batched_best_s={payload['gpu_batched_best_s']:.6g}")
            print(f"gpu_batch_speedup={payload['gpu_batch_speedup']:.6g}")
            print(
                "max_abs_coefficient_diff="
                f"{payload['max_abs_coefficient_diff']:.6g}"
            )
            print(
                "max_abs_eigenvalue_diff="
                f"{payload['max_abs_eigenvalue_diff']:.6g}"
            )
    return 0


def _linear_prediction_p2_benchmark(args):
    from .linear_prediction import benchmark_linear_prediction_p2

    result = benchmark_linear_prediction_p2(
        samples=args.samples,
        traces=args.traces,
        n_components=args.components,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
    )
    payload = dataclass_asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"modes={payload['modes']}")
        print(f"design_columns={payload['design_columns']}")
        print(f"repeat={payload['repeat']}")
        print(f"cpu_serial_best_s={payload['cpu_serial_best_s']:.6g}")
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_batched_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(f"gpu_serial_best_s={payload['gpu_serial_best_s']:.6g}")
            print(f"gpu_batched_best_s={payload['gpu_batched_best_s']:.6g}")
            print(f"gpu_batch_speedup={payload['gpu_batch_speedup']:.6g}")
            print(
                "max_abs_reconstruction_diff="
                f"{payload['max_abs_reconstruction_diff']:.6g}"
            )
            print(
                "rms_reconstruction_diff="
                f"{payload['rms_reconstruction_diff']:.6g}"
            )
    return 0


def _linear_prediction_savgol_benchmark(args):
    from .linear_prediction import benchmark_linear_prediction_savgol

    result = benchmark_linear_prediction_savgol(
        samples=args.samples,
        traces=args.traces,
        window_length=args.window_length,
        polyorder=args.polyorder,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
    )
    payload = dataclass_asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"repeat={payload['repeat']}")
        print(f"cpu_serial_best_s={payload['cpu_serial_best_s']:.6g}")
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_batched_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(f"gpu_serial_best_s={payload['gpu_serial_best_s']:.6g}")
            print(f"gpu_batched_best_s={payload['gpu_batched_best_s']:.6g}")
            print(f"gpu_batch_speedup={payload['gpu_batch_speedup']:.6g}")
            print(f"max_abs_filter_diff={payload['max_abs_filter_diff']:.6g}")
            print(f"rms_filter_diff={payload['rms_filter_diff']:.6g}")
    return 0


def _linear_prediction_fixed_stages_benchmark(args):
    from .linear_prediction import benchmark_linear_prediction_fixed_stages

    result = benchmark_linear_prediction_fixed_stages(
        samples=args.samples,
        traces=args.traces,
        n_components=args.components,
        window_length=args.window_length,
        polyorder=args.polyorder,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
    )
    payload = dataclass_asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"modes={payload['modes']}")
        print(f"design_columns={payload['design_columns']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"repeat={payload['repeat']}")
        print(f"cpu_serial_best_s={payload['cpu_serial_best_s']:.6g}")
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_batched_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(f"gpu_serial_best_s={payload['gpu_serial_best_s']:.6g}")
            print(f"gpu_batched_best_s={payload['gpu_batched_best_s']:.6g}")
            print(f"gpu_batch_speedup={payload['gpu_batch_speedup']:.6g}")
            print(f"max_abs_filter_diff={payload['max_abs_filter_diff']:.6g}")
            print(f"rms_filter_diff={payload['rms_filter_diff']:.6g}")
            print(
                "max_abs_reconstruction_diff="
                f"{payload['max_abs_reconstruction_diff']:.6g}"
            )
            print(
                "rms_reconstruction_diff="
                f"{payload['rms_reconstruction_diff']:.6g}"
            )
    return 0


def _linear_prediction_variable_p2_benchmark(args):
    from .linear_prediction import benchmark_linear_prediction_variable_p2

    time = None
    trace_rows = None
    source = {"kind": "synthetic"}
    trace_paths = _trace_npz_batch_paths_from_args(args, required=False)
    if trace_paths:
        time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
        source = {
            "kind": "trace-npz-batch",
            "trace_count": len(sources),
            "traces": sources,
        }

    result = benchmark_linear_prediction_variable_p2(
        samples=args.samples,
        traces=args.traces,
        n_components=args.components,
        window_length=args.window_length,
        polyorder=args.polyorder,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
        batched_solver=args.batched_solver,
        time=time,
        trace_rows=trace_rows,
    )
    payload = dataclass_asdict(result)
    payload["source"] = source
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"batched_solver={payload['batched_solver']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"repeat={payload['repeat']}")
        print(f"mode_count_min={payload['mode_count_min']}")
        print(f"mode_count_max={payload['mode_count_max']}")
        print(f"mode_count_unique={payload['mode_count_unique']}")
        groups = ",".join(
            f"{group['mode_count']}:{group['trace_count']}"
            for group in payload["mode_groups"]
        )
        print(f"mode_groups={groups}")
        print(f"max_design_columns={payload['max_design_columns']}")
        print(f"padded_design_entries={payload['padded_design_entries']}")
        print(f"grouped_design_entries={payload['grouped_design_entries']}")
        print(
            f"padding_overhead_ratio={payload['padding_overhead_ratio']:.6g}"
        )
        print(
            "mode_reference_elapsed_s="
            f"{payload['mode_reference_elapsed_s']:.6g}"
        )
        print(f"cpu_serial_best_s={payload['cpu_serial_best_s']:.6g}")
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_batched_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(f"gpu_serial_best_s={payload['gpu_serial_best_s']:.6g}")
            print(f"gpu_batched_best_s={payload['gpu_batched_best_s']:.6g}")
            print(f"gpu_batch_speedup={payload['gpu_batch_speedup']:.6g}")
            print(
                "max_abs_reconstruction_diff="
                f"{payload['max_abs_reconstruction_diff']:.6g}"
            )
            print(
                "rms_reconstruction_diff="
                f"{payload['rms_reconstruction_diff']:.6g}"
            )
    return 0


def _linear_prediction_variable_stages_benchmark(args):
    from .linear_prediction import benchmark_linear_prediction_variable_stages

    time = None
    trace_rows = None
    source = {"kind": "synthetic"}
    trace_paths = _trace_npz_batch_paths_from_args(args, required=False)
    if trace_paths:
        time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
        source = {
            "kind": "trace-npz-batch",
            "trace_count": len(sources),
            "traces": sources,
        }

    result = benchmark_linear_prediction_variable_stages(
        samples=args.samples,
        traces=args.traces,
        n_components=args.components,
        window_length=args.window_length,
        polyorder=args.polyorder,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
        batched_solver=args.batched_solver,
        time=time,
        trace_rows=trace_rows,
    )
    payload = dataclass_asdict(result)
    payload["source"] = source
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"batched_solver={payload['batched_solver']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"repeat={payload['repeat']}")
        print(f"mode_count_min={payload['mode_count_min']}")
        print(f"mode_count_max={payload['mode_count_max']}")
        print(f"mode_count_unique={payload['mode_count_unique']}")
        groups = ",".join(
            f"{group['mode_count']}:{group['trace_count']}"
            for group in payload["mode_groups"]
        )
        print(f"mode_groups={groups}")
        print(f"max_design_columns={payload['max_design_columns']}")
        print(f"padded_design_entries={payload['padded_design_entries']}")
        print(f"grouped_design_entries={payload['grouped_design_entries']}")
        print(
            f"padding_overhead_ratio={payload['padding_overhead_ratio']:.6g}"
        )
        print(
            "mode_reference_elapsed_s="
            f"{payload['mode_reference_elapsed_s']:.6g}"
        )
        print(f"cpu_serial_best_s={payload['cpu_serial_best_s']:.6g}")
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_batched_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(f"gpu_serial_best_s={payload['gpu_serial_best_s']:.6g}")
            print(f"gpu_batched_best_s={payload['gpu_batched_best_s']:.6g}")
            print(f"gpu_batch_speedup={payload['gpu_batch_speedup']:.6g}")
            print(f"max_abs_filter_diff={payload['max_abs_filter_diff']:.6g}")
            print(f"rms_filter_diff={payload['rms_filter_diff']:.6g}")
            print(
                "max_abs_reconstruction_diff="
                f"{payload['max_abs_reconstruction_diff']:.6g}"
            )
            print(
                "rms_reconstruction_diff="
                f"{payload['rms_reconstruction_diff']:.6g}"
            )
    return 0


def _linear_prediction_variable_artifacts_benchmark(args):
    from .linear_prediction import (
        benchmark_linear_prediction_variable_artifacts,
    )

    time = None
    trace_rows = None
    source = {"kind": "synthetic"}
    trace_paths = _trace_npz_batch_paths_from_args(args, required=False)
    if trace_paths:
        time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
        source = {
            "kind": "trace-npz-batch",
            "trace_count": len(sources),
            "traces": sources,
        }

    result = benchmark_linear_prediction_variable_artifacts(
        samples=args.samples,
        traces=args.traces,
        n_components=args.components,
        window_length=args.window_length,
        polyorder=args.polyorder,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
        batched_solver=args.batched_solver,
        time=time,
        trace_rows=trace_rows,
    )
    payload = dataclass_asdict(result)
    payload["source"] = source
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"batched_solver={payload['batched_solver']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"repeat={payload['repeat']}")
        print(f"mode_count_min={payload['mode_count_min']}")
        print(f"mode_count_max={payload['mode_count_max']}")
        print(f"mode_count_unique={payload['mode_count_unique']}")
        groups = ",".join(
            f"{group['mode_count']}:{group['trace_count']}"
            for group in payload["mode_groups"]
        )
        print(f"mode_groups={groups}")
        print(f"max_design_columns={payload['max_design_columns']}")
        print(
            "mode_reference_elapsed_s="
            f"{payload['mode_reference_elapsed_s']:.6g}"
        )
        print(f"cpu_serial_best_s={payload['cpu_serial_best_s']:.6g}")
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_batched_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(f"gpu_serial_best_s={payload['gpu_serial_best_s']:.6g}")
            print(f"gpu_batched_best_s={payload['gpu_batched_best_s']:.6g}")
            print(f"gpu_batch_speedup={payload['gpu_batch_speedup']:.6g}")
            print(f"max_abs_filter_diff={payload['max_abs_filter_diff']:.6g}")
            print(f"rms_filter_diff={payload['rms_filter_diff']:.6g}")
            print(
                "max_abs_coefficient_diff="
                f"{payload['max_abs_coefficient_diff']:.6g}"
            )
            print(
                "max_abs_amplitude_diff="
                f"{payload['max_abs_amplitude_diff']:.6g}"
            )
            print(f"max_abs_phase_diff={payload['max_abs_phase_diff']:.6g}")
            print(
                "max_abs_frequency_center_diff="
                f"{payload['max_abs_frequency_center_diff']:.6g}"
            )
            print(
                "max_abs_time_component_diff="
                f"{payload['max_abs_time_component_diff']:.6g}"
            )
            print(
                "max_abs_reconstruction_diff="
                f"{payload['max_abs_reconstruction_diff']:.6g}"
            )
            print(
                "rms_reconstruction_diff="
                f"{payload['rms_reconstruction_diff']:.6g}"
            )
            print(
                "max_abs_spectrum_component_diff="
                f"{payload['max_abs_spectrum_component_diff']:.6g}"
            )
            print(
                "max_abs_spectrum_total_diff="
                f"{payload['max_abs_spectrum_total_diff']:.6g}"
            )
            print(f"max_abs_chi2_diff={payload['max_abs_chi2_diff']:.6g}")
    return 0


def _linear_prediction_runtime_bridge_benchmark(args):
    from .linear_prediction import benchmark_linear_prediction_runtime_bridge

    time = None
    trace_rows = None
    source = {"kind": "synthetic"}
    trace_paths = _trace_npz_batch_paths_from_args(args, required=False)
    if trace_paths:
        time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
        source = {
            "kind": "trace-npz-batch",
            "trace_count": len(sources),
            "traces": sources,
        }

    result = benchmark_linear_prediction_runtime_bridge(
        samples=args.samples,
        tiles=args.tiles,
        rows_per_tile=args.rows_per_tile,
        n_components=args.components,
        window_length=args.window_length,
        polyorder=args.polyorder,
        repeat=args.repeat,
        run_gpu=not args.no_gpu,
        batched_solver=args.batched_solver,
        time=time,
        trace_rows=trace_rows,
    )
    payload = dataclass_asdict(result)
    payload["source"] = source
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"tiles={payload['tiles']}")
        print(f"rows_per_tile={payload['rows_per_tile']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"batched_solver={payload['batched_solver']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"repeat={payload['repeat']}")
        print(f"mode_count_min={payload['mode_count_min']}")
        print(f"mode_count_max={payload['mode_count_max']}")
        print(f"mode_count_unique={payload['mode_count_unique']}")
        groups = ",".join(
            f"{group['mode_count']}:{group['trace_count']}"
            for group in payload["mode_groups"]
        )
        print(f"mode_groups={groups}")
        print(f"max_design_columns={payload['max_design_columns']}")
        print(
            f"p1_reference_elapsed_s={payload['p1_reference_elapsed_s']:.6g}"
        )
        if payload["gpu_error"]:
            print("gpu_status=unavailable")
            print(f"gpu_error={payload['gpu_error']}")
        elif payload["gpu_serial_per_tile_best_s"] is None:
            print("gpu_status=skipped")
        else:
            print(
                "gpu_serial_per_tile_best_s="
                f"{payload['gpu_serial_per_tile_best_s']:.6g}"
            )
            print(
                "gpu_row_batched_per_tile_best_s="
                f"{payload['gpu_row_batched_per_tile_best_s']:.6g}"
            )
            print(
                "gpu_multi_tile_grouped_best_s="
                f"{payload['gpu_multi_tile_grouped_best_s']:.6g}"
            )
            print(f"row_batched_speedup={payload['row_batched_speedup']:.6g}")
            print(f"multi_tile_speedup={payload['multi_tile_speedup']:.6g}")
            print(
                "multi_vs_row_batched_speedup="
                f"{payload['multi_vs_row_batched_speedup']:.6g}"
            )
            print(
                "max_abs_frequency_center_diff="
                f"{payload['max_abs_frequency_center_diff']:.6g}"
            )
            print(
                "max_abs_time_component_diff="
                f"{payload['max_abs_time_component_diff']:.6g}"
            )
            print(
                "max_abs_reconstruction_diff="
                f"{payload['max_abs_reconstruction_diff']:.6g}"
            )
            print(
                "max_abs_amplitude_diff="
                f"{payload['max_abs_amplitude_diff']:.6g}"
            )
            print(f"max_abs_phase_diff={payload['max_abs_phase_diff']:.6g}")
            print(f"max_abs_chi2_diff={payload['max_abs_chi2_diff']:.6g}")
    return 0


def _linear_prediction_variable_artifacts_acceptance(args):
    from .linear_prediction import (
        benchmark_linear_prediction_variable_artifacts,
    )

    thresholds = {
        "max_filter_diff_ratio": args.max_filter_diff_ratio,
        "max_time_component_diff_ratio": args.max_time_component_diff_ratio,
        "max_reconstruction_diff_ratio": args.max_reconstruction_diff_ratio,
        "max_coefficient_diff": args.max_coefficient_diff,
        "max_amplitude_diff": args.max_amplitude_diff,
        "max_phase_diff": args.max_phase_diff,
        "max_frequency_center_diff": args.max_frequency_center_diff,
        "max_spectrum_component_diff": args.max_spectrum_component_diff,
        "max_spectrum_total_diff": args.max_spectrum_total_diff,
        "max_chi2_diff": args.max_chi2_diff,
        "min_gpu_speedup": args.min_gpu_speedup,
    }
    for name, value in thresholds.items():
        if value < 0:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be non-negative")

    trace_paths = _trace_npz_batch_paths_from_args(args, required=True)
    time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
    source = {
        "kind": "trace-npz-batch",
        "trace_count": len(sources),
        "traces": sources,
    }
    solvers = (
        ("grouped-pinv",)
        if args.batched_solver is None
        else tuple(args.batched_solver)
    )
    trace_scale = _trace_range_scale(trace_rows)
    solver_results = []
    accepted_solvers = []

    for solver in solvers:
        benchmark = benchmark_linear_prediction_variable_artifacts(
            n_components=args.components,
            window_length=args.window_length,
            polyorder=args.polyorder,
            repeat=args.repeat,
            run_gpu=not args.no_gpu,
            batched_solver=solver,
            time=time,
            trace_rows=trace_rows,
        )
        result = dataclass_asdict(benchmark)
        filter_diff_ratio = _optional_finite_ratio(
            benchmark.max_abs_filter_diff,
            trace_scale,
        )
        time_component_diff_ratio = _optional_finite_ratio(
            benchmark.max_abs_time_component_diff,
            trace_scale,
        )
        reconstruction_diff_ratio = _optional_finite_ratio(
            benchmark.max_abs_reconstruction_diff,
            trace_scale,
        )
        coefficient_diff = _optional_abs_float(
            benchmark.max_abs_coefficient_diff
        )
        amplitude_diff = _optional_abs_float(benchmark.max_abs_amplitude_diff)
        phase_diff = _optional_abs_float(benchmark.max_abs_phase_diff)
        frequency_center_diff = _optional_abs_float(
            benchmark.max_abs_frequency_center_diff
        )
        spectrum_component_diff = _optional_abs_float(
            benchmark.max_abs_spectrum_component_diff
        )
        spectrum_total_diff = _optional_abs_float(
            benchmark.max_abs_spectrum_total_diff
        )
        chi2_diff = _optional_abs_float(benchmark.max_abs_chi2_diff)
        speedup = benchmark.gpu_batch_speedup
        checks = {
            "filter_diff_ratio": (
                filter_diff_ratio <= args.max_filter_diff_ratio
            ),
            "time_component_diff_ratio": (
                time_component_diff_ratio
                <= args.max_time_component_diff_ratio
            ),
            "reconstruction_diff_ratio": (
                reconstruction_diff_ratio
                <= args.max_reconstruction_diff_ratio
            ),
            "coefficient_diff": coefficient_diff <= args.max_coefficient_diff,
            "amplitude_diff": amplitude_diff <= args.max_amplitude_diff,
            "phase_diff": phase_diff <= args.max_phase_diff,
            "frequency_center_diff": (
                frequency_center_diff <= args.max_frequency_center_diff
            ),
            "spectrum_component_diff": (
                spectrum_component_diff <= args.max_spectrum_component_diff
            ),
            "spectrum_total_diff": (
                spectrum_total_diff <= args.max_spectrum_total_diff
            ),
            "chi2_diff": chi2_diff <= args.max_chi2_diff,
            "gpu_speedup": (
                speedup is not None and speedup >= args.min_gpu_speedup
            ),
        }
        accepted = bool(
            benchmark.gpu_error is None
            and speedup is not None
            and all(checks.values())
        )
        result["filter_diff_ratio"] = filter_diff_ratio
        result["time_component_diff_ratio"] = time_component_diff_ratio
        result["reconstruction_diff_ratio"] = reconstruction_diff_ratio
        result["artifact_checks"] = checks
        result["accepted"] = accepted
        solver_results.append(result)
        if accepted:
            accepted_solvers.append(solver)

    accepted = bool(solver_results) and len(accepted_solvers) == len(
        solver_results
    )
    payload = {
        "accepted": accepted,
        "accepted_solvers": tuple(accepted_solvers),
        "thresholds": thresholds,
        "source": source,
        "trace_count": len(sources),
        "trace_range": trace_scale,
        "samples": int(time.shape[0]),
        "components": args.components,
        "window_length": args.window_length,
        "polyorder": args.polyorder,
        "repeat": args.repeat,
        "solvers": tuple(solver_results),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "accepted" if payload["accepted"] else "rejected"
        print(f"status={status}")
        print(f"trace_count={payload['trace_count']}")
        print(f"samples={payload['samples']}")
        print(f"components={payload['components']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"max_filter_diff_ratio={args.max_filter_diff_ratio:.6g}")
        print(
            "max_time_component_diff_ratio="
            f"{args.max_time_component_diff_ratio:.6g}"
        )
        print(
            "max_reconstruction_diff_ratio="
            f"{args.max_reconstruction_diff_ratio:.6g}"
        )
        print(f"min_gpu_speedup={args.min_gpu_speedup:.6g}")
        for result in payload["solvers"]:
            print(f"solver={result['batched_solver']}")
            print(f"  accepted={result['accepted']}")
            print(f"  gpu_error={result['gpu_error']}")
            print(f"  gpu_batch_speedup={result['gpu_batch_speedup']}")
            print(f"  filter_diff_ratio={result['filter_diff_ratio']:.6g}")
            print(
                "  time_component_diff_ratio="
                f"{result['time_component_diff_ratio']:.6g}"
            )
            print(
                "  reconstruction_diff_ratio="
                f"{result['reconstruction_diff_ratio']:.6g}"
            )
            print(
                "  max_abs_coefficient_diff="
                f"{result['max_abs_coefficient_diff']}"
            )
            print(
                "  max_abs_spectrum_component_diff="
                f"{result['max_abs_spectrum_component_diff']}"
            )
            print(f"  max_abs_chi2_diff={result['max_abs_chi2_diff']}")
    return 0 if payload["accepted"] else 1


def _linear_prediction_variable_stages_acceptance(args):
    from .linear_prediction import benchmark_linear_prediction_variable_stages

    if args.max_filter_diff_ratio < 0:
        raise ValueError("--max-filter-diff-ratio must be non-negative")
    if args.max_reconstruction_diff_ratio < 0:
        raise ValueError(
            "--max-reconstruction-diff-ratio must be non-negative"
        )
    if args.min_gpu_speedup < 0:
        raise ValueError("--min-gpu-speedup must be non-negative")

    trace_paths = _trace_npz_batch_paths_from_args(args, required=True)
    time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
    source = {
        "kind": "trace-npz-batch",
        "trace_count": len(sources),
        "traces": sources,
    }
    solvers = (
        ("grouped-pinv",)
        if args.batched_solver is None
        else tuple(args.batched_solver)
    )
    trace_scale = _trace_range_scale(trace_rows)
    solver_results = []
    accepted_solvers = []

    for solver in solvers:
        benchmark = benchmark_linear_prediction_variable_stages(
            n_components=args.components,
            window_length=args.window_length,
            polyorder=args.polyorder,
            repeat=args.repeat,
            run_gpu=not args.no_gpu,
            batched_solver=solver,
            time=time,
            trace_rows=trace_rows,
        )
        result = dataclass_asdict(benchmark)
        filter_diff_ratio = _optional_finite_ratio(
            benchmark.max_abs_filter_diff,
            trace_scale,
        )
        reconstruction_diff_ratio = _optional_finite_ratio(
            benchmark.max_abs_reconstruction_diff,
            trace_scale,
        )
        speedup = benchmark.gpu_batch_speedup
        filter_within_threshold = (
            filter_diff_ratio <= args.max_filter_diff_ratio
        )
        reconstruction_within_threshold = (
            reconstruction_diff_ratio <= args.max_reconstruction_diff_ratio
        )
        accepted = bool(
            benchmark.gpu_error is None
            and speedup is not None
            and speedup >= args.min_gpu_speedup
            and filter_within_threshold
            and reconstruction_within_threshold
        )
        result["filter_diff_ratio"] = filter_diff_ratio
        result["reconstruction_diff_ratio"] = reconstruction_diff_ratio
        result["accepted"] = accepted
        solver_results.append(result)
        if accepted:
            accepted_solvers.append(solver)

    accepted = bool(solver_results) and len(accepted_solvers) == len(
        solver_results
    )
    payload = {
        "accepted": accepted,
        "accepted_solvers": tuple(accepted_solvers),
        "thresholds": {
            "max_filter_diff_ratio": args.max_filter_diff_ratio,
            "max_reconstruction_diff_ratio": (
                args.max_reconstruction_diff_ratio
            ),
            "min_gpu_speedup": args.min_gpu_speedup,
        },
        "source": source,
        "trace_count": len(sources),
        "trace_range": trace_scale,
        "samples": int(time.shape[0]),
        "components": args.components,
        "window_length": args.window_length,
        "polyorder": args.polyorder,
        "repeat": args.repeat,
        "solvers": tuple(solver_results),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "accepted" if payload["accepted"] else "rejected"
        print(f"status={status}")
        print(f"trace_count={payload['trace_count']}")
        print(f"samples={payload['samples']}")
        print(f"components={payload['components']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"max_filter_diff_ratio={args.max_filter_diff_ratio:.6g}")
        print(
            "max_reconstruction_diff_ratio="
            f"{args.max_reconstruction_diff_ratio:.6g}"
        )
        print(f"min_gpu_speedup={args.min_gpu_speedup:.6g}")
        for result in payload["solvers"]:
            print(f"solver={result['batched_solver']}")
            print(f"  accepted={result['accepted']}")
            print(f"  gpu_error={result['gpu_error']}")
            print(f"  gpu_batch_speedup={result['gpu_batch_speedup']}")
            print(f"  filter_diff_ratio={result['filter_diff_ratio']:.6g}")
            print(
                "  reconstruction_diff_ratio="
                f"{result['reconstruction_diff_ratio']:.6g}"
            )
    return 0 if payload["accepted"] else 1


def _linear_prediction_variable_p2_acceptance(args):
    from .linear_prediction import benchmark_linear_prediction_variable_p2

    if args.max_diff_ratio < 0:
        raise ValueError("--max-diff-ratio must be non-negative")
    if args.min_gpu_speedup < 0:
        raise ValueError("--min-gpu-speedup must be non-negative")

    trace_paths = _trace_npz_batch_paths_from_args(args, required=True)
    time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
    source = {
        "kind": "trace-npz-batch",
        "trace_count": len(sources),
        "traces": sources,
    }
    solvers = (
        ("grouped-pinv",)
        if args.batched_solver is None
        else tuple(args.batched_solver)
    )
    trace_scale = _trace_range_scale(trace_rows)
    solver_results = []
    accepted_solvers = []

    for solver in solvers:
        benchmark = benchmark_linear_prediction_variable_p2(
            n_components=args.components,
            window_length=args.window_length,
            polyorder=args.polyorder,
            repeat=args.repeat,
            run_gpu=not args.no_gpu,
            batched_solver=solver,
            time=time,
            trace_rows=trace_rows,
        )
        result = dataclass_asdict(benchmark)
        if benchmark.max_abs_reconstruction_diff is None:
            diff_ratio = float("inf")
        else:
            diff_ratio = _finite_ratio(
                benchmark.max_abs_reconstruction_diff,
                trace_scale,
            )
        speedup = benchmark.gpu_batch_speedup
        accepted = bool(
            benchmark.gpu_error is None
            and speedup is not None
            and speedup >= args.min_gpu_speedup
            and diff_ratio <= args.max_diff_ratio
        )
        result["diff_ratio"] = diff_ratio
        result["accepted"] = accepted
        solver_results.append(result)
        if accepted:
            accepted_solvers.append(solver)

    accepted = bool(solver_results) and len(accepted_solvers) == len(
        solver_results
    )
    payload = {
        "accepted": accepted,
        "accepted_solvers": tuple(accepted_solvers),
        "thresholds": {
            "max_diff_ratio": args.max_diff_ratio,
            "min_gpu_speedup": args.min_gpu_speedup,
        },
        "source": source,
        "trace_count": len(sources),
        "trace_range": trace_scale,
        "samples": int(time.shape[0]),
        "components": args.components,
        "window_length": args.window_length,
        "polyorder": args.polyorder,
        "repeat": args.repeat,
        "solvers": tuple(solver_results),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "accepted" if payload["accepted"] else "rejected"
        print(f"status={status}")
        print(f"trace_count={payload['trace_count']}")
        print(f"samples={payload['samples']}")
        print(f"components={payload['components']}")
        print(f"window_length={payload['window_length']}")
        print(f"polyorder={payload['polyorder']}")
        print(f"max_diff_ratio={args.max_diff_ratio:.6g}")
        print(f"min_gpu_speedup={args.min_gpu_speedup:.6g}")
        for result in payload["solvers"]:
            print(f"solver={result['batched_solver']}")
            print(f"  accepted={result['accepted']}")
            print(f"  gpu_error={result['gpu_error']}")
            print(f"  gpu_batch_speedup={result['gpu_batch_speedup']}")
            print(f"  diff_ratio={result['diff_ratio']:.6g}")
            print(
                "  max_abs_reconstruction_diff="
                f"{result['max_abs_reconstruction_diff']}"
            )
    return 0 if payload["accepted"] else 1


def _optional_finite_ratio(value, scale: float) -> float:
    if value is None:
        return float("inf")
    return _finite_ratio(value, scale)


def _optional_abs_float(value) -> float:
    if value is None:
        return float("inf")
    return abs(float(value))


def _prediction_roots_benchmark(args):
    from .linear_prediction import benchmark_prediction_roots

    backends = None if args.backend is None else tuple(args.backend)
    result = benchmark_prediction_roots(
        samples=args.samples,
        traces=args.traces,
        n_components=args.components,
        repeat=args.repeat,
        backends=backends,
        run_gpu=not args.no_gpu,
    )
    payload = dataclass_asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"samples={payload['samples']}")
        print(f"traces={payload['traces']}")
        print(f"components={payload['components']}")
        print(f"repeat={payload['repeat']}")
        for backend in payload["backends"]:
            print(f"backend={backend['backend']}")
            print(f"  array_module={backend['array_module']}")
            print(f"  batched={backend['batched']}")
            print(f"  matrix_size={backend['matrix_size']}")
            print(f"  row_count={backend['row_count']}")
            print(f"  failures={backend['failures']}")
            if backend["error"]:
                print(f"  error={backend['error']}")
            else:
                print(f"  best_s={backend['best_s']:.6g}")
                print(
                    f"  max_abs_root_diff={backend['max_abs_root_diff']:.6g}"
                )
    return 0


def _model_order_sweep(args):
    from .linear_prediction import model_order_sweep

    components = _component_sweep_values(args)
    trace_paths = _trace_npz_batch_paths_from_args(args, required=False)
    if trace_paths:
        _reject_trace_npz_hdf5_options(args)
        if len(trace_paths) == 1:
            time, trace, source = _load_trace_npz(trace_paths[0])
            payload = _single_model_order_sweep_payload(
                model_order_sweep(
                    time,
                    trace,
                    components,
                    roots_backend=args.roots_backend,
                    relative_tolerance=args.relative_tolerance,
                ),
                source=source,
            )
        else:
            time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
            payload = _model_order_sweep_batch_payload(
                time,
                trace_rows,
                sources,
                components=components,
                roots_backend=args.roots_backend,
                relative_tolerance=args.relative_tolerance,
            )
    else:
        time, trace, source = _trace_for_workbench(args)
        payload = _single_model_order_sweep_payload(
            model_order_sweep(
                time,
                trace,
                components,
                roots_backend=args.roots_backend,
                relative_tolerance=args.relative_tolerance,
            ),
            source=source,
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["source"]["kind"] == "trace-npz-batch":
        _print_model_order_sweep_batch(payload)
    else:
        _print_model_order_sweep_payload(payload)
    return 0


def _single_model_order_sweep_payload(result, *, source):
    payload = dataclass_asdict(result)
    payload["source"] = source
    return payload


def _model_order_sweep_batch_payload(
    time,
    trace_rows,
    sources,
    *,
    components,
    roots_backend: str,
    relative_tolerance: float,
):
    from .linear_prediction import model_order_sweep

    trace_payloads = []
    best_components = []
    best_selected_model_orders = []
    best_rms_residuals = []
    best_reconstruction_errors = []
    for trace_index, (trace, source) in enumerate(
        zip(trace_rows, sources, strict=True)
    ):
        result = model_order_sweep(
            time,
            trace,
            components,
            roots_backend=roots_backend,
            relative_tolerance=relative_tolerance,
        )
        trace_payload = _single_model_order_sweep_payload(
            result,
            source=source,
        )
        trace_payload["trace_index"] = trace_index
        trace_payloads.append(trace_payload)
        best_components.append(int(result.best_components))
        best_selected_model_orders.append(
            int(result.best_selected_model_order)
        )
        best_rms_residuals.append(float(result.best_rms_residual))
        best_reconstruction_errors.append(
            float(result.best_reconstruction_rms_error)
        )

    return {
        "source": {
            "kind": "trace-npz-batch",
            "trace_count": len(sources),
            "traces": sources,
        },
        "trace_count": len(sources),
        "samples": int(time.shape[0]),
        "roots_backend": roots_backend,
        "relative_tolerance": float(relative_tolerance),
        "component_counts": tuple(int(item) for item in components),
        "best_components_by_trace": tuple(best_components),
        "best_components_unique": tuple(sorted(set(best_components))),
        "best_selected_model_orders_by_trace": tuple(
            best_selected_model_orders
        ),
        "best_selected_model_orders_unique": tuple(
            sorted(set(best_selected_model_orders))
        ),
        "best_rms_residual_min": min(best_rms_residuals),
        "best_rms_residual_max": max(best_rms_residuals),
        "best_reconstruction_rms_error_min": min(best_reconstruction_errors),
        "best_reconstruction_rms_error_max": max(best_reconstruction_errors),
        "traces": tuple(trace_payloads),
    }


def _print_model_order_sweep_payload(payload):
    print(f"source={payload['source']['kind']}")
    print(f"samples={payload['samples']}")
    print(f"roots_backend={payload['roots_backend']}")
    print(f"best_components={payload['best_components']}")
    print(f"best_selected_model_order={payload['best_selected_model_order']}")
    print(f"best_rms_residual={payload['best_rms_residual']:.6g}")
    print(
        "best_reconstruction_rms_error="
        f"{payload['best_reconstruction_rms_error']:.6g}"
    )
    for entry in payload["entries"]:
        print(
            f"components={entry['components']} "
            f"selected_model_order={entry['selected_model_order']} "
            f"rms_residual={entry['rms_residual']:.6g} "
            "reconstruction_rms_error="
            f"{entry['reconstruction_rms_error']:.6g} "
            f"chi2={entry['chi2']:.6g} "
            f"selected_roots={entry['selected_root_count']} "
            f"decaying_roots={entry['decaying_root_count']}"
        )


def _print_model_order_sweep_batch(payload):
    print(f"source={payload['source']['kind']}")
    print(f"trace_count={payload['trace_count']}")
    print(f"samples={payload['samples']}")
    print(f"roots_backend={payload['roots_backend']}")
    component_counts = ",".join(
        str(item) for item in payload["component_counts"]
    )
    print(f"component_counts={component_counts}")
    print(
        "best_components_unique="
        + ",".join(str(item) for item in payload["best_components_unique"])
    )
    print(
        "best_selected_model_orders_unique="
        + ",".join(
            str(item) for item in payload["best_selected_model_orders_unique"]
        )
    )
    for trace_payload in payload["traces"]:
        print(f"trace_index={trace_payload['trace_index']}")
        print(f"  best_components={trace_payload['best_components']}")
        print(
            "  best_selected_model_order="
            f"{trace_payload['best_selected_model_order']}"
        )
        print(
            "  best_reconstruction_rms_error="
            f"{trace_payload['best_reconstruction_rms_error']:.6g}"
        )


def _subspace_benchmark(args):
    from .subspace import compare_subspace_methods

    trace_paths = _trace_npz_batch_paths_from_args(args, required=False)
    svd_backends = _subspace_svd_backends(args)
    methods = (
        ("matrix-pencil", "esprit")
        if args.method is None
        else tuple(args.method)
    )
    if trace_paths:
        _reject_trace_npz_hdf5_options(args)
        if len(trace_paths) == 1:
            time, trace, source = _load_trace_npz(trace_paths[0])
            payload = _single_subspace_benchmark_payload(
                compare_subspace_methods(
                    time,
                    trace,
                    model_order=args.model_order,
                    components=args.components,
                    methods=methods,
                    svd_backends=svd_backends,
                    pencil_rows=args.pencil_rows,
                    randomized_oversamples=args.randomized_oversamples,
                    power_iterations=args.power_iterations,
                    random_seed=args.random_seed,
                ),
                source=source,
                svd_backends=svd_backends,
            )
        else:
            time, trace_rows, sources = _load_trace_npz_batch(trace_paths)
            payload = _subspace_benchmark_batch_payload(
                time,
                trace_rows,
                sources,
                model_order=args.model_order,
                components=args.components,
                methods=methods,
                svd_backends=svd_backends,
                pencil_rows=args.pencil_rows,
                randomized_oversamples=args.randomized_oversamples,
                power_iterations=args.power_iterations,
                random_seed=args.random_seed,
            )
    else:
        time, trace, source = _trace_for_workbench(args)
        payload = _single_subspace_benchmark_payload(
            compare_subspace_methods(
                time,
                trace,
                model_order=args.model_order,
                components=args.components,
                methods=methods,
                svd_backends=svd_backends,
                pencil_rows=args.pencil_rows,
                randomized_oversamples=args.randomized_oversamples,
                power_iterations=args.power_iterations,
                random_seed=args.random_seed,
            ),
            source=source,
            svd_backends=svd_backends,
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["source"]["kind"] == "trace-npz-batch":
        _print_subspace_benchmark_batch(payload)
    else:
        print(f"source={payload['source']['kind']}")
        print(f"samples={payload['samples']}")
        print(f"model_order={payload['model_order']}")
        print(f"components={payload['components']}")
        print(f"baseline_chi2={payload['baseline_chi2']:.6g}")
        print(f"baseline_rms_residual={payload['baseline_rms_residual']:.6g}")
        for method in payload["methods"]:
            print(f"method={method['method']}")
            print(f"  svd_backend={method['svd_backend']}")
            print(f"  svd_rank={method['svd_rank']}")
            print(f"  rms_residual={method['rms_residual']:.6g}")
            print(
                "  max_abs_reconstruction_diff="
                f"{method['max_abs_reconstruction_diff']:.6g}"
            )
            print(f"  elapsed_s={method['elapsed_s']:.6g}")
    return 0


def _single_subspace_benchmark_payload(result, *, source, svd_backends):
    payload = dataclass_asdict(result)
    payload["source"] = source
    payload["svd_backends"] = svd_backends
    return payload


def _subspace_benchmark_batch_payload(
    time,
    trace_rows,
    sources,
    *,
    model_order: int,
    components: int,
    methods,
    svd_backends,
    pencil_rows: int | None,
    randomized_oversamples: int,
    power_iterations: int,
    random_seed: int,
):
    from .subspace import compare_subspace_methods

    method_summary = {
        (method, svd_backend): {
            "method": method,
            "svd_backend": svd_backend,
            "trace_count": 0,
            "rms_residual_min": float("inf"),
            "rms_residual_max": 0.0,
            "max_abs_reconstruction_diff_max": 0.0,
            "elapsed_s_total": 0.0,
        }
        for method in methods
        for svd_backend in svd_backends
    }
    trace_payloads = []
    baseline_chi2_values = []
    baseline_rms_values = []

    for trace_index, (trace, source) in enumerate(
        zip(trace_rows, sources, strict=True)
    ):
        result = compare_subspace_methods(
            time,
            trace,
            model_order=model_order,
            components=components,
            methods=methods,
            svd_backends=svd_backends,
            pencil_rows=pencil_rows,
            randomized_oversamples=randomized_oversamples,
            power_iterations=power_iterations,
            random_seed=random_seed,
        )
        trace_payload = _single_subspace_benchmark_payload(
            result,
            source=source,
            svd_backends=svd_backends,
        )
        trace_payload["trace_index"] = trace_index
        trace_payloads.append(trace_payload)
        baseline_chi2_values.append(float(result.baseline_chi2))
        baseline_rms_values.append(float(result.baseline_rms_residual))

        for method_result in result.methods:
            summary = method_summary[
                (method_result.method, method_result.svd_backend)
            ]
            summary["trace_count"] += 1
            summary["rms_residual_min"] = min(
                float(summary["rms_residual_min"]),
                float(method_result.rms_residual),
            )
            summary["rms_residual_max"] = max(
                float(summary["rms_residual_max"]),
                float(method_result.rms_residual),
            )
            summary["max_abs_reconstruction_diff_max"] = max(
                float(summary["max_abs_reconstruction_diff_max"]),
                float(method_result.max_abs_reconstruction_diff),
            )
            summary["elapsed_s_total"] += float(method_result.elapsed_s)

    summaries = tuple(
        method_summary[(method, svd_backend)]
        for method in methods
        for svd_backend in svd_backends
    )
    return {
        "source": {
            "kind": "trace-npz-batch",
            "trace_count": len(sources),
            "traces": sources,
        },
        "trace_count": len(sources),
        "samples": int(time.shape[0]),
        "model_order": int(model_order),
        "components": int(components),
        "svd_backends": svd_backends,
        "baseline_chi2_min": min(baseline_chi2_values),
        "baseline_chi2_max": max(baseline_chi2_values),
        "baseline_rms_residual_min": min(baseline_rms_values),
        "baseline_rms_residual_max": max(baseline_rms_values),
        "method_summary": summaries,
        "traces": tuple(trace_payloads),
    }


def _print_subspace_benchmark_batch(payload):
    print(f"source={payload['source']['kind']}")
    print(f"trace_count={payload['trace_count']}")
    print(f"samples={payload['samples']}")
    print(f"model_order={payload['model_order']}")
    print(f"components={payload['components']}")
    print(
        "baseline_rms_residual_range="
        f"{payload['baseline_rms_residual_min']:.6g}:"
        f"{payload['baseline_rms_residual_max']:.6g}"
    )
    for summary in payload["method_summary"]:
        print(f"method={summary['method']}")
        print(f"  svd_backend={summary['svd_backend']}")
        print(f"  trace_count={summary['trace_count']}")
        print(
            "  rms_residual_range="
            f"{summary['rms_residual_min']:.6g}:"
            f"{summary['rms_residual_max']:.6g}"
        )
        print(
            "  max_abs_reconstruction_diff_max="
            f"{summary['max_abs_reconstruction_diff_max']:.6g}"
        )


def _subspace_acceptance(args):
    from .subspace import compare_subspace_methods

    if args.max_rms_ratio < 0:
        raise ValueError("--max-rms-ratio must be non-negative")
    if args.max_diff_ratio < 0:
        raise ValueError("--max-diff-ratio must be non-negative")

    requested_methods = (
        ("matrix-pencil", "esprit")
        if args.method is None
        else tuple(args.method)
    )
    requested_svd_backends = _subspace_svd_backends(args)
    method_summary = {
        (method, svd_backend): {
            "method": method,
            "svd_backend": svd_backend,
            "accepted": True,
            "accepted_traces": 0,
            "trace_count": 0,
            "max_rms_ratio": 0.0,
            "max_diff_ratio": 0.0,
        }
        for method in requested_methods
        for svd_backend in requested_svd_backends
    }
    trace_payloads = []

    trace_paths = _trace_npz_batch_paths_from_args(args, required=True)
    for trace_path in trace_paths:
        time, trace, source = _load_trace_npz(trace_path)
        result = compare_subspace_methods(
            time,
            trace,
            model_order=args.model_order,
            components=args.components,
            methods=requested_methods,
            svd_backends=requested_svd_backends,
            pencil_rows=args.pencil_rows,
            randomized_oversamples=args.randomized_oversamples,
            power_iterations=args.power_iterations,
            random_seed=args.random_seed,
        )
        trace_scale = _trace_range_scale(trace)
        methods = []
        for method_result in result.methods:
            rms_ratio = _finite_ratio(
                method_result.rms_residual,
                result.baseline_rms_residual,
            )
            diff_ratio = _finite_ratio(
                method_result.max_abs_reconstruction_diff,
                trace_scale,
            )
            accepted = (
                rms_ratio <= args.max_rms_ratio
                and diff_ratio <= args.max_diff_ratio
            )
            method_payload = dataclass_asdict(method_result)
            method_payload["rms_ratio"] = rms_ratio
            method_payload["diff_ratio"] = diff_ratio
            method_payload["accepted"] = accepted
            methods.append(method_payload)

            summary = method_summary[
                (method_result.method, method_result.svd_backend)
            ]
            summary["trace_count"] += 1
            summary["accepted_traces"] += int(accepted)
            summary["accepted"] = bool(summary["accepted"] and accepted)
            summary["max_rms_ratio"] = max(
                float(summary["max_rms_ratio"]),
                rms_ratio,
            )
            summary["max_diff_ratio"] = max(
                float(summary["max_diff_ratio"]),
                diff_ratio,
            )

        trace_payloads.append(
            {
                "source": source,
                "samples": result.samples,
                "model_order": result.model_order,
                "components": result.components,
                "baseline_chi2": result.baseline_chi2,
                "baseline_rms_residual": result.baseline_rms_residual,
                "trace_range": trace_scale,
                "methods": methods,
            }
        )

    summaries = tuple(
        method_summary[(method, svd_backend)]
        for method in requested_methods
        for svd_backend in requested_svd_backends
    )
    accepted_methods = tuple(
        _subspace_variant_label(
            summary["method"],
            summary["svd_backend"],
            svd_backends=requested_svd_backends,
        )
        for summary in summaries
        if summary["accepted"]
    )
    payload = {
        "accepted": bool(accepted_methods),
        "accepted_methods": accepted_methods,
        "thresholds": {
            "max_rms_ratio": args.max_rms_ratio,
            "max_diff_ratio": args.max_diff_ratio,
        },
        "trace_count": len(trace_payloads),
        "model_order": args.model_order,
        "components": args.components,
        "svd_backends": requested_svd_backends,
        "method_summary": summaries,
        "traces": tuple(trace_payloads),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "accepted" if payload["accepted"] else "rejected"
        print(f"status={status}")
        print(f"trace_count={payload['trace_count']}")
        print(f"model_order={payload['model_order']}")
        print(f"components={payload['components']}")
        print(f"svd_backends={','.join(payload['svd_backends'])}")
        print(f"max_rms_ratio={args.max_rms_ratio:.6g}")
        print(f"max_diff_ratio={args.max_diff_ratio:.6g}")
        for summary in payload["method_summary"]:
            print(f"method={summary['method']}")
            print(f"  svd_backend={summary['svd_backend']}")
            print(f"  accepted={summary['accepted']}")
            print(
                "  accepted_traces="
                f"{summary['accepted_traces']}/{summary['trace_count']}"
            )
            print(f"  max_rms_ratio={summary['max_rms_ratio']:.6g}")
            print(f"  max_diff_ratio={summary['max_diff_ratio']:.6g}")
    return 0 if payload["accepted"] else 1


def _linear_prediction_payload(
    comparison,
    *,
    samples: int,
    components: int,
    gpu_error: str | None,
):
    payload = {
        "samples": samples,
        "components": components,
        "cpu": _linear_prediction_result_payload(comparison.cpu),
        "gpu": None,
        "max_abs_reconstruction_diff": (
            comparison.max_abs_reconstruction_diff
        ),
        "rms_reconstruction_diff": comparison.rms_reconstruction_diff,
    }
    if comparison.gpu is not None:
        payload["gpu"] = _linear_prediction_result_payload(comparison.gpu)
    elif gpu_error:
        payload["gpu"] = {"error": gpu_error}
    return payload


def _linear_prediction_result_payload(result):
    payload = {
        "backend": result.backend,
        "elapsed_s": result.elapsed_s,
        "chi2": result.chi2,
        "selected_model_order": result.selected_model_order,
        "decaying_root_count": result.decaying_root_count,
        "modes": len(result.angular_frequency),
        "reconstruction_samples": len(result.reconstruction),
    }
    if result.roots_stats is not None:
        payload["roots"] = dataclass_asdict(result.roots_stats)
    return payload


def _gpu_policy(_args):
    print(gpu_first_policy())
    return 0


def _component_sweep_values(args):
    if args.component_values:
        values = tuple(int(item) for item in args.component_values)
    else:
        if args.step <= 0:
            raise ValueError("--step must be positive")
        if args.min_components <= 0:
            raise ValueError("--min-components must be positive")
        if args.max_components < args.min_components:
            raise ValueError(
                "--max-components must be greater than or equal to "
                "--min-components"
            )
        values = tuple(
            range(args.min_components, args.max_components + 1, args.step)
        )
    if any(value <= 0 for value in values):
        raise ValueError("component counts must be positive")
    return values


def _subspace_svd_backends(args):
    if args.randomized_oversamples < 0:
        raise ValueError("--randomized-oversamples must be non-negative")
    if args.power_iterations < 0:
        raise ValueError("--power-iterations must be non-negative")
    if args.svd_backend is None:
        return ("full",)
    return tuple(args.svd_backend)


def _subspace_variant_label(method: str, svd_backend: str, *, svd_backends):
    if len(svd_backends) == 1 and svd_backend == "full":
        return method
    return f"{method}:{svd_backend}"


def _trace_npz_batch_paths_from_args(args, *, required: bool):
    trace_paths = tuple(args.trace_npz or ())
    trace_dir = getattr(args, "trace_dir", None)
    if trace_dir is not None:
        if trace_paths:
            raise ValueError(
                "--trace-dir cannot be combined with --trace-npz"
            )
        directory = Path(trace_dir)
        if not directory.is_dir():
            raise ValueError("--trace-dir must be a directory")
        trace_paths = tuple(
            sorted(directory.glob("*.npz"), key=_trace_npz_sort_key)
        )
        if not trace_paths:
            raise ValueError(
                "--trace-dir must contain at least one .npz file"
            )
    if required and not trace_paths:
        raise ValueError("provide --trace-npz or --trace-dir")
    return trace_paths


def _trace_npz_sort_key(path: Path):
    parts = re.split(r"(\d+)", Path(path).name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _reject_trace_npz_hdf5_options(args):
    h5_inputs = (
        args.h5dir is not None,
        args.fon is not None,
        args.foff is not None,
        args.roi_lower is not None,
        args.roi_dim is not None,
        args.row_y is not None,
        bool(args.exclude_y),
        args.drop_leading != 1,
        args.chunk_frames != 16,
        args.no_reference_shift,
    )
    if any(h5_inputs):
        raise ValueError(
            "--trace-npz cannot be combined with HDF5 input options"
        )


def _trace_for_workbench(args):
    if args.trace_npz is not None:
        _reject_trace_npz_hdf5_options(args)
        return _load_trace_npz(args.trace_npz)

    provided = (
        args.h5dir is not None,
        args.fon is not None,
        args.foff is not None,
    )
    if any(provided) and not all(provided):
        raise ValueError(
            "--h5dir, --fon, and --foff must be provided together"
        )
    if all(provided):
        from .detector_mask import format_y_ranges, parse_y_ranges
        from .hdf5 import load_hdf5_pair_roi_trace, load_hdf5_pair_trace

        exclude_y = parse_y_ranges(args.exclude_y)
        has_roi = args.roi_lower is not None or args.roi_dim is not None
        has_row_selection = args.row_y is not None or bool(exclude_y)
        if has_row_selection and not has_roi:
            raise ValueError(
                "--roi-lower and --roi-dim are required with row/mask "
                "selection"
            )
        if has_roi:
            if args.roi_lower is None or args.roi_dim is None:
                raise ValueError(
                    "--roi-lower and --roi-dim must be provided together"
                )
            loaded = load_hdf5_pair_roi_trace(
                h5dir=args.h5dir,
                fon=args.fon,
                foff=args.foff,
                roi_x=args.roi_lower[0],
                roi_y=args.roi_lower[1],
                roi_width=args.roi_dim[0],
                roi_height=args.roi_dim[1],
                row_y=args.row_y,
                exclude_y=exclude_y,
                drop_leading=args.drop_leading,
                chunk_frames=args.chunk_frames,
                reference_shift=not args.no_reference_shift,
            )
            source = {
                "kind": "hdf5-roi",
                "h5dir": str(args.h5dir),
                "fon": args.fon,
                "foff": args.foff,
                "drop_leading": args.drop_leading,
                "roi_lower": list(args.roi_lower),
                "roi_dim": list(args.roi_dim),
                "row_y": args.row_y,
                "exclude_y": format_y_ranges(exclude_y),
            }
        else:
            loaded = load_hdf5_pair_trace(
                h5dir=args.h5dir,
                fon=args.fon,
                foff=args.foff,
                drop_leading=args.drop_leading,
                chunk_frames=args.chunk_frames,
                reference_shift=not args.no_reference_shift,
            )
            source = {
                "kind": "hdf5-pair",
                "h5dir": str(args.h5dir),
                "fon": args.fon,
                "foff": args.foff,
                "drop_leading": args.drop_leading,
            }
        return loaded.delay, loaded.ratio_minus_one, source

    from .linear_prediction import synthetic_trace

    time, trace = synthetic_trace(args.samples)
    return time, trace, {"kind": "synthetic"}


def _load_trace_npz(path: Path):
    import numpy as np

    trace_path = Path(path)
    with np.load(trace_path, allow_pickle=False) as loaded:
        if "time" not in loaded or "trace" not in loaded:
            raise ValueError(
                "trace NPZ must contain 'time' and 'trace' arrays"
            )
        time = np.asarray(loaded["time"], dtype=float)
        trace = np.asarray(loaded["trace"], dtype=float)
        source = {
            "kind": "trace-npz",
            "path": str(trace_path),
        }
        if "summary" in loaded:
            summary = np.asarray(loaded["summary"])
            if summary.shape == ():
                raw_summary = str(summary.item())
            else:
                raw_summary = json.dumps(summary.tolist())
            try:
                source["summary"] = json.loads(raw_summary)
            except json.JSONDecodeError:
                source["summary"] = raw_summary
    if time.ndim != 1 or trace.ndim != 1:
        raise ValueError("trace NPZ time and trace arrays must be 1D")
    if time.shape != trace.shape:
        raise ValueError("trace NPZ time and trace arrays must match")
    source["samples"] = int(time.shape[0])
    return time, trace, source


def _load_trace_npz_batch(paths):
    import numpy as np

    loaded = [_load_trace_npz(path) for path in paths]
    if not loaded:
        raise ValueError("at least one trace NPZ is required")
    reference_time = loaded[0][0]
    for time, _trace, source in loaded[1:]:
        if time.shape != reference_time.shape or not np.allclose(
            time,
            reference_time,
        ):
            raise ValueError(
                "all trace NPZ files must use the same time array"
            )
        source["time_matches_first"] = True
    traces = np.stack([trace for _time, trace, _source in loaded])
    sources = tuple(source for _time, _trace, source in loaded)
    return reference_time, traces, sources


def _trace_range_scale(trace) -> float:
    import numpy as np

    values = np.asarray(trace, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    scale = float(np.max(finite) - np.min(finite))
    if scale > 1.0e-12:
        return scale
    max_abs = float(np.max(np.abs(finite)))
    return max(max_abs, 1.0)


def _finite_ratio(numerator: float, denominator: float) -> float:
    denominator = abs(float(denominator))
    numerator = abs(float(numerator))
    floor = 1.0e-12
    if denominator <= floor:
        if numerator <= floor:
            return 0.0
        return numerator / floor
    return numerator / denominator
