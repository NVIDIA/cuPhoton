/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime_api.h>

#include "native_api.h"

#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuphoton::xfit::Batch;
using cuphoton::xfit::KernelPolicy;
using cuphoton::xfit::Settings;
using cuphoton::xfit::Timings;
using cuphoton::xfit::Workspace;

constexpr const char* capsule_name = "cuphoton.xfit.Workspace";

// Destruction of these owners always occurs with the Python state attached.
struct PythonOwner {
    PyObject* value;
    explicit PythonOwner(PyObject* object)
        : value(object) {}
    ~PythonOwner() {
        Py_XDECREF(value);
    }
    PythonOwner(const PythonOwner&) = delete;
    PythonOwner& operator=(const PythonOwner&) = delete;
};

// RAII also restores Python state if a CUDA or allocation operation throws.
struct Detached {
    PyThreadState* state = PyEval_SaveThread();
    ~Detached() {
        PyEval_RestoreThread(state);
    }
    Detached(const Detached&) = delete;
    Detached& operator=(const Detached&) = delete;
    Detached() = default;
};

struct State {
    int device;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    Workspace workspace;
    std::unique_ptr<cuphoton::xfit::GaussianPipeline> pipeline;
    explicit State(int id, KernelPolicy policy = KernelPolicy::Automatic)
        : device(id),
          workspace(id, policy) {}
};

struct BusyGuard {
    State* state;
    ~BusyGuard() {
        state->busy.clear(std::memory_order_release);
    }
};

PyObject* translate_exception() {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::invalid_argument& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown native xFit error");
    }
    return nullptr;
}

void destroy_workspace(PyObject* capsule) {
    auto* state =
        static_cast<State*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (state == nullptr) {
        PyErr_WriteUnraisable(capsule);
        return;
    }
    // An in-flight call owns the capsule through its argument tuple, so its
    // destructor cannot race run(). Workspace destruction never calls Python.
    Detached detached;
    delete state;
}

constexpr const char* policy_names[] = {
    "auto", "portable", "test_sm120", "test_sm121"};

bool parse_policy(const char* name, KernelPolicy& policy) {
    for (int i = 0; i < 4; ++i) {
        if (std::strcmp(name, policy_names[i]) == 0) {
            policy = static_cast<KernelPolicy>(i);
            return true;
        }
    }
    PyErr_SetString(PyExc_ValueError, "unknown native xFit test kernel policy");
    return false;
}

PyObject* make_workspace(long device, KernelPolicy policy) {
    if (device < 0 || device > INT_MAX) {
        PyErr_SetString(
            PyExc_ValueError, "device must be a nonnegative integer");
        return nullptr;
    }
    State* state = nullptr;
    try {
        Detached detached;
        state = new State(static_cast<int>(device), policy);
    } catch (...) {
        return translate_exception();
    }
    PyObject* capsule = PyCapsule_New(state, capsule_name, destroy_workspace);
    if (capsule == nullptr) {
        Detached detached;
        delete state;
    }
    return capsule;
}

PyObject* create_workspace(PyObject*, PyObject* argument) {
    long device = PyLong_AsLong(argument);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return make_workspace(device, KernelPolicy::Automatic);
}

PyObject* create_test_workspace(PyObject*, PyObject* arguments) {
    long device;
    const char* name;
    if (!PyArg_ParseTuple(
            arguments, "ls:_create_test_workspace", &device, &name)) {
        return nullptr;
    }
    KernelPolicy policy;
    if (!parse_policy(name, policy)) {
        return nullptr;
    }
    return make_workspace(device, policy);
}

PyObject* test_kernel_policy(PyObject*, PyObject* arguments) {
    const char* name;
    int major, minor, version;
    if (!PyArg_ParseTuple(
            arguments,
            "siii:_test_kernel_policy",
            &name,
            &major,
            &minor,
            &version)) {
        return nullptr;
    }
    KernelPolicy policy;
    if (!parse_policy(name, policy)) {
        return nullptr;
    }
    const auto selected =
        cuphoton::xfit::select_kernels(policy, major, minor, version);
    return Py_BuildValue(
        "{s:O,s:O,s:O}",
        "tiled_jacobian",
        selected.tiled_jacobian ? Py_True : Py_False,
        "fused_trial_residual",
        selected.fused_trial_residual ? Py_True : Py_False,
        "small_hessian",
        selected.small_hessian ? Py_True : Py_False);
}

struct Array {
    std::uintptr_t pointer = 0;
    std::size_t bytes = 0;
    int itemsize = 0;
    char kind = 0;
    bool writable = false;
    std::vector<Py_ssize_t> shape;
    std::vector<Py_ssize_t> strides;
};

// The interface dictionary is copied under CPython's dictionary lock. All
// remaining borrowed items belong to that private snapshot or immutable tuples.
bool read_array(
    PyObject* owner,
    Array& result,
    std::uintptr_t producer,
    bool strided = false) {
    PythonOwner interface(
        PyObject_GetAttrString(owner, "__cuda_array_interface__"));
    if (interface.value == nullptr) {
        return false;
    }
    if (!PyDict_Check(interface.value)) {
        PyErr_SetString(
            PyExc_TypeError, "CUDA array interface must be a dictionary");
        return false;
    }
    PythonOwner snapshot(PyDict_Copy(interface.value));
    if (snapshot.value == nullptr) {
        return false;
    }
    auto get = [&](const char* key) {
        return PyDict_GetItemString(snapshot.value, key);
    };
    PyObject* version = get("version");
    long version_number = version == nullptr ? -1 : PyLong_AsLong(version);
    if (PyErr_Occurred()) {
        return false;
    }
    if (version_number != 2 && version_number != 3) {
        PyErr_SetString(
            PyExc_ValueError, "CUDA array interface version must be 2 or 3");
        return false;
    }
    PyObject* mask = get("mask");
    if (mask != nullptr && mask != Py_None) {
        PyErr_SetString(
            PyExc_ValueError, "CUDA interface masks are unsupported");
        return false;
    }
    PyObject* typestr = get("typestr");
    if (typestr == nullptr || !PyUnicode_Check(typestr)) {
        PyErr_SetString(PyExc_TypeError, "CUDA array typestr must be a string");
        return false;
    }
    const char* dtype = PyUnicode_AsUTF8(typestr);
    if (dtype == nullptr) {
        return false;
    }
    int length = int(std::strlen(dtype));
    if ((length != 3 && length != 4) || (dtype[0] != '<' && dtype[0] != '|')) {
        PyErr_SetString(
            PyExc_ValueError, "unsupported CUDA array dtype or byte order");
        return false;
    }
    result.kind = dtype[1];
    result.itemsize =
        length == 4 ? 10 * (dtype[2] - '0') + dtype[3] - '0' : dtype[2] - '0';
    bool supported =
        (result.kind == 'f' && (result.itemsize == 4 || result.itemsize == 8))
        || (result.kind == 'i'
            && (result.itemsize == 1 || result.itemsize == 4
                || result.itemsize == 8))
        || (result.kind == 'b' && result.itemsize == 1);
    if (strided) {
        supported = supported || (result.kind == 'f' && result.itemsize == 2)
            || ((result.kind == 'i' || result.kind == 'u')
                && (result.itemsize == 1 || result.itemsize == 2
                    || result.itemsize == 4 || result.itemsize == 8))
            || (result.kind == 'c'
                && (result.itemsize == 8 || result.itemsize == 16));
    }
    if (!supported) {
        PyErr_SetString(PyExc_ValueError, "unsupported CUDA array dtype");
        return false;
    }
    PyObject* shape = get("shape");
    PyObject* data = get("data");
    if (shape == nullptr || !PyTuple_Check(shape) || PyTuple_GET_SIZE(shape) > 4
        || data == nullptr || !PyTuple_Check(data)
        || PyTuple_GET_SIZE(data) != 2) {
        PyErr_SetString(
            PyExc_ValueError, "invalid CUDA array shape or data tuple");
        return false;
    }
    result.bytes = result.itemsize;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(shape); ++i) {
        Py_ssize_t dimension = PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, i));
        if (PyErr_Occurred()) {
            return false;
        }
        if (dimension < 0
            || static_cast<std::size_t>(dimension)
                > std::numeric_limits<std::size_t>::max() / result.bytes) {
            PyErr_SetString(
                PyExc_ValueError, "CUDA array shape is out of range");
            return false;
        }
        result.shape.push_back(dimension);
        result.bytes *= dimension;
        if (result.bytes == 0) {
            PyErr_SetString(
                PyExc_ValueError, "native xFit requires nonempty arrays");
            return false;
        }
    }
    auto pointer = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(data, 0));
    int readonly = PyObject_IsTrue(PyTuple_GET_ITEM(data, 1));
    if (PyErr_Occurred() || readonly < 0) {
        return false;
    }
    result.pointer = static_cast<std::uintptr_t>(pointer);
    result.writable = !readonly;
    if (!result.pointer || result.pointer % result.itemsize != 0
        || result.pointer
            > std::numeric_limits<std::uintptr_t>::max() - result.bytes) {
        PyErr_SetString(
            PyExc_ValueError, "invalid or unaligned CUDA array pointer");
        return false;
    }
    result.strides.resize(result.shape.size());
    Py_ssize_t contiguous_stride = result.itemsize;
    for (std::size_t i = result.shape.size(); i-- > 0;) {
        result.strides[i] = contiguous_stride;
        contiguous_stride *= result.shape[i];
    }
    PyObject* strides = get("strides");
    if (strides != nullptr && strides != Py_None) {
        if (!PyTuple_Check(strides)
            || PyTuple_GET_SIZE(strides) != PyTuple_GET_SIZE(shape)) {
            PyErr_SetString(PyExc_ValueError, "invalid CUDA array strides");
            return false;
        }
        std::size_t expected = result.itemsize;
        for (Py_ssize_t i = PyTuple_GET_SIZE(shape); i-- > 0;) {
            Py_ssize_t stride = PyLong_AsSsize_t(PyTuple_GET_ITEM(strides, i));
            if (PyErr_Occurred()) {
                return false;
            }
            result.strides[i] = stride;
            if (!strided && result.shape[i] > 1
                && (stride < 0
                    || static_cast<std::size_t>(stride) != expected)) {
                PyErr_SetString(
                    PyExc_ValueError, "CUDA arrays must be C contiguous");
                return false;
            }
            expected *= result.shape[i];
        }
    }
    PyObject* stream = get("stream");
    if (stream != nullptr && stream != Py_None) {
        auto producer_value = PyLong_AsUnsignedLongLong(stream);
        if (PyErr_Occurred()) {
            return false;
        }
        if (producer_value != (producer == 0 ? 1 : producer)) {
            PyErr_SetString(
                PyExc_ValueError,
                "CUDA arrays must be ready on the supplied producer stream");
            return false;
        }
    }
    return true;
}

bool same_shape(const Array& array, std::initializer_list<Py_ssize_t> shape) {
    return array.shape == std::vector<Py_ssize_t>(shape);
}

void check_device(const Array& array, int device) {
    cudaPointerAttributes attributes{};
    cudaError_t error = cudaPointerGetAttributes(
        &attributes, reinterpret_cast<const void*>(array.pointer));
    if (error != cudaSuccess) {
        cudaGetLastError();
        throw std::invalid_argument(
            std::string("invalid CUDA array pointer: ")
            + cudaGetErrorString(error));
    }
    if (attributes.type != cudaMemoryTypeDevice
        || attributes.device != device) {
        throw std::invalid_argument(
            "CUDA arrays must be device memory on the workspace device");
    }
}

PyObject* run_impl(PyObject* arguments) {
    PyObject *capsule, *owners, *configuration, *dimensions;
    PyObject* producer_object;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOOO:run",
            &capsule,
            &owners,
            &configuration,
            &dimensions,
            &producer_object)) {
        return nullptr;
    }
    unsigned long long producer = PyLong_AsUnsignedLongLong(producer_object);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    if (!PyTuple_Check(owners) || PyTuple_GET_SIZE(owners) != 7
        || !PyTuple_Check(configuration) || !PyTuple_Check(dimensions)) {
        PyErr_SetString(
            PyExc_TypeError,
            "run expects seven arrays and settings/shape tuples");
        return nullptr;
    }
    Settings settings{};
    Batch batch{};
    if (!PyArg_ParseTuple(
            configuration,
            "ddddddi|p",
            &settings.f_tol,
            &settings.x_tol,
            &settings.g_tol,
            &settings.initial_damping,
            &settings.damping_increase,
            &settings.damping_decrease,
            &settings.max_evaluations,
            &settings.legacy)
        || !PyArg_ParseTuple(
            dimensions, "iii", &batch.height, &batch.width, &batch.planes)) {
        return nullptr;
    }
    if (!std::isfinite(settings.f_tol) || settings.f_tol < 0
        || !std::isfinite(settings.x_tol) || settings.x_tol < 0
        || !std::isfinite(settings.g_tol) || settings.g_tol < 0
        || !std::isfinite(settings.initial_damping)
        || settings.initial_damping <= 0
        || !std::isfinite(settings.damping_increase)
        || settings.damping_increase <= 1
        || !std::isfinite(settings.damping_decrease)
        || settings.damping_decrease <= 0 || settings.damping_decrease >= 1
        || settings.max_evaluations < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid native LM settings");
        return nullptr;
    }
    if (batch.height < 1 || batch.width < 1
        || (batch.planes != 1 && batch.planes != 3)
        || batch.height > INT_MAX / batch.width / batch.planes) {
        PyErr_SetString(PyExc_ValueError, "invalid native image dimensions");
        return nullptr;
    }
    std::array<Array, 7> arrays;
    for (int i = 0; i < 7; ++i) {
        if (!read_array(PyTuple_GET_ITEM(owners, i), arrays[i], producer)) {
            return nullptr;
        }
    }
    if (arrays[0].shape.size() != 2 || arrays[0].shape[1] != 8
        || arrays[0].shape[0] > INT_MAX / 64 || arrays[0].kind != 'f') {
        PyErr_SetString(
            PyExc_ValueError, "x must have floating shape (count, 8)");
        return nullptr;
    }
    batch.count = static_cast<int>(arrays[0].shape[0]);
    batch.fp64 = arrays[0].itemsize == 8;
    const Py_ssize_t observations = batch.height * batch.width * batch.planes;
    for (int i = 1; i < 4; ++i) {
        bool valid_shape = same_shape(arrays[i], {batch.count, observations})
            || same_shape(
                arrays[i],
                {batch.count, batch.planes, batch.height, batch.width})
            || (batch.planes == 1
                && same_shape(
                    arrays[i], {batch.count, batch.height, batch.width}));
        if (!valid_shape || arrays[i].kind != 'f'
            || arrays[i].itemsize != arrays[0].itemsize) {
            PyErr_SetString(
                PyExc_ValueError,
                "image buffers must match x dtype and batch/image shape");
            return nullptr;
        }
    }
    for (int i = 4; i < 7; ++i) {
        if (!same_shape(arrays[i], {batch.count})
            || arrays[i].kind != (i == 6 ? 'b' : 'i')
            || arrays[i].itemsize != (i == 5 ? 4 : 1)) {
            PyErr_SetString(
                PyExc_ValueError,
                "status/evaluations/current must be int8/int32/bool vectors");
            return nullptr;
        }
    }
    for (int i : {0, 3, 4, 5, 6}) {
        if (!arrays[i].writable) {
            PyErr_SetString(
                PyExc_ValueError, "native xFit output arrays must be writable");
            return nullptr;
        }
        for (int j = 0; j < 7; ++j) {
            if (i != j
                && arrays[i].pointer < arrays[j].pointer + arrays[j].bytes
                && arrays[j].pointer < arrays[i].pointer + arrays[i].bytes) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native xFit writable buffers must not overlap");
                return nullptr;
            }
        }
    }
    auto* state =
        static_cast<State*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (state == nullptr) {
        return nullptr;
    }
    if (state->busy.test_and_set(std::memory_order_acquire)) {
        PyErr_SetString(
            PyExc_RuntimeError, "native xFit workspace is already in use");
        return nullptr;
    }
    BusyGuard guard{state};
    batch.x = reinterpret_cast<void*>(arrays[0].pointer);
    batch.images = reinterpret_cast<const void*>(arrays[1].pointer);
    batch.weights = reinterpret_cast<const void*>(arrays[2].pointer);
    batch.residuals = reinterpret_cast<void*>(arrays[3].pointer);
    batch.status = reinterpret_cast<std::int8_t*>(arrays[4].pointer);
    batch.evaluations = reinterpret_cast<std::int32_t*>(arrays[5].pointer);
    batch.diagnostic_current = reinterpret_cast<bool*>(arrays[6].pointer);
    Timings timings{};
    {
        // arguments owns capsule and the immutable owners tuple throughout;
        // run drains submitted work on success and on every exception path.
        Detached detached;
        for (const auto& array : arrays) {
            check_device(array, state->device);
        }
        timings = state->workspace.run(batch, settings, producer);
    }
    if (PyErr_CheckSignals() < 0) {
        return nullptr;
    }
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:i,s:s,s:i,s:i,s:i,s:i,s:i}",
        "host_seconds",
        timings.host_seconds,
        "synchronize_seconds",
        timings.synchronize_seconds,
        "gpu_milliseconds",
        static_cast<double>(timings.gpu_milliseconds),
        "iterations",
        timings.iterations,
        "kernel_policy",
        policy_names[static_cast<int>(timings.kernel_policy)],
        "cublas_version",
        timings.cublas_version,
        "tiled_jacobian_launches",
        timings.tiled_jacobian_launches,
        "fused_trial_residual_launches",
        timings.fused_trial_residual_launches,
        "hessian_single_launches",
        timings.hessian_single_launches,
        "hessian_batch_launches",
        timings.hessian_batch_launches);
}

PyObject* run(PyObject*, PyObject* arguments) {
    try {
        return run_impl(arguments);
    } catch (...) {
        return translate_exception();
    }
}

PyObject* fit_impl(PyObject* arguments) {
    PyObject *capsule, *inputs, *outputs, *configuration, *dimensions, *split,
        *producer_object;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOOOOO:fit",
            &capsule,
            &inputs,
            &outputs,
            &configuration,
            &dimensions,
            &split,
            &producer_object)) {
        return nullptr;
    }
    auto producer = PyLong_AsUnsignedLongLong(producer_object);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    if (!PyTuple_Check(inputs) || PyTuple_GET_SIZE(inputs) != 4
        || !PyTuple_Check(outputs) || PyTuple_GET_SIZE(outputs) != 29
        || !PyTuple_Check(configuration) || !PyTuple_Check(dimensions)
        || !PyTuple_Check(split)) {
        PyErr_SetString(
            PyExc_TypeError,
            "fit expects four inputs, 29 outputs and settings tuples");
        return nullptr;
    }
    cuphoton::xfit::GaussianFit fit{};
    Settings settings{};
    if (!PyArg_ParseTuple(
            configuration,
            "ddddddi|p",
            &settings.f_tol,
            &settings.x_tol,
            &settings.g_tol,
            &settings.initial_damping,
            &settings.damping_increase,
            &settings.damping_decrease,
            &settings.max_evaluations,
            &settings.legacy)
        || !PyArg_ParseTuple(
            dimensions,
            "iiii",
            &fit.count,
            &fit.height,
            &fit.width,
            &fit.planes)
        || !PyArg_ParseTuple(
            split,
            "ddd",
            &fit.split_weights[0],
            &fit.split_weights[1],
            &fit.split_weights[2])) {
        return nullptr;
    }
    if (fit.count < 1 || fit.count > INT_MAX / 64 || fit.height < 1
        || fit.width < 1 || (fit.planes != 1 && fit.planes != 3)
        || fit.height > INT_MAX / fit.width / fit.planes) {
        PyErr_SetString(PyExc_ValueError, "invalid native image dimensions");
        return nullptr;
    }
    if (!std::isfinite(settings.f_tol) || settings.f_tol < 0
        || !std::isfinite(settings.x_tol) || settings.x_tol < 0
        || !std::isfinite(settings.g_tol) || settings.g_tol < 0
        || !std::isfinite(settings.initial_damping)
        || settings.initial_damping <= 0
        || !std::isfinite(settings.damping_increase)
        || settings.damping_increase <= 1
        || !std::isfinite(settings.damping_decrease)
        || settings.damping_decrease <= 0 || settings.damping_decrease >= 1
        || settings.max_evaluations < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid native LM settings");
        return nullptr;
    }
    for (double weight : fit.split_weights) {
        if (!std::isfinite(weight) || weight < 0) {
            PyErr_SetString(PyExc_ValueError, "invalid native split weights");
            return nullptr;
        }
    }
    std::array<Array, 4> input_arrays;
    cuphoton::xfit::Input* descriptors[] = {
        &fit.images, &fit.mask, &fit.variance, &fit.initial};
    for (int i = 0; i < 4; ++i) {
        PyObject* value = PyTuple_GET_ITEM(inputs, i);
        if (value == Py_None && i > 0) {
            continue;
        }
        if (!read_array(value, input_arrays[i], producer, true)) {
            return nullptr;
        }
        auto& array = input_arrays[i];
        if (array.shape.size() != 4) {
            PyErr_SetString(
                PyExc_ValueError,
                "native inputs require four-dimensional broadcast views");
            return nullptr;
        }
        int expected[] = {fit.count, fit.planes, fit.height, fit.width};
        if (i == 3) {
            expected[1] = expected[2] = 1;
            expected[3] = 8;
        }
        for (int axis = 0; axis < 4; ++axis) {
            if (array.shape[axis] != 1 && array.shape[axis] != expected[axis]) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native input broadcast shape does not match batch");
                return nullptr;
            }
            descriptors[i]->shape[axis] = int(array.shape[axis]);
            descriptors[i]->strides[axis] = array.strides[axis];
        }
        descriptors[i]->data = reinterpret_cast<const void*>(array.pointer);
        descriptors[i]->kind = array.kind;
        descriptors[i]->itemsize = array.itemsize;
    }
    std::array<Array, 29> arrays;
    int m = fit.height * fit.width * fit.planes;
    for (int i = 0; i < 29; ++i) {
        if (!read_array(PyTuple_GET_ITEM(outputs, i), arrays[i], producer)) {
            return nullptr;
        }
        auto& array = arrays[i];
        if (i == 0) {
            if (array.kind != 'f'
                || (array.itemsize != 4 && array.itemsize != 8)) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native fit outputs require float32 or float64");
                return nullptr;
            }
            fit.fp64 = array.itemsize == 8;
        }
        char kind = 'f';
        int itemsize = fit.fp64 ? 8 : 4;
        if (i == 4 || i == 25) {
            kind = 'i';
            itemsize = 1;
        }
        if (i == 5 || i == 28) {
            kind = 'i';
            itemsize = 4;
        }
        if (i == 6 || i == 7 || i == 10 || i == 24) {
            kind = 'b';
            itemsize = 1;
        }
        if (i == 11 || i == 14 || i == 19) {
            kind = 'i';
            itemsize = 8;
        }
        if (i == 15 || i == 18 || i == 20) {
            itemsize = 8;
        }
        bool shape = same_shape(array, {fit.count});
        if (i == 0 || i == 8 || i == 23 || i == 26) {
            shape = same_shape(array, {fit.count, 8});
        }
        if (i == 1 || i == 2 || i == 3 || i == 7 || i == 9) {
            shape = same_shape(array, {fit.count, m});
        }
        if (i == 21 || i == 22 || i == 27) {
            shape = same_shape(array, {fit.count, 8, 8});
        }
        if (!shape || array.kind != kind || array.itemsize != itemsize
            || !array.writable) {
            PyErr_SetString(
                PyExc_ValueError,
                "native fit output shape, dtype or writeability mismatch");
            return nullptr;
        }
        fit.outputs[i] = reinterpret_cast<void*>(array.pointer);
    }
    for (int i = 0; i < 29; ++i) {
        for (int j = i + 1; j < 29; ++j) {
            if (arrays[i].pointer < arrays[j].pointer + arrays[j].bytes
                && arrays[j].pointer < arrays[i].pointer + arrays[i].bytes) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native fit output buffers must not overlap");
                return nullptr;
            }
        }
    }
    for (const auto& input : input_arrays) {
        if (!input.pointer) {
            continue;
        }
        std::int64_t low = 0, high = 0;
        for (std::size_t axis = 0; axis < input.shape.size(); ++axis) {
            auto span = (input.shape[axis] - 1) * input.strides[axis];
            if (span < 0) {
                low += span;
            } else {
                high += span;
            }
        }
        auto first = input.pointer + low;
        auto last = input.pointer + high + input.itemsize;
        for (const auto& output : arrays) {
            if (output.pointer < last
                && first < output.pointer + output.bytes) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native fit outputs must not overlap inputs");
                return nullptr;
            }
        }
    }
    auto* state =
        static_cast<State*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!state) {
        return nullptr;
    }
    if (state->busy.test_and_set(std::memory_order_acquire)) {
        PyErr_SetString(
            PyExc_RuntimeError, "native xFit workspace is already in use");
        return nullptr;
    }
    BusyGuard guard{state};
    Timings timings{};
    {
        Detached detached;
        for (const auto& array : input_arrays) {
            if (array.pointer) {
                check_device(array, state->device);
            }
        }
        for (const auto& array : arrays) {
            check_device(array, state->device);
        }
        if (!state->pipeline) {
            state->pipeline =
                std::make_unique<cuphoton::xfit::GaussianPipeline>(
                    state->device);
        }
        timings =
            state->pipeline->run(state->workspace, fit, settings, producer);
    }
    if (PyErr_CheckSignals() < 0) {
        return nullptr;
    }
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:i,s:s,s:i,s:i,s:i,s:i,s:i,s:O}",
        "host_seconds",
        timings.host_seconds,
        "synchronize_seconds",
        timings.synchronize_seconds,
        "gpu_milliseconds",
        double(timings.gpu_milliseconds),
        "iterations",
        timings.iterations,
        "kernel_policy",
        policy_names[int(timings.kernel_policy)],
        "cublas_version",
        timings.cublas_version,
        "tiled_jacobian_launches",
        timings.tiled_jacobian_launches,
        "fused_trial_residual_launches",
        timings.fused_trial_residual_launches,
        "hessian_single_launches",
        timings.hessian_single_launches,
        "hessian_batch_launches",
        timings.hessian_batch_launches,
        "native_pipeline",
        Py_True);
}

PyObject* fit(PyObject*, PyObject* arguments) {
    try {
        return fit_impl(arguments);
    } catch (...) {
        return translate_exception();
    }
}

PyMethodDef methods[] = {
    {"fit",
        fit,
        METH_VARARGS,
        "Prepare, solve and finalize a Gaussian fit with precompiled CUDA."},
    {"create_workspace",
        create_workspace,
        METH_O,
        "Create one worker's CUDA workspace."},
    {"_create_test_workspace",
        create_test_workspace,
        METH_VARARGS,
        "Create an isolated test-policy workspace using the real CUDA provider."},
    {"_test_kernel_policy",
        test_kernel_policy,
        METH_VARARGS,
        "Inspect pure kernel eligibility for supplied architecture/version inputs."},
    {"run",
        run,
        METH_VARARGS,
        "Run a synchronous batch with independent native state."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef_Slot slots[] = {
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native_ext",
    "Optional synchronous CUDA xFit; no CUDA initialization at module import.",
    0,
    methods,
    slots,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__native_ext() {
    return PyModuleDef_Init(&module);
}
