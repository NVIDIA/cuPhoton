/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuphoton::imaging {

inline void check(cudaError_t status) {
    if (status != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct Detached {
    PyThreadState* state = PyEval_SaveThread();
    ~Detached() {
        PyEval_RestoreThread(state);
    }
    Detached() = default;
    Detached(const Detached&) = delete;
    Detached& operator=(const Detached&) = delete;
};

// Arguments retain the Python owners until the stream has completed. Calls
// accept the consumer stream explicitly and join producer streams with events.
struct Array {
    void* pointer = nullptr;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::int64_t size = 1;
    int itemsize = 0;
    char kind = 0;
    bool writable = false;
    cudaStream_t producer = nullptr;
    bool has_producer = false;

    explicit Array(PyObject* owner) {
        PyObject* original =
            PyObject_GetAttrString(owner, "__cuda_array_interface__");
        if (!original) {
            PyErr_Clear();
            throw std::invalid_argument(
                "expected a CUDA Array Interface buffer");
        }
        PyObject* interface =
            PyDict_Check(original) ? PyDict_Copy(original) : nullptr;
        Py_DECREF(original);
        if (!interface) {
            PyErr_Clear();
            throw std::invalid_argument(
                "invalid CUDA Array Interface dictionary");
        }
        struct Owner {
            PyObject* value;
            ~Owner() {
                Py_DECREF(value);
            }
        } guard{interface};
        PyObject* typestr = PyDict_GetItemString(interface, "typestr");
        PyObject* dimensions = PyDict_GetItemString(interface, "shape");
        PyObject* data = PyDict_GetItemString(interface, "data");
        const char* dtype = typestr && PyUnicode_Check(typestr)
            ? PyUnicode_AsUTF8(typestr)
            : nullptr;
        if (!dtype || std::strlen(dtype) != 3
            || (dtype[0] != '<' && dtype[0] != '|' && dtype[0] != '=')
            || !dimensions || !data || !PyTuple_Check(dimensions)
            || !PyTuple_Check(data) || PyTuple_Size(data) != 2) {
            throw std::invalid_argument(
                "invalid CUDA Array Interface shape, data or dtype");
        }
        kind = dtype[1];
        try {
            itemsize = std::stoi(dtype + 2);
        } catch (...) {
            throw std::invalid_argument("invalid CUDA array dtype");
        }
        if ((kind != 'f' && kind != 'i' && kind != 'u' && kind != 'b')
            || (itemsize != 1 && itemsize != 2 && itemsize != 4
                && itemsize != 8)) {
            throw std::invalid_argument("unsupported CUDA array dtype");
        }
        pointer = PyLong_AsVoidPtr(PyTuple_GetItem(data, 0));
        writable = PyObject_IsTrue(PyTuple_GetItem(data, 1)) == 0;
        std::int64_t layout_bytes = itemsize;
        for (Py_ssize_t i = 0; i < PyTuple_Size(dimensions); ++i) {
            const auto dimension =
                PyLong_AsLongLong(PyTuple_GetItem(dimensions, i));
            if (PyErr_Occurred() || dimension < 0
                || (dimension
                    && size
                        > std::numeric_limits<std::int64_t>::max() / dimension)
                || (dimension > 0
                    && layout_bytes > std::numeric_limits<std::int64_t>::max()
                            / dimension)) {
                PyErr_Clear();
                throw std::invalid_argument("invalid CUDA array dimensions");
            }
            shape.push_back(dimension);
            size *= dimension;
            if (dimension > 0) {
                layout_bytes *= dimension;
            }
        }
        if (shape.size() > 8
            || size > std::numeric_limits<std::int64_t>::max() / itemsize
            || (size && !pointer)) {
            throw std::invalid_argument("invalid CUDA array size or pointer");
        }
        PyObject* layout = PyDict_GetItemString(interface, "strides");
        strides.resize(shape.size());
        if (layout && layout != Py_None) {
            if (!PyTuple_Check(layout)
                || PyTuple_Size(layout)
                    != static_cast<Py_ssize_t>(shape.size())) {
                throw std::invalid_argument("invalid CUDA array strides");
            }
            for (std::size_t i = 0; i < shape.size(); ++i) {
                strides[i] = PyLong_AsLongLong(PyTuple_GetItem(layout, i));
            }
        } else {
            std::int64_t stride = itemsize;
            for (std::size_t i = shape.size(); i-- > 0;) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }
        PyObject* stream = PyDict_GetItemString(interface, "stream");
        if (stream && stream != Py_None) {
            const auto value = PyLong_AsUnsignedLongLong(stream);
            if (!value || PyErr_Occurred()) {
                PyErr_Clear();
                throw std::invalid_argument("invalid CUDA producer stream");
            }
            producer = value == 1 ? cudaStreamLegacy
                : value == 2      ? cudaStreamPerThread
                                  : reinterpret_cast<cudaStream_t>(value);
            has_producer = true;
        }
        if (PyErr_Occurred()) {
            PyErr_Clear();
            throw std::invalid_argument("invalid CUDA Array Interface value");
        }
        if (size) {
            cudaPointerAttributes attributes{};
            check(cudaPointerGetAttributes(&attributes, pointer));
            int device;
            check(cudaGetDevice(&device));
            if ((attributes.type != cudaMemoryTypeDevice
                    && attributes.type != cudaMemoryTypeManaged)
                || attributes.device != device) {
                throw std::invalid_argument(
                    "buffer must reside on the active CUDA device");
            }
        }
    }

    bool contiguous() const {
        if (!size) {
            return true;
        }
        std::int64_t stride = itemsize;
        for (std::size_t i = shape.size(); i-- > 0;) {
            if (shape[i] > 1 && strides[i] != stride) {
                return false;
            }
            stride *= shape[i];
        }
        return true;
    }

    void require(
        char expected_kind,
        int expected_size,
        int dimensions,
        bool output = false) const {
        if (kind != expected_kind || itemsize != expected_size
            || (dimensions >= 0
                && shape.size() != static_cast<std::size_t>(dimensions))
            || !contiguous() || (output && !writable)) {
            throw std::invalid_argument(
                "CUDA buffer has incompatible dtype, shape, layout or writeability");
        }
    }

    void wait(cudaStream_t consumer) const {
        if (has_producer && producer != consumer) {
            cudaEvent_t event;
            check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            const auto recorded = cudaEventRecord(event, producer);
            const auto waited = recorded == cudaSuccess
                ? cudaStreamWaitEvent(consumer, event, 0)
                : recorded;
            cudaEventDestroy(event);
            check(waited);
        }
    }

    template <class T>
    T* data() const {
        return static_cast<T*>(pointer);
    }
};

template <class T>
struct DeviceBuffer {
    T* pointer = nullptr;
    std::size_t count;
    explicit DeviceBuffer(std::size_t size)
        : count(size) {
        if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        if (size) {
            check(cudaMalloc(
                reinterpret_cast<void**>(&pointer), sizeof(T) * size));
        }
    }
    ~DeviceBuffer() {
        if (pointer) {
            cudaFree(pointer);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const {
        return pointer;
    }
};

// Drain on errors too, before argument owners or intermediate buffers leave scope.
struct StreamGuard {
    cudaStream_t stream;
    explicit StreamGuard(std::uintptr_t value)
        : stream(reinterpret_cast<cudaStream_t>(value)) {}
    ~StreamGuard() {
        cudaStreamSynchronize(stream);
    }
    void finish() {
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
    }
};

inline PyObject* translate_exception() {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::invalid_argument& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown native imaging error");
    }
    return nullptr;
}

PyObject* copy_array(PyObject*, PyObject*);
PyObject* gather_bytes(PyObject*, PyObject*);
PyObject* validate_mask(PyObject*, PyObject*);
PyObject* selection_image(PyObject*, PyObject*);
PyObject* selection_candidates(PyObject*, PyObject*);
PyObject* stamp_mask(PyObject*, PyObject*);
PyObject* prepare_mapping(PyObject*, PyObject*);
PyObject* reproject(PyObject*, PyObject*);
PyObject* mapping_edge_score(PyObject*, PyObject*);
PyObject* sample_mapping(PyObject*, PyObject*);
PyObject* solve_constant_kernel(PyObject*, PyObject*);
PyObject* restore_fits_tiles(PyObject*, PyObject*);
PyObject* dequantize_fits_tiles(PyObject*, PyObject*);

}  // namespace cuphoton::imaging
