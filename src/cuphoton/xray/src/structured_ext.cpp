// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <new>
#include <stdexcept>

#include "structured_api.h"

namespace {

// Retain exporters while native workers run, and release with Python attached.
struct Buffer {
    Py_buffer view{};
    ~Buffer() {
        if (view.obj) {
            PyBuffer_Release(&view);
        }
    }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    bool acquire(
        PyObject* object,
        const char* format,
        Py_ssize_t itemsize,
        int dimensions,
        bool writable) {
        if (PyObject_GetBuffer(
                object,
                &view,
                PyBUF_FORMAT | PyBUF_C_CONTIGUOUS
                    | (writable ? PyBUF_WRITABLE : 0))
            < 0) {
            return false;
        }
        if (view.ndim != dimensions || view.itemsize != itemsize || !view.format
            || std::strcmp(view.format, format) != 0
            || reinterpret_cast<std::uintptr_t>(view.buf) % itemsize != 0) {
            PyErr_SetString(
                PyExc_ValueError,
                "Unexpected native xRay buffer dtype, dimensions or alignment");
            return false;
        }
        return true;
    }

    bool shape(std::initializer_list<Py_ssize_t> expected) const {
        int axis = 0;
        for (const auto size : expected) {
            if (view.shape[axis++] != size) {
                PyErr_SetString(
                    PyExc_ValueError, "Unexpected native xRay buffer shape");
                return false;
            }
        }
        return true;
    }
};

bool distinct(std::initializer_list<const Buffer*> buffers) {
    for (auto a = buffers.begin(); a != buffers.end(); ++a) {
        auto start = reinterpret_cast<std::uintptr_t>((*a)->view.buf);
        for (auto b = a + 1; b != buffers.end(); ++b) {
            auto other = reinterpret_cast<std::uintptr_t>((*b)->view.buf);
            if (start < other + (*b)->view.len
                && other < start + (*a)->view.len) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Native xRay input and output buffers must not overlap");
                return false;
            }
        }
    }
    return true;
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

PyObject* solve_roots_impl(PyObject* arguments) {
    PyObject *coefficients, *roots, *stats, *errors;
    int workers, cap;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOOii",
            &coefficients,
            &roots,
            &stats,
            &errors,
            &workers,
            &cap)) {
        return nullptr;
    }
    Buffer a, r, s, e;
    if (!a.acquire(coefficients, "d", 8, 2, false)
        || !r.acquire(roots, "d", 8, 3, true)
        || !s.acquire(stats, "i", 4, 2, true)
        || !e.acquire(errors, "B", 1, 2, true)) {
        return nullptr;
    }
    const auto batch = a.view.shape[0], degree = a.view.shape[1] - 1;
    if (degree < 1 || degree > 4096 || batch < 1 || batch > INT32_MAX
        || workers < 1 || workers > 256 || cap < -1) {
        PyErr_SetString(
            PyExc_ValueError, "Invalid native xRay root dimensions or limits");
        return nullptr;
    }
    if (!r.shape({batch, degree, 2}) || !s.shape({batch, 8})
        || !e.shape({batch, 256}) || !distinct({&a, &r, &s, &e})) {
        return nullptr;
    }
    int code;
    {
        Detached detached;
        code = cuphoton::xray::solve_roots(
            static_cast<const double*>(a.view.buf),
            static_cast<double*>(r.view.buf),
            static_cast<std::int32_t*>(s.view.buf),
            static_cast<char*>(e.view.buf),
            static_cast<std::int32_t>(degree),
            static_cast<std::int32_t>(batch),
            cap,
            workers);
    }
    if (code != 0) {
        PyErr_SetString(
            PyExc_RuntimeError, "Native xRay root worker dispatch failed");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* refine_selected_impl(PyObject* arguments) {
    PyObject *coefficients, *seeds, *separation, *output, *history, *status;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOOOO",
            &coefficients,
            &seeds,
            &separation,
            &output,
            &history,
            &status)) {
        return nullptr;
    }
    Buffer a, z, sep, out, h, st;
    if (!a.acquire(coefficients, "d", 8, 1, false)
        || !z.acquire(seeds, "d", 8, 2, false)
        || !sep.acquire(separation, "d", 8, 1, false)
        || !out.acquire(output, "d", 8, 2, true)
        || !h.acquire(history, "d", 8, 3, true)
        || !st.acquire(status, "i", 4, 2, true)) {
        return nullptr;
    }
    const auto count = z.view.shape[0];
    if (count < 1 || count > 36) {
        PyErr_SetString(
            PyExc_ValueError, "Native xRay refinement requires 1..36 seeds");
        return nullptr;
    }
    if (!a.shape({598}) || !z.shape({count, 2}) || !sep.shape({count})
        || !out.shape({count, 2}) || !h.shape({count, 8, 8})
        || !st.shape({count, 2}) || !distinct({&a, &z, &sep, &out, &h, &st})) {
        return nullptr;
    }
    int code;
    {
        Detached detached;
        code = cuphoton::xray::refine_selected(
            static_cast<const double*>(a.view.buf),
            static_cast<const double*>(z.view.buf),
            static_cast<const double*>(sep.view.buf),
            static_cast<double*>(out.view.buf),
            static_cast<double*>(h.view.buf),
            static_cast<std::int32_t*>(st.view.buf),
            597,
            static_cast<std::int32_t>(count));
    }
    if (code != 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native xRay refinement requires nearest rounding and valid buffers");
        return nullptr;
    }
    Py_RETURN_NONE;
}

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
        PyErr_SetString(PyExc_RuntimeError, "Unknown native xRay error");
    }
    return nullptr;
}

PyObject* solve_roots(PyObject*, PyObject* arguments) {
    try {
        return solve_roots_impl(arguments);
    } catch (...) {
        return translate_exception();
    }
}

PyObject* refine_selected(PyObject*, PyObject* arguments) {
    try {
        return refine_selected_impl(arguments);
    } catch (...) {
        return translate_exception();
    }
}

PyMethodDef methods[] = {
    {"solve_roots",
        solve_roots,
        METH_VARARGS,
        "Solve independent real polynomials with the GIL released."},
    {"refine_selected",
        refine_selected,
        METH_VARARGS,
        "Refine selected degree-597 roots with double-double arithmetic."},
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
    "_structured_ext",
    "Native structured xRay roots and mandatory double-double refinement.",
    0,
    methods,
    slots,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__structured_ext() {
    return PyModuleDef_Init(&module);
}
