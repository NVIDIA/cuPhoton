/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvcomp_batch_ext.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <fitsio.h>
#include <kvikio/file_handle.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace xdr_gpu {

struct ReadSpan {
    std::uint64_t file_offset = 0;
    std::uint64_t nbytes = 0;
    std::uint64_t host_offset = 0;
};

struct DirectReadSpan {
    std::uint64_t file_offset = 0;
    std::uint64_t nbytes = 0;
    std::uintptr_t device_ptr = 0;
};

struct FilePlan {
    std::string path;
    std::int64_t file_index = 0;
    std::size_t total_bytes = 0;
    std::vector<ReadSpan> spans;
    std::vector<DirectReadSpan> direct_spans;
};

struct ReadyFile {
    std::int64_t file_index = 0;
    std::string path;
    std::size_t batch_offset = 0;
    std::size_t nbytes = 0;
};

struct ReadyBatch {
    std::uint64_t batch_id = 0;
    NativeDeviceAllocation device_buffer;
    std::vector<ReadyFile> files;
};

struct BatchState {
    std::uint64_t batch_id = 0;
    NativeDeviceAllocation device_buffer;
    std::vector<ReadyFile> files;
    std::size_t remaining = 0;
};

struct ReadTask {
    std::shared_ptr<BatchState> batch;
    std::size_t slot = 0;
    std::size_t batch_offset = 0;
    FilePlan plan;
};

std::uint64_t as_u64(const py::handle& obj, const char* name) {
    auto value = py::cast<std::uint64_t>(obj);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw std::overflow_error(std::string(name) + " exceeds Py_ssize_t");
    }
    return value;
}

std::vector<FilePlan> parse_file_plans(py::sequence file_plans) {
    std::vector<FilePlan> plans;
    plans.reserve(static_cast<std::size_t>(py::len(file_plans)));

    for (py::handle obj : file_plans) {
        py::sequence item = py::reinterpret_borrow<py::sequence>(obj);
        if (py::len(item) != 4 && py::len(item) != 5) {
            throw std::invalid_argument(
                "native file plans must be (path, file_index, total_bytes, spans[, direct_spans])");
        }

        FilePlan plan;
        plan.path = py::cast<std::string>(item[0]);
        plan.file_index = py::cast<std::int64_t>(item[1]);
        auto total = as_u64(item[2], "total_bytes");
        plan.total_bytes = static_cast<std::size_t>(total);

        py::sequence spans = py::reinterpret_borrow<py::sequence>(item[3]);
        plan.spans.reserve(static_cast<std::size_t>(py::len(spans)));
        for (py::handle span_obj : spans) {
            py::sequence span = py::reinterpret_borrow<py::sequence>(span_obj);
            if (py::len(span) != 3) {
                throw std::invalid_argument("native read spans must be (file_offset, nbytes, host_offset)");
            }
            ReadSpan rs;
            rs.file_offset = as_u64(span[0], "file_offset");
            rs.nbytes = as_u64(span[1], "nbytes");
            rs.host_offset = as_u64(span[2], "host_offset");
            if (rs.host_offset + rs.nbytes < rs.host_offset
                || rs.host_offset + rs.nbytes > static_cast<std::uint64_t>(plan.total_bytes)) {
                throw std::invalid_argument("read span exceeds planned host buffer size");
            }
            plan.spans.push_back(rs);
        }

        if (py::len(item) == 5) {
            py::sequence direct_spans = py::reinterpret_borrow<py::sequence>(item[4]);
            plan.direct_spans.reserve(static_cast<std::size_t>(py::len(direct_spans)));
            for (py::handle span_obj : direct_spans) {
                py::sequence span = py::reinterpret_borrow<py::sequence>(span_obj);
                if (py::len(span) != 3) {
                    throw std::invalid_argument(
                        "native direct read spans must be (file_offset, nbytes, device_ptr)");
                }
                DirectReadSpan rs;
                rs.file_offset = as_u64(span[0], "direct_file_offset");
                rs.nbytes = as_u64(span[1], "direct_nbytes");
                rs.device_ptr = py::cast<std::uintptr_t>(span[2]);
                if (rs.nbytes > 0 && rs.device_ptr == 0) {
                    throw std::invalid_argument("direct read device pointer is null");
                }
                plan.direct_spans.push_back(rs);
            }
        }
        plans.push_back(std::move(plan));
    }

    return plans;
}

struct SectionSpec {
    bool enabled = false;
    bool row_stop_set = false;
    bool col_stop_set = false;
    std::int64_t row_start = 0;
    std::int64_t row_stop = 0;
    std::int64_t col_start = 0;
    std::int64_t col_stop = 0;
};

struct NativeHduPlan {
    std::int64_t hdu_index = 0;
    std::string kind;
    std::vector<std::int64_t> tile_idx;
    std::vector<std::int64_t> sel_abs_offsets;
    std::vector<std::int64_t> sel_lengths;
    std::vector<std::int64_t> out_bytes;
    std::vector<std::int32_t> origins_r;
    std::vector<std::int32_t> origins_c;
    std::vector<std::int32_t> heights;
    std::vector<std::int32_t> widths;
    std::vector<std::int32_t> src_off_r;
    std::vector<std::int32_t> src_off_c;
    std::vector<std::int32_t> tile_full_w;
    std::vector<double> sel_zscale;
    std::vector<double> sel_zzero;
    std::vector<std::int64_t> heap_rel_offsets;
    std::uint64_t heap_span_start = 0;
    std::uint64_t heap_span_len = 0;
    std::uint64_t file_host_offset = 0;
    std::int64_t out_h = 0;
    std::int64_t out_w = 0;
    std::string compression_type;
    std::string dtype_char;
    int itemsize = 0;
    int zbitpix = 0;
    bool quantized = false;
    std::vector<std::int64_t> image_shape;
    double image_bzero = 0.0;
    double image_bscale = 1.0;
};

struct NativePlannedFile {
    std::string path;
    std::int64_t file_index = 0;
    std::size_t total_bytes = 0;
    std::vector<ReadSpan> spans;
    std::vector<NativeHduPlan> hdus;
};

struct FitsFileCloser {
    void operator()(fitsfile* fptr) const noexcept {
        if (fptr != nullptr) {
            int status = 0;
            fits_close_file(fptr, &status);
        }
    }
};

using FitsFilePtr = std::unique_ptr<fitsfile, FitsFileCloser>;

std::string fits_status_text(int status) {
    char text[FLEN_STATUS] = {0};
    fits_get_errstatus(status, text);
    return std::string(text);
}

[[noreturn]] void throw_fits_error(
    const std::string& path,
    const char* context,
    int status) {
    throw std::runtime_error(
        path + ": CFITSIO error in " + context + " (status="
        + std::to_string(status) + ", " + fits_status_text(status) + ")");
}

void check_fits(int status, const std::string& path, const char* context) {
    if (status != 0) {
        throw_fits_error(path, context, status);
    }
}

std::string uppercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return value;
}

std::pair<std::string, int> dtype_for_bitpix(int bitpix) {
    switch (bitpix) {
    case 8:
        return {"u1", 1};
    case 16:
        return {"i2", 2};
    case 32:
        return {"i4", 4};
    case 64:
        return {"i8", 8};
    case -32:
        return {"f4", 4};
    case -64:
        return {"f8", 8};
    default:
        throw std::invalid_argument("unsupported BITPIX/ZBITPIX " + std::to_string(bitpix));
    }
}

FitsFilePtr open_fits_readonly(const std::string& path) {
    fitsfile* raw = nullptr;
    int status = 0;
    fits_open_file(&raw, path.c_str(), READONLY, &status);
    check_fits(status, path, "fits_open_file");
    return FitsFilePtr(raw);
}

LONGLONG read_key_lnglng_required(fitsfile* fptr, const std::string& path, const char* key) {
    LONGLONG value = 0;
    int status = 0;
    fits_read_key_lnglng(fptr, key, &value, nullptr, &status);
    check_fits(status, path, key);
    return value;
}

std::optional<LONGLONG> read_key_lnglng_optional(
    fitsfile* fptr,
    const std::string& path,
    const char* key) {
    LONGLONG value = 0;
    int status = 0;
    fits_read_key_lnglng(fptr, key, &value, nullptr, &status);
    if (status == KEY_NO_EXIST) {
        return std::nullopt;
    }
    check_fits(status, path, key);
    return value;
}

double read_key_dbl_optional(
    fitsfile* fptr,
    const std::string& path,
    const char* key,
    double default_value) {
    double value = default_value;
    int status = 0;
    fits_read_key_dbl(fptr, key, &value, nullptr, &status);
    if (status == KEY_NO_EXIST) {
        return default_value;
    }
    check_fits(status, path, key);
    return value;
}

std::optional<std::string> read_key_str_optional(
    fitsfile* fptr,
    const std::string& path,
    const char* key) {
    char value[FLEN_VALUE] = {0};
    int status = 0;
    fits_read_key_str(fptr, key, value, nullptr, &status);
    if (status == KEY_NO_EXIST) {
        return std::nullopt;
    }
    check_fits(status, path, key);
    return std::string(value);
}

std::string read_key_str_required(fitsfile* fptr, const std::string& path, const char* key) {
    auto value = read_key_str_optional(fptr, path, key);
    if (!value) {
        throw std::runtime_error(path + ": required FITS keyword missing: " + key);
    }
    return *value;
}

bool read_key_logical_optional(
    fitsfile* fptr,
    const std::string& path,
    const char* key,
    bool default_value) {
    int value = default_value ? 1 : 0;
    int status = 0;
    fits_read_key_log(fptr, key, &value, nullptr, &status);
    if (status == KEY_NO_EXIST) {
        return default_value;
    }
    check_fits(status, path, key);
    return value != 0;
}

int get_colnum_required(fitsfile* fptr, const std::string& path, const char* name) {
    int colnum = 0;
    int status = 0;
    char colname[FLEN_VALUE] = {0};
    std::strncpy(colname, name, sizeof(colname) - 1);
    fits_get_colnum(fptr, CASEINSEN, colname, &colnum, &status);
    check_fits(status, path, name);
    return colnum;
}

std::optional<int> get_colnum_optional(fitsfile* fptr, const std::string& path, const char* name) {
    int colnum = 0;
    int status = 0;
    char colname[FLEN_VALUE] = {0};
    std::strncpy(colname, name, sizeof(colname) - 1);
    fits_get_colnum(fptr, CASEINSEN, colname, &colnum, &status);
    if (status == COL_NOT_FOUND) {
        return std::nullopt;
    }
    check_fits(status, path, name);
    return colnum;
}

void append_compact_read_spans(
    const std::vector<std::int64_t>& abs_offsets,
    const std::vector<std::int64_t>& lengths,
    std::uint64_t file_host_base,
    std::vector<ReadSpan>& file_spans,
    std::vector<std::int64_t>& rel_offsets,
    std::uint64_t& heap_span_start,
    std::uint64_t& heap_span_len) {
    if (abs_offsets.size() != lengths.size()) {
        throw std::invalid_argument("abs_offsets and lengths must have the same size");
    }
    rel_offsets.assign(abs_offsets.size(), 0);
    heap_span_start = 0;
    heap_span_len = 0;
    if (abs_offsets.empty()) {
        return;
    }

    std::vector<std::size_t> order(abs_offsets.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return abs_offsets[a] < abs_offsets[b];
    });

    std::uint64_t cursor = 0;
    std::uint64_t span_start = static_cast<std::uint64_t>(abs_offsets[order[0]]);
    std::uint64_t span_end = span_start + static_cast<std::uint64_t>(lengths[order[0]]);
    std::uint64_t span_host_offset = cursor;
    heap_span_start = span_start;
    rel_offsets[order[0]] = static_cast<std::int64_t>(span_host_offset);

    for (std::size_t pos = 1; pos < order.size(); ++pos) {
        const auto tile_pos = order[pos];
        const auto off = static_cast<std::uint64_t>(abs_offsets[tile_pos]);
        const auto length = static_cast<std::uint64_t>(lengths[tile_pos]);
        const auto end = off + length;
        if (off > span_end) {
            const auto span_len = span_end - span_start;
            file_spans.push_back(
                ReadSpan{span_start, span_len, file_host_base + span_host_offset});
            cursor += span_len;
            span_start = off;
            span_end = end;
            span_host_offset = cursor;
        }
        rel_offsets[tile_pos] =
            static_cast<std::int64_t>(span_host_offset + (off - span_start));
        if (end > span_end) {
            span_end = end;
        }
    }

    const auto span_len = span_end - span_start;
    file_spans.push_back(ReadSpan{span_start, span_len, file_host_base + span_host_offset});
    cursor += span_len;
    heap_span_len = cursor;
}

SectionSpec parse_section(py::object section_obj) {
    SectionSpec section;
    if (section_obj.is_none()) {
        return section;
    }
    if (!py::isinstance<py::tuple>(section_obj) || py::len(section_obj) != 2) {
        throw std::invalid_argument("section must be None or a tuple of two slices");
    }

    auto parse_one = [](py::handle obj,
                        std::int64_t& start,
                        std::int64_t& stop,
                        bool& stop_set,
                        const char* axis) {
        if (!py::isinstance<py::slice>(obj)) {
            throw std::invalid_argument(std::string("section ") + axis + " entry must be a slice");
        }
        py::object slice = py::reinterpret_borrow<py::object>(obj);
        py::object start_obj = slice.attr("start");
        py::object stop_obj = slice.attr("stop");
        py::object step_obj = slice.attr("step");
        start = start_obj.is_none() ? 0 : py::cast<std::int64_t>(start_obj);
        if (stop_obj.is_none()) {
            stop_set = false;
            stop = 0;
        } else {
            stop_set = true;
            stop = py::cast<std::int64_t>(stop_obj);
        }
        if (!step_obj.is_none() && py::cast<std::int64_t>(step_obj) != 1) {
            throw std::invalid_argument("strided section slices are not supported");
        }
    };

    py::tuple tuple = py::reinterpret_borrow<py::tuple>(section_obj);
    section.enabled = true;
    parse_one(tuple[0], section.row_start, section.row_stop, section.row_stop_set, "row");
    parse_one(tuple[1], section.col_start, section.col_stop, section.col_stop_set, "column");
    return section;
}

NativeHduPlan plan_image_hdu(
    fitsfile* fptr,
    const std::string& path,
    int hdu_index,
    std::uint64_t file_host_offset,
    std::vector<ReadSpan>& file_spans) {
    int status = 0;
    int bitpix = 0;
    int naxis = 0;
    LONGLONG naxes[16] = {0};
    fits_get_img_paramll(fptr, 16, &bitpix, &naxis, naxes, &status);
    check_fits(status, path, "fits_get_img_paramll");
    if (naxis < 2) {
        throw std::runtime_error(path + ": ImageHDU has fewer than 2 axes");
    }
    if (naxis > 16) {
        throw std::runtime_error(path + ": ImageHDU has too many axes");
    }

    auto dtype = dtype_for_bitpix(bitpix);
    std::uint64_t nbytes = static_cast<std::uint64_t>(dtype.second);
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(naxis));
    for (int axis = naxis; axis >= 1; --axis) {
        shape.push_back(static_cast<std::int64_t>(naxes[axis - 1]));
        nbytes *= static_cast<std::uint64_t>(naxes[axis - 1]);
    }

    LONGLONG headstart = 0;
    LONGLONG datastart = 0;
    LONGLONG dataend = 0;
    status = 0;
    fits_get_hduaddrll(fptr, &headstart, &datastart, &dataend, &status);
    check_fits(status, path, "fits_get_hduaddrll");

    NativeHduPlan hdu;
    hdu.hdu_index = hdu_index;
    hdu.kind = "image";
    hdu.heap_span_start = static_cast<std::uint64_t>(datastart);
    hdu.heap_span_len = nbytes;
    hdu.file_host_offset = file_host_offset;
    hdu.image_shape = std::move(shape);
    hdu.dtype_char = dtype.first;
    hdu.itemsize = dtype.second;
    hdu.image_bzero = read_key_dbl_optional(fptr, path, "BZERO", 0.0);
    hdu.image_bscale = read_key_dbl_optional(fptr, path, "BSCALE", 1.0);

    file_spans.push_back(ReadSpan{
        static_cast<std::uint64_t>(datastart),
        nbytes,
        file_host_offset});
    return hdu;
}

void build_tile_grid_2d(
    NativeHduPlan& hdu,
    std::int64_t H,
    std::int64_t W,
    std::int64_t th,
    std::int64_t tw,
    const SectionSpec& section) {
    std::int64_t roi_r0 = 0;
    std::int64_t roi_r1 = H;
    std::int64_t roi_c0 = 0;
    std::int64_t roi_c1 = W;
    if (section.enabled) {
        roi_r0 = section.row_start;
        roi_r1 = section.row_stop_set ? section.row_stop : H;
        roi_c0 = section.col_start;
        roi_c1 = section.col_stop_set ? section.col_stop : W;
        if (!(0 <= roi_r0 && roi_r0 < roi_r1 && roi_r1 <= H
              && 0 <= roi_c0 && roi_c0 < roi_c1 && roi_c1 <= W)) {
            throw std::runtime_error("section out of bounds for compressed image");
        }
    }

    hdu.out_h = roi_r1 - roi_r0;
    hdu.out_w = roi_c1 - roi_c0;
    const auto n_cols_in_grid = (W + tw - 1) / tw;

    for (std::int64_t tr0 = 0; tr0 < H; tr0 += th) {
        const auto t_h_full = std::min(th, H - tr0);
        for (std::int64_t tc0 = 0; tc0 < W; tc0 += tw) {
            const auto t_w_full = std::min(tw, W - tc0);
            const auto inter_r0 = std::max(tr0, roi_r0);
            const auto inter_r1 = std::min(tr0 + t_h_full, roi_r1);
            const auto inter_c0 = std::max(tc0, roi_c0);
            const auto inter_c1 = std::min(tc0 + t_w_full, roi_c1);
            if (inter_r0 >= inter_r1 || inter_c0 >= inter_c1) {
                continue;
            }
            const auto row_i = (tr0 / th) * n_cols_in_grid + (tc0 / tw);
            hdu.tile_idx.push_back(row_i);
            hdu.origins_r.push_back(static_cast<std::int32_t>(inter_r0 - roi_r0));
            hdu.origins_c.push_back(static_cast<std::int32_t>(inter_c0 - roi_c0));
            hdu.heights.push_back(static_cast<std::int32_t>(inter_r1 - inter_r0));
            hdu.widths.push_back(static_cast<std::int32_t>(inter_c1 - inter_c0));
            hdu.out_bytes.push_back(t_h_full * t_w_full * hdu.itemsize);
            hdu.src_off_r.push_back(static_cast<std::int32_t>(inter_r0 - tr0));
            hdu.src_off_c.push_back(static_cast<std::int32_t>(inter_c0 - tc0));
            hdu.tile_full_w.push_back(static_cast<std::int32_t>(t_w_full));
        }
    }
}

NativeHduPlan plan_compressed_hdu(
    fitsfile* fptr,
    const std::string& path,
    int hdu_index,
    const SectionSpec& section,
    std::uint64_t file_host_offset,
    std::vector<ReadSpan>& file_spans) {
    auto compression = uppercase_ascii(read_key_str_required(fptr, path, "ZCMPTYPE"));
    if (compression != "GZIP_1" && compression != "GZIP_2") {
        throw std::runtime_error(
            path + ": " + compression + " is not supported by the native streaming path");
    }

    const auto znaxis = read_key_lnglng_required(fptr, path, "ZNAXIS");
    if (znaxis != 2) {
        throw std::runtime_error(path + ": native compressed path supports only 2D images");
    }
    const auto W = read_key_lnglng_required(fptr, path, "ZNAXIS1");
    const auto H = read_key_lnglng_required(fptr, path, "ZNAXIS2");
    const auto tw = read_key_lnglng_required(fptr, path, "ZTILE1");
    const auto th = read_key_lnglng_required(fptr, path, "ZTILE2");
    if (W <= 0 || H <= 0) {
        throw std::runtime_error(path + ": ZNAXIS1/ZNAXIS2 must be positive");
    }
    if (tw <= 0 || th <= 0) {
        throw std::runtime_error(path + ": ZTILE1/ZTILE2 must be positive");
    }
    const auto zbitpix = static_cast<int>(read_key_lnglng_required(fptr, path, "ZBITPIX"));
    auto dtype = dtype_for_bitpix(zbitpix);

    NativeHduPlan hdu;
    hdu.hdu_index = hdu_index;
    hdu.kind = "comp";
    hdu.compression_type = compression;
    hdu.zbitpix = zbitpix;
    hdu.dtype_char = dtype.first;
    hdu.itemsize = dtype.second;
    hdu.file_host_offset = file_host_offset;

    build_tile_grid_2d(hdu, H, W, th, tw, section);

    LONGLONG headstart = 0;
    LONGLONG datastart = 0;
    LONGLONG dataend = 0;
    int status = 0;
    fits_get_hduaddrll(fptr, &headstart, &datastart, &dataend, &status);
    check_fits(status, path, "fits_get_hduaddrll");

    const auto naxis1 = read_key_lnglng_required(fptr, path, "NAXIS1");
    const auto naxis2 = read_key_lnglng_required(fptr, path, "NAXIS2");
    const auto theap = read_key_lnglng_optional(fptr, path, "THEAP").value_or(naxis1 * naxis2);
    const auto heap_base = datastart + theap;

    const int comp_col = get_colnum_required(fptr, path, "COMPRESSED_DATA");
    const auto zscale_col = get_colnum_optional(fptr, path, "ZSCALE");
    const auto zzero_col = get_colnum_optional(fptr, path, "ZZERO");
    hdu.quantized = zscale_col.has_value();
    if (hdu.quantized && !zzero_col.has_value()) {
        throw std::runtime_error(path + ": ZSCALE column exists but ZZERO is missing");
    }
    if (hdu.quantized) {
        const auto zquantiz = uppercase_ascii(
            read_key_str_optional(fptr, path, "ZQUANTIZ").value_or("NO_DITHER"));
        if (zquantiz != "NO_DITHER" && zquantiz != "NONE") {
            throw std::runtime_error(path + ": dithered ZQUANTIZ is not supported");
        }
    }

    LONGLONG nrows = 0;
    status = 0;
    fits_get_num_rowsll(fptr, &nrows, &status);
    check_fits(status, path, "fits_get_num_rowsll");

    hdu.sel_abs_offsets.reserve(hdu.tile_idx.size());
    hdu.sel_lengths.reserve(hdu.tile_idx.size());
    hdu.sel_zscale.reserve(hdu.tile_idx.size());
    hdu.sel_zzero.reserve(hdu.tile_idx.size());
    for (const auto tile : hdu.tile_idx) {
        const auto row = tile + 1;
        if (row < 1 || row > nrows) {
            throw std::runtime_error(path + ": compressed tile index exceeds table rows");
        }
        LONGLONG length = 0;
        LONGLONG heapaddr = 0;
        status = 0;
        fits_read_descriptll(fptr, comp_col, row, &length, &heapaddr, &status);
        check_fits(status, path, "fits_read_descriptll COMPRESSED_DATA");
        hdu.sel_lengths.push_back(static_cast<std::int64_t>(length));
        hdu.sel_abs_offsets.push_back(static_cast<std::int64_t>(heap_base + heapaddr));

        if (hdu.quantized) {
            int anynul = 0;
            double zscale = 0.0;
            double zzero = 0.0;
            status = 0;
            fits_read_col_dbl(fptr, *zscale_col, row, 1, 1, 0.0, &zscale, &anynul, &status);
            check_fits(status, path, "fits_read_col_dbl ZSCALE");
            status = 0;
            fits_read_col_dbl(fptr, *zzero_col, row, 1, 1, 0.0, &zzero, &anynul, &status);
            check_fits(status, path, "fits_read_col_dbl ZZERO");
            hdu.sel_zscale.push_back(zscale);
            hdu.sel_zzero.push_back(zzero);
        }
    }

    append_compact_read_spans(
        hdu.sel_abs_offsets,
        hdu.sel_lengths,
        file_host_offset,
        file_spans,
        hdu.heap_rel_offsets,
        hdu.heap_span_start,
        hdu.heap_span_len);
    return hdu;
}

NativePlannedFile plan_native_file_cfitsio(
    const std::string& path,
    std::int64_t file_index,
    const std::vector<int>& hdu_indices,
    const SectionSpec& section) {
    auto fptr = open_fits_readonly(path);
    NativePlannedFile planned;
    planned.path = path;
    planned.file_index = file_index;

    std::uint64_t cursor = 0;
    for (const auto hdu_index : hdu_indices) {
        int status = 0;
        int hdu_type = 0;
        fits_movabs_hdu(fptr.get(), hdu_index + 1, &hdu_type, &status);
        check_fits(status, path, "fits_movabs_hdu");

        NativeHduPlan hdu;
        const bool zimage = read_key_logical_optional(fptr.get(), path, "ZIMAGE", false);
        if (zimage) {
            hdu = plan_compressed_hdu(
                fptr.get(), path, hdu_index, section, cursor, planned.spans);
        } else if (hdu_type == IMAGE_HDU) {
            hdu = plan_image_hdu(fptr.get(), path, hdu_index, cursor, planned.spans);
        } else {
            throw std::runtime_error(
                path + ": HDU[" + std::to_string(hdu_index)
                + "] is not a supported ImageHDU or CompImageHDU");
        }

        cursor += hdu.heap_span_len;
        planned.hdus.push_back(std::move(hdu));
    }

    planned.total_bytes = static_cast<std::size_t>(cursor);
    return planned;
}

template <typename T>
py::array_t<T> array_from_vector(const std::vector<T>& values) {
    py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return out;
}

py::tuple tuple_from_shape(const std::vector<std::int64_t>& shape) {
    py::tuple out(static_cast<py::ssize_t>(shape.size()));
    for (std::size_t i = 0; i < shape.size(); ++i) {
        out[static_cast<py::ssize_t>(i)] = shape[i];
    }
    return out;
}

py::tuple py_hdu_plan(const NativeHduPlan& hdu) {
    py::object plan_obj = py::none();
    py::object heap_rel_obj = py::none();
    py::object image_shape_obj = py::none();
    py::object image_dtype_obj = py::none();

    if (hdu.kind == "comp") {
        py::dict plan;
        plan["tile_idx"] = array_from_vector(hdu.tile_idx);
        plan["sel_abs_offsets"] = array_from_vector(hdu.sel_abs_offsets);
        plan["sel_lengths"] = array_from_vector(hdu.sel_lengths);
        plan["origins_r"] = array_from_vector(hdu.origins_r);
        plan["origins_c"] = array_from_vector(hdu.origins_c);
        plan["heights"] = array_from_vector(hdu.heights);
        plan["widths"] = array_from_vector(hdu.widths);
        plan["out_bytes"] = array_from_vector(hdu.out_bytes);
        plan["src_off_r"] = array_from_vector(hdu.src_off_r);
        plan["src_off_c"] = array_from_vector(hdu.src_off_c);
        plan["tile_full_w"] = array_from_vector(hdu.tile_full_w);
        plan["out_shape"] = py::make_tuple(hdu.out_h, hdu.out_w);
        plan["compression_type"] = hdu.compression_type;
        plan["itemsize"] = hdu.itemsize;
        plan["dtype_char"] = hdu.dtype_char;
        plan["zbitpix"] = hdu.zbitpix;
        plan["quantized"] = hdu.quantized;
        if (hdu.quantized) {
            plan["sel_zscale"] = array_from_vector(hdu.sel_zscale);
            plan["sel_zzero"] = array_from_vector(hdu.sel_zzero);
        }
        plan_obj = std::move(plan);
        heap_rel_obj = array_from_vector(hdu.heap_rel_offsets);
    } else {
        image_shape_obj = tuple_from_shape(hdu.image_shape);
        image_dtype_obj = py::cast(hdu.dtype_char);
    }

    return py::make_tuple(
        hdu.hdu_index,
        hdu.kind,
        plan_obj,
        heap_rel_obj,
        hdu.heap_span_start,
        hdu.heap_span_len,
        image_shape_obj,
        image_dtype_obj,
        hdu.itemsize,
        hdu.image_bzero,
        hdu.image_bscale,
        hdu.file_host_offset);
}

py::tuple py_planned_file(const NativePlannedFile& planned) {
    py::list hdus;
    for (const auto& hdu : planned.hdus) {
        hdus.append(py_hdu_plan(hdu));
    }

    py::list spans;
    for (const auto& span : planned.spans) {
        spans.append(py::make_tuple(span.file_offset, span.nbytes, span.host_offset));
    }

    return py::make_tuple(
        planned.path,
        planned.file_index,
        hdus,
        spans,
        planned.total_bytes);
}

py::list plan_native_files(
    py::sequence paths_obj,
    py::sequence file_indices_obj,
    py::sequence hdu_indices_obj,
    std::size_t native_plan_threads,
    py::object section_obj) {
    if (native_plan_threads == 0) {
        throw std::invalid_argument("native_plan_threads must be >= 1");
    }
    if (!fits_is_reentrant()) {
        throw std::runtime_error("CFITSIO is not compiled in reentrant/thread-safe mode");
    }

    std::vector<std::string> paths;
    std::vector<std::int64_t> file_indices;
    std::vector<int> hdu_indices;
    paths.reserve(static_cast<std::size_t>(py::len(paths_obj)));
    file_indices.reserve(static_cast<std::size_t>(py::len(file_indices_obj)));
    hdu_indices.reserve(static_cast<std::size_t>(py::len(hdu_indices_obj)));

    if (py::len(paths_obj) != py::len(file_indices_obj)) {
        throw std::invalid_argument("paths and file_indices must have the same length");
    }
    for (py::handle obj : paths_obj) {
        paths.push_back(py::cast<std::string>(obj));
    }
    for (py::handle obj : file_indices_obj) {
        file_indices.push_back(py::cast<std::int64_t>(obj));
    }
    for (py::handle obj : hdu_indices_obj) {
        hdu_indices.push_back(py::cast<int>(obj));
    }
    if (paths.empty()) {
        return py::list();
    }
    if (hdu_indices.empty()) {
        throw std::invalid_argument("hdu_indices is empty");
    }
    const auto section = parse_section(section_obj);

    std::vector<NativePlannedFile> results(paths.size());
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;

    const auto worker_count = std::min<std::size_t>(native_plan_threads, paths.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    {
        py::gil_scoped_release release;
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&] {
                while (!failed.load(std::memory_order_relaxed)) {
                    const auto i = next.fetch_add(1);
                    if (i >= paths.size()) {
                        break;
                    }
                    try {
                        results[i] = plan_native_file_cfitsio(
                            paths[i], file_indices[i], hdu_indices, section);
                    } catch (...) {
                        bool expected = false;
                        if (failed.compare_exchange_strong(expected, true)) {
                            std::lock_guard<std::mutex> lock(error_mutex);
                            error = std::current_exception();
                        }
                        break;
                    }
                }
            });
        }
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    if (error) {
        std::rethrow_exception(error);
    }

    py::list out;
    for (const auto& planned : results) {
        out.append(py_planned_file(planned));
    }
    return out;
}

class NativeBatchBuilder {
public:
    NativeBatchBuilder(std::size_t decode_batch_files,
                      std::size_t batch_queue_depth,
                      std::size_t native_read_threads,
                      int device_id)
        : decode_batch_files_(decode_batch_files),
          batch_queue_depth_(batch_queue_depth),
          native_read_threads_(native_read_threads),
          device_id_(device_id) {
        if (decode_batch_files_ == 0) {
            throw std::invalid_argument("decode_batch_files must be >= 1");
        }
        if (batch_queue_depth_ == 0) {
            throw std::invalid_argument("batch_queue_depth must be >= 1");
        }
        if (native_read_threads_ == 0) {
            throw std::invalid_argument("native_read_threads must be >= 1");
        }
        max_outstanding_batches_ = batch_queue_depth_ + native_read_threads_;
        workers_.reserve(native_read_threads_);
        for (std::size_t i = 0; i < native_read_threads_; ++i) {
            workers_.emplace_back(&NativeBatchBuilder::worker_loop, this);
        }
    }

    NativeBatchBuilder(const NativeBatchBuilder&) = delete;
    NativeBatchBuilder& operator=(const NativeBatchBuilder&) = delete;

    ~NativeBatchBuilder() {
        request_stop();
    }

    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_not_empty_.notify_all();
        cv_submit_.notify_all();
        cv_ready_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void submit_batch(std::uint64_t batch_id, py::sequence file_plans) {
        auto plans = parse_file_plans(file_plans);
        auto batch = std::make_shared<BatchState>();
        batch->batch_id = batch_id;
        batch->files.resize(plans.size());
        batch->remaining = plans.size();
        std::size_t device_bytes = 0;
        for (const auto& plan : plans) {
            device_bytes += plan.total_bytes;
        }
        if (device_bytes > 0) {
            py::gil_scoped_release release;
            batch->device_buffer = acquire_native_device_allocation(device_id_, device_bytes);
            device_batches_.fetch_add(1);
        }

        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            cv_submit_.wait(lock, [&] {
                return stop_ || error_ || outstanding_batches_ < max_outstanding_batches_;
            });
            if (stop_ || error_) {
                return;
            }
            if (input_closed_) {
                throw std::runtime_error("submit_batch called after close_input");
            }
            if (batch_id != next_submit_batch_id_) {
                throw std::invalid_argument(
                    "batch_id must be submitted in increasing consecutive order");
            }

            ++next_submit_batch_id_;
            ++outstanding_batches_;
            if (plans.empty()) {
                auto ready = std::make_shared<ReadyBatch>();
                ready->batch_id = batch_id;
                completed_[batch_id] = std::move(ready);
                cv_ready_.notify_all();
                return;
            }

            std::size_t device_cursor = 0;
            for (std::size_t i = 0; i < plans.size(); ++i) {
                const std::size_t plan_bytes = plans[i].total_bytes;
                ReadTask task;
                task.batch = batch;
                task.slot = i;
                task.batch_offset = device_cursor;
                task.plan = std::move(plans[i]);
                pending_.push_back(std::move(task));
                device_cursor += plan_bytes;
            }
        }
        cv_not_empty_.notify_all();
    }

    void close_input() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            input_closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_ready_.notify_all();
    }

    py::object next_batch() {
        std::shared_ptr<ReadyBatch> batch;
        std::exception_ptr error;
        bool finished = false;

        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            cv_ready_.wait(lock, [&] {
                return stop_ || error_ || completed_.count(next_emit_batch_id_) > 0
                       || (input_closed_ && outstanding_batches_ == 0);
            });

            if (completed_.count(next_emit_batch_id_) > 0) {
                auto it = completed_.find(next_emit_batch_id_);
                batch = it->second;
                completed_.erase(it);
                ++next_emit_batch_id_;
                --outstanding_batches_;
                cv_submit_.notify_all();
            } else if (error_) {
                error = error_;
            } else {
                finished = true;
            }
        }

        if (error) {
            std::rethrow_exception(error);
        }
        if (finished || !batch) {
            return py::none();
        }

        py::list files;
        for (const auto& file : batch->files) {
            files.append(py::make_tuple(file.file_index, file.batch_offset));
        }
        auto holder = new std::shared_ptr<ReadyBatch>(batch);
        py::capsule owner(holder, [](void* p) {
            delete reinterpret_cast<std::shared_ptr<ReadyBatch>*>(p);
        });
        std::uintptr_t ptr = 0;
        std::size_t nbytes = 0;
        if (batch->device_buffer.owner) {
            ptr = reinterpret_cast<std::uintptr_t>(batch->device_buffer.data);
            nbytes = batch->device_buffer.size;
        }
        return py::make_tuple(owner, ptr, nbytes, files);
    }

    py::dict pool_stats() {
        return native_device_pool_stats_dict(device_id_);
    }

    py::dict device_pool_stats() {
        return native_device_pool_stats_dict(device_id_);
    }

    py::dict pinned_pool_stats() {
        return native_pinned_pool_stats_dict(device_id_);
    }

    py::dict io_stats() {
        py::dict out;
        out["backend"] = "kvikio";
        out["kvikio_reads"] = kvikio_reads_.load();
        out["kvikio_bytes"] = kvikio_bytes_.load();
        out["device_batches"] = device_batches_.load();
        return out;
    }

private:
    ReadyFile read_file(const ReadTask& task) {
        const auto& plan = task.plan;
        kvikio::FileHandle file_handle(plan.path, "r");
        struct PendingRead {
            std::future<std::size_t> future;
            std::size_t expected = 0;
        };
        std::vector<PendingRead> pending_reads;
        pending_reads.reserve(plan.spans.size() + plan.direct_spans.size());

        for (const auto& span : plan.spans) {
            if (span.nbytes == 0) {
                continue;
            }
            auto batch = task.batch;
            if (!batch || !batch->device_buffer.owner) {
                throw std::runtime_error("native batch device buffer is not allocated");
            }
            auto dst = static_cast<void*>(
                batch->device_buffer.data + task.batch_offset + static_cast<std::size_t>(span.host_offset));
            auto future = file_handle.pread(
                dst,
                static_cast<std::size_t>(span.nbytes),
                static_cast<std::size_t>(span.file_offset),
                kvikio::defaults::task_size(),
                kvikio::defaults::gds_threshold(),
                false);
            pending_reads.push_back(
                PendingRead{std::move(future), static_cast<std::size_t>(span.nbytes)});
        }

        for (const auto& span : plan.direct_spans) {
            if (span.nbytes == 0) {
                continue;
            }
            auto dst = reinterpret_cast<void*>(span.device_ptr);
            auto future = file_handle.pread(
                dst,
                static_cast<std::size_t>(span.nbytes),
                static_cast<std::size_t>(span.file_offset),
                kvikio::defaults::task_size(),
                kvikio::defaults::gds_threshold(),
                false);
            pending_reads.push_back(
                PendingRead{std::move(future), static_cast<std::size_t>(span.nbytes)});
        }

        for (auto& read : pending_reads) {
            const auto got = read.future.get();
            if (got != read.expected) {
                throw std::runtime_error("short KvikIO read on " + plan.path);
            }
            kvikio_reads_.fetch_add(1);
            kvikio_bytes_.fetch_add(static_cast<std::uint64_t>(got));
        }

        ReadyFile file;
        file.file_index = plan.file_index;
        file.path = plan.path;
        file.batch_offset = task.batch_offset;
        file.nbytes = plan.total_bytes;
        return file;
    }

    void set_error(std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) {
                error_ = error;
            }
            stop_ = true;
        }
        cv_not_empty_.notify_all();
        cv_submit_.notify_all();
        cv_ready_.notify_all();
    }

    void complete_task(ReadTask task, ReadyFile file) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            return;
        }

        auto batch = task.batch;
        batch->files[task.slot] = std::move(file);
        --batch->remaining;
        if (batch->remaining == 0) {
            auto ready = std::make_shared<ReadyBatch>();
            ready->batch_id = batch->batch_id;
            ready->device_buffer = batch->device_buffer;
            ready->files = std::move(batch->files);
            completed_[batch->batch_id] = std::move(ready);
            cv_ready_.notify_all();
        }
    }

    void worker_loop() {
        try {
            if (device_id_ >= 0) {
                check_cuda(cudaSetDevice(device_id_), "cudaSetDevice native batch reader");
            }

            while (true) {
                ReadTask task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_not_empty_.wait(lock, [&] {
                        return stop_ || !pending_.empty() || input_closed_;
                    });
                    if (stop_ || (pending_.empty() && input_closed_)) {
                        break;
                    }
                    task = std::move(pending_.front());
                    pending_.pop_front();
                }

                auto file = read_file(task);
                complete_task(std::move(task), std::move(file));
            }
        } catch (...) {
            set_error(std::current_exception());
        }
        cv_ready_.notify_all();
    }

    std::size_t decode_batch_files_ = 1;
    std::size_t batch_queue_depth_ = 1;
    std::size_t native_read_threads_ = 1;
    std::size_t max_outstanding_batches_ = 2;
    int device_id_ = -1;

    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_submit_;
    std::condition_variable cv_ready_;
    std::deque<ReadTask> pending_;
    std::map<std::uint64_t, std::shared_ptr<ReadyBatch>> completed_;
    std::vector<std::thread> workers_;
    std::uint64_t next_submit_batch_id_ = 0;
    std::uint64_t next_emit_batch_id_ = 0;
    std::size_t outstanding_batches_ = 0;
    bool stop_ = false;
    bool input_closed_ = false;
    std::exception_ptr error_;
    std::atomic<std::uint64_t> kvikio_reads_{0};
    std::atomic<std::uint64_t> kvikio_bytes_{0};
    std::atomic<std::uint64_t> device_batches_{0};
};



void bind_io(py::module_& m) {
    py::class_<NativeBatchBuilder>(m, "NativeBatchBuilder")
        .def(py::init<std::size_t, std::size_t, std::size_t, int>(),
             py::arg("decode_batch_files"),
             py::arg("batch_queue_depth"),
             py::arg("native_read_threads"),
             py::arg("device_id") = -1,
             "Background native worker pool that builds KvikIO device file batches.")
        .def("submit_batch", &NativeBatchBuilder::submit_batch,
             py::arg("batch_id"),
             py::arg("file_plans"),
             "Submit one planned batch of files for native KvikIO reads.")
        .def("close_input", &NativeBatchBuilder::close_input,
             "Signal that no more batches will be submitted.")
        .def("next_batch", &NativeBatchBuilder::next_batch,
             "Return the next ready batch as (owner, device_ptr, device_nbytes, files), or None.")
        .def("request_stop", &NativeBatchBuilder::request_stop,
             "Stop the background builder and join its native threads.")
        .def("pool_stats", &NativeBatchBuilder::pool_stats,
             "Return native device buffer pool statistics for this builder's CUDA device.")
        .def("device_pool_stats", &NativeBatchBuilder::device_pool_stats,
             "Return native device buffer pool statistics for this builder's CUDA device.")
        .def("pinned_pool_stats", &NativeBatchBuilder::pinned_pool_stats,
             "Return legacy pinned host buffer pool statistics for this builder's CUDA device.")
        .def("io_stats", &NativeBatchBuilder::io_stats,
             "Return native KvikIO read statistics for this builder.");
    m.attr("NativeBatchReader") = m.attr("NativeBatchBuilder");

    m.def("plan_native_files", &plan_native_files,
          py::arg("paths"),
          py::arg("file_indices"),
          py::arg("hdu_indices"),
          py::arg("native_plan_threads"),
          py::arg("section") = py::none(),
          "Plan ImageHDU/CompImageHDU GDS reads with CFITSIO in native worker threads.");
}


}  // namespace xdr_gpu
