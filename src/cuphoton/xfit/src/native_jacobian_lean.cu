// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused native xFit Jacobian evaluation and packing for short split-plane rows.
// Each CTA owns a row or a cluster portion of one row. Warp-local cp.async
// double buffers hold weights/residuals; 80/64-register variants evaluate
// lobes sequentially and write J^T directly with two 256-bit stores per pixel.
// Output alignment selects scalar stores when a 32-byte vector is unsafe.
// All model arithmetic retains the native FP64 operation sequence, runtime
// cube exponent, and exact quotient corrections with IEEE fallback.
// The host wrapper matches the fused launcher; N is unused by this path.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>

namespace cuphoton::xfit::detail {

namespace cg = cooperative_groups;

namespace lean_path {

constexpr int kMaxCluster = 8;

// Register-cap variants. Residency per SM follows from the register budget
// (65536 registers), the block size and the prefetch buffers' shared memory.
// cost is the relative per-chunk time: 128 registers interleave both lobes;
// 80 registers run spill-free with sequential lobes; at 64 registers the
// three-plane kernel spills (measured much slower, so effectively excluded).
struct Variant {
    int regs;
    double cost[2];  // one-plane, three-plane
};
constexpr Variant kVariants[3] = {
    {128, {1.0, 1.0}}, {80, {1.04, 1.04}}, {64, {1.10, 2.0}}};
// This path serves multi-wave (throughput) launches only: capped variants.
constexpr int kFirstVariant = 1;

__device__ __forceinline__ double mul(double a, double b) {
    return __dmul_rn(a, b);
}
__device__ __forceinline__ double add(double a, double b) {
    return __dadd_rn(a, b);
}
__device__ __forceinline__ double sub(double a, double b) {
    return __dsub_rn(a, b);
}

struct Star {
    double unit, width_x, width_y, angle, position_x, position_y;
};

// Per-row scalars are computed once into shared memory. The few used in every
// lobe/derivative (cos, sin, 1/s^2, their difference) are held in registers;
// the others are re-read from shared memory at their point of use (volatile, so
// the compiler does not hoist them into registers), which keeps long-lived
// values out of the register-capped variants.
enum RowSlot {
    kAmplitude,
    kSx,
    kSy,
    kCos,
    kSin,
    kIx,
    kIy,
    kIxMinusIy,
    kSxCube,
    kSyCube,
    kRcpSxCube,
    kRcpSyCube,  // RN(1 / s^3), used only to form exact quotients
    kXp,
    kYp,
    kXn,
    kYn,
    kSlots
};

struct RowScalars {
    double cosine, sine, ix, iy, ix_minus_iy;
    const volatile double*
        s;          // shared-memory slots (RowSlot), read at their point of use
    bool fast_div;  // divisors and reciprocals within [2^-400, 2^401)
    __device__ __forceinline__ double operator[](int i) const {
        return s[i];
    }
};

// Exponent field (sign masked) of a double, still in place: comparable as an unsigned.
__device__ __forceinline__ unsigned exponent_field(double v) {
    return (unsigned)__double2hiint(v) & 0x7ff00000u;
}
constexpr unsigned kScaledLo = (1023u - 400u) << 20;  // 2^-400
constexpr unsigned kScaledHi = (1023u + 400u)
    << 20;  // 2^400 (inclusive exponent)

__device__ __forceinline__ bool well_scaled(double v) {
    const unsigned e = exponent_field(v);
    return e >= kScaledLo && e <= kScaledHi;
}

// Correctly rounded a / d, bit-identical to IEEE division, for |a| and |d| in
// [2^-400, 2^401). y = RN(1/d) is hoisted per row. One Newton correction makes
// the quotient faithful; the second correction (Markstein's theorem: faithful
// q, y within half an ulp of 1/d, exact FMA remainder) yields RN(a/d). In this
// range no quotient, remainder or product can overflow or lose bits to underflow.
__device__ __forceinline__ double divide_scaled(double a, double d, double y) {
    const double q0 = __dmul_rn(a, y);
    const double q1 = __fma_rn(__fma_rn(-q0, d, a), y, q0);
    return __fma_rn(__fma_rn(-q1, d, a), y, q1);
}

// One Gaussian lobe. The four width quotients of both lobes are formed together
// so a single range test selects the reciprocal path or IEEE division.
struct Lobe {
    double unit, value, xr, yr, width_x_num, width_y_num;
};

__device__ __forceinline__ Lobe lobe(
    const RowScalars& r, double cx, double cy, int origin) {
    const double dx = sub(cx, r[origin]);
    const double dy = sub(cy, r[origin + 1]);
    Lobe l;
    l.xr = add(mul(r.cosine, dx), mul(r.sine, dy));
    l.yr = sub(mul(r.cosine, dy), mul(r.sine, dx));
    l.unit = exp(
        mul(-0.5, add(mul(mul(l.xr, l.xr), r.ix), mul(mul(l.yr, l.yr), r.iy))));
    l.value = mul(r[kAmplitude], l.unit);
    l.width_x_num = mul(mul(l.value, l.xr), l.xr);
    l.width_y_num = mul(mul(l.value, l.yr), l.yr);
    return l;
}

__device__ __forceinline__ Star finish(
    const RowScalars& r, const Lobe& l, double width_x, double width_y) {
    Star s;
    s.unit = l.unit;
    s.width_x = width_x;
    s.width_y = width_y;
    s.angle = mul(mul(mul(-l.value, l.xr), l.yr), r.ix_minus_iy);
    s.position_x =
        mul(l.value,
            sub(mul(mul(r.cosine, l.xr), r.ix), mul(mul(r.sine, l.yr), r.iy)));
    s.position_y =
        mul(l.value,
            add(mul(mul(r.sine, l.xr), r.ix), mul(mul(r.cosine, l.yr), r.iy)));
    return s;
}

__device__ __forceinline__ void stars(
    const RowScalars& r, double cx, double cy, Star& pos, Star& neg) {
    const Lobe lp = lobe(r, cx, cy, kXp);
    const Lobe ln = lobe(r, cx, cy, kXn);
    const double sx_cube = r[kSxCube], sy_cube = r[kSyCube];
    const double rcp_sx_cube = r[kRcpSxCube], rcp_sy_cube = r[kRcpSyCube];
    double q[4] = {divide_scaled(lp.width_x_num, sx_cube, rcp_sx_cube),
        divide_scaled(lp.width_y_num, sy_cube, rcp_sy_cube),
        divide_scaled(ln.width_x_num, sx_cube, rcp_sx_cube),
        divide_scaled(ln.width_y_num, sy_cube, rcp_sy_cube)};
    const unsigned e0 = exponent_field(lp.width_x_num),
                   e1 = exponent_field(lp.width_y_num);
    const unsigned e2 = exponent_field(ln.width_x_num),
                   e3 = exponent_field(ln.width_y_num);
    const unsigned lo = min(min(e0, e1), min(e2, e3)),
                   hi = max(max(e0, e1), max(e2, e3));
    if (!(r.fast_div && lo >= kScaledLo && hi <= kScaledHi)) {
        q[0] = __ddiv_rn(lp.width_x_num, sx_cube);
        q[1] = __ddiv_rn(lp.width_y_num, sy_cube);
        q[2] = __ddiv_rn(ln.width_x_num, sx_cube);
        q[3] = __ddiv_rn(ln.width_y_num, sy_cube);
    }
    pos = finish(r, lp, q[0], q[1]);
    neg = finish(r, ln, q[2], q[3]);
}

// One lobe on its own, with its own division range test. Evaluating the two
// lobes one after the other keeps fewer values live than stars() (used where
// register pressure, not latency, limits).
__device__ __forceinline__ Star star(
    const RowScalars& r, double cx, double cy, int origin) {
    const Lobe l = lobe(r, cx, cy, origin);
    const double sx_cube = r[kSxCube], sy_cube = r[kSyCube];
    double qx = divide_scaled(l.width_x_num, sx_cube, r[kRcpSxCube]);
    double qy = divide_scaled(l.width_y_num, sy_cube, r[kRcpSyCube]);
    const unsigned ex = exponent_field(l.width_x_num),
                   ey = exponent_field(l.width_y_num);
    if (!(r.fast_div && min(ex, ey) >= kScaledLo && max(ex, ey) <= kScaledHi)) {
        qx = __ddiv_rn(l.width_x_num, sx_cube);
        qy = __ddiv_rn(l.width_y_num, sy_cube);
    }
    return finish(r, l, qx, qy);
}

__device__ __forceinline__ double valid_width(double s) {
    return (isfinite(s) && s > 0.0)
        ? s
        : __longlong_as_double(0x7ff8000000000000LL);
}

// 256-bit global store (sm_100+): one 32-byte sector per lane.
__device__ __forceinline__ void store_v4(
    double* p, double a, double b, double c, double d) {
    asm volatile(
        "st.global.v4.f64 [%0], {%1, %2, %3, %4};" ::"l"(p),
        "d"(a),
        "d"(b),
        "d"(c),
        "d"(d));
}

// Weight one plane's derivatives, store the packed layout (coalesced across the
// warp) and the pixel's 64-byte transposed record (two full-sector 256-bit
// stores; scalar fallback if the output base is not 32-byte aligned), and fold
// finiteness into `worst`. Non-finite (Inf/NaN) iff |hi word| >= 0x7ff00000;
// the integer test keeps the check off the FP64 pipe.
__device__ __forceinline__ void emit(
    const double (&d)[8],
    double weight,
    double* packed_m,
    int64_t M,
    double* record,
    bool vec,
    unsigned& worst) {
    double v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = mul(d[j], weight);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        worst = max(worst, (unsigned)__double2hiint(v[j]) & 0x7fffffffu);
        packed_m[j * M] = v[j];
    }
    if (vec) {
        store_v4(record, v[0], v[1], v[2], v[3]);
        store_v4(record + 4, v[4], v[5], v[6], v[7]);
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            record[j] = v[j];
        }
    }
}

__device__ __forceinline__ void cp_async8(unsigned smem, const double* g) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait_prev() {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
}

// grid.x = K * cluster; the `cluster` blocks of one thread-block cluster cover
// one selected row. Each warp owns 32-pixel chunks of that row, one pixel per
// lane, and evaluates both lobes once for all planes. The next chunk's weights
// and residuals are prefetched by cp.async into a per-warp shared double buffer
// [2][2P][32] while the current chunk is evaluated, so no registers are held
// across the loop. MaxRegs is the register cap of the variant; capped variants
// evaluate the two lobes one after the other (fewer live values, no spills at
// 80 registers), the uncapped one interleaves them for latency.
template <int P, int MaxRegs>
__global__ void __maxnreg__(MaxRegs) jacobian_pack_kernel(
    const double* __restrict__ x,
    const double* __restrict__ weights,
    const int* __restrict__ indices,
    const double* __restrict__ residuals,
    double* __restrict__ packed,
    double* __restrict__ transposed,
    double* __restrict__ compact,
    bool* __restrict__ finite_row,
    int H,
    int W,
    double cube_exponent,
    int cluster,
    bool vec) {
    constexpr bool kSequential = MaxRegs < 128;
    extern __shared__ double prefetch_all[];
    __shared__ double row_scalars[kSlots];
    __shared__ int block_finite;

    const int T = blockDim.x;
    const int t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5, warps = T >> 5;
    const int k = blockIdx.x / cluster;
    const int c = blockIdx.x - k * cluster;
    const int HW = H * W;
    const int64_t M = (int64_t)P * HW;
    const int64_t src = indices[k];
    const double* xrow = x + src * 8;

    const double* wrow = weights + src * M;
    const double* rrow = residuals + src * M;

    const int chunks = (HW + 31) >> 5;
    const int stride = cluster * warps;
    int chunk = c * warps + warp;
    // Issue the chunk's loads into shared buffer b (always commits a group,
    // possibly empty, so wait_group 1 always refers to the current chunk).
    double* prefetch = prefetch_all + warp * (2 * 2 * P * 32) + lane;
    const unsigned prefetch_s = (unsigned)__cvta_generic_to_shared(prefetch);
    auto load_async = [&](int ch, int b) {
        if (ch < chunks) {
            const int px = min((ch << 5) + lane, HW - 1);
#pragma unroll
            for (int p = 0; p < P; ++p) {
                cp_async8(
                    prefetch_s + 8u * 32u * (b * 2 * P + p),
                    wrow + (int64_t)p * HW + px);
                cp_async8(
                    prefetch_s + 8u * 32u * (b * 2 * P + P + p),
                    rrow + (int64_t)p * HW + px);
            }
        }
        cp_async_commit();
    };
    load_async(chunk, 0);
    if (t < 16 && (t == 0 || t >= 12)) {
        row_scalars[t == 0 ? kAmplitude : t] = xrow[t == 0 ? 0 : t - 8];
    }
    // Per-row scalar work: the four independent transcendental chains run on
    // lane 0 of different warps so they overlap.
    for (int job = warp; job < 4; job += warps) {
        if (lane != 0) {
            continue;
        }
        if (job < 2) {
            const double s = valid_width(exp(xrow[1 + job]));
            row_scalars[kSx + job] = s;
            row_scalars[kIx + job] = 1.0 / mul(s, s);
            const double cube = pow(s, cube_exponent);
            row_scalars[kSxCube + job] = cube;
            row_scalars[kRcpSxCube + job] = 1.0 / cube;
        } else {
            row_scalars[kCos + job - 2] =
                job == 2 ? cos(xrow[3]) : sin(xrow[3]);
        }
    }
    __syncthreads();
    RowScalars r;
    r.s = row_scalars;
    r.cosine = row_scalars[kCos];
    r.sine = row_scalars[kSin];
    r.ix = row_scalars[kIx];
    r.iy = row_scalars[kIy];
    r.ix_minus_iy = sub(r.ix, r.iy);
    r.fast_div = well_scaled(row_scalars[kSxCube])
        && well_scaled(row_scalars[kSyCube])
        && well_scaled(row_scalars[kRcpSxCube])
        && well_scaled(row_scalars[kRcpSyCube]);
    double* prow = packed + (int64_t)k * 8 * M;
    double* trow = transposed + (int64_t)k * 8 * M;
    double* crow = compact + (int64_t)k * M;
    unsigned worst = 0;
    for (int b = 0; chunk < chunks; chunk += stride, b ^= 1) {
        load_async(chunk + stride, b ^ 1);
        cp_async_wait_prev();
        const volatile double* buf =
            prefetch + b * (2 * P * 32);  // [weights P][residuals P] x 32 lanes
        const int pix = (chunk << 5) + lane;
        if (pix >= HW) {
            continue;  // idle lanes of the row's last, partial chunk
        }
        const int h = pix / W;
        // Exact pixel-centered coordinates: (2*col - (W-1)) / 2 == col - (W-1)/2.
        const double cx = (double)(pix - h * W) - 0.5 * (double)(W - 1);
        const double cy = (double)h - 0.5 * (double)(H - 1);
        Star pos, neg;
        if constexpr (kSequential) {
            pos = star(r, cx, cy, kXp);
            neg = star(r, cx, cy, kXn);
        } else {
            stars(r, cx, cy, pos, neg);
        }
        const double sx = r[kSx], sy = r[kSy];
        {
            // Plane 0: dipole difference.
            const double d[8] = {sub(pos.unit, neg.unit),
                mul(sub(pos.width_x, neg.width_x), sx),
                mul(sub(pos.width_y, neg.width_y), sy),
                sub(pos.angle, neg.angle),
                pos.position_x,
                pos.position_y,
                -neg.position_x,
                -neg.position_y};
            emit(d, buf[0], prow + pix, M, trow + (int64_t)pix * 8, vec, worst);
        }
        if constexpr (P == 3) {
            // Plane 1: positive lobe.
            const double d1[8] = {pos.unit,
                mul(pos.width_x, sx),
                mul(pos.width_y, sy),
                pos.angle,
                pos.position_x,
                pos.position_y,
                0.0,
                0.0};
            emit(
                d1,
                buf[32],
                prow + HW + pix,
                M,
                trow + ((int64_t)HW + pix) * 8,
                vec,
                worst);
            // Plane 2: negative lobe.
            const double d2[8] = {neg.unit,
                mul(neg.width_x, sx),
                mul(neg.width_y, sy),
                neg.angle,
                0.0,
                0.0,
                neg.position_x,
                neg.position_y};
            emit(
                d2,
                buf[64],
                prow + 2 * HW + pix,
                M,
                trow + (2 * (int64_t)HW + pix) * 8,
                vec,
                worst);
        }
        // Residual copy (exact) from the prefetch buffer.
#pragma unroll
        for (int p = 0; p < P; ++p) {
            crow[(int64_t)p * HW + pix] = buf[(P + p) * 32];
        }
    }

    const int ok = __syncthreads_and(worst < 0x7ff00000u);
    if (cluster == 1) {
        if (t == 0) {
            finite_row[k] = ok != 0;
        }
        return;
    }
    if (t == 0) {
        block_finite = ok;
    }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    if (c == 0 && t == 0) {
        int all = 1;
        for (int b = 0; b < cluster; ++b) {
            all &= *cl.map_shared_rank(&block_finite, b);
        }
        finite_row[k] = all != 0;
    }
    cl.sync();
}

// Launch geometry. A row is split into 32-pixel warp chunks; the `cluster`
// blocks of one thread-block cluster share a row so finite_row can be reduced
// through distributed shared memory, and each warp takes up to `per_warp`
// chunks of its row. The cost model (in units of one chunk's latency):
//   waves * (per_warp + 1/2) * chunk_cost   critical path of the busiest warp
//                                           plus the per-row prologue per wave
//   + 1/4 if cluster > 1                    cluster launch / synchronization
//   + 0.02 * P * (warp-chunks on the busiest SM)   per-SM store/issue load
// Blocks larger than 8 warps are only used to avoid a cluster; three-plane rows
// keep warps short-lived (per_warp <= 2), which measured faster for their
// store-heavy stream. Ties prefer balanced rows (fewest idle warp slots).
struct Geometry {
    int cluster, warps, variant;
};

static size_t prefetch_bytes(int warps, int P) {
    return (size_t)warps * 2 * 2 * P * 32 * sizeof(double);
}

static int blocks_per_sm(int warps, int regs, int P) {
    const int by_regs = 65536 / (regs * 32 * warps);
    const int by_smem = (int)(233472 / (prefetch_bytes(warps, P) + 1024 + 256));
    return std::min(std::min(by_regs, by_smem), std::min(32, 64 / warps));
}

static Geometry choose_geometry(int K, int HW, int P, int sms) {
    const int chunks = std::max(1, (HW + 31) / 32);
    const int max_per_warp = P == 1 ? chunks : std::min(chunks, 2);
    Geometry best = {1, 1, kFirstVariant};
    double best_cost = 1e30, best_eff = 0.0;
    for (int v = kFirstVariant; v < 3; ++v) {
        const double chunk_cost = kVariants[v].cost[P == 1 ? 0 : 1];
        for (int per_warp = 1; per_warp <= max_per_warp; ++per_warp) {
            // w only shrinks with per_warp; stop once even c = 1 leaves too few warps.
            if ((chunks + per_warp - 1) / per_warp < std::min(4, chunks)) {
                break;
            }
            for (int c = 1; c <= kMaxCluster; ++c) {
                const int w = (chunks + c * per_warp - 1) / (c * per_warp);
                if (w > (c == 1 ? 16 : 8)) {
                    continue;
                }
                // The prologue's four transcendental chains run on separate warps.
                if (w < std::min(4, chunks)) {
                    continue;
                }
                if (c * w * per_warp - chunks >= w * per_warp) {
                    continue;  // a whole block would idle
                }
                const long long blocks = (long long)K * c;
                const long long per_wave =
                    (long long)blocks_per_sm(w, kVariants[v].regs, P) * sms;
                if (per_wave == 0) {
                    continue;
                }
                const long long waves = (blocks + per_wave - 1) / per_wave;
                const long long sm_load =
                    (blocks + sms - 1) / sms * w * per_warp;
                const double cost =
                    (double)waves * (per_warp + 0.5) * chunk_cost
                    + (c > 1 ? 0.25 : 0.0) + 0.02 * P * (double)sm_load;
                const double eff = (double)chunks / ((double)c * w * per_warp);
                if (cost < best_cost - 1e-9
                    || (cost < best_cost + 1e-9 && eff > best_eff)) {
                    best_cost = cost;
                    best_eff = eff;
                    best = {c, w, v};
                }
            }
        }
    }
    return best;
}

template <int P>
cudaError_t launch_variant(
    const Geometry& g,
    cudaLaunchConfig_t& cfg,
    const double* x,
    const double* weights,
    const int* indices,
    const double* residuals,
    double* packed,
    double* transposed,
    double* compact,
    bool* finite_row,
    int H,
    int W,
    bool vec) {
    // The native source passes the runtime cube exponent (3) as a kernel argument.
    const double cube_exponent = 3.0;
    cfg.dynamicSmemBytes = prefetch_bytes(g.warps, P);
    switch (g.variant) {
    case 1:
        return cudaLaunchKernelEx(
            &cfg,
            jacobian_pack_kernel<P, 80>,
            x,
            weights,
            indices,
            residuals,
            packed,
            transposed,
            compact,
            finite_row,
            H,
            W,
            cube_exponent,
            g.cluster,
            vec);
    default:
        return cudaLaunchKernelEx(
            &cfg,
            jacobian_pack_kernel<P, 64>,
            x,
            weights,
            indices,
            residuals,
            packed,
            transposed,
            compact,
            finite_row,
            H,
            W,
            cube_exponent,
            g.cluster,
            vec);
    }
}

template <int P>
cudaError_t launch(
    const double* x,
    const double* weights,
    const int* indices,
    const double* residuals,
    double* packed,
    double* transposed,
    double* compact,
    bool* finite_row,
    int K,
    int H,
    int W,
    int device,
    cudaStream_t stream) {
    // Host-side launch configuration only (never results) is memoized: the SM
    // count per device and the geometry of the most recent shape on this thread.
    thread_local int sm_count[64] = {};
    int sms = device >= 0 && device < 64 ? sm_count[device] : 0;
    if (sms == 0) {
        const cudaError_t err = cudaDeviceGetAttribute(
            &sms, cudaDevAttrMultiProcessorCount, device);
        if (err != cudaSuccess) {
            return err;
        }
        if (device >= 0 && device < 64) {
            sm_count[device] = sms;
        }
    }
    struct Memo {
        int K = -1, HW = -1, sms = -1;
        Geometry g;
    };
    thread_local Memo memo;
    if (memo.K != K || memo.HW != H * W || memo.sms != sms) {
        memo.g = choose_geometry(K, H * W, P, sms);
        memo.K = K;
        memo.HW = H * W;
        memo.sms = sms;
    }
    const Geometry g = memo.g;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(K * g.cluster));
    cfg.blockDim = dim3((unsigned)(g.warps * 32));
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = g.cluster;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = g.cluster > 1 ? 1 : 0;
    // 256-bit transposed stores need a 32-byte aligned base (rows are 64-byte multiples).
    const bool vec = (reinterpret_cast<uintptr_t>(transposed) & 31) == 0;
    return launch_variant<P>(
        g,
        cfg,
        x,
        weights,
        indices,
        residuals,
        packed,
        transposed,
        compact,
        finite_row,
        H,
        W,
        vec);
}

}  // namespace lean_path

namespace {
__global__ void lean_empty_rows(bool* finite_row, int K) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        finite_row[k] = true;
    }
}
}  // namespace

cudaError_t launch_jacobian_pack_lean_fp64(
    const double* x,
    const double* weights,
    const int* indices,
    const double* residuals,
    double* packed,
    double* transposed,
    double* compact,
    bool* finite_row,
    long long N,
    int K,
    int P,
    int H,
    int W,
    int device,
    cudaStream_t stream) {
    (void)N;
    if (K == 0) {
        return cudaSuccess;
    }
    if (H == 0 || W == 0) {
        lean_empty_rows<<<(K + 255) / 256, 256, 0, stream>>>(finite_row, K);
        return cudaGetLastError();
    }
    if (P == 1) {
        return lean_path::launch<1>(
            x,
            weights,
            indices,
            residuals,
            packed,
            transposed,
            compact,
            finite_row,
            K,
            H,
            W,
            device,
            stream);
    }
    return lean_path::launch<3>(
        x,
        weights,
        indices,
        residuals,
        packed,
        transposed,
        compact,
        finite_row,
        K,
        H,
        W,
        device,
        stream);
}

}  // namespace cuphoton::xfit::detail
