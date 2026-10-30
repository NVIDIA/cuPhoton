// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// Bounded real companion QR convergence; no CUDA or BLAS.
// Adapted mathematical routines: jverzani/AMRVW.jl, commit
// 8cb7afc20abd9dff4216380faa3a149cce16421f, src/{transformations,companion,
// RDS,chains,rfactorization,qfactorization,diagonal-block,create-bulge,
// AMRVW_algorithm,bulge-step,utils}.jl.
// Rotation convention is [c s; -s c]; indices remain one-based deliberately.
// The original MIT notice applies to adapted portions; the Apache notice
// above applies to new validation/error-handling code.
//
// The MIT License (MIT)
// Copyright (c) 2019 John Verzani
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "structured_api.h"
#include <atomic>
#include <cstdio>
#include <thread>

namespace cuphoton::xray {
namespace {

enum class Status { invalid_input, bounds, numerical_range, identity_failure };
struct Failure : std::runtime_error {
    Status status;
    Failure(Status s, const char* text)
        : std::runtime_error(text),
          status(s) {}
};
void require(bool condition, Status status, const char* text) {
    if (!condition) {
        throw Failure(status, text);
    }
}
double checked_finite(double x) {
    require(
        std::isfinite(x), Status::numerical_range, "nonfinite intermediate");
    return x;
}
double checked_ratio(double a, double b) {
    require(b != 0, Status::numerical_range, "zero coefficient divisor");
    double value = checked_finite(a / b);
    require(
        a == 0 || value != 0,
        Status::numerical_range,
        "nonzero coefficient ratio underflow");
    return value;
}
struct Rot {
    double c, s;
    int i;
};
Rot checked(Rot u) {
    checked_finite(u.c);
    checked_finite(u.s);
    require(u.i >= 1, Status::bounds, "rotation index");
    require(
        std::abs(u.c * u.c + u.s * u.s - 1) < 2e-12,
        Status::numerical_range,
        "rotation lost orthogonality");
    return u;
}
Rot adj(Rot u) {
    return {u.c, -u.s, u.i};
}
std::tuple<double, double, double> givens(double a, double b) {
    checked_finite(a);
    checked_finite(b);
    double r = checked_finite(std::hypot(a, b));
    if (r == 0) {
        return {1, 0, 0};
    }
    return {a / r, b / r, r};
}
Rot polish(double a, double b, int i) {
    double delta = checked_finite(a * a + b * b - 1);
    require(
        std::abs(delta) < 1e-8,
        Status::numerical_range,
        "Taylor normalization outside near-unit domain");
    double scale = 1 - delta / 2;
    return checked({a * scale, b * scale, i});
}
Rot fuse(Rot a, Rot b) {
    checked(a);
    checked(b);
    require(a.i == b.i, Status::bounds, "fusion indices differ");
    return polish(a.c * b.c - a.s * b.s, a.c * b.s + a.s * b.c, a.i);
}
std::array<Rot, 3> turnover(Rot u, Rot v, Rot w) {
    checked(u);
    checked(v);
    checked(w);
    require(
        u.i == w.i && std::abs(u.i - v.i) == 1,
        Status::bounds,
        "turnover indices");
    double a = -v.c * w.s * u.c - w.c * u.s, b = v.s * w.s;
    if (a == 0 && b == 0 && v.s * w.c == 0) {
        return {checked({1, 0, v.i}),
            checked({u.c * w.c - v.c * u.s * w.s, 0, u.i}),
            checked({v.c, 0, v.i})};
    }
    auto [c4, s4, r4] = givens(a, b);
    s4 = -s4;
    Rot r5 = polish(u.c * w.c - v.c * u.s * w.s, r4, u.i);
    r5.s = -r5.s;
    Rot r6;
    if (r5.s != 0) {
        // The improved turnover's sine-product invariant (2018 Eq. 5.3).
        r6 = polish(u.c * v.s * s4 + v.c * c4, u.s * v.s / r5.s, v.i);
    } else {
        // Correct the upstream zero-s5 branch. Read the residual 2x2
        // rotation from the transformed third column, not a fresh Givens
        // annihilation. In particular turnover(I,V,I) must return V, not V^T.
        double c6 = u.c * v.s * s4 + v.c * c4;
        double s6 = r5.c * (u.c * v.s * c4 - v.c * s4);
        r6 = polish(c6, s6, v.i);
    }
    return {checked({c4, s4, v.i}), r5, r6};
}

struct Factors {
    int n;
    std::vector<Rot> q, ct,
        b;  // ct indexed by spatial index, product ct[n]..ct[1].
    explicit Factors(const std::vector<double>& ps)
        : n(int(ps.size()) - 1) {
        require(
            n >= 3 && n <= 4096,
            Status::invalid_input,
            "structured factorization supports degree 3..4096");
        for (double a : ps) {
            require(
                std::isfinite(a),
                Status::invalid_input,
                "nonfinite coefficient");
        }
        require(
            ps.back() != 0 && ps.front() != 0,
            Status::invalid_input,
            "leading/constant coefficient zero after explicit deflation");
        q.resize(n);
        ct.resize(n + 1);
        b.resize(n + 1);
        std::vector<double> xs(n + 1);
        double parity = n % 2 ? -1 : 1;
        for (int i = 1; i < n; ++i) {
            xs[i] = parity * (i % 2 ? 1 : -1) * checked_ratio(ps[i], ps[n]);
        }
        xs[n] = parity * checked_ratio(ps[0], ps[n]);
        auto [c, s, tmp] = givens(xs[n], -1);
        ct[n] = checked({c, -s, n});
        b[n] = checked({s, -c, n});
        for (int i = n - 1; i >= 1; --i) {
            std::tie(c, s, tmp) = givens(xs[i], tmp);
            ct[i] = checked({c, -s, i});
            b[i] = checked({c, s, i});
            q[i] = {0, 1, i};
        }
    }
    Rot& qr(int i) {
        require(i >= 1 && i < n, Status::bounds, "Q index");
        return q.at(i);
    }
    double w(int j, int k) const {
        if (j == k) {
            return -b.at(k).s;
        }
        double prod = b.at(j).c * b.at(k).c;
        for (int i = j + 1; i < k; ++i) {
            prod *= b.at(i).s;
        }
        return prod;
    }
    double r(int j, int k) const {
        if (j <= 0 || k < j) {
            return 0;
        }
        require(k <= n, Status::bounds, "R index");
        double sj = ct.at(j).s, prod = sj;
        require(
            prod != 0, Status::numerical_range, "singular R reconstruction");
        double total = checked_finite(w(j, k) / prod), par = -1;
        for (int i = j + 1; i <= k; ++i) {
            prod *= ct.at(i).s;
            require(
                prod != 0,
                Status::numerical_range,
                "underflow in R reconstruction");
            total = checked_finite(
                total + par * ct.at(j).c * ct.at(i).c / prod * w(i, k));
            par = -par;
        }
        return total;
    }
    double qentry(int i, int j) const {
        if (i <= 0 || j <= 0 || i > j + 1) {
            return 0;
        }
        require(i <= n && j <= n, Status::bounds, "Q entry");
        if (i == j + 1) {
            return -q.at(j).s;
        }
        double prod = (i > 1 ? q.at(i - 1).c : 1);
        if (j < n) {
            prod *= q.at(j).c;
        }
        for (int k = i; k < j; ++k) {
            prod *= q.at(k).s;
        }
        return prod;
    }
    std::array<double, 4> block(int j) const {
        require(j >= 1 && j < n, Status::bounds, "block index");
        int k = j + 1;
        return {checked_finite(
                    qentry(j, j - 1) * r(j - 1, j) + qentry(j, j) * r(j, j)),
            checked_finite(
                qentry(j, j - 1) * r(j - 1, k) + qentry(j, j) * r(j, k)
                + qentry(j, k) * r(k, k)),
            checked_finite(qentry(k, j) * r(j, j)),
            checked_finite(qentry(k, j) * r(j, k) + qentry(k, k) * r(k, k))};
    }
    Rot pass_desc(std::vector<Rot>& chain, Rot u, int last) {
        int i = u.i;
        require(i >= 1 && i < last, Status::bounds, "descending pass");
        auto out = turnover(chain.at(i), chain.at(i + 1), u);
        chain[i] = out[1];
        chain[i + 1] = out[2];
        return out[0];
    }
    Rot pass_r(Rot u) {
        u = pass_desc(b, u, n);
        int i = u.i;
        require(i > 1 && i <= n, Status::bounds, "ascending pass");
        auto out = turnover(ct.at(i), ct.at(i - 1), u);
        ct[i] = out[1];
        ct[i - 1] = out[2];
        return out[0];
    }
    double parity(int k) const {
        if (k == 0 || k == n) {
            return 1;
        }
        require(
            k > 0 && k < n && q.at(k).s == 0,
            Status::bounds,
            "non-diagonal block boundary");
        return std::copysign(1.0, q.at(k).c);
    }
    static Rot flip(Rot u, double sign) {
        u.s *= sign;
        return checked(u);
    }
    void sweep(int lo, int hi, unsigned exceptional = 0) {
        require(
            lo >= 1 && hi <= n && hi - lo >= 2,
            Status::bounds,
            "active block bounds");
        const double left_sign = parity(lo - 1), right_sign = parity(hi);
        Rot u, v;
        if (exceptional) {
            // Deterministic version of upstream's paired exceptional rotations.
            // Unsigned arithmetic is defined modulo 2^32; no global RNG state.
            unsigned word = exceptional * 2654435761u
                + unsigned(lo) * 2246822519u + unsigned(hi) * 3266489917u;
            double angle =
                (double(word) + 0.5) * (3.14159265358979323846 / 4294967296.0);
            u = checked({std::sin(angle), std::cos(angle), lo});
            v = checked({u.c, -u.s, lo + 1});
        } else {
            auto tail = block(hi - 1), top = block(lo), next = block(lo + 1);
            double scale = 0;
            for (double x : tail) {
                scale = std::max(scale, std::abs(x));
            }
            for (double x : top) {
                scale = std::max(scale, std::abs(x));
            }
            scale = std::max(scale, std::abs(next[2]));
            require(
                scale > 0,
                Status::numerical_range,
                "zero shift scale in unreduced block");
            for (double& x : tail) {
                x /= scale;
            }
            for (double& x : top) {
                x /= scale;
            }
            double tr = tail[0] + tail[3],
                   det = tail[0] * tail[3] - tail[1] * tail[2];
            double x = checked_finite(
                top[0] * top[0] + top[1] * top[2] - tr * top[0] + det);
            double y = checked_finite(top[2] * (top[0] + top[3] - tr));
            double z = checked_finite(top[2] * (next[2] / scale));
            auto [cv, sv, nv] = givens(y, z);
            v = checked({cv, -sv, lo + 1});
            auto [cu, su, nu] = givens(x, nv);
            (void)nu;
            u = checked({cu, -su, lo});
        }
        auto left = turnover(adj(u), adj(v), flip(qr(lo), left_sign));
        Rot limb = left[0];
        qr(lo) = flip(left[1], left_sign);
        qr(lo + 1) = fuse(left[2], qr(lo + 1));
        for (int step = 0; step < hi - lo + 1; ++step) {
            v = pass_r(v);
            u = pass_r(u);
            if (v.i < hi - 1) {
                v = pass_desc(q, v, n - 1);
                u = pass_desc(q, u, n - 1);
                auto moved = turnover(limb, v, u);
                v = moved[0];
                u = moved[1];
                limb = moved[2];
            } else {
                require(
                    v.i == hi - 1, Status::bounds, "sweep termination index");
                qr(v.i) = fuse(qr(v.i), flip(v, right_sign));
                u = pass_desc(q, u, n - 1);
                u = fuse(limb, u);
                u = pass_r(u);
                qr(u.i) = fuse(qr(u.i), flip(u, right_sign));
                return;
            }
        }
        throw Failure(
            Status::bounds, "sweep chase exceeded active block length");
    }
};

using Complex = std::complex<double>;
// Product differences use a shared binary exponent and an FMA residual. This
// avoids forming a*d or b*c at their unscaled magnitudes (overflow/underflow).
struct Scaled {
    double value;
    int exponent;
};
Scaled determinant(double a, double d, double b, double c) {
    int ea = 0, ed = 0, eb = 0, ec = 0;
    double ma = std::frexp(a, &ea), md = std::frexp(d, &ed);
    double mb = std::frexp(b, &eb), mc = std::frexp(c, &ec);
    int ad = (a == 0 || d == 0) ? -8192 : ea + ed;
    int bc = (b == 0 || c == 0) ? -8192 : eb + ec;
    int exponent = std::max(ad, bc);
    if (exponent == -8192) {
        return {0, 0};
    }
    double p0 = ma * md, q0 = mb * mc;
    double p = std::scalbn(p0, ad - exponent),
           q = std::scalbn(q0, bc - exponent);
    double pe = std::scalbn(std::fma(ma, md, -p0), ad - exponent);
    double qe = std::scalbn(std::fma(mb, mc, -q0), bc - exponent);
    double h = p - q, z = h - p;
    double low = (p - (h - z)) + (-q - z) + pe - qe;
    return {h + low, exponent};
}
std::array<Complex, 2> eigen2(double a, double b, double c, double d) {
    checked_finite(a);
    checked_finite(b);
    checked_finite(c);
    checked_finite(d);
    if (b == 0 || c == 0) {
        return {Complex(a, 0), Complex(d, 0)};
    }
    bool same_sign = std::signbit(a) == std::signbit(d);
    double center, delta;
    if (same_sign) {
        center = a + (d - a) / 2;
        delta = (a - d) / 2;
    } else {
        double half_a = checked_ratio(a, 2), half_d = checked_ratio(d, 2);
        center = half_a + half_d;
        delta = half_a - half_d;
    }
    require(
        (a == -d || center != 0) && (a == d || delta != 0),
        Status::numerical_range,
        "2x2 midpoint or half-difference underflow");
    int edelta = 0, eb = 0, ec = 0;
    std::frexp(delta, &edelta);
    double mb = std::frexp(b, &eb), mc = std::frexp(c, &ec);
    int sum = eb + ec, half = sum >= 0 ? (sum + 1) / 2 : sum / 2;
    int exponent =
        std::max(delta == 0 ? -8192 : edelta, b == 0 || c == 0 ? -8192 : half);
    if (exponent == -8192) {
        return {Complex(center, 0), Complex(center, 0)};
    }
    double v = std::scalbn(delta, -exponent), p = v * v,
           p_error = std::fma(v, v, -p);
    double q0 = mb * mc;
    int shift = sum - 2 * exponent;
    double q = (b == 0 || c == 0) ? 0 : std::scalbn(q0, shift);
    double q_error =
        (b == 0 || c == 0) ? 0 : std::scalbn(std::fma(mb, mc, -q0), shift);
    double h = p + q, z = h - p;
    double discriminant = h + ((p - (h - z)) + (q - z) + p_error + q_error);
    double width = checked_finite(
        std::scalbn(std::sqrt(std::abs(discriminant)), exponent));
    require(
        discriminant == 0 || width != 0,
        Status::numerical_range,
        "2x2 root width underflow");
    if (discriminant < 0) {
        return {Complex(center, width), Complex(center, -width)};
    }
    if (width == 0) {
        return {Complex(center, 0), Complex(center, 0)};
    }
    double large = checked_finite(center + std::copysign(width, center));
    if (large == 0) {
        return {Complex(0, 0), Complex(0, 0)};
    }
    auto det = determinant(a, d, b, c);
    int er = 0;
    double mr = std::frexp(large, &er);
    double small =
        checked_finite(std::scalbn(det.value / mr, det.exponent - er));
    require(
        det.value == 0 || small != 0,
        Status::numerical_range,
        "2x2 small root underflow");
    return {Complex(large, 0), Complex(small, 0)};
}

enum class SolveStatus { converged, iteration_limit };
struct Result {
    SolveStatus status = SolveStatus::iteration_limit;
    int degree = 0, unconverged = 0, sweeps = 0, deflations = 0,
        exceptional = 0;
    std::vector<Complex> roots;
};
Result solve_descending(const std::vector<double>& input, int max_sweeps = -1) {
    require(
        input.size() >= 2 && input.size() <= 4097,
        Status::invalid_input,
        "coefficient count 2..4097 required");
    require(
        input.front() != 0,
        Status::invalid_input,
        "leading coefficient must be nonzero");
    for (double x : input) {
        require(
            std::isfinite(x), Status::invalid_input, "nonfinite coefficient");
    }
    require(max_sweeps >= -1, Status::invalid_input, "invalid sweep cap");
    Result out;
    out.degree = int(input.size()) - 1;
    std::vector<double> descending = input;
    while (descending.size() > 1 && descending.back() == 0) {
        out.roots.emplace_back(0, 0);
        descending.pop_back();
    }
    int n = int(descending.size()) - 1;
    if (n == 1) {
        out.roots.emplace_back(-checked_ratio(descending[1], descending[0]), 0);
    }
    if (n == 2) {
        auto e = eigen2(
            0,
            -checked_ratio(descending[2], descending[0]),
            1,
            -checked_ratio(descending[1], descending[0]));
        out.roots.insert(out.roots.end(), e.begin(), e.end());
    }
    if (n <= 2) {
        out.status = SolveStatus::converged;
        return out;
    }
    // Input is explicitly polynomial coefficients in descending order, never
    // LP recurrence coefficients. The caller supplies [1,-LPcoefficients].
    std::vector<double> ascending(descending.rbegin(), descending.rend());
    Factors f(ascending);
    if (max_sweeps < 0) {
        max_sweeps = 20 * n;
    }
    int hi = n, previous_lo = -1, previous_hi = -1, since_exceptional = 0;
    while (hi > 0) {
        if (hi == 1) {
            out.roots.emplace_back(
                checked_finite(f.qentry(1, 1) * f.r(1, 1)), 0);
            --hi;
            break;
        }
        int lo = 1;
        for (int k = hi - 1; k >= 1; --k) {
            if (std::abs(f.qr(k).s) <= std::numeric_limits<double>::epsilon()) {
                if (f.qr(k).s != 0) {
                    ++out.deflations;
                }
                f.qr(k) = {std::copysign(1.0, f.qr(k).c), 0, k};
                lo = k + 1;
                break;
            }
        }
        if (lo != previous_lo || hi != previous_hi) {
            since_exceptional = 0;
        }
        previous_lo = lo;
        previous_hi = hi;
        if (hi - lo == 0) {
            auto block = f.block(hi - 1);
            out.roots.emplace_back(block[3], 0);
            --hi;
            continue;
        }
        if (hi - lo == 1) {
            auto block = f.block(lo);
            auto e = eigen2(block[0], block[1], block[2], block[3]);
            out.roots.insert(out.roots.end(), e.begin(), e.end());
            hi = lo - 1;
            continue;
        }
        if (out.sweeps >= max_sweeps) {
            out.unconverged = hi;
            return out;
        }
        unsigned kick = 0;
        if (since_exceptional >= 15) {
            kick = unsigned(++out.exceptional);
            since_exceptional = 0;
        }
        f.sweep(lo, hi, kick);
        ++out.sweeps;
        ++since_exceptional;
    }
    require(
        int(out.roots.size()) == out.degree,
        Status::identity_failure,
        "root count mismatch");
    out.status = SolveStatus::converged;
    return out;
}

}  // namespace

// Caller supplies distinct contiguous arrays: descending[B][degree+1],
// roots[B][degree][2], stats[B][8], errors[B][256]. No CUDA or BLAS calls.
// stats columns: status,count,unconverged,sweeps,deflations,kicks,0,effective_n.
// Status: converged0, cap1, invalid2, bounds3, range4, identity5,
// std_exception7, unknown_exception8. CPU exceptions expose no partial Result:
// their roots remain NaN and counters stay zero (unconverged=degree).
// Return: 0=dispatch complete (inspect every row), -1=invalid ABI arguments,
// -2=thread/setup failure after joining all successfully created threads.
// Fresh thread creation, input copies and all output writes are inside call.
std::int32_t solve_roots(
    const double* descending,
    double* roots,
    std::int32_t* stats,
    char* errors,
    std::int32_t degree,
    std::int32_t batch,
    std::int32_t max_sweeps,
    std::int32_t workers) noexcept {
    if (!descending || !roots || !stats || !errors || degree < 1
        || degree > 4096 || batch < 1 || max_sweeps < -1 || workers < 1
        || workers > 256) {
        return -1;
    }
    std::atomic<std::int64_t> next{0};
    std::vector<std::thread> threads;
    auto work = [&]() noexcept {
        for (;;) {
            const auto row = next.fetch_add(1, std::memory_order_relaxed);
            if (row >= batch) {
                return;
            }
            const std::size_t index = static_cast<std::size_t>(row);
            auto* out = roots + index * static_cast<std::size_t>(degree) * 2;
            auto* st = stats + index * 8;
            auto* message = errors + index * 256;
            std::fill(
                out,
                out + static_cast<std::size_t>(degree) * 2,
                std::numeric_limits<double>::quiet_NaN());
            std::fill(st, st + 8, 0);
            st[0] = 8;
            st[2] = degree;
            st[7] = degree;
            std::fill(message, message + 256, '\0');
            try {
                const auto* begin =
                    descending + index * (static_cast<std::size_t>(degree) + 1);
                std::vector<double> input(begin, begin + degree + 1);
                while (st[7] > 0 && input[st[7]] == 0) {
                    --st[7];
                }
                const auto result = solve_descending(input, max_sweeps);
                require(
                    result.roots.size() <= static_cast<std::size_t>(degree),
                    Status::identity_failure,
                    "root count exceeds output capacity");
                for (std::size_t i = 0; i < result.roots.size(); ++i) {
                    out[2 * i] = result.roots[i].real();
                    out[2 * i + 1] = result.roots[i].imag();
                }
                st[0] = result.status == SolveStatus::converged ? 0 : 1;
                st[1] = static_cast<std::int32_t>(result.roots.size());
                st[2] = result.unconverged;
                st[3] = result.sweeps;
                st[4] = result.deflations;
                st[5] = result.exceptional;
            } catch (const Failure& failure) {
                switch (failure.status) {
                case Status::invalid_input:
                    st[0] = 2;
                    break;
                case Status::bounds:
                    st[0] = 3;
                    break;
                case Status::numerical_range:
                    st[0] = 4;
                    break;
                case Status::identity_failure:
                    st[0] = 5;
                    break;
                }
                std::snprintf(message, 256, "%s", failure.what());
            } catch (const std::exception& failure) {
                st[0] = 7;
                std::snprintf(message, 256, "%s", failure.what());
            } catch (...) {
                st[0] = 8;
                std::snprintf(message, 256, "unknown CPU exception");
            }
        }
    };
    std::int32_t status = 0;
    try {
        threads.reserve(workers);
        for (int i = 0; i < workers; ++i) {
            threads.emplace_back(work);
        }
    } catch (...) {
        status = -2;
    }
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    return status;
}

}  // namespace cuphoton::xray
