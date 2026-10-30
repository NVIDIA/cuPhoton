// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Selected-root double-double refinement. Explicit FMA; contraction off.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "structured_api.h"

namespace cuphoton::xray {
namespace {
static_assert(
    std::numeric_limits<double>::is_iec559
        && std::numeric_limits<double>::digits == 53,
    "Double-double refinement requires IEEE binary64");
static_assert(
    sizeof(double) == 8 && sizeof(std::uint64_t) == 8
        && std::numeric_limits<double>::radix == 2
        && std::numeric_limits<double>::min_exponent == -1021
        && std::numeric_limits<double>::max_exponent == 1024,
    "Bit classification requires the binary64 storage and exponent format");
enum Status {
    ok = 0,
    nonfinite = 1,
    range = 2,
    derivative = 3,
    association = 4,
    cap = 5,
    invalid = 6
};
struct Failure {
    Status status;
};
struct DD {
    double hi = 0, lo = 0;
};
struct CD {
    DD re, im;
};
constexpr int iterations = 8, columns = 8;
constexpr double tolerance = 0x1p-80;

double checked(double x) {
    std::uint64_t magnitude;
    std::memcpy(&magnitude, &x, sizeof(magnitude));
    magnitude &= UINT64_C(0x7fffffffffffffff);
    const unsigned exponent = static_cast<unsigned>(magnitude >> 52);
    if (exponent == 2047) {
        throw Failure{nonfinite};
    }
    if (magnitude != 0 && (exponent < 573 || exponent > 1473)) {
        throw Failure{range};
    }
    return x;
}
DD sum(double a, double b) {
    checked(a);
    checked(b);
    double s = checked(a + b), bv = checked(s - a), av = checked(s - bv);
    return {s, checked(checked(a - av) + checked(b - bv))};
}
DD product(double a, double b) {
    checked(a);
    checked(b);
    // Exponents of operands lie in [-450,450]. A nonzero exact product
    // residual has exponent >= -1004 (two 53-bit significands), above the
    // normal binary64 boundary. Reject results outside the narrower window;
    // never mistake a silently lost subnormal residual for an exact product.
    double p = checked(a * b);
    return {p, checked(std::fma(a, b, -p))};
}
DD add(DD a, DD b) {
    DD s = sum(a.hi, b.hi), t = sum(a.lo, b.lo);
    DD u = sum(s.hi, checked(s.lo + t.hi));
    return sum(u.hi, checked(u.lo + t.lo));
}
DD neg(DD a) {
    return {-a.hi, -a.lo};
}
DD sub(DD a, DD b) {
    return add(a, neg(b));
}
DD mul(DD a, DD b) {
    return add(
        add(product(a.hi, b.hi), product(a.hi, b.lo)),
        add(product(a.lo, b.hi), product(a.lo, b.lo)));
}
double rounded(DD a) {
    return checked(a.hi + a.lo);
}
DD divide(DD a, DD b) {
    if (b.hi == 0) {
        throw Failure{derivative};
    }
    DD q{checked(a.hi / b.hi), 0};
    for (int i = 0; i < 2; ++i) {
        DD residual = sub(a, mul(b, q));
        q = add(q, {checked(residual.hi / b.hi), 0});
    }
    return q;
}
DD scale(DD a, int exponent) {
    auto part = [=](double x) {
        if (x == 0) {
            return x;
        }
        int e = std::ilogb(checked(x)) + exponent;
        if (e < -450 || e > 450) {
            throw Failure{range};
        }
        return checked(std::scalbn(x, exponent));
    };
    return {part(a.hi), part(a.lo)};
}
CD add(CD a, CD b) {
    return {add(a.re, b.re), add(a.im, b.im)};
}
CD sub(CD a, CD b) {
    return {sub(a.re, b.re), sub(a.im, b.im)};
}
CD mul(CD a, CD b) {
    return {sub(mul(a.re, b.re), mul(a.im, b.im)),
        add(mul(a.re, b.im), mul(a.im, b.re))};
}
double magnitude(CD a) {
    return checked(std::hypot(rounded(a.re), rounded(a.im)));
}
CD divide(CD a, CD b) {
    double largest = std::max(std::abs(b.re.hi), std::abs(b.im.hi));
    if (largest == 0) {
        throw Failure{derivative};
    }
    int exponent = -std::ilogb(checked(largest));
    a = {scale(a.re, exponent), scale(a.im, exponent)};
    b = {scale(b.re, exponent), scale(b.im, exponent)};
    DD denominator = add(mul(b.re, b.re), mul(b.im, b.im));
    return {divide(add(mul(a.re, b.re), mul(a.im, b.im)), denominator),
        divide(sub(mul(a.im, b.re), mul(a.re, b.im)), denominator)};
}
struct Eval {
    CD p, d;
    double scale_p, scale_d;
};
Eval evaluate(const double* a, int degree, CD z) {
    Eval v{{{checked(a[0]), 0}, {0, 0}}, {{0, 0}, {0, 0}}, std::abs(a[0]), 0};
    double radius = magnitude(z);
    for (int i = 1; i <= degree; ++i) {
        // D consumes the OLD P; both use the exact original coefficient.
        v.d = add(mul(v.d, z), v.p);
        v.p = add(mul(v.p, z), CD{{checked(a[i]), 0}, {0, 0}});
        v.scale_d = checked(checked(v.scale_d * radius) + v.scale_p);
        v.scale_p = checked(checked(v.scale_p * radius) + std::abs(a[i]));
    }
    return v;
}
CD correction(const Eval& v) {
    if (v.scale_p == 0 || v.scale_d == 0
        || magnitude(v.d) <= tolerance * v.scale_d) {
        throw Failure{derivative};
    }
    return divide(v.p, v.d);
}
void record(
    double* history,
    int step,
    CD z,
    double relative_step,
    double residual,
    double derivative_ratio,
    int stable) {
    double values[] = {z.re.hi,
        z.re.lo,
        z.im.hi,
        z.im.lo,
        relative_step,
        residual,
        derivative_ratio,
        double(stable)};
    std::copy(values, values + columns, history + step * columns);
}
int refine_dd(
    const double* a,
    int degree,
    CD seed,
    double separation,
    double* output,
    double* history,
    int& steps) {
    CD z = seed;
    int stable = 0;
    Eval v = evaluate(a, degree, z);
    for (int i = 0; i < iterations; ++i) {
        CD candidate = sub(z, correction(v));
        if (magnitude(sub(candidate, seed)) >= separation / 10) {
            throw Failure{association};
        }
        if (seed.im.hi == 0 && (candidate.im.hi != 0 || candidate.im.lo != 0)) {
            throw Failure{invalid};
        }
        stable = (rounded(candidate.re) == rounded(z.re)
                     && rounded(candidate.im) == rounded(z.im))
            ? stable + 1
            : 0;
        z = candidate;
        // Both convergence quantities are evaluated at the RETURNED iterate.
        Eval final = evaluate(a, degree, z);
        double relative_step =
            magnitude(correction(final)) / std::max(1.0, magnitude(z));
        double residual = magnitude(final.p) / final.scale_p;
        record(
            history,
            i,
            z,
            relative_step,
            residual,
            magnitude(final.d) / final.scale_d,
            stable);
        output[0] = rounded(z.re);
        output[1] = rounded(z.im);
        steps = i + 1;
        if (stable >= 2 && relative_step <= tolerance
            && residual <= tolerance) {
            return ok;
        }
        v = final;
    }
    return cap;
}
}  // namespace

// ABI: descending[degree+1], seeds[count,2], separation[count], output[count,2],
// history[count,8,8], status_steps[count,2]. Double-double is mandatory.
// A nonzero row status is never eligible for replacing any roots in that row.
std::int32_t refine_selected(
    const double* descending,
    const double* seeds,
    const double* separation,
    double* output,
    double* history,
    int32_t* status_steps,
    int32_t degree,
    int32_t count) {
    if (!descending || !seeds || !separation || !output || !history
        || !status_steps || degree != 597 || count < 1 || count > 36
        || std::fegetround() != FE_TONEAREST) {
        return invalid;
    }
    std::fill(
        history,
        history + int64_t(count) * iterations * columns,
        std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < count; ++i) {
        output[2 * i] = seeds[2 * i];
        output[2 * i + 1] = seeds[2 * i + 1];
        int steps = 0, status = invalid;
        try {
            checked(seeds[2 * i]);
            checked(seeds[2 * i + 1]);
            if (!std::isfinite(separation[i]) || separation[i] <= 0) {
                throw Failure{invalid};
            }
            status = refine_dd(
                descending,
                degree,
                {{seeds[2 * i], 0}, {seeds[2 * i + 1], 0}},
                separation[i],
                output + 2 * i,
                history + i * iterations * columns,
                steps);
        } catch (Failure error) {
            status = error.status;
        } catch (...) {
            status = invalid;
        }
        status_steps[2 * i] = status;
        status_steps[2 * i + 1] = steps;
    }
    return ok;
}

}  // namespace cuphoton::xray
