"""Canonical inverse NLFT on causal analytic prefixes.

The exterior complement is defined by R=log(1-|b|²)/2 and its
negative-frequency Schwarz transform. F9/F10 in the phase derivation use
the reflected ratio prefix (1, conj(reverse(chi))) instead of recompleting
recursive subproblems. See arXiv:2505.12615v1 Algorithm 1 and Lemma 5.2,
arXiv:2407.05634v3 Propositions 17--23, and
docs/algorithms/qsp_phase_derivation.md for the conventions and qualifications.
"""

from __future__ import annotations

import math

import numpy as np


# Matched non-normalized prefixes at 16, 32 and 64 factors did not locate
# a direct/split crossover. Use the smallest tested leaf schedule, 16.
# The convolution cutoff bounds direct product work to small arrays.
# Neither fixed threshold claims a universal timing optimum.
LEAF_SIZE = 16
CONVOLUTION_CUTOFF = 64


def power_two(length):
    return 1 << (int(length) - 1).bit_length()


def convolution(a, b):
    """Linear convolution with explicit padding to its complete support."""
    length = len(a) + len(b) - 1
    if min(len(a), len(b)) == 1 or length <= CONVOLUTION_CUTOFF:
        return np.convolve(a, b)
    n = power_two(length)
    left = np.fft.fft(a, n)
    right = np.fft.fft(b, n)
    left *= right
    del right
    result = np.fft.ifft(left)
    return result[:length].copy()


def leaf(alpha, beta):
    """PF-I1 peeling, including the normalized transfer pair it generates.

Common analytic rescaling does not change a ratio beta[0]/alpha[0].
Consequently these prefixes need not satisfy a normalization identity.
    """
    alpha, beta = alpha.copy(), beta.copy()
    count = len(alpha)
    factors = np.empty(count, complex)
    for k in range(count):
        if alpha[0] == 0 or not np.isfinite(alpha[0]):
            raise FloatingPointError("inverse NLFT has a zero or nonfinite prefix pivot")
        factor = beta[0] / alpha[0]
        if not np.isfinite(factor):
            raise FloatingPointError("inverse NLFT produced a nonfinite factor")
        factors[k] = factor
        norm = math.hypot(1.0, abs(factor))
        next_alpha = (alpha + factor.conjugate() * beta) / norm
        next_beta = (beta - factor * alpha) / norm
        alpha, beta = next_alpha[:-1], next_beta[1:]
    # This transfer is formed from the extracted factors, never from the
    # generally unnormalized input prefixes. Its polynomial degree is h-1.
    p, b = np.ones(1, complex), np.zeros(1, complex)
    for k, factor in enumerate(factors):
        if k:
            p, b = np.r_[0j, p], np.r_[b, 0j]
        norm = math.hypot(1.0, abs(factor))
        p, b = (p - factor.conjugate() * b) / norm, (factor * p + b) / norm
    return factors, p[::-1].conj(), b


def inverse_prefix(alpha, beta, leaf_size=LEAF_SIZE):
    """F9 causal recursion returning factors and their transfer pair."""
    h = len(alpha)
    if h <= leaf_size:
        return leaf(alpha, beta)
    m = (h + 1) // 2
    yl, al, bl = inverse_prefix(alpha[:m], beta[:m], leaf_size)
    # Laurent conjugation contributes a shift -(m-1). F9a's second
    # shift is -m, where m counts factors, not polynomial degree.
    ar = convolution(al[::-1].conj(), alpha)
    ar += convolution(bl[::-1].conj(), beta)
    ar = ar[m - 1:m - 1 + h - m].copy()
    br = convolution(al, beta)
    br -= convolution(bl, alpha)
    br = br[m:h].copy()
    yr, at, bt = inverse_prefix(ar, br, leaf_size)
    del ar, br
    a = np.r_[convolution(al, at), 0j]
    a[1:] -= convolution(bl[::-1].conj(), bt)
    b = np.r_[0j, convolution(al[::-1].conj(), bt)]
    b[:-1] += convolution(bl, at)
    return np.r_[yl, yr], a, b


def weiss_ratio(b, n):
    """F1: four FFTs with the Nyquist frequency on the negative side."""
    values = np.fft.ifft(b, n)
    values *= n
    modulus = np.abs(values)
    modulus *= modulus
    if not np.all(np.isfinite(modulus)) or np.any(modulus >= 1):
        raise FloatingPointError("canonical completion requires positive 1-|b|² at every FFT node")
    modulus *= -1
    np.log1p(modulus, out=modulus)
    modulus *= 0.5
    spectrum = np.fft.fft(modulus)
    spectrum /= n
    del modulus
    schwarz = np.zeros(n, complex)
    schwarz[0] = spectrum[0]
    schwarz[n // 2:] = 2 * spectrum[n // 2:]
    del spectrum
    logarithm = np.fft.ifft(schwarz)
    logarithm *= -n
    del schwarz
    np.exp(logarithm, out=logarithm)
    values *= logarithm
    del logarithm
    ratio = np.fft.fft(values)
    ratio /= n
    return ratio[:len(b)].copy()


def factors_from_target(b, n):
    if len(b) == 1:
        if abs(b[0]) >= 1:
            raise FloatingPointError("canonical completion requires |b| < 1")
        return b / math.sqrt((1 - abs(b[0])) * (1 + abs(b[0])))
    chi = weiss_ratio(b, n)
    one = np.zeros(len(b), complex)
    one[0] = 1
    y, _, _ = inverse_prefix(one, chi[::-1].conj())
    return y[::-1].conj()


def matrix_product(rotations, start=0, stop=None):
    """Balanced product A0 (D A1) ... (D Ad), degree on the last axis."""
    if stop is None:
        stop = len(rotations)
    if stop - start == 1:
        a = rotations[start]
        if start == 0:
            return a[:, :, None].copy()
        out = np.zeros((2, 2, 2), complex)
        out[0, :, 1] = a[0]
        out[1, :, 0] = a[1]
        return out
    middle = (start + stop) // 2
    left = matrix_product(rotations, start, middle)
    right = matrix_product(rotations, middle, stop)
    out = np.empty((2, 2, left.shape[2] + right.shape[2] - 1), complex)
    for row in range(2):
        for column in range(2):
            out[row, column] = convolution(left[row, 0], right[0, column])
            out[row, column] += convolution(left[row, 1], right[1, column])
    return out


def known_array_bytes(degree, n):
    """Coarse bytes of the dominant arrays of one synthesis attempt at FFT length ``n``.

Weiss: four complex-N buffers, two real-N buffers and a complex-N FFT copy
(96N), plus four h buffers, h = degree + 1. The depth-first inverse and the
reconstruction hold at most a few hundred complex-h buffers, with the
largest padded convolution below 4h. This is an estimate of the arrays of
this implementation, not process memory or FFT library workspace.
    """
    h = degree + 1
    return max(96 * n + 64 * h, (2048 + 256 + 128) * h)


def generic_ratio_bound(degree, rho, n):
    """PF-G2's ideal per-coefficient error, evaluated without overflow."""
    if degree == 0 or rho == 0:
        return 0.0
    eta = 1 - rho
    log_r = -math.log(rho) / (2 * degree)
    if n <= 2 * degree or n * log_r <= math.log(2):
        return math.inf
    log_xi = (math.log(5) + math.log(-math.log1p(-rho)) - n * log_r / 2
              - math.log(-math.expm1(-log_r)))
    if log_xi > math.log(700):
        return math.inf
    # A positive discretization bound cannot become a claim of exact zero
    # when its floating-point evaluation underflows.
    log_tiny = math.log(float(np.nextafter(0.0, 1.0)))
    xi = math.exp(max(log_xi, log_tiny))
    first = 4 * math.exp(max((2 * degree - n) * log_r, log_tiny)) / eta
    return first + math.expm1(xi) / math.sqrt(eta)


def propagated_ratio_bound(degree, rho, delta):
    slack = (1 - rho) * (1 + rho)
    if delta >= slack or not math.isfinite(delta):
        return math.inf
    return (degree + 1) * delta * (1 + slack) / (slack * (slack - delta))


def _near_flat_bound(degree, rho, b_l1, sigma, delta, n):
    """PF-J1--J6 at FFT length n for a full complex unit-modulus approximation only.

    Returns the smallest propagated bound over the orders j whose length
    next_power_of_two(max(2j, j²+1)d+1) is at most n, or inf.
    """
    flat = 1 - sigma ** -2
    q = (2 * delta + delta * delta) / (sigma * sigma - 1)
    if not (flat > 0 and 0 <= q < 1):
        return math.inf
    best, v, j = math.inf, 0.0, 1
    while power_two(max(2 * j, j * j + 1) * degree + 1) <= n:
        if q == 0:
            ratio_error = 0.0
        else:
            v += math.sqrt(2 * j * degree + 1) * q ** j / j
            remainder = math.sqrt(2 * degree + 1) * q ** (j + 1) / (
                2 * math.sqrt(j + 1) * (1 - q))
            if 2 * remainder > 700 or v > 700:
                ratio_error = math.inf
            else:
                taylor_log = (j + 1) * math.log(v) - math.lgamma(j + 2) if v else -math.inf
                if taylor_log > 700:
                    ratio_error = math.inf
                else:
                    ratio_error = (2 * b_l1 / math.sqrt(flat) * math.exp(v)
                                   * (math.expm1(2 * remainder) + math.exp(taylor_log)))
                    if ratio_error == 0 and b_l1:
                        ratio_error = float(np.nextafter(0.0, 1.0))
        best = min(best, propagated_ratio_bound(degree, rho, ratio_error))
        j += 1
    return best


def real_log_radius(coefficients, rho):
    """R4--R8 real-Wx annulus, from actual normalized Chebyshev coefficients.

For log r=2y, bb*=P(cos(theta+i v))² on z=exp(2i(theta+i v)).
With K=cosh(1/2), C1=sum k|ck|, C2=sum k²|ck|, the strip bound is
rho²+y² C, C=rho K C2+K² C1²+K² C2²/(16d²). Choosing
y=min(1/(2d),sqrt((1-rho²)/(2C))) leaves gap h=(1-rho²)/2.
See the real-Wx annulus derivation in qsp_phase_derivation.md.
Returns None when the annulus arithmetic is unavailable.
    """
    degree = len(coefficients) - 1
    if degree == 0 or not np.any(coefficients[1:] != 0):
        return None
    orders = np.arange(1, degree + 1, dtype=float)
    weighted = orders * abs(coefficients[1:])
    first = float(np.sum(weighted))
    second = float(np.sum(orders * weighted))
    k = math.cosh(0.5)
    c = rho * k * second + k * k * first * first + k * k * second * second / (16 * degree * degree)
    slack = (1 - rho) * (1 + rho)
    if not math.isfinite(c) or c <= 0 or not math.isfinite(slack) or slack <= 0:
        return None
    y = min(0.5 / degree, math.sqrt(slack / (2 * c)))
    if not math.isfinite(y) or y <= 0:
        return None
    return 2 * y


def real_ratio_bound(degree, rho, log_r, n):
    """R10--R13: log/Schwarz error and the true ratio's negative aliases."""
    if degree == 0:
        return 0.0
    slack = (1 - rho) * (1 + rho)
    gap = slack / 2
    log_tiny = math.log(float(np.nextafter(0.0, 1.0)))
    log_xi = math.log(5) + math.log(math.log(1 / gap)) - n * log_r / 2 - math.log(-math.expm1(-log_r))
    if log_xi > math.log(700):
        return math.inf
    xi = math.exp(max(log_xi, log_tiny))
    log_alias = math.log(rho / gap) + (degree - n) * log_r - math.log(-math.expm1(-n * log_r))
    alias = math.exp(max(log_alias, log_tiny))
    return alias + rho / math.sqrt(slack) * math.expm1(xi)


def ideal_rule(degree, rho, tolerance, n, *, log_r=None, near_flat=None):
    """Return ``(rule, bound)``: an ideal-arithmetic rule that calls FFT length ``n`` sufficient, or ``(None, None)``.

    Only scalar formulas are evaluated, at the length actually reached.
    ``generic_sufficient`` (PF-G2) applies to every target with norm cap
    rho < 1; ``real_wx_annulus`` (R15, allowance
    min(h/(2(d+1)), tol h²/(4(d+1)²(1+h))) with h = 1 - rho²) needs the
    real-Wx log radius ``log_r``; ``full_ja_sufficient`` (PF-J) needs
    ``near_flat = (sum|b|, sigma, delta)`` of a full unit-modulus
    approximation. Among the rules whose propagated ideal ratio bound at
    ``n`` is at most tolerance/2, the smallest bound is returned. Each bound
    decreases with n, so it also holds at every longer length. These are
    conditional ideal-arithmetic statements, not floating-point
    certificates; synthesis accepts only the actual coefficient residual.
    """
    if degree == 0:
        return "constant", 0.0
    bounds = {"generic_sufficient": propagated_ratio_bound(
        degree, rho, (degree + 1) * generic_ratio_bound(degree, rho, n))}
    if log_r is not None:
        count = degree + 1
        slack = (1 - rho) * (1 + rho)
        allowance = min(slack / (2 * count), tolerance * slack * slack / (4 * count * count * (1 + slack)))
        ratio = real_ratio_bound(degree, rho, log_r, n)
        bounds["real_wx_annulus"] = (propagated_ratio_bound(degree, rho, count * ratio)
                                     if ratio <= allowance else math.inf)
    if near_flat is not None:
        bounds["full_ja_sufficient"] = _near_flat_bound(degree, rho, *near_flat, n)
    rule = min(bounds, key=bounds.get)
    return (rule, bounds[rule]) if bounds[rule] <= tolerance / 2 else (None, None)
