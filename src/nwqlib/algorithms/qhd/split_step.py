"""Split-step classical evolution of the restricted QHD grid model in the kinetic eigenbasis.

The ``split_step`` flavor of ``method.QHD`` evolves the ``D = K**d``
amplitudes of the grid model ``H(t) = a(t) K + b(t) V`` with one symmetric
(Strang) product per step. In ``H(t)`` and the operator formulas of this
docstring K is the kinetic operator, and in ``K**d`` it is the number of
grid points per variable.

    ``S_k = exp(-i beta_k V) exp(-i alpha_k K) exp(-i beta_k V)``,

applied to a column vector from the right, so the chronological order is
potential half, kinetic, potential half. With ``dt = total_time/num_steps``
and the step weights ``(t_k, a_k, b_k)`` of ``schedules.step_weights``,
``alpha_k = dt a_k`` and ``beta_k = dt b_k/2``. Under the ``"midpoint"``
rule ``a_k = a(t_k)`` and ``b_k = b(t_k)`` at the step midpoint. Under the
``"integrated"`` rule ``alpha_k = A_k`` and the two potential halves are
equal, ``B_k/2`` each, with ``A_k`` and ``B_k`` the interval integrals of a
and b over the step. The equal halves use only the whole-step integrals.
Reversing the interval changes the sign of A and B, so the reversed step
is the inverse step and the step is time symmetric. Splitting B into its
chronological half-integrals, which exchange with opposite signs under
reversal, gives another time-symmetric second-order method whose leading
time-ordering defect differs by ``(dt**3/8)(a b')_m [K, V]``, and neither
convention is uniformly more accurate.

The kinetic factor is applied exactly in the eigenbasis of the selected
kinetic operator, so its cost per step does not depend on ``alpha_k``. This
is the pseudo-spectral step of Leng et al., arXiv:2303.01471v1, Eq. (C.3),
p. 32, which is first order (potential, then kinetic, with the
left-endpoint weights ``a_j = exp(phi_(j s))``, ``b_j = exp(chi_(j s))`` of
their Eq. (C.2), where s = T/N is their step size, dt here). The symmetric
placement is Strang's splitting (G. Strang, SIAM J. Numer. Anal. 5 (1968)
506-517, doi:10.1137/0705041). Wu et al., arXiv:2605.12066v1, Sec. VI,
p. 7, simulate their benchmarks with this split-step Fourier method and
Strang splitting.

Time accuracy has three separate parts, for a fixed grid, fixed V, fixed
schedule parameters and coefficients a and b with four bounded continuous
derivatives. The constants depend on operator norms, commutators and
schedule derivatives and are not uniform as s tends to zero, the grid is
refined or a penalty grows.

1. Coefficient quadrature. The midpoint rule's remainder
   ``∫ f = dt f(m) + (dt**3/24) f''(m) + O(dt**5)`` over a step of midpoint
   m means that midpoint freezing misses the first Magnus term's correction
   ``-i (dt**3/24)(a'' K + b'' V)_m``. The integrated rule's exact integrals
   remove it. A left-endpoint rule, ``∫ f = dt f(t) + (dt**2/2) f'(t) +
   O(dt**3)``, would miss ``-i (dt**2/2)(a' K + b' V)`` and be first order,
   which symmetric placement does not repair, so it is not offered.
2. Time ordering. For sufficiently small dt, the exact step propagator is
   the Magnus exponential ``exp(Omega_1 + Omega_2 + ...)`` of ``-i H(t)``,
   with ``Omega_1 = -i (A_k K + B_k V)`` and
   ``Omega_2 = (1/2) ∫∫_{v<u} [-i H(u), -i H(v)] du dv`` over the step.
   Since ``[H(u), H(v)] = (a(u) b(v) - b(u) a(v)) [K, V]``,
   ``Omega_2 = -(1/2)[K, V] ∫∫_{v<u} (a(u) b(v) - b(u) a(v)) du dv``.
   Write ``x = u - m`` and ``y = v - m``. Expanding a and b about m gives
   the linear term ``(a b' - a' b)_m (y - x)`` and the quadratic term
   ``(a b'' - a'' b)_m (y**2 - x**2)/2``. The quadratic term integrates to
   zero because the centered triangle ``-dt/2 <= y <= x <= dt/2`` is
   invariant under ``(x, y) -> (-y, -x)``, which reverses its sign. The
   remaining integrand is O(dt**3), so its double integral is O(dt**5).
   With ``∫∫_{v<u} (u - v) du dv = dt**3/6``, this gives
   ``Omega_2 = (dt**3/12)(a b' - a' b)_m [K, V] + O(dt**5)``. Neither
   coefficient rule supplies this term. Exact coefficient integrals alone
   therefore do not generally resolve time ordering when ``[K, V] != 0``.
3. Splitting. The symmetric Baker-Campbell-Hausdorff formula
   ``log(exp(X/2) exp(Y) exp(X/2)) = X + Y + [Y, [Y, X]]/12 -
   [X, [X, Y]]/24 + O(5)``, which follows from expanding the three
   exponentials to third order, gives with ``X = -i B V`` and
   ``Y = -i A K``
   ``log(exp(-i B V/2) exp(-i A K) exp(-i B V/2)) = -i (A K + B V) +
   (i A**2 B/12)[K, [K, V]] + (i A B**2/24)[V, [K, V]] + O(dt**5)``,
   with A and B of order dt. The ``schrodinger`` flavor applies
   ``exp(-i(A K + B V))`` and has no such term.

Each step therefore has local error O(dt**3), and the global error at fixed
total time is O(dt**2) under both rules. For ``[K, V] = 0``, which on a
connected grid means a constant V, the integrated rule is exact in time.
"""

from __future__ import annotations

from math import fsum, isfinite

import numpy as np
import scipy.fft

from nwqlib._validation import GLOBAL_PHASE_STATE_ROUNDOFF, UNIT_ROUNDOFF

# Qualification constant C of the transform accuracy assumption
# ``||fl(T x) - T x||_2 <= C u L(K) ||x||_2`` for each normalized length-K
# transform T that the kernel calls, with ``L(K) = ceil(log2 K)`` for the
# FFT and its inverse and ``L(K) = ceil(log2(2 (K + 1)))`` for DST-I, whose
# logical Fourier length is 2(K + 1) (``transform_levels``). It must cover
# the normalization, the twiddle factors and every factorization of the
# installed backend, including Bluestein's algorithm for lengths with large
# prime factors, and for the complex DST the transforms of its real and
# imaginary parts. SciPy 1.18.1 routes scipy.fft.fft, ifft and dst through
# ducc0.fft (the extension scipy.fft._duccfft.pyduccfft, SciPy 1.18.0 release
# notes, "scipy.fft improvements"), and neither SciPy nor ducc0 publishes such
# a bound, so C is a stated qualification assumption about that backend, not
# a derived constant, and the budget that uses it is first order in u.
# The canary test_scipy_transforms_meet_the_qualification_constant in
# tests/test_qhd_split_step.py compares SciPy 1.18.1's FFT and inverse FFT
# at ten lengths and its DST-I at eight lengths between 2 and 127,
# including primes and a DST-I logical length with a large prime factor,
# with 80-digit mpmath sums on three unit inputs each. The largest error it
# measured was 0.80 in these units, for the forward FFT at K = 2
# (2026-09-26, revision aede9fd119563d63562c9afed4b00021920560c6, Python
# 3.12.14, NumPy 2.5.2, mpmath 1.3.0, macOS arm64). A measurement can
# falsify C but cannot prove a bound for every length. Registered in
# docs/ENGINEERING_CONSTANTS.md. Revisit when SciPy changes its FFT backend
# or the canary fails.
TRANSFORM_ROUNDOFF = 5

# Relative error, in units of u, of one kinetic eigenvalue
# ``E = 2 (g(x)/h)**2`` from ``kinetic_eigenvalues``, with x in [0, pi/2]
# and g = sin or the identity. Forming ``x = pi (m/n)`` rounds m/n and the
# product, and np.pi differs from pi by less than u, so x carries at most 3u.
# On [0, pi/2] the condition number ``x cot x`` of sin is at most 1, and one
# ulp of sin adds 2u, so ``g(x)`` carries at most 5u. The division by the
# exact binary64 spacing h adds u, squaring doubles the 6u and adds u, and
# doubling is exact: 13u to first order. The identity g carries 3u, so the
# spectral eigenvalue is within 9u. Derived, not tuned. Registered in
# docs/ENGINEERING_CONSTANTS.md. Revisit when the eigenvalue formula or the
# sine model changes.
EIGENVALUE_ROUNDOFF = 13

# Transform-scratch allowance of the byte law (``sizes``), measured, not
# derived, with SciPy 1.18.1 and NumPy 2.5.2 on macOS arm64 on 2026-09-26.
# SciPy 1.18.1's transform backend, ducc0.fft (scipy.fft._duccfft), keeps its
# plans and fiber buffers in native memory, which tracemalloc does not see.
# The native observation is the growth of the process high-water mark
# (resource.getrusage(RUSAGE_SELF).ru_maxrss) across one in-place transform
# pair along either axis of a K-by-K array in a fresh process with one
# transform worker, less the traced allocations. The measuring script called
# scipy.fft alone, so this figure does not depend on the NWQLib revision. It
# was at most 688 bytes per point of K over two runs at K = 1008, 1009, 1023
# and 1024, which include DST-I lengths whose logical length has large prime
# factors, and at most 201 per point for one-dimensional lengths near 10**6.
# A high-water mark grows only when the process exceeds its earlier peak and
# then in whole pages, so an allocation that fits below an earlier peak does
# not show, and these figures are samples, not an upper bound on the
# backend's scratch.
# SCRATCH_BYTES_PER_POINT charges 1024 per point, for one worker, which
# _dst and _transform fix whatever scipy.fft.set_workers selects. Beyond the
# arrays that the law counts, tracemalloc saw at most 7034 bytes of Python
# and NumPy objects (array headers, the einsum buffer of the kinetic
# moments, per-axis index and phase temporaries) at d = 1, rising to 17340
# at d = 8, once the process had made its first scipy.fft call, which loads
# about 1.6 MB of modules and caches once. The cases were three-step
# evolutions on both boundaries and both kinetic models, with d = 1 to 3 and
# K = 4 to 128 and with d = 4 to 8 and K = 2 and 4, at most 300000 points,
# in two runs on 2026-09-27 at develop
# 21d75e3c13a38c10988eb5e649a9567897751f87, whose evolve forms the scaled
# kinetic and potential moments. The two runs differed by up to 1426 bytes
# at one d, so the traced figures vary between runs by that much.
# SCRATCH_BYTES_PER_AXIS charges 4096 (d + 1), above every measured case.
# Registered in docs/ENGINEERING_CONSTANTS.md. Revisit when SciPy changes
# its FFT backend, when evolve changes its temporaries, or when a workload's
# admission approaches max_bytes.
SCRATCH_BYTES_PER_POINT = 1024
SCRATCH_BYTES_PER_AXIS = 4096


def transform_levels(grid):
    """Return the level count L of the kernel's transforms on ``grid``.

    The periodic grid uses the length-K FFT, ``L = ceil(log2 K)``. The
    Dirichlet grids use DST-I, which is the discrete Fourier transform of
    the odd extension of its input, of logical length 2(K + 1), so
    ``L = ceil(log2(2 (K + 1)))``. L is the length measure of the
    qualification assumption, not a count of the backend's passes. For an integer n >= 1, ``ceil(log2 n)`` is the bit length of
    n - 1. L scales both the transform roundoff (``TRANSFORM_ROUNDOFF``) and
    the transform work of ``sizes``.
    """
    k = grid.num_grid_points
    length = k if grid.boundary == "periodic" else 2 * (k + 1)
    return (length - 1).bit_length()


def kinetic_eigenvalues(grid, var_index, kinetic_model):
    """Return the eigenvalues of variable ``var_index``'s kinetic operator in the order of its transform.

    Entry m is the eigenvalue of the mode that output index m of the
    kernel's forward transform holds, for the grid's binary64 spacing h
    (``OneHotGrid.spacing``). Every value has the form ``2 (g(x)/h)**2`` with
    x in [0, pi/2] and carries at most ``EIGENVALUE_ROUNDOFF`` units of
    relative roundoff.

    - Dirichlet grids, ``"finite_difference"``. The stencil of
      ``theory.restricted_kinetic_sparse`` is ``tridiag(-1, 2, -1)/(2 h**2)``
      with its own h, K + 1 intervals on the interior grid and K - 1 on the
      endpoint grid, where the endpoint amplitudes stay dynamical. Its
      eigenvectors ``S[j, r] = sqrt(2/(K + 1)) sin(pi (j + 1) r/(K + 1))``,
      r = 1..K, follow from ``sin((j + 2) t) + sin(j t) = 2 cos(t) sin((j +
      1) t)`` with ``t = pi r/(K + 1)``, whose sine vanishes at j = -1 and
      j = K, so ``E_r = (1 - cos t)/h**2 = (2/h**2) sin(pi r/(2 (K +
      1)))**2``. Orthonormal DST-I (``_dst``) is S, and its output index m
      holds r = m + 1.
    - Periodic grid, ``"finite_difference"``. The circulant stencil has
      diagonal ``1/h**2`` and ``-1/(2 h**2)`` at ``(i, i +- 1 mod K)``. The
      Fourier mode ``exp(2 pi i k j/K)`` has eigenvalue
      ``(1 - cos(2 pi k/K))/h**2 = (2/h**2) sin(pi k/K)**2``, the value
      that the unitary FFT's output index k holds.
    - Periodic grid, ``"spectral"``. The Fourier mode k is the grid sample
      of ``exp(2 pi i q x/L)`` with period ``L = K h`` for every integer q
      congruent to k modulo K. The signed index ``q = k`` for
      ``k < ceil(K/2)`` and ``q = k - K`` otherwise, the order of
      ``scipy.fft.fftfreq(K) * K``, picks the alias of least magnitude, so
      for even K the Nyquist mode K/2 has ``q = -K/2``. The eigenvalue of
      ``-(1/2) d**2/dx**2`` on that alias is
      ``E_k = 2 pi**2 q**2/L**2 = 2 (pi |q|/(K h))**2``, which is
      ``(2 pi fftfreq(K, h))**2/2``. The unsigned index k would give the
      mode K - 1 the energy of momentum K - 1 instead of 1.

    Both periodic formulas depend on q only through ``|q| = min(k, K - k)``,
    since ``sin(pi k/K)**2 = sin(pi (K - k)/K)**2``, which keeps x in
    [0, pi/2], where sin is well conditioned. At the Nyquist mode
    ``E_sp/E_FD = pi**2/4``, and for ``|q| <= K/2``,
    ``0 <= E_sp - E_FD = (2/h**2)(x**2 - sin(x)**2) <= (2/h**2) x**4/3 =
    2 pi**4 q**4 h**2/(3 L**4)`` with ``x = pi q/K``, because
    ``x**2 - x**4/3 <= sin(x)**2 <= x**2``.

    Raises:
        ValueError: The model is not ``"finite_difference"`` or
            ``"spectral"``, or it is ``"spectral"`` on a Dirichlet grid
            (``method.QHD._spectral_kinetic_grid``).
    """
    if kinetic_model not in ("finite_difference", "spectral"):
        raise ValueError(f"unknown kinetic model {kinetic_model!r}")
    if kinetic_model == "spectral" and grid.boundary != "periodic":
        raise ValueError("the spectral kinetic model requires the periodic grid")
    k, h = grid.num_grid_points, grid.spacing(var_index)
    if grid.boundary == "periodic":
        index = np.arange(k)
        # Signed Fourier index q_k: k below ceil(K/2), k - K from there on.
        signed = np.where(index < -(-k // 2), index, index - k)
        # x = pi |q|/K in [0, pi/2]
        x = np.pi * (np.abs(signed) / k)
        g = x if kinetic_model == "spectral" else np.sin(x)
    else:
        # x = pi r/(2 (K + 1)) for the DST-I mode r = m + 1 of output index m
        x = np.pi * (np.arange(1, k + 1) / (2 * (k + 1)))
        g = np.sin(x)
    # E = 2 (g(x)/h)**2
    return 2.0 * (g / h) ** 2


def potential_diagonal(reconstruction, grid):
    """Return the potential ``V_0`` on the grid as a float64 array of shape ``(K,) * d``, axis j for variable j.

    ``V_0[n_0, ..., n_(d-1)] = sum_S v_S(n_S)`` with the stored support
    tables v_S of the reconstruction (``records.SupportValues``), whose values
    are in lexicographic order of their sorted support, first variable most
    significant. Reshaping a table to K along its support axes and 1 along the
    others keeps that order, and broadcasting adds it to every grid point.
    The objective is not evaluated again. The objective constant c is left
    out, because ``c I`` commutes with every factor and contributes only the
    global phase that ``method._constant_phase`` restores. Starting from
    zero, the sum ``v_1 + ... + v_M`` takes M - 1 roundings, so each entry is
    within ``gamma_(M-1) sum_S |v_S|`` of the exact sum of the stored values
    (Higham, doi:10.1137/1.9780898718027, Sec. 4.2). ``state_error`` charges
    that term. One table is converted to an array at a time.
    """
    d, k = grid.num_variables, grid.num_grid_points
    potential = np.zeros((k,) * d)
    for table in reconstruction.support_values:
        shape = [k if j in table.support else 1 for j in range(d)]
        # V += v_S broadcast along the axes outside S
        potential += np.asarray(table.values, dtype=float).reshape(shape)
    return potential


def _dst(state, axis):
    """Return the orthonormal DST-I of a C-contiguous complex array along ``axis``.

    DST-I is real and linear, so the complex transform is the real transform
    of the real and imaginary parts. Viewing the complex128 array as float64
    of shape ``state.shape + (2,)``, whose last axis holds the two parts,
    transforms both parts in one call along the same axis without copying
    them apart. With ``norm="ortho"`` DST-I is its own inverse (SciPy's
    ``scipy.fft.dst`` documentation), so ``_kinetic_axis`` calls it on both
    sides of the phase. ``workers=1`` fixes the one-worker setting under
    which the transform-scratch allowance was measured, whatever
    ``scipy.fft.set_workers`` the caller has set.
    """
    parts = state.view(np.float64).reshape(state.shape + (2,))
    parts = scipy.fft.dst(parts, type=1, axis=axis, norm="ortho", overwrite_x=True, workers=1)
    return np.ascontiguousarray(parts).view(np.complex128).reshape(state.shape)


def _transform(state, axis, periodic, inverse=False):
    """Return the kernel's unitary kinetic transform T, or its inverse, of ``state`` along ``axis``.

    On the periodic grid T is the unitary DFT ``(F x)_k = K**(-1/2) sum_j
    exp(-2 pi i k j/K) x_j``, ``scipy.fft.fft(norm="ortho")``, and its inverse
    ``scipy.fft.ifft(norm="ortho")``. With ``norm="ortho"`` both are unitary,
    so no intermediate array grows by a factor K and one relative roundoff
    constant covers both (``TRANSFORM_ROUNDOFF``). On the Dirichlet grids T
    is the orthonormal DST-I (``_dst``), which is symmetric and its own
    inverse. The input may be overwritten. Both run with one worker
    (``SCRATCH_BYTES_PER_POINT``).
    """
    if periodic:
        call = scipy.fft.ifft if inverse else scipy.fft.fft
        return call(state, axis=axis, norm="ortho", overwrite_x=True, workers=1)
    return _dst(state, axis)


def _kinetic_axis(state, axis, angles, scale, periodic):
    """Return ``T^-1 diag(exp(-i angles)) T`` applied to ``state`` along ``axis``, and the moment ``||angles z||``.

    ``angles`` holds ``alpha E_m`` for the eigenvalues E of
    ``kinetic_eigenvalues`` in the transform's output order, and the phase
    broadcasts along the other axes (``_transform``). The moment is the
    root mean square of the angles weighted by the mode populations of the
    transformed input ``z = T state/||T state||``, observed after the forward
    transform and before the phase multiplication, where
    ``||angles z||**2 = sum_m angles_m**2 s_m/sum_m s_m`` with
    ``s_m = sum_l |(T state)_(m,l)|**2`` summed over the other axes l. The
    reduction forms the K entries of s and reads the state in place, and it
    does not change the state.

    Scaling. ``scale`` is the computed cap
    ``A = fl(|alpha| max_m E_m)`` for the computed eigenvalues E.
    Monotone, sign-symmetric rounding gives ``|fl(alpha E_m)| <= A``.
    For finite positive A the moment is evaluated as
    ``A (sum_m (angles_m/A)**2 s_m/sum_m s_m)**(1/2)``, whose squared ratios
    lie in [0, 1], so no square overflows even where ``angles_m**2`` would.
    For the exact eigenvalues, every nonzero mode has
    ``x >= pi/(2 (K + 1))`` in ``E = 2 (g(x)/h)**2``, with x in [0, pi/2].
    Both sine and the identity satisfy ``g(x)/g(pi/2) >= 2 x/pi``, so
    ``E_m/E_max >= (K + 1)**-2``. Under the normal relative-rounding
    premises of evolve, the computed ratio differs by a relative O(u).
    On a 64-bit array host ``K + 1 <= 2**63``, so its square is of order
    at least ``2**-252``, far above the normal-range floor ``2**-1022``.
    This argument does not cover underflow in the population products.
    The division and square add two roundings per mode. Including the
    population reductions over the other axes, a conservative relative
    estimate for the diagnostic is ``O(D u)``, with ``D = state.size``.
    For ``D u << 1`` this affects its budget charge only at second order
    under the same arithmetic premises (``_potential_moment``).
    """
    shape = [1] * state.ndim
    shape[axis] = -1
    state = _transform(state, axis, periodic)
    # s_m, the squared magnitudes of the real and imaginary parts summed over every axis but this one
    parts = state.view(np.float64).reshape(state.shape + (2,))
    axes = list(range(state.ndim + 1))
    populations = np.einsum(parts, axes, parts, axes, [axis])
    total = float(populations.sum())
    moment = 0.0
    if total != 0 and scale != 0:
        # ||angles z|| = A (sum_m (angles_m/A)**2 s_m/sum_m s_m)**(1/2)
        moment = scale * (float(np.dot(np.square(angles / scale), populations)) / total) ** 0.5
    state *= np.exp(-1j * angles).reshape(shape)
    return _transform(state, axis, periodic, inverse=True), moment


def _potential_moment(state, potential, scratch):
    """Return the unscaled potential moment ``mu = ||V z||`` for ``z = state/||state||``.

    With ``w(x) = V(x) |state(x)|``, ``mu**2 = T/S`` for
    ``S = sum_x |state(x)|**2`` and ``T = sum_x w(x)**2``. ``scratch`` is a
    real array of the state's shape that holds no live data, the angle array
    of ``evolve`` after its phase array is formed. One call writes
    ``|state|`` into it (``np.abs``), forms S as its ``vdot`` with itself,
    overwrites it with w, finds ``M = max_x |w(x)|`` from its maximum and
    minimum, divides it by M and forms ``T/M**2`` as its ``vdot`` with
    itself, and returns ``mu = M (T/(M**2 S))**(1/2)``. It reads the state
    once and forms no new array of the state's size.

    Scaling. The squares are taken after the division by M, so they lie in
    [0, 1] and none overflows, and the largest is exactly 1, so the scaled
    sum is at least 1. A square that falls below the normal binary64 range
    is below ``2**-1022``, so all such squares together change that sum by
    a relative amount below ``D 2**-1022``, far below the relative
    ``O(D u)`` of the sums. The scaling is needed because the unscaled
    terms ``q V**2`` with ``q = |state|**2`` underflow for an objective of
    order 1e-200 and overflow for one of order 1e200. The largest weight
    has ``|w| <= |V| (1 + delta)`` for a state of norm about ``1 + delta``,
    so M is finite unless ``|V|`` is within that factor of the largest
    binary64 number. The caller replaces a moment that is not finite by
    its Plan maximum W. A product w(x) below the normal range carries an
    absolute error of at most ``2**-1075``, so all of them move the moment
    by at most ``(D/S)**(1/2) 2**-1075``, and since a potential half charges
    ``2 u |beta| mu`` with ``|beta| < 2**1024``, their effect on each charge
    stays below ``2**-50 u (D/S)**(1/2)``.

    Rounding. Each ``|state(x)|`` carries one ulp of ``np.abs``, each scaled
    entry ``w(x)/M`` two more roundings (the product with V and the
    division) and each square one, and the sums, whose order NumPy and the
    BLAS ``vdot`` do not specify, at most ``gamma_h`` for h at most D - 1
    additions along any term's path, all terms being nonnegative. Under
    evolve's arithmetic premises, with S of order one and ``D u << 1``,
    normal rounding contributes a relative ``O(D u)`` moment error.
    Subnormal weighted products add the absolute allowance derived above,
    and a subnormal final rescaling adds at most ``2**-1075``. Thus the
    error is relative plus absolute, including when every weighted entry
    rounds to zero. In the charge ``2 u |beta| mu``, the relative part
    contributes ``O(D u**2 |beta| mu)`` and the absolute part contributes
    ``O(u**2 sqrt(D/S))`` for finite beta. The first-order budget omits
    these second-order contributions (``evolve``), so it does not require
    upward evaluation of this diagnostic. A zero state or a zero potential
    on the state's support gives the moment 0.
    """
    np.abs(state, out=scratch)
    # S = sum_x |state(x)|**2
    total = float(np.vdot(scratch, scratch))
    # w(x) = V(x) |state(x)| and M = max_x |w(x)|, from two reductions without a temporary array
    np.multiply(scratch, potential, out=scratch)
    largest = max(float(scratch.max()), -float(scratch.min()))
    if total == 0 or largest == 0:
        return 0.0
    np.divide(scratch, largest, out=scratch)
    # mu = M (sum_x (w(x)/M)**2/S)**(1/2)
    return largest * (float(np.vdot(scratch, scratch)) / total) ** 0.5


def _capped(moment, maximum):
    """Return an observed moment capped at its Plan maximum, the maximum itself for a moment that is not finite.

    The scaled evaluations of ``_kinetic_axis`` and ``_potential_moment``
    leave a moment that is not finite only when a weight overflows or the
    state is not finite, and the Plan maximum bounds every unit-input moment
    (``evolve``).
    """
    return min(moment, maximum) if isfinite(moment) else maximum


def evolve(grid, kinetic_model, reconstruction, dt, state, *, start, ceiling=None):
    """Return the final restricted state after one Strang step per stored step weight, and its observed state budget.

    ``state`` is the unit start vector of length ``K**d`` in lexicographic
    grid order (``theory.restricted_basis``), and the potential is
    ``potential_diagonal`` of ``reconstruction``, whose ``step_weights``
    give one step each. Reshaped in C order to ``(K,) * d``, axis j is
    variable j, because the lexicographic position is
    ``sum_j n_j K**(d-1-j)``. The kinetic operator is ``sum_j K_j`` with K_j
    acting on axis j alone, so its exponential is the commuting product of
    the per-axis exponentials, each one transform pair and one mode-phase
    multiplication. Each step applies ``exp(-i beta V)``, the d kinetic
    factors and ``exp(-i beta V)`` again, with ``alpha = dt a`` and
    ``beta = dt b/2`` (the module docstring). The two potential halves use
    one phase array formed once per step, and ``sizes`` counts these
    operations. A C-contiguous complex128 ``state`` is updated in place.

    Observed state budget. The second value is the first-order 2-norm
    error budget delta of the returned state that this evolution observed
    on its own trajectory. Its reference is the exact Strang product of
    ``state_error``, applied to the exact initial unit state. Norms are
    Euclidean norms over every entry of the restricted tensor. Each moment
    uses the actual input to its phase operation, or for a first potential
    half the moment that the preceding diagonal phase keeps (below), divided
    by that input's norm for error accounting only, and the evolving state
    is unchanged by this normalization.

    Write each computed operation as its exact unitary action on the
    computed input plus a local defect, ``x_l = U_l x_(l-1) + r_l``. Then
    ``x_L - U_L ... U_1 x_0 = U_L ... U_1 (x_0 - exact x_0) + sum_l
    U_L ... U_(l+1) r_l``, an exact identity, and since the exact later
    operations keep each defect's norm, ``||x_L - exact x_L|| <= delta_0 +
    sum_l ||r_l||``. A defect bounded on the actual input to its operation
    therefore needs no further charge for the earlier errors. For a phase
    ``diag(exp(-i theta_hat))`` in place of ``diag(exp(-i theta))`` with
    ``|theta_hat_m - theta_m| <= q_m``, the pointwise inequality
    ``|exp(i a) - exp(i b)| <= min(2, |a - b|)`` gives
    ``||(diag(exp(-i theta_hat)) - diag(exp(-i theta))) c||**2 <= sum_m
    |c_m|**2 min(2, q_m)**2``, so for a unit input the defect is the root
    mean square of the angle-error bounds weighted by the input's
    probabilities, not the largest one.

    - Kinetic phase of axis j in step k. The eigenvalue carries at most
      ``EIGENVALUE_ROUNDOFF`` = 13 units of relative roundoff, forming
      ``alpha_k = fl(dt a_k)`` 1 and the angle product 1, so
      ``eta_(j,k) = 15 u |alpha_k| ||E_j z||`` with
      ``||E_j z||**2 = sum_(m,l) E_(j,m)**2 |z_(m,l)|**2`` for z observed
      after the forward transform of axis j and before its phase, with l
      over the other axes (``_kinetic_axis``). The kernel applies one axis
      at a time, so each step has d such moments.
    - Potential half of step k. With M support tables v_S,
      ``W = sum_S max |v_S|`` and ``m = max(M - 1, 0)``, the summed
      potential is within ``gamma_m sum_S |v_S(x)| <= gamma_m W`` of
      ``V_0(x)`` (``potential_diagonal``), and forming ``beta_k`` and the
      angle product add two relative roundings, so
      ``eta_(V,k)(z) = u c_k (2 mu(z) + m W)`` with ``c_k = |beta_k|`` and
      the unscaled moment ``mu(z) = ||V_0 z||`` (``_potential_moment``).
      The m W term covers the summation error even when the tables cancel,
      it vanishes for one table, and the whole term is zero without tables.
      The phase array is shared, but the two inputs of a step differ, so
      each half is charged on its own input. The second half of step k is
      observed on the state ``y_k`` after all kinetic factors, giving
      ``mu_k = mu(y_k)``. The first half of step k + 1 acts on the state
      that step k's second half produced from ``y_k``, an exact diagonal
      unitary that keeps every magnitude ``|y_k(x)|`` and therefore the
      moment, so it is charged with ``mu_k`` again, and the first half of
      step 1 with ``mu_0`` of the initial state. Each interior pair is thus
      charged ``u (c_k + c_(k+1)) (2 mu_k + m W)``, the absolute
      coefficients added so that signed weights cannot cancel, and N steps
      need N + 1 moments instead of 2N. The computed multiplication changes
      each magnitude by a relative factor within ``1 +- p``,
      ``p <= GLOBAL_PHASE_STATE_ROUNDOFF u`` (the phase-application charge
      below), so the moment of the actual input to the next first half lies
      within the relative factor ``(1 + p)/(1 - p)`` of ``mu_k``. Charging
      ``mu_k`` omits at most ``2 u (2p/(1 - p)) c_(k+1) mu_k``, about
      ``20 u**2 c_(k+1) mu_k``, a second-order product of local charges that
      the first-order budget omits. So does measuring the stored ``V_0``
      instead of the computed angle array ``fl(beta_k V_0)``, which changes
      the moment by a relative factor within ``1 +- u``.
    - Transforms and phase applications. Each of the 2d transforms of a
      step adds ``eps_T = TRANSFORM_ROUNDOFF * L * u`` (``transform_levels``)
      and each of the d + 2 phase applications
      ``GLOBAL_PHASE_STATE_ROUNDOFF`` u, the same terms as ``state_error``.

    To first order in u, with ``delta_0 = start u``,
    ``delta = delta_0 + N_s (2 d eps_T + (d + 2) 5 u) + sum_k [sum_j
    eta_(j,k) + u c_k (2 mu_(k-1) + m W) + u c_k (2 mu_k + m W)]``, where the
    kinetic moments use the computed angles ``alpha E_j`` in place of the
    exact ones, which changes them at second order. The per-step charges
    are summed with ``math.fsum``. The first-order budget also
    bounds the error up to a global phase. Norm growth of the computed
    state comes from the transforms and phase applications alone, since an
    angle error keeps the norm, so normalizing the observed inputs differs
    from charging their raw norms only by products of the local charges.

    Each unit-input moment is at most its Plan maximum,
    ``||E_j z|| <= max E_j`` and ``2 ||V_0 z|| + m W <= (M + 1) W`` for
    ``M >= 1``, since ``||V_0 z|| <= max |V_0| <= W``, so the budget is at
    most ``state_error`` under the same first-order model. Each moment is
    therefore capped at its maximum, ``|alpha_k| max E_j`` for the computed
    kinetic angles and W for ``mu``, a moment that is not finite is replaced
    by that maximum (``_capped``), and the returned budget is the smaller of the
    observed sum and ``state_error``, so that a last-bit discrepancy of
    the floating-point sums cannot exceed the Plan's first-order ceiling.
    ``ceiling`` is that ``state_error`` value when the caller already holds
    it: a split-step Plan stores it at planning as
    ``QHDReconstruction.host_state_error``, from the same grid, kinetic
    model, tables, step weights, dt and start (``method._admit_host_windows``),
    so the evolution does not
    recompute it. With ``ceiling`` None, as for a reference evolution of a
    Plan of another flavor, whose stored budget is another flavor's or
    None, it is computed here.
    Both quantities are first order in u. A budget valid at every order in u
    would need a Plan ceiling valid at every order too, because enclosing
    some moments from above and keeping the linear ceiling of
    ``state_error`` does not bound the state error at every order.
    The transform constant and the one-ulp elementary-function premises are
    qualification assumptions, as in ``state_error``. The relative-roundoff
    charges assume round-to-nearest binary64 arithmetic without overflow,
    with normal nonzero quantities wherever a relative error is used, so an
    evolution whose intermediate values underflow would need absolute terms
    that the budget does not charge. The budget excludes schedule,
    table-input, spacing and discretization errors.
    ``method._execute_theory`` converts it with
    ``probability_difference_window(delta, 5u)`` into the tie window of the
    most probable point. The observations add the reductions and arrays
    that ``sizes`` lists, and no transform, operator application or second
    evolution.
    """
    d, k = grid.num_variables, grid.num_grid_points
    shape = (k,) * d
    state = np.ascontiguousarray(state, dtype=complex).reshape(shape)
    periodic = grid.boundary == "periodic"
    potential = potential_diagonal(reconstruction, grid)
    energies = [kinetic_eigenvalues(grid, j, kinetic_model) for j in range(d)]
    largest = [float(np.max(values)) for values in energies]
    tables = reconstruction.support_values
    # W = sum_S max |v_S| and m = max(M - 1, 0)
    magnitude = fsum(t.magnitude for t in tables)
    spare = max(len(tables) - 1, 0)
    # Per step, in units of u: 2d transforms and d + 2 phase applications.
    fixed = 2 * d * TRANSFORM_ROUNDOFF * transform_levels(grid) + (d + 2) * GLOBAL_PHASE_STATE_ROUNDOFF
    steps = reconstruction.step_weights
    charges = np.empty(len(steps))
    angle = np.empty(shape)
    phase = np.empty(shape, dtype=complex)

    def potential_charge(moment, beta):
        """Return eta_V/u = |beta| (2 mu + m W) for the unscaled moment mu of the half's input, capped at W."""
        return abs(beta) * (2 * _capped(moment, magnitude) + spare * magnitude)

    # mu_0 of the initial state. The angle array holds no data until the first step forms its angles.
    moment = _potential_moment(state, potential, angle)
    for step, (_time, kinetic_weight, potential_weight) in enumerate(steps):
        alpha = dt * kinetic_weight
        # beta = fl(dt b)/2, halving being exact
        beta = dt * potential_weight / 2
        # exp(-i beta V). Multiplying the real angle by -1j is exact, and np.exp
        # of the pure imaginary result is (cos, -sin) of the angle.
        np.multiply(potential, beta, out=angle)
        np.multiply(angle, -1j, out=phase)
        np.exp(phase, out=phase)
        # First half: the cached moment, mu_0 or the previous step's second-half input,
        # which the previous step's diagonal phase kept (2 mu + m W with this step's |beta|).
        terms = [fixed, potential_charge(moment, beta)]
        state *= phase
        for axis in range(d):
            # exp(-i alpha E) on the modes of this axis, and 15 ||alpha E_j z|| capped at |alpha| max E_j
            cap = abs(alpha) * largest[axis]
            state, kinetic = _kinetic_axis(state, axis, alpha * energies[axis], cap, periodic)
            terms.append((EIGENVALUE_ROUNDOFF + 2) * _capped(kinetic, cap))
        # Second half: mu_k of the state after the kinetic factors, cached for the next step's
        # first half. The angle array is free once the phase array is formed.
        moment = _potential_moment(state, potential, angle)
        terms.append(potential_charge(moment, beta))
        state *= phase
        charges[step] = fsum(terms)
    # delta = (start + sum_k rho_k/u) u, at most the Plan's first-order ceiling
    observed = (start + fsum(charges)) * UNIT_ROUNDOFF
    if ceiling is None:
        ceiling = state_error(grid, kinetic_model, tables, steps, dt, start=start)
    return state.reshape(-1), min(observed, ceiling)


def state_error(grid, kinetic_model, tables, step_weights, dt, *, start):
    """Return the first-order 2-norm error budget delta of ``evolve`` on the Method's initial state.

    ``tables`` are the Plan's ``SupportValues`` and ``step_weights`` its step
    rows (``QHDReconstruction``). The reference is the exact product of the step operators with
    ``alpha_k = dt a_k`` and ``beta_k = dt b_k/2`` for the binary64 dt and
    the stored step weights, the exact eigenvalues of ``kinetic_eigenvalues``
    at the grid's binary64 spacing h, and ``V_0`` the exact sum of the stored
    support tables (``potential_diagonal``), applied to the exact initial
    unit state. The objective constant only multiplies that product by a
    global phase (``method._constant_phase``), which the probabilities do not
    read. Relative to that product the computed state errs by at most

    ``delta = delta_0 + sum_k rho_k``,
    ``rho_k = 2 d eps_T + (d + 2) (2 + 2 sqrt(2)) u + sum_j eta_(j,k) +
    2 eta_(V,k)``,

    to first order in u, because each exact step is unitary and carries an
    earlier error to the final state unchanged in norm (telescoping). The
    sum omits products of the charges, which the product form
    ``(1 + delta_0) prod_k (1 + rho_k) - 1`` would add. They add at most
    about ``delta**2/2``, 0.3% of delta at ``delta = 6.7e-3``, the budget on
    the 64-by-64 periodic spectral grid of the unit square with
    ``ShiftedCubicSchedule(s=2e-4)``, 1000 integrated steps and total time
    10, where the kinetic angles dominate for an objective of order one
    (2026-09-26, revision aede9fd119563d63562c9afed4b00021920560c6). The
    ``expm_multiply`` budget is first order in the same way. The
    relative-roundoff charges assume round-to-nearest binary64 arithmetic
    without overflow or underflow, as in ``evolve``.

    - ``delta_0 = start u`` bounds the construction of the initial vector,
      the value of ``initial_state.restricted_state_error`` that
      ``method._host_state_error`` receives, the same start term as
      ``_validation.expm_multiply_state_error`` receives. It is 2 for the
      uniform state, whose D equal entries ``1/sqrt(D)`` take one correctly
      rounded square root and one division of an exactly representable D.
    - ``eps_T = TRANSFORM_ROUNDOFF * L * u`` for each of the 2d transforms
      of a step (``transform_levels``). A bound for every length-K fiber
      bounds the whole tensor with the same relative factor, because the
      squared fiber errors add.
    - Each of the d kinetic and two potential phase multiplications adds
      ``(2 + 2 sqrt(2)) u``, because the phase ``exp(i theta)`` at a binary64
      angle is within 2u under one-ulp cos and sin and a complex product
      errs by at most ``sqrt(2) gamma_2`` relative. This is
      ``GLOBAL_PHASE_STATE_ROUNDOFF``, 5 after rounding up, applied
      entrywise.
    - ``eta`` bounds the error of the computed phase angles, since
      ``|exp(i a) - exp(i b)| <= |a - b|``. For ``theta = alpha E`` with
      ``alpha = fl(dt a)`` (u), E within ``EIGENVALUE_ROUNDOFF`` u and one
      rounding of the product,
      ``eta_(j,k) = (EIGENVALUE_ROUNDOFF + 2) u |alpha_k| max_m E_(j,m)``.
      For ``theta = beta V_0`` with ``beta = fl(dt b)/2`` (u), ``V_0``
      within ``gamma_(M-1) W`` (``potential_diagonal``) and one rounding of
      the product, ``eta_(V,k) = (M + 1) u |beta_k| W`` with
      ``W = sum_S max |v_S|``, which bounds both ``|V_0|`` and the summands'
      absolute sum at every point. The same phase array serves both halves,
      and each application is charged separately.

    A large angle loses absolute accuracy when it is formed, which wrapping
    it afterwards cannot restore, so the eta terms grow with ``sum_k
    |alpha_k|``, the kinetic integral, and with ``sum_k |beta_k| W``, and
    they dominate for schedules whose kinetic weight is large near t = 0.
    The transform term is an assumption about SciPy's transform backend,
    ducc0.fft in SciPy 1.18.1 (``TRANSFORM_ROUNDOFF``), not a proved bound. The budget excludes the
    error of the step weights, of h and of the stored tables relative to the
    exact schedule, box and objective, and every time, splitting or grid
    discretization error. ``method._host_probability_window`` turns delta
    into the mass window, the same conversion as for the ``expm_multiply``
    flavors. The charges use the largest eigenvalue and the largest table
    magnitudes for every state, so this Plan budget is a ceiling. The kernel
    selects its most probable point with the budget that ``evolve`` observes
    on the actual trajectory, which weights each angle error by the state's
    own mode and point probabilities and is at most this one
    (``method._host_tie_window`` gives the ceiling of its tie window).

    Raises:
        ValueError: An eigenvalue, weight product or the budget is not finite.
    """
    d = grid.num_variables
    largest = fsum(float(np.max(kinetic_eigenvalues(grid, j, kinetic_model))) for j in range(d))
    # W = sum_S max |v_S|
    magnitude = fsum(t.magnitude for t in tables)
    # Per step, in units of u: 2d transforms and d + 2 phase multiplications.
    fixed = 2 * d * TRANSFORM_ROUNDOFF * transform_levels(grid) + (d + 2) * GLOBAL_PHASE_STATE_ROUNDOFF
    # rho_k/u = fixed + sum_j eta_(j,k)/u + 2 eta_(V,k)/u, with
    # sum_j eta_(j,k) = (EIGENVALUE_ROUNDOFF + 2) |alpha_k| sum_j max E_j and
    # 2 eta_(V,k) = 2 (M + 1) |beta_k| W. fsum reads the generator one step at
    # a time, so the budget holds no per-step array.
    charges = (
        fixed
        + (EIGENVALUE_ROUNDOFF + 2) * abs(dt * kinetic_weight) * largest
        + 2 * (len(tables) + 1) * (abs(dt * potential_weight) / 2) * magnitude
        for _time, kinetic_weight, potential_weight in step_weights
    )
    # delta = (start + sum_k rho_k/u) u
    delta = (start + fsum(charges)) * UNIT_ROUNDOFF
    if not np.isfinite(delta):
        raise ValueError(
            "split-step phase angles exceed the binary64 range. The kinetic eigenvalues, of order "
            "1/h**2, the schedule weights or the objective values are too large for this grid"
        )
    return delta


def sizes(grid, tables, num_steps, start_work):
    """Return ``(size_units, workspace_bytes)`` of one ``evolve`` call, its start vector and its potential diagonal.

    With ``D = K**d``, M support tables of at most ``T_max`` entries, N_s
    steps, the transform levels L of ``transform_levels`` and the work
    ``W_start`` of forming the start vector
    (``initial_state.start_vector_work``, ``start_work`` here),

    ``size_units = W_start + D (d + M + 7) + d K + N_s (2 d L D + (2 d + 13) D + 7 d K + d + 4)``.

    ``W_start`` is D for the direct uniform fill and
    ``d K + sum_(j=2)^d K**j + D`` for the stored real factors, charged once
    per evolution. ``D (d + M)`` counts forming V from the tables (M
    additions and d index positions per point), 7 D the potential moment
    of the initial state and ``d K`` the eigenvalues,
    separate from the ``d K`` inside ``W_start``. These once-per-evolution
    terms are nominal counts, not one unit per pass: ``kinetic_eigenvalues``
    runs several passes of K per axis, ``evolve`` forms the eigenvalues and
    the table magnitudes again when it compares its budget with
    ``state_error``, and that comparison reads the N_s step rows once
    more. Per step, each of the
    2d transforms is charged the nominal proxy ``L D``, the level count of
    ``transform_levels`` times the D entries it transforms. It is not a count
    of the backend's radix passes, twiddle products, Bluestein convolutions
    or planning work, which SciPy does not report. The per-step linear terms count
    one unit per element of each array pass: the d D kinetic and 2 D
    potential complex multiplications, 2 D for the potential phases (the
    product of the angles with -i and the exponential), 2 D potential
    angle products, d K kinetic angle products and 2 d K for the kinetic
    phases (the product with -i and the exponential in ``_kinetic_axis``).
    ``evolve`` forms the potential angles once per step for both halves, so
    their term charges D more than it runs. The observed state budget adds,
    per step, d D for the mode populations of the d kinetic moments (one
    ``einsum`` pass over the D entries per axis), 4 d K for weighting them
    (per axis the sum of the populations, the division of the angles by
    their cap, the square and the reduction against the populations,
    ``_kinetic_axis``), 7 D for the one potential moment of
    the step (``_potential_moment``: the magnitudes, their sum of squares,
    the product with V, its maximum and minimum, the division by the
    largest magnitude and the sum of squares, the first half reusing the
    previous moment), and ``d + 4`` for summing the step's d + 3 charges and adding
    the step to the final sum. N_s steps take N_s + 1 potential moments. The whole law is a
    nominal proxy in the units of the other QHD laws, not an upper bound on
    the backend's arithmetic, a FLOP count or a runtime bound. At a fixed
    step count it depends on neither a, b nor the objective's range, whereas
    the work of ``expm_multiply`` grows with them
    (``method.restricted_sizes``). The selected resource law records it as an
    estimate with the backend's internal work unknown
    (``method.QHD._host_construction``).

    ``workspace_bytes = 64 D + 32 (d + 1) K + 8 T_max + 8 N_s + 1024 K + 4096 (d + 1)``
    counts the arrays of ``evolve`` and ``potential_diagonal``, namely the
    complex state (16 D), a returned transform or state copy (16 D), V
    (8 D), the real angle array (8 D) and the complex phase array (16 D),
    the d per-axis eigenvalue arrays (8 K each) with, for the current axis,
    its kinetic angles (8 K), their mode populations (8 K), the complex
    product with -i (16 K) and the phases (16 K), in all at most
    ``8 d K + 48 K <= 32 (d + 1) K`` for d >= 1, one table converted to an
    array, and the float64 per-step charges of the observed budget (8 N_s).
    The kinetic moment (``_kinetic_axis``) reads the state in place, and
    beside the populations NumPy 2.5.2's ``einsum`` traced at most 1368
    bytes on arrays of up to 2**20 entries, released before the phases are
    formed. The potential moment (``_potential_moment``) writes into the
    angle array, which holds no live data at that point, and traced 296
    bytes beyond it. The traced excess that ``SCRATCH_BYTES_PER_AXIS``
    covers includes both (2026-09-26, revision
    aede9fd119563d63562c9afed4b00021920560c6). The scaled potential moment
    traced the same 296 bytes on arrays of 256 to 2**20 entries
    (2026-09-27, develop a6db3600b682c84575c6b892005c9d3cbe427bcf with
    this change, Python 3.12.14 and NumPy 2.5.2 on macOS arm64). The last two terms are the measured transform-scratch
    allowance, ``SCRATCH_BYTES_PER_POINT`` for the native plans and buffers
    of ducc0.fft with one worker and ``SCRATCH_BYTES_PER_AXIS`` for Python
    and NumPy objects. The start vector is formed before these arrays, and
    its explicit payload, at most ``24 D + 8 d K`` bytes
    (``initial_state.restricted_state``), fits within
    ``64 D + 32 (d + 1) K`` term by term. Its complex result is the state already counted
    here. The readout of ``method._execute_theory`` runs after these arrays
    are released and has its own allowance.
    """
    d, k = grid.num_variables, grid.num_grid_points
    dimension = k**d
    passes = 2 * d * transform_levels(grid) * dimension
    size = (start_work + dimension * (d + len(tables) + 7) + d * k
            + num_steps * (passes + (2 * d + 13) * dimension + 7 * d * k + d + 4))
    largest = max((len(t.values) for t in tables), default=0)
    workspace = (64 * dimension + 32 * (d + 1) * k + 8 * largest + 8 * num_steps
                 + SCRATCH_BYTES_PER_POINT * k + SCRATCH_BYTES_PER_AXIS * (d + 1))
    return size, workspace
