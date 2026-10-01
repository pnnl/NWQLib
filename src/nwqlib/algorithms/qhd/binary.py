"""Binary encoding of the periodic QHD grid: register order, QFT kinetic conjugation and diagonal synthesis.

Register. With ``K = 2**b`` grid points per variable, variable j occupies the
b qubits ``j*b`` to ``j*b + b - 1``, little-endian within its block, so its
grid index is ``n_j = sum_l 2**l n_(j,l)`` with bit l on qubit ``j*b + l``.
Every one of the ``2**(d*b) = K**d`` register states is a grid point, so the
encoding has no invalid outcomes and its valid mass is 1. Qiskit's
statevector index of a grid tuple is ``z = sum_j n_j K**j`` (variable 0 in the
low-order qubits), whereas NWQLib's restricted arrays use the lexicographic
index ``i = sum_j n_j K**(d-1-j)`` (variable 0 most significant, as in
``theory.restricted_basis``). The permutation ``P|i> = |z>`` reverses the order
of the variable axes and keeps the bits within each variable. A comparison of
a circuit U with a lexicographic operator uses ``P^dagger U P``.

Kinetic term. On the periodic grid ``x_i = lower + i h``, ``h = L/K``, the
kinetic operator of variable j is diagonal in the discrete Fourier basis. The
QFT circuit on the variable's block implements
``F_+ |n> = K**(-1/2) sum_k exp(+2 pi i k n/K) |k>`` (Qiskit's convention), and
one kinetic step is ``F_+^dagger diag(exp(-i alpha E_k)) F_+``, with E_k the
periodic eigenvalues of ``QHD.kinetic_model`` in Fourier-index order
(``split_step.kinetic_eigenvalues``). ``kinetic_table`` proves that this sign
convention gives the same operator as the classical ``F_-^dagger D F_-`` and
builds the permuted phase table that removes the swap layers, and
``qft_gates`` lists the circuit and its bit-reversal convention.

Potential term. Each admitted support table is a real diagonal on ``|S| b``
qubits, synthesized either as an exact phase diagonal (the recursion of
Shende, Bullock and Markov, quant-ph/0406176v5, Theorem 7 on p. 10, whose
multiplexed Rz with k select bits costs ``2**k`` CX by the count after their
Theorem 8 on p. 11, so the diagonal costs ``2**n - 2`` CX on n qubits) or as
one parity-ladder Z rotation per Walsh coefficient (Liu et al.,
arXiv:2607.16996v1, Sec. IV A, Eqs. (53)-(58), ``2 (w - 1)`` CX for a
weight-w string). ``PhaseTable.synthesize`` owns both constructions and
their CX and rotation counts, ``BinarySynthesis`` selects between them, and
``native.append_binary_steps`` and ``theory.run_binary_product`` apply the
same emitted constructions to a circuit and to a restricted state.

The derivations in the docstrings below are NWQLib's unless a source is
named. ``tests/test_qhd_binary.py`` compares the emitted circuits with
independently written products of the finite model and the CX laws with
Qiskit's transpiled circuits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal, NamedTuple

import numpy as np
from pydantic import Field, StrictInt

from nwqlib._validation import UNIT_ROUNDOFF
from nwqlib.core.records import Record
from nwqlib.algorithms.qhd.validation import (
    _admitted_products,
    _kinetic_range_error,
    _normal_product,
    _normal_products,
    _normal_quotient,
    _normal_range,
    _underflows,
    angle_units,
)

SynthesisChoice = Literal["min_cx", "dense_diagonal", "walsh_rotations"]

# The smallest normal and the largest finite binary64 numbers, the ends of the
# admitted open range that the array range checks test (validation._normal_range).
_NU = 2.0**-1022
_OMEGA = 1.7976931348623157e308


class BinarySynthesis(Record):
    """Circuit choices of the binary encoding: phase diagonals, QFT bit reversal and AQFT cutoff.

    Build it with keyword arguments and pass it as
    `QHD(encoding="binary", binary_synthesis=BinarySynthesis(...))`. Every argument is
    optional, and the defaults build exact circuits. Every choice builds the same exact
    operator unless `aqft_cutoff` or a positive `QHD.rotation_threshold` selects an
    approximation, whose operator-norm bound the Plan records as `aqft_error_bound` and
    `pruning_error_bound`. The
    [circuit synthesis](../../algorithms/qhd.md#circuit-synthesis) section of the guide
    gives the CX counts of each choice.

    Attributes:
        potential: Default `"min_cx"`. Synthesis of each support table's phase diagonal:
            `"dense_diagonal"`, `"walsh_rotations"`, or `"min_cx"`, the one of the two
            with fewer CX for each table, Walsh rotations on a tie. The Synthesis
            choices note below gives their CX counts and what `"min_cx"` does not
            minimize.
        kinetic_phase: Default `"min_cx"`. The same choice for each variable's kinetic
            phase diagonal in the Fourier basis.
        qft_bit_reversal: Default `"relabel"`, which omits the swap layers of both QFTs
            and applies the phase table permuted by bit reversal between the swap-free
            QFT and its exact inverse, the same operator. `"swap"` keeps Qiskit's swap
            layers, `3 floor(b/2)` CX per QFT.
        aqft_cutoff: Default `None`, which keeps every controlled phase of the QFT. A
            nonnegative integer m keeps the controlled phases between wires at distance
            `r <= m`, whose angles are `pi/2**r`, and drops the rest, Qiskit's
            `approximation_degree = max(0, b - 1 - m)`. m = 0 drops every controlled
            phase.

    Synthesis choices:
        `"dense_diagonal"` is the exact phase diagonal of Shende, Bullock and Markov,
        quant-ph/0406176v5, Theorem 7, `2**n - 2` CX on n qubits for a nonzero table and
        none for an all-zero one. `"walsh_rotations"` applies one parity-ladder Z
        rotation per nonzero Walsh coefficient, `2 (w - 1)` CX for a weight-w string.
        `"min_cx"` compares the two emitted CX counts of each table at its step exponent
        and takes the smaller, preferring `"walsh_rotations"` on a tie. It minimizes
        that CX count table by table, not the Rz or T count, the depth, routed CX on
        limited connectivity, or the CX of a jointly synthesized potential.
    """

    potential: SynthesisChoice = "min_cx"
    kinetic_phase: SynthesisChoice = "min_cx"
    qft_bit_reversal: Literal["relabel", "swap"] = "relabel"
    aqft_cutoff: Annotated[StrictInt, Field(ge=0)] | None = None


def register_bits(num_grid_points):
    """Return b with ``K = 2**b``, or raise when K is not a power of two of at least 2."""
    k = int(num_grid_points)
    bits = k.bit_length() - 1
    if k < 2 or 1 << bits != k:
        raise ValueError(f"the binary encoding requires num_grid_points = 2**b with b >= 1, got {k}")
    return bits


def decode_register_index(index, num_variables, bits):
    """Return the grid tuple ``(n_0, ..., n_(d-1))`` with ``n_j = (z >> j b) mod K``.

    Every index below ``2**(d b)`` decodes to a grid point, so no outcome is invalid.
    """
    mask = (1 << bits) - 1
    return tuple((index >> (j * bits)) & mask for j in range(num_variables))


def lexicographic_register_array(register, num_variables, bits):
    """Return a full binary-register array in lexicographic grid order, by reversing its variable axes.

    Convert a full binary-register array to lexicographic grid order by
    reversing its variable axes. Variable j is the base-K digit of weight
    ``K**j`` in the register and of weight ``K**(d-1-j)`` in the output. The
    transformation preserves the local bit order within each variable.

    Derivation. The full-register index is
    ``z = sum_j n_j*K**j``, since variable j occupies qubits
    ``j*b..j*b+b-1``. The restricted flat index is
    ``i = sum_j n_j*K**(d-1-j)``. Viewing a register array in C order gives
    its axes as ``(n_(d-1),...,n_0)``. Reversing those variable axes gives
    ``(n_0,...,n_(d-1))``, with the internal b-bit order of each n_j
    unchanged, so the result equals
    ``register[lexicographic_register_indices(d, b)]`` entry by entry. For
    d = 1 it is the identity. K = 2 has one bit per axis and the same proof.
    Reversing all individual bits for K > 2 would be a different
    permutation. The proof applies to real probabilities, complex states and
    integer counts. Use an array in the documented full-register qubit
    order, after any backend bit-layout conversion, and only once per
    readout. The final reshape generally makes one contiguous output copy,
    8D bytes for float64 probabilities and 16D for complex128 states. It
    need not copy at d = 1. It replaces three D-length integer buffers and
    the gather.
    """
    k = 1 << bits
    d = num_variables
    return np.asarray(register).reshape((k,) * d).transpose(tuple(reversed(range(d)))).reshape(-1)


def lexicographic_register_indices(num_variables, bits):
    """Return z for each lexicographic position i, the permutation ``P|i> = |z>`` as an integer array.

    Position ``i = sum_j n_j K**(d-1-j)`` has digit ``n_j = (i // K**(d-1-j)) mod K``,
    and ``z = sum_j n_j 2**(j b)``. ``state[result]`` is the lexicographic
    restriction of a full-register state.
    """
    k = 1 << bits
    positions = np.arange(k**num_variables, dtype=np.int64)
    indices = np.zeros_like(positions)
    for j in range(num_variables):
        digit = (positions // k ** (num_variables - 1 - j)) % k
        indices |= digit << (j * bits)
    return indices


def bit_reversal(bits):
    """Return ``rev_b(k)`` for k = 0..2**b - 1, the reversal of the b bits of k.

    b shift/OR passes over the int64 indices move bit l to bit b-1-l, exact
    integer operations; b = 0 gives the identity ``[0]``.
    """
    indices = np.arange(1 << bits, dtype=np.int64)
    reversed_indices = np.zeros_like(indices)
    for level in range(bits):
        reversed_indices |= ((indices >> level) & 1) << (bits - 1 - level)
    return reversed_indices


def qft_gates(bits, *, cutoff=None, swaps):
    """Return the gates of Qiskit's QFT circuit on local qubits ``0..b-1``.

    The gates are ``("h", q)``, ``("cp", angle, control, target)`` and
    ``("swap", q1, q2)`` in application order. For j from b-1 down to 0 the
    circuit applies H to qubit j and then ``CP(pi/2**r)`` between qubits j and
    ``j - r`` for ``r = 1..min(j, m)``, and with ``swaps`` it ends with the
    swaps ``(i, b-1-i)``, i < floor(b/2). This is the gate sequence of
    Qiskit 2.5.2 ``synth_qft_full(b, do_swaps, approximation_degree)`` with
    ``approximation_degree = max(0, b - 1 - m)``: its loop keeps
    ``min(j, b - 1 - approximation_degree)`` controlled phases on qubit j.

    With every controlled phase (``cutoff`` None or at least b - 1) and the
    swaps, the circuit is ``F_+|n> = K**(-1/2) sum_k exp(2 pi i k n/K)|k>``
    by the product representation of the Fourier transform (Nielsen and
    Chuang, *Quantum Computation and Quantum Information*, 2000,
    ISBN 978-0-521-63503-5, Sec. 5.1 and Fig. 5.1, whose output order the
    swaps restore). Without the swaps it is ``W = R F_+``, with R the bit
    reversal ``R|k> = |rev_b(k)>``.

    A cutoff m keeps the phases ``pi/2**r`` with ``r <= m`` and drops the
    ``b - r`` gates of each larger distance, the approximate QFT of
    Coppersmith, arXiv:quant-ph/0201067v1.
    ``qft_error_bound`` bounds the omitted gates. The angle
    ``ldexp(pi, -r)`` is binary64 pi scaled by an exact power of two, the
    value Qiskit passes.
    """
    cutoff = bits - 1 if cutoff is None else min(cutoff, bits - 1)
    gates = []
    for j in reversed(range(bits)):
        gates.append(("h", j))
        for k in reversed(range(max(0, j - cutoff), j)):
            gates.append(("cp", math.ldexp(math.pi, k - j), j, k))
    if swaps:
        gates.extend(("swap", i, bits - 1 - i) for i in range(bits // 2))
    return tuple(gates)


def inverse_gates(gates):
    """Return the exact inverse of a ``qft_gates`` list: reversed order, negated controlled-phase angles."""
    return tuple(("cp", -g[1], g[2], g[3]) if g[0] == "cp" else g for g in reversed(gates))


def qft_controlled_phases(bits, cutoff):
    """Return the controlled phases of one QFT, ``sum_(r=1..m) (b - r)`` with ``m = min(cutoff, b - 1)``.

    Distance r has ``b - r`` gates (``qft_gates``), so the exact QFT has
    ``b (b - 1)/2``.
    """
    cutoff = bits - 1 if cutoff is None else min(cutoff, bits - 1)
    return sum(bits - r for r in range(1, cutoff + 1))


def qft_cx(bits, cutoff, swaps):
    """Return the CX count of one QFT at optimization level 0, ``2 C_p + 3 floor(b/2) [swaps]``.

    Qiskit lowers each controlled phase to two CX (the ``CPhaseGate``
    definition applies ``P(lambda/2)`` to the control, a CX, ``P(-lambda/2)``
    to the target, a second CX and ``P(lambda/2)`` to the target) and each
    swap to three CX, and H needs none. The exact QFT therefore costs
    ``b (b - 1)`` CX without swaps and ``b (b - 1) + 3 floor(b/2)`` with
    them. Its inverse has the same count.
    """
    return 2 * qft_controlled_phases(bits, cutoff) + (3 * (bits // 2) if swaps else 0)


def qft_error_bound(bits, cutoff):
    """Return e_F, a bound on ``||F~ - F||_2`` for the QFT F and the truncated QFT F~ with this cutoff.

    ``CP(theta) = diag(1, 1, 1, exp(i theta))``, so
    ``||CP(theta) - I|| = 2 |sin(theta/2)|``. Replacing the omitted gates by
    the identity one at a time in the unitary gate product changes it by at
    most that norm per gate (telescoping), so with m the cutoff

    ``e_F = min(2, sum_(r=m+1..b-1) (b - r) 2 sin(pi/2**(r+1)))``,

    which is zero when nothing is omitted. The 2 is the diameter of the
    unitary group in the operator norm. Each term is formed from binary64 pi
    with one sine (at most one ulp, a platform assumption) and one product,
    and ``fsum`` rounds the sum once, so the computed sum has relative error
    below 5u, and the returned value is multiplied by ``1 + 8u``, which
    rounds it outward. ``aqft_error_bound`` of a Plan charges ``2 e_F`` per
    kinetic conjugation. The one-ulp premise for ``math.sin`` is
    conditional. The bound was tested on macOS arm64 and Linux aarch64, and
    the sine of Linux x86-64 hosts has not been checked
    (docs/dependency_issues.md, "Platform math libraries").
    """
    cutoff = bits - 1 if cutoff is None else min(cutoff, bits - 1)
    terms = [(bits - r) * 2.0 * math.sin(math.ldexp(math.pi, -(r + 1))) for r in range(cutoff + 1, bits)]
    # fsum of the omitted gate norms, rounded outward by 1 + 8u.
    return min(2.0, math.fsum(terms) * (1.0 + 8.0 * UNIT_ROUNDOFF))


def walsh_coefficients(values):
    """Return ``c_m = 2**-n sum_z (-1)**popcount(m & z) v[z]`` for a table v of length ``2**n``.

    With ``Z_m`` the product of Pauli Z on the set bits of m, whose eigenvalue
    on the basis state ``|z>`` is ``(-1)**popcount(m & z)``, Walsh
    orthogonality gives ``diag(v) = sum_m c_m Z_m``. The fast Walsh-Hadamard
    butterfly uses ``n 2**n`` additions and ``O(2**n)`` storage.

    Range admission (``walsh_admission``). Every butterfly addition and
    subtraction must be finite. An overflow cannot become finite again in
    later additions, so the final outputs show it. A subnormal scratch
    result is exact under gradual underflow. A nonzero output s with
    ``|s|/N < 2**-1022`` is omitted as coefficient zero instead of being
    normalized, as for the normal entries ``2**-1022`` and
    ``nextafter(2**-1022, inf)``, whose normalized difference would be
    ``-2**-1075``, and every other output is normalized exactly. For
    ``N = 2**n`` the admitted coefficient errs from the exact Walsh
    coefficient of the stored table by at most ``gamma_n a + d_m``, with a
    the mean absolute table value, ``gamma_n = n u/(1 - n u)``, ``n u < 1``,
    and ``d_m = |s_m|/N`` for an omitted coefficient and zero otherwise, and
    computed zero coefficients carry the ``gamma_n a`` term too. The stored
    table is the exact input of this statement. Write ``c_m*`` for the exact
    Walsh coefficient of the stored table (``c0_star`` below for m = 0). The
    ``gamma_n a`` term holds because each output ``s_m`` is a signed pairwise
    sum in which every table entry passes through the same number n of
    rounded additions. The product of their rounding factors gives
    ``|s_m - N c_m*| <= gamma_n sum_z |v_z|``, the pairwise summation bound
    of Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
    SIAM 2002, doi:10.1137/1.9780898718027, Sec. 4.2, Eq. (4.6), p. 83,
    whose proof is unchanged by the exact sign flips. Dividing by N gives
    ``|z_m - c_m*| <= gamma_n a`` for ``z_m = s_m/N``, and an omission adds
    ``d_m = |z_m|`` (``walsh_admission``). Liu et al.,
    arXiv:2607.16996v1, Sec. IV A, Eqs. (53)-(54), expand each support-local
    potential block this way to count one Z rotation per nonzero string,
    which is cheap for the sparse spectra of low-degree polynomials and
    costly for a coupled transcendental table such as Ackley's, where every
    string is nonzero. ``BinarySynthesis.potential`` therefore keeps the
    exact phase diagonal as the alternative and compares the two per table.

    Identity coefficient. For N = 2**n stored binary64 entries let
    ``a = sum_r |v_r|/N`` and ``c0_star = sum_r v_r/N``, both sums read
    exactly. For the admitted, computed identity coefficient c0 the bound
    above at m = 0 gives ``|c0 - c0_star| <= gamma_n a + d_0``, with
    ``gamma_r = r u/(1 - r u)``, ``u = 2**-53``, ``r u < 1`` and ``d_0``
    from the normalization above. The bound needs finite intermediates and
    depends on the mean absolute value a even when the computed identity is
    zero. For example, the table ``(2**53, 1, -2**53, 0)`` has exact mean
    1/4 and computed identity 0, so at exponent 16 the computed phase misses
    ``-4`` by 4 rad. ``compile_binary_steps`` charges the identity phase
    formed from ``c0`` in each Walsh block.
    """
    return walsh_admission(values)[0]


class WalshOmission(NamedTuple):
    """The Walsh coefficients of one table that the lower-range normalization omitted (``walsh_admission``).

    Attributes:
        coefficients: Omitted nonidentity coefficients.
        identity: Whether the identity coefficient was omitted.
        identity_charge: ``d_0``, the exact ``|s_0|/N`` of an omitted identity, else zero.
        charge: The exact ``sum_(m != 0) d_m`` over the omitted nonidentity coefficients.
    """

    coefficients: int
    identity: bool
    identity_charge: Fraction
    charge: Fraction


NO_WALSH_OMISSION = WalshOmission(0, False, Fraction(0), Fraction(0))


def walsh_admission(values):
    """Return ``(coefficients, omission)``: the admitted Walsh coefficients of a table and what their normalization omitted.

    With the butterfly outputs s_m of ``walsh_sum``, ``z_m = s_m/N`` and
    ``N = 2**n``, the admitted coefficient is zero when ``0 < |z_m| < 2**-1022``
    and ``z_m`` otherwise, decided by the exact test ``|s_m| < 2**(n - 1022)``
    before any division, so the kept ``z_m`` is an exact power-of-two scaling
    and ``|z_m| = 2**-1022`` is admitted. The omission ``d_m = |z_m|`` of an
    omitted coefficient enters the coefficient error bound
    ``gamma_n a + d_m`` (``walsh_coefficients``), ``d_0`` for the identity
    and the nonidentity sum for ``circuit_errors.binary_angle_formation``.
    The triangle inequality through ``z_m`` gives that bound. A nonfinite
    butterfly output raises ValueError.
    """
    # walsh_sum copies the table once and transforms that copy in place.
    # An overflowing butterfly is reported by the range check below, not by a NumPy warning.
    with np.errstate(over="ignore", invalid="ignore"):
        sums = walsh_sum(np.asarray(values, dtype=float))
    finite = np.isfinite(sums)
    if not finite.all():
        _normal_range(float(sums[~finite][0]), "a Walsh butterfly sum of the objective table")
    size = sums.size
    # |s|/N < 2**-1022 exactly when |s| < 2**(n - 1022), both sides exact binary64 numbers.
    omitted = (sums != 0.0) & (np.abs(sums) < math.ldexp(1.0, size.bit_length() - 1 - 1022))
    with np.errstate(under="ignore"):
        coefficients = np.where(omitted, 0.0, sums / size)
    losses = [abs(Fraction(float(s))) / size for s in sums[omitted].tolist()]
    identity = bool(omitted[0])
    return coefficients, WalshOmission(
        coefficients=int(np.count_nonzero(omitted[1:])), identity=identity,
        identity_charge=losses[0] if identity else Fraction(0),
        charge=sum(losses[1:] if identity else losses, Fraction(0)))


def walsh_sum(coefficients):
    """Return ``v[z] = sum_m (-1)**popcount(m & z) c_m``, the unnormalized butterfly and inverse of ``walsh_coefficients``."""
    values = np.array(coefficients, dtype=complex if np.iscomplexobj(coefficients) else float)
    size = values.size
    if size & (size - 1):
        raise ValueError("a Walsh table needs a power-of-two length")
    stride = 1
    while stride < size:
        blocks = values.reshape(-1, 2 * stride)
        left, right = blocks[:, :stride], blocks[:, stride:]
        # One half-length temporary per level: left + right and left - right from the saved left.
        saved = left.copy()
        np.add(saved, right, out=left)
        np.subtract(saved, right, out=right)
        stride *= 2
    return values


def signed_square_walsh(bits):
    """Return the Walsh coefficients of ``q**2`` for the signed index q of a b-bit register.

    The signed (two's-complement) index of ``n = sum_l 2**l n_l`` is
    ``q = sum_l w_l n_l`` with ``w_l = 2**l`` for ``l < b - 1`` and
    ``w_(b-1) = -2**(b-1)``, so ``q = n`` for ``n < K/2`` and ``q = n - K``
    otherwise. Writing ``n_l = (1 - Z_l)/2`` and ``W = sum_l w_l`` gives
    ``q = (W - sum_l w_l Z_l)/2``, and squaring with ``Z_l**2 = I``,

    ``q**2 = (W**2 + sum_l w_l**2)/4 I - (W/2) sum_l w_l Z_l + (1/2) sum_(l<m) w_l w_m Z_l Z_m``.

    For the signed weights ``W = -1``. Only the identity, the b single-Z and
    the ``b (b-1)/2`` two-Z masks are nonzero, so ``walsh_rotations`` costs
    ``b (b - 1)`` CX. Every coefficient is a multiple of 1/2: the single-Z
    coefficients are ``w_l/2``, the two-Z ones ``w_l w_m/2`` and the identity
    ``(4**b + 2)/12``. Their absolute values sum to ``2**(2b-2)``, which bounds
    every partial sum of the ``walsh_sum`` butterfly. A multiple of 1/2 of
    magnitude at most ``2**52`` is exact in binary64, so for ``b <= 27`` the
    coefficients and ``walsh_sum`` of them, the integers ``q**2``, are exact.
    At ``b = 28`` the identity coefficient ``6004799503160661.5`` already
    rounds. The algebraic expansion holds for every b. Liu et al., arXiv:2607.16996v1,
    Sec. IV E, Eqs. (89)-(91), introduce this quadratic phase to reduce the
    kinetic diagonal to one- and two-bit strings, and expand ``k**2`` with the
    unsigned index ``k = sum_l 2**l k_l`` (``W = K - 1``), which gives
    ``(K - 1)**2`` instead of 1 at the index ``K - 1`` of momentum -1 and is
    not symmetric under ``k -> -k``. The low-momentum form of their Eq. (89)
    is the signed one.
    """
    weights = [1 << level for level in range(bits - 1)] + [-(1 << (bits - 1))]
    total = sum(weights)
    coefficients = np.zeros(1 << bits)
    coefficients[0] = (total * total + sum(w * w for w in weights)) / 4
    for level, weight in enumerate(weights):
        coefficients[1 << level] = -total * weight / 2
        for other in range(level + 1, bits):
            coefficients[(1 << level) | (1 << other)] = weight * weights[other] / 2
    return coefficients


def kinetic_walsh(model, bits, spacing):
    """Return the Walsh coefficients of the periodic kinetic energies with the exact zeros of their Pauli expansion.

    The energies are ``split_step.kinetic_eigenvalues`` of the periodic grid,
    ``E_k = 2 pi**2 q_k**2/L**2`` (spectral, signed index q_k, period
    ``L = K h``) and ``E_k = 2 sin**2(pi k/K)/h**2`` (finite difference).
    ``"spectral"`` divides ``signed_square_walsh``, which is exactly ``q**2``
    for ``b <= 27``, by the power of two ``(K/2)**2`` and multiplies it by the
    Nyquist energy ``E_Nyq = 2 (pi/(2h))**2``, which equals
    ``2 pi**2 (K/2)**2/L**2``, so the ``b (b - 1)/2`` two-Z strings and b
    single-Z strings are its only nonzero masks. No step squares the period
    L, whose square can overflow while every energy is finite.

    ``"finite_difference"``: ``E_n = (1 - cos(2 pi n/K))/h**2`` and, with
    ``a_l = pi 2**l/K`` and ``n_l = (1 - Z_l)/2``,
    ``exp(2 pi i n/K) = exp(i pi (K-1)/K) prod_l (cos a_l - i Z_l sin a_l)``.
    The most significant bit has ``a_(b-1) = pi/2``, whose cosine is zero, so
    every nonconstant mask with a nonzero coefficient contains that bit and
    ``c_0 = 1/h**2``. For such a mask m of weight w,

    ``c_m = s_w(pi/K) prod_(l in m, l < b-1) sin a_l prod_(l not in m, l < b-1) cos a_l / h**2``,

    where ``s_w`` is ``cos, -sin, -cos, sin`` for ``w mod 4 = 0, 1, 2, 3``,
    because ``Re(exp(i pi (K-1)/K) (-i)**w) = -cos(pi/K + pi w/2)`` and
    ``c_m = -Re(...)/h**2``. For ``b >= 2`` every factor is nonzero, and at
    ``b = 1`` the single mask has ``s_1(pi/2) = -1``, so all ``2**(b-1)`` masks
    that contain the top bit are emitted and
    ``walsh_rotations`` costs ``(b - 1) 2**(b-1)`` CX. A Walsh transform of
    the rounded table would instead leave rounding noise of order ``u/h**2``
    on the masks without the top bit, which are exactly zero.

    The identity coefficient is ``fl(1/fl(h**2))`` or the rounded Nyquist
    energy times the exact ``(K**2 + 2)/12`` divided by ``(K/2)**2``, not a
    butterfly output, so its formation allowance comes from
    ``kinetic_identity_enclosure``.

    Range admission (``validation._normal_range``). The Nyquist energy and
    its factor ``pi/(2 h)``, every trigonometric-factor product, every
    division by ``h**2`` and every nonzero coefficient are normal binary64
    numbers. The known zero masks stay exactly zero, and a computed zero in
    a mask whose formula is nonzero is refused with the kinetic-scale remedy
    (``validation._kinetic_range_error``), since these coefficients depend
    only on the grid.
    """
    try:
        return _kinetic_walsh(model, bits, spacing)
    except ValueError as error:
        raise _kinetic_range_error(error) from error


def _kinetic_walsh(model, bits, spacing):
    """Return the coefficients of ``kinetic_walsh``, raising ValueError for a value outside the admitted range."""
    if model == "spectral":
        # E_Nyq c_m(q**2)/(K/2)**2 with E_Nyq = 2 (pi/(2h))**2
        wave = _normal_quotient(math.pi, 2.0 * spacing, "the spectral Nyquist wave number pi/(2 h)")
        nyquist = _normal_range(2.0 * wave**2, "the spectral Nyquist energy 2 (pi/(2 h))**2")
        return _normal_products(nyquist, signed_square_walsh(bits) / float(1 << (2 * bits - 2)),
                                "a spectral kinetic Walsh coefficient")
    return kinetic_finite_difference_array(bits, spacing)


def checked_product_level(previous, factors, masks):
    """Return one level of the finite-difference partial products, admitting each exact product.

    One elementwise multiplication executes the same binary64 operation for
    every mask as the scalar loop. The normal-range check is on the exact
    product of the stored previous partial and factor. Min/max of the
    rounded output certify an open normal interval only, so an entry at or
    beyond nu or Omega, or nonfinite, calls the exact boundary test. A zero
    output on these structurally nonzero masks is a refusal, as it is in
    ``_normal_product``.
    """
    with np.errstate(all="ignore"):
        product = previous * factors
    mag = np.abs(product)
    doubtful = (~np.isfinite(product)) | (mag <= _NU) | (mag >= _OMEGA)
    for pos in np.flatnonzero(doubtful):
        a, b = float(previous[pos]), float(factors[pos])
        _normal_range(float(product[pos]),
                      f"a trigonometric factor product of the kinetic Walsh mask {int(masks[pos])}",
                      lambda a=a, b=b: Fraction(a) * Fraction(b), zero=False)
    return product


def kinetic_finite_difference_array(bits, spacing):
    """Return the finite-difference kinetic Walsh coefficients of ``kinetic_walsh`` by array levels.

    The finite-difference coefficients use the source's scalar
    trigonometric factors and multiply them in increasing level order. Each
    entry therefore follows the scalar binary64 product sequence. Range
    admission checks exact products at the normal-range endpoints. The
    summary reuses one analytic enclosure per distinct trigonometric factor
    and propagates interval products outward. Its radii enclose the exact
    analytic coefficients under the stated one-ulp premise and can exceed
    the exact-rational enclosure's radii.

    For top-bit masks, ``s_w = cos(pi/K), -sin(pi/K), -cos(pi/K), sin(pi/K)``
    by popcount modulo four, then at levels 0 through b-2 a multiplication
    by the scalar ``math.sin`` or ``math.cos`` evaluated at exactly
    ``math.pi*(1<<level)/K``, followed by division by the already formed
    ``spacing**2``. Calling NumPy sine on an angle array would not prove the
    same coefficients, so the scalar values are reused. One elementwise
    multiplication at each level executes the same binary64 operation for
    every mask as the scalar loop, and induction on the level proves bit
    equality of each partial product; ``np.prod`` or reordered levels would
    not. At b = 1 the level loop is empty and the sole top mask uses the same
    ``-sin(pi/2)``. All other nonidentity masks stay exactly zero.
    """
    k, top = 1 << bits, bits - 1
    masks = np.arange(1 << top, k, dtype=np.int64)
    base = math.pi / k
    signs = np.array([math.cos(base), -math.sin(base), -math.cos(base), math.sin(base)])
    partial = signs[np.bitwise_count(masks) % 4]
    for level in range(top):
        angle = math.pi * (1 << level) / k
        factors = np.where((masks >> level) & 1, math.sin(angle), math.cos(angle))
        partial = checked_product_level(partial, factors, masks)
    square = spacing**2
    result = np.zeros(k, dtype=np.float64)
    result[0] = _normal_quotient(1.0, square, "the kinetic identity coefficient 1/h**2")
    with np.errstate(all="ignore"):
        coefficients = partial / square
    mag = np.abs(coefficients)
    for pos in np.flatnonzero((~np.isfinite(coefficients)) | (mag <= _NU) | (mag >= _OMEGA)):
        a = float(partial[pos])
        _normal_range(float(coefficients[pos]), f"the kinetic Walsh coefficient of mask {int(masks[pos])}",
                      lambda a=a: Fraction(a) / Fraction(square), zero=False)
    result[1 << top:] = coefficients
    return result


def kinetic_identity_enclosure(model, bits, spacing):
    """Return ``(low, high)`` enclosing the analytic identity coefficient of the kinetic table.

    The identity coefficient of the periodic kinetic energies is ``h**-2``
    for finite differences and ``2 pi**2 (K**2 + 2)/(12 L**2)`` for the
    signed spectral model, with the stored spacing h, ``K = 2**bits`` and the
    period ``L = K h``. Mathematical pi lies between ``math.pi`` and the
    next binary64 number, and the exact rational ``(K**2 + 2)/12`` is
    enclosed by rounding its conversion outward. Each rounded product or
    quotient of positive values errs by at most half an ulp, so stepping it
    one binary64 number down (lower end) or up (upper end) keeps the exact
    value inside. ``compile_binary_steps`` bounds the computed coefficient's
    formation error by its distance to the farther endpoint.
    """

    def down(value):
        return math.nextafter(value, -math.inf)

    def up(value):
        return math.nextafter(value, math.inf)

    if model == "finite_difference":
        return down(down(1.0 / spacing) / spacing), up(up(1.0 / spacing) / spacing)
    ratio = Fraction(4**bits + 2, 12)
    rounded = float(ratio)
    ratio_low = rounded if Fraction(rounded) <= ratio else down(rounded)
    ratio_high = rounded if Fraction(rounded) >= ratio else up(rounded)
    pi_high = up(math.pi)
    period = (1 << bits) * spacing
    period_low, period_high = down(period), up(period)
    squared_low, squared_high = down(period_low * period_low), up(period_high * period_high)
    numerator_low = down(2.0 * down(math.pi * math.pi) * ratio_low)
    numerator_high = up(2.0 * up(pi_high * pi_high) * ratio_high)
    return down(numerator_low / squared_high), up(numerator_high / squared_low)


def address_table(values, support_size, num_grid_points):
    """Reorder a lexicographic support table to the address order of its synthesized diagonal.

    A support ``S = (j_0, ..., j_(s-1))`` stores its values in lexicographic
    order, ``j_0`` most significant (``records.SupportValues``). Its diagonal
    acts on the qubits of the variables in S in increasing order, local bit
    ``t b + l`` on qubit ``j_t b + l``, so the address of a tuple is
    ``z_S = sum_t n_(j_t) K**t``. The C-order reshape to ``(K,)*s`` makes axis t
    the variable ``j_t``, and reversing the axes before flattening puts
    ``j_0`` in the least significant digit. For two variables this is
    ``V.T.ravel()`` of the lexicographic ``V[n0, n1]``.
    """
    shape = (num_grid_points,) * support_size
    return np.asarray(values, dtype=float).reshape(shape).transpose(tuple(reversed(range(support_size)))).ravel()


def lexicographic_table(values, support_size, num_grid_points):
    """Return an address-ordered table as the ``(K,)*s`` array whose axis t is variable ``j_t``, the inverse of ``address_table``."""
    shape = (num_grid_points,) * support_size
    return np.asarray(values).reshape(shape).transpose(tuple(reversed(range(support_size))))


@dataclass(frozen=True)
class DiagonalConstruction:
    """The emitted construction of ``exp(-i x diag(v))`` on n qubits for one exponent x.

    Attributes:
        synthesis: ``"dense_diagonal"`` or ``"walsh_rotations"``.
        qubits: Number of qubits n.
        phases: Dense construction: the phase angles ``-x v[z]`` in address
            order, passed to ``append_control_diagonal_phases``. None for
            Walsh rotations.
        identity_phase: Walsh construction: the circuit global phase
            ``-x c_0`` of the identity coefficient. Zero for the dense
            construction, whose helper keeps the identity inside its phases.
        masks: Walsh construction: the emitted masks, in increasing order.
        angles: Walsh construction: the ``Rz`` angle ``2 x c_m`` of each
            emitted mask.
        cx: CX count of the construction at optimization level 0.
        rotations: Rotation gates emitted: one per Walsh string, or the
            ``2**n - 1`` angles of the dense construction's multiplexor
            schedule, of which exact-zero angles are omitted in the circuit.
        dropped_count: Nonzero Walsh rotations removed by the angle threshold.
        dropped_angle_units: Exact sum of the absolute ``Rz`` angles removed,
            as an integer count of ``2**-1074`` (``validation.angle_units``).
        range_omitted: Walsh rotations, or dense phase entries, omitted
            because their product would fall below the normal range.
        range_charge: Their exact charge: ``sum |x c_m|`` over the omitted
            rotations, or ``max |x v_z|`` over the omitted dense entries.
        identity_omitted: Whether the Walsh identity phase ``-x c_0`` was
            omitted for the same reason.
    """

    synthesis: str
    qubits: int
    phases: np.ndarray | None
    identity_phase: float
    masks: np.ndarray
    angles: np.ndarray
    cx: int
    rotations: int
    dropped_count: int
    dropped_angle_units: int
    range_omitted: int = 0
    range_charge: Fraction = Fraction(0)
    identity_omitted: bool = False

    def emitted_phases(self):
        """Return binary64 diagonal phase parameters in address order for the host reference.

        Dense constructions return their stored ``phases``. For Walsh
        constructions, ``Rz(theta) = exp(-i theta Z/2)`` gives the exact
        angle ``phi_z = identity_phase - sum_m theta_m chi_m(z)/2`` of the
        separate kept rotations, with ``chi_m(z) = (-1)**popcount(m & z)``.
        This routine evaluates that expression as
        ``identity_phase - walsh_sum(theta)/2`` in binary64. Its returned
        ``hat_phi_z`` can differ from ``phi_z`` by a z-dependent residual.
        The two ideal diagonals then differ in operator norm by
        ``max_z 2*abs(sin((hat_phi_z-phi_z)/2))``.

        ``theory.run_binary_product`` applies phases from this array.
        ``theory.binary_product_state_error`` treats its entries as exact
        parameters and prices their application, not their formation or
        their difference from native gate lowering.
        """
        if self.phases is not None:
            return self.phases
        emitted = np.zeros(1 << self.qubits)
        emitted[self.masks] = self.angles
        return self.identity_phase - walsh_sum(emitted) / 2.0


class PhaseTable:
    """One real diagonal ``diag(v)`` on n qubits, held as its address-order values and Walsh coefficients.

    ``synthesize`` returns the construction of ``exp(-i x diag(v))`` for a
    given exponent x, which is the only quantity that changes from step to
    step, so the Walsh coefficients are computed once per table. A table
    built by ``dense`` has no coefficients and is synthesized only as the
    dense diagonal.
    """

    def __init__(self, values, coefficients, omission=NO_WALSH_OMISSION, popcounts=None, largest=None):
        self.values = np.asarray(values, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.omission = omission
        self.qubits = self.values.size.bit_length() - 1
        # Mask popcounts by qubit count, formed only for Walsh synthesis and shared by the
        # tables of one BinaryModel (and by a bit-reversed copy, whose masks keep their weights).
        self._popcounts = {} if popcounts is None else popcounts
        self._nonzero = None
        # max |v|, the cached exact extremum of a stored support table when the caller has it.
        if largest is None:
            largest = float(np.max(np.abs(self.values))) if self.values.size else 0.0
        self._largest = float(largest)

    def popcounts(self):
        """Return the read-only popcounts of the masks ``1..2**n - 1``, formed once per qubit count.

        ``np.bitwise_count`` of the mask indices gives exact integer
        popcounts. The array is shared by every table holding the same
        popcount dictionary (``BinaryModel``) and is formed only when a Walsh
        construction first needs it.
        """
        array = self._popcounts.get(self.qubits)
        if array is None:
            array = _mask_popcounts(self.qubits)
            self._popcounts[self.qubits] = array
        return array

    def _nonzero_summary(self):
        """Return ``(nonzero, cx, low, high)`` of the nonidentity coefficients, formed once.

        ``nonzero`` marks ``c_m != 0``, ``cx = 2 sum (popcount(m) - 1)`` over
        those masks, and ``(low, high)`` the minimum nonzero and the maximum
        magnitudes, None without a nonzero coefficient (``all_scaled_normal``).
        """
        if self._nonzero is None:
            nonzero = self.coefficients[1:] != 0.0
            magnitudes = np.abs(self.coefficients[1:][nonzero])
            self._nonzero = (nonzero, int(2 * np.sum(self.popcounts()[nonzero] - 1)),
                             float(magnitudes.min()) if magnitudes.size else None,
                             float(magnitudes.max()) if magnitudes.size else None)
        return self._nonzero

    @classmethod
    def from_values(cls, values, largest=None):
        """Return the table of ``values`` with its admitted Walsh coefficients and their omission (``walsh_admission``).

        ``largest`` is ``max |v|`` when the caller already holds it, such as
        ``SupportValues.magnitude`` of the stored table.
        """
        return cls(values, *walsh_admission(values), largest=largest)

    @classmethod
    def dense(cls, values, largest=None):
        """Return the table of ``values`` without Walsh coefficients, for a diagonal that only the dense construction synthesizes.

        The dense construction reads the values alone, so neither the Walsh
        transform nor its range admission (``walsh_admission``) applies, and
        a table whose butterfly sums would leave the binary64 range, such as
        ``1e308 x`` on eight points, keeps its dense Plan.
        """
        return cls(values, np.zeros(0), largest=largest)

    def permuted(self, permutation):
        """Return the table ``v[perm[z]]``, whose Walsh coefficients are ``c[perm[m]]`` for a bit permutation.

        For the bit reversal R, ``(R D R)[z] = v[rev(z)]`` and
        ``(-1)**popcount(m & rev(z)) = (-1)**popcount(rev(m) & z)``, so the
        coefficient of mask m is ``c_(rev(m))``. String weights, and hence the
        CX counts, are unchanged.
        """
        return PhaseTable(self.values[permutation], self.coefficients[permutation], self.omission, self._popcounts,
                          self._largest)

    def dense_cx(self, exponent):
        """Return ``2**n - 2`` CX when some nonzero dense phase survives range admission, and 0 otherwise.

        ``append_control_diagonal_phases`` builds no gate when every admitted
        phase is zero. A nonzero exact product below ``2**-1022`` is omitted
        even when its rounded binary64 value would be a nonzero subnormal.
        Otherwise it builds one uniformly controlled Rz on each level
        k = 1..n-1 of the Theorem 7 recursion of
        Shende, Bullock and Markov, quant-ph/0406176v5 (p. 10). A multiplexed
        Rz with k select bits costs ``2**k`` CX by the count stated after
        their Theorem 8 (p. 11), and the last level, with no select bit, is
        one Rz without CX, so the diagonal costs
        ``sum_k 2**k = 2**n - 2``. The Gray-code construction of Mottonen et
        al., quant-ph/0407010v1, Sec. II, Fig. 2 and Eq. (3), places its CX
        gates independently of the angles, and the helper omits only the Rz
        of an exactly zero angle, so a zero angle leaves the CX count
        unchanged. The count is a structural count of that construction, not
        a lower bound for every diagonal.
        """
        return (1 << self.qubits) - 2 if self._dense_nonzero(exponent) else 0

    def _dense_nonzero(self, exponent):
        """Whether some admitted dense phase ``fl(-x v[z])`` is nonzero.

        On the admitted finite range, a nonzero phase product survives
        exactly when its exact magnitude is at least ``2**-1022``. If the
        largest exact magnitude is smaller, every nonzero product is
        omitted. Otherwise at least that largest product survives. This
        decides whether the admitted diagonal is nonzero without forming
        all phases.
        """
        product = abs(exponent) * self._largest
        return product != 0.0 and not _underflows(
            product, "the largest dense phase |x| max |v|",
            lambda: abs(Fraction(exponent)) * Fraction(self._largest))

    def synthesize(self, exponent, choice, threshold):
        """Return the construction of ``exp(-i exponent diag(v))`` for ``choice``.

        Walsh rotations. All Z strings commute, so
        ``exp(-i x V) = exp(-i x c_0) prod_(m != 0) exp(-i x c_m Z_m)``, and
        ``exp(-i x c_m Z_m)`` is ``Rz(2 x c_m)`` on the parity of the string's
        w qubits, computed with ``w - 1`` CX and undone with ``w - 1`` more.
        The identity coefficient is the global phase ``-x c_0``, whose
        formation error ``compile_binary_steps`` charges to kept-state phase
        admission (``walsh_coefficients``). A mask with ``c_m = 0``, or with
        an angle omitted below the normal range (below), emits nothing. A
        nonzero angle below ``threshold`` is dropped, the rule of
        ``QHD.rotation_threshold``, and counted with its absolute angle. A
        small floating-point coefficient is not a proved zero, so without a
        threshold it is emitted.

        Dense diagonal. The phase angles ``-x v[z]`` go to the exact phase
        diagonal (``dense_cx``), which keeps every nonzero phase that the
        range admission below keeps, and the threshold never prunes it.

        ``"min_cx"`` compares the dense CX count with the Walsh count at this
        exponent and takes Walsh rotations when they cost at most as many.

        Range admission happens before the threshold and the ``min_cx``
        choice. The exponent and ``2 x`` are normal (``compile_binary_steps``).
        A Walsh angle ``fl(2 x c_m)`` of a nonzero coefficient, a dense phase
        ``fl(-x v[z])`` of a nonzero value or the identity phase
        ``fl(-x c_0)`` whose exact product is nonzero and below ``2**-1022``
        (``validation._admitted_products``, ``validation._underflows``) is
        omitted as zero, and a nonfinite or overflowing one raises. Since
        ``||exp(-i a G) - I|| <= |a| ||G||`` for a Hermitian G, removing
        ``Rz(2 x c_m) = exp(-i x c_m Z_m)`` costs ``|x c_m|``, and zeroing
        dense entries costs their largest ``|x v_z|``, because a diagonal's
        operator norm is its largest entry magnitude. Both charges belong to
        the ``rotation_pruning`` entry, while an omitted identity phase
        enters ``F_W`` (``compile_binary_steps``). Only the selected
        construction carries these omissions, and dense cost selection reads
        the admitted table.

        Range admission applies to the exact products represented by the
        stored operands. Walsh rotations use the doubled exponent, and the
        identity phase uses the undoubled exponent. When scalar extrema
        establish that every nonzero product lies in the admitted normal
        range and the threshold is zero, the nonidentity mask and its CX count
        are independent of the nonzero exponent. The phase angles remain
        exponent-dependent. Reusing a construction preserves the count and
        error contribution of every occurrence.

        That scalar test is ``all_scaled_normal(2x, c_min, c_max)`` on the
        minimum nonzero and maximum nonidentity coefficient magnitudes. Its
        success, with threshold zero and nonzero x, proves that the kept
        nonidentity set is exactly ``c_m != 0``, the omission set is empty and
        the CX sum is the cached count of that set, so the angles
        ``fl(2x c_m)`` are formed without the per-entry range pass. A positive
        threshold, a zero exponent, a failed test and every dense
        construction take the full pass, which keeps the zero and
        identity-omission bookkeeping.
        """
        n = self.qubits
        x = abs(Fraction(exponent))
        if choice != "dense_diagonal":
            doubled = _normal_product(2.0, exponent, "the doubled Walsh exponent 2 x")
            nonzero_set, nonzero_cx, low, high = self._nonzero_summary()
            if threshold == 0 and exponent != 0 and all_scaled_normal(doubled, low, high):
                with np.errstate(all="ignore"):
                    angles = doubled * self.coefficients[1:]
                kept = nonzero_set
                omitted = dropped = np.zeros_like(nonzero_set)
                walsh_cx = nonzero_cx
            else:
                angles, omitted = _admitted_products(doubled, self.coefficients[1:],
                                                     "a Walsh rotation angle 2 x c_m")
                nonzero = angles != 0.0
                kept = nonzero & (np.abs(angles) >= threshold)
                dropped = nonzero & ~kept
                # 2 (w - 1) CX per kept weight-w string.
                walsh_cx = int(2 * np.sum(self.popcounts()[kept] - 1))
            if choice == "walsh_rotations" or walsh_cx <= self.dense_cx(exponent):
                c0 = float(self.coefficients[0])
                identity = -exponent * c0
                identity_omitted = _underflows(identity, "the Walsh identity phase -x c_0",
                                               lambda: Fraction(exponent) * Fraction(c0), zero=c0 == 0.0)
                return DiagonalConstruction(
                    synthesis="walsh_rotations", qubits=n, phases=None,
                    identity_phase=0.0 if identity_omitted else identity,
                    masks=np.flatnonzero(kept) + 1, angles=angles[kept], cx=walsh_cx,
                    rotations=int(np.count_nonzero(kept)), dropped_count=int(np.count_nonzero(dropped)),
                    dropped_angle_units=sum(angle_units(angle) for angle in angles[dropped].tolist()),
                    range_omitted=int(np.count_nonzero(omitted)),
                    # sum |x c_m| over the omitted rotations
                    range_charge=x * sum((abs(Fraction(c)) for c in self.coefficients[1:][omitted].tolist()),
                                         Fraction(0)),
                    identity_omitted=identity_omitted)
        phases, omitted = _admitted_products(-exponent, self.values, "a dense diagonal phase -x v")
        # The multiplexor schedule of level k has 2**k angles, 2**n - 1 over the n levels.
        rotations = (1 << n) - 1 if self._dense_nonzero(exponent) else 0
        return DiagonalConstruction(
            synthesis="dense_diagonal", qubits=n, phases=phases, identity_phase=0.0,
            masks=np.zeros(0, dtype=np.int64), angles=np.zeros(0), cx=self.dense_cx(exponent),
            rotations=rotations, dropped_count=0, dropped_angle_units=0,
            range_omitted=int(np.count_nonzero(omitted)),
            # max |x v_z| over the omitted entries, the norm of the omitted diagonal
            range_charge=x * max((abs(Fraction(v)) for v in self.values[omitted].tolist()), default=Fraction(0)))


def _mask_popcounts(qubits):
    """Return the read-only int64 popcounts of the masks ``1..2**qubits - 1`` (``np.bitwise_count``)."""
    array = np.bitwise_count(np.arange(1, 1 << qubits, dtype=np.uint64)).astype(np.int64)
    array.flags.writeable = False
    return array


def all_scaled_normal(factor, nonzero_min, maximum):
    """Return whether every nonzero exact product ``factor * c`` lies in ``[nu, Omega]``.

    A sufficient all-normal test is ``nu <= |2x| c_min`` and
    ``|2x| c_max <= Omega`` for the admitted ``2x``, with ``(c_min, c_max)``
    the minimum nonzero and maximum magnitudes, ``nu = 2**-1022`` and
    ``Omega`` the largest finite binary64 number. These comparisons concern
    exact products of stored operands; the test ``|x| c_max <= Omega`` would be
    insufficient (x = 2, ``c_max = Omega/2`` passes while ``2 x c_max``
    overflows). Only two extremal products are checked, as exact Fraction
    comparisons that avoid every boundary ambiguity. None means there are no
    nonzero entries. A zero factor is handled by the full zero-bookkeeping
    path of the synthesis owner.
    """
    if nonzero_min is None:
        return True
    if factor == 0 or not math.isfinite(factor):
        return False
    f = abs(Fraction(float(factor)))
    return Fraction(_NU) <= f * Fraction(nonzero_min) and f * Fraction(maximum) <= Fraction(_OMEGA)


def _integer_storage(bits):
    """Return ``L(b) = 32 + 4 ceil(max(1, b)/30)`` bytes for a CPython integer of b magnitude bits.

    The qualified heap allowance of one integer on 64-bit CPython 3.12.14.
    The resource-inspection census (``resources._census_sizes``), the level-table files
    (``refinement._level_table_phase_bytes``) and the construction cache
    (``_construction_entry_cap``) size their integers with it.
    """
    return 32 + 4 * ((max(1, int(bits)) + 29) // 30)


def _require_bytes(limit, *, held, local, stage, later=False):
    """Return ``held + local`` bytes after checking them against ``limit``, a QHD ``max_bytes``.

    ``held`` prices the data that stay live through the phase and ``local``
    the phase's own workspace. The refusal names the phase and the smallest
    ``max_bytes`` that admits it; with ``later`` it says that the phases
    admitted after this one need more, so that value is a lower bound.

    Raises:
        ValueError: ``held + local`` exceeds ``limit``.
    """
    required = int(held) + int(local)
    if required > int(limit):
        raise ValueError(
            f"QHD {stage} requires {required} bytes "
            f"({held} held + {local} phase), with QHD(max_bytes={limit}); "
            + (f"QHD(max_bytes>={required}) admits this phase only; the phases admitted after it need more" if later
               else f"use QHD(max_bytes>={required})"))
    return required


def _construction_entry_cap(entries, support_size):
    """Return ``(16 E + h, h)``: one construction's payload bound and its metadata h, for a table of E entries.

    A construction's final payload is at most 16E: E real phases (8E) for a
    dense one, an int64 mask and a float64 angle per kept rotation for a
    Walsh one. For h use a qualified fixed-object allowance plus support
    slots and exact scalar widths. A conservative width for the current
    construction integers is ``b = 4200 + ceil(log2(E))``: finite binary64
    magnitudes expressed in units of ``2**-1074`` have at most 2098 bits; a
    sum of at most E such values adds at most ``ceil(log2 E)`` bits;
    multiplying two dyadic binary64 operands gives a numerator of at most
    ``4196 + ceil(log2 E)`` bits and a denominator of at most 2149 bits. This
    covers the dropped-unit sum and range-charge numerator/denominator, as
    well as the much smaller counts. The allowance is

    ``h = 4096 + 16*len(variables) + 8L(4200 + ceil(log2(E)))``.

    The 4096 covers the construction, arrays' headers, key tuple/hex string,
    mapping-entry resize storage and fixed scalars on this stack. The
    support-index objects already belong to the source/model; the slot
    charge is conservative. Eight L-width slots also cover count/cx/scalar
    integers. The same h serves the baseline and the optional-entry
    accounting (``BinaryModel.construction``). This intentionally avoids a
    production recursive object-size walk or a synthesis trial to discover
    the budget.
    """
    entries = int(entries)
    if entries < 1:
        raise ValueError("a phase table is nonempty")
    bits = (entries - 1).bit_length()
    metadata = 4096 + 16 * int(support_size) + 8 * _integer_storage(4200 + bits)
    return 16 * entries + metadata, metadata


def _construction_cache_capacity(max_bytes, *, held_bytes, model_bytes, latest_bytes, build_bytes, use_bytes):
    """Admit a ``BinaryModel``'s baseline ``B_base`` and return the capacity left for optional cache entries.

    ``B_base = held + model + latest + max(build, use)`` covers the caller's
    residents, the model, one latest construction per table and the largest
    construction or use workspace, and is admitted against ``max_bytes``
    before any model or build allocation. Only the remaining capacity
    ``cache_capacity = max_bytes - B_base`` funds optional whole-model
    retention, ``cache_bytes + new_entry_bytes <= cache_capacity``. Because
    the baseline already funds a miss and its latest-per-table result, the
    retention decision may use the actual payload after synthesis. A
    negative residual is a refusal of the baseline, never clamped to zero.

    Raises:
        ValueError: ``B_base`` exceeds ``max_bytes``.
    """
    baseline = (int(held_bytes) + int(model_bytes) + int(latest_bytes)
                + max(int(build_bytes), int(use_bytes)))
    _require_bytes(max_bytes, held=held_bytes,
                   local=baseline - int(held_bytes), stage="binary construction and use")
    return int(max_bytes) - baseline


def _construction_entry_bytes(built, *, entries, support_size):
    """Return the optional-cache charge of one construction: its phase or mask/angle payload plus its metadata h."""
    _, metadata = _construction_entry_cap(entries, support_size)
    data = (int(built.phases.nbytes) if built.phases is not None
            else int(built.masks.nbytes) + int(built.angles.nbytes))
    return data + metadata


def _model_metadata(d, bits, table_specs, *, cutoff=None, swaps=False):
    """Qualified headers and containers, separate from all numerical payloads."""
    specs = [(1 << bits, 1)] * d + [(int(e), int(s)) for e, s in table_specs]
    widths = {e.bit_length() - 1 for e, _ in specs}
    m = bits - 1 if cutoff is None else min(int(cutoff), bits - 1)
    cp = m * bits - m * (m + 1) // 2
    swap_count = bits // 2 if swaps else 0
    gates = bits + cp + swap_count
    index_bytes = _integer_storage(max(0, bits - 1).bit_length())
    qft = (256 + 32 * gates + 56 * bits + 64 * swap_count + 192 * cp
           + index_bytes * (bits + 2 * cp + 2 * swap_count))
    table_headers = sum(4096 + 8 * _integer_storage(4200 + e.bit_length() - 1)
                        for e, _ in specs)
    supports = sum(s for _, s in table_specs)
    slots = (16 + _integer_storage(max(0, d - 1).bit_length())) * supports
    return 65536 + table_headers + slots + 64 * d + 256 * len(widths) + qft


def _model_reservation(d, bits, table_specs, *, cutoff=None, swaps=False, potential_walsh=True):
    """Return the model, latest-construction, build and use byte reservations.

    Potential table_specs are (E, support_size). Add d kinetic tables of
    K=2**bits entries. For A=sum(E), reserve 16A bytes of values and
    coefficients, sum(8*2**n) over distinct widths n for shared popcounts,
    and sum(E-1) for the persistent nonzero masks. _model_metadata adds
    the PhaseTable objects, exact omission scalars, maps, support slots,
    spacings and the single shared QFT pair. With m=min(cutoff,b-1), or
    b-1 for None, Cp=m*b-m*(m+1)//2 and w=b//2 for swaps, the QFT has
    g=b+Cp+w slots per direction. Its H/swap tuples are shared between
    directions and its CP tuples and angle floats are distinct.

    The metadata allowance is 65536 + sum(4096+8L(4200+n)) over all
    tables, plus (16+L(bit_length(max(0,d-1)))) times the total potential
    support length, 64d, 256 per distinct width, and the QFT allowance
    256+32g+56b+64w+192Cp+L(bit_length(max(0,b-1)))*(b+2Cp+2w).
    L is _integer_storage. The per-table constant covers fixed objects,
    array headers, omission wrappers and mapping/sequence entries.
    These are qualified object allowances on 64-bit CPython 3.12.14,
    NumPy 2.5.2, SymPy 1.14.0 and Pydantic 2.13.5, not process RSS.

    latest sums 16E+h for one current construction per table. build is
    the maximum of 80E+h across all tables, preserving the qualified 64E
    synthesis scratch premise, and (192+2L(2200+2n))*E+h across
    Walsh-capable potential tables. The latter covers walsh_admission's
    simultaneous Fraction-loss list, its numerator/denominator objects,
    float/list staging and arrays. A finite binary64 dyadic uses at most
    2098 numerator bits and 1075 denominator bits. Division by 2**n and
    summation of at most 2**n entries fit 2200+2n bits. A forced dense
    potential bypasses this list. h is _construction_entry_cap's metadata.
    use reserves 64E_max for phase reconstruction and application.

    No arrays or gates are built here. BinaryModel admits the caller's
    held_bytes + model + latest + max(build,use) before allocation.
    Step/block records and content-identity serialization belong to that
    caller. Construction entries use their separate cache allowance.
    """
    table_specs = tuple((int(e), int(s)) for e, s in table_specs)
    specs = [(1 << bits, 1)] * d + list(table_specs)
    numeric = (16 * sum(e for e, _ in specs)
               + sum(8 << n for n in {e.bit_length() - 1 for e, _ in specs})
               + sum(e - 1 for e, _ in specs))
    model = numeric + _model_metadata(d, bits, table_specs, cutoff=cutoff, swaps=swaps)
    latest = sum(_construction_entry_cap(e, s)[0] for e, s in specs)
    build = max(80 * e + _construction_entry_cap(e, s)[1] for e, s in specs)
    if potential_walsh:
        initialization = max(((192 + 2 * _integer_storage(2200 + 2 * (e.bit_length() - 1))) * e
                              + _construction_entry_cap(e, s)[1] for e, s in table_specs), default=0)
        build = max(build, initialization)
    use = 64 * max(e for e, _ in specs)
    return model, latest, build, use


def kinetic_setup_work(bits):
    """Bound explicit numerical value operations for one periodic kinetic table.

    Count one unit per scalar or array element for arithmetic, absolute
    value, square, sine or cosine. Range tests, index bookkeeping and data
    movement are excluded. Elementary functions count as calls, not as
    the arithmetic internal to their implementations.

    Derivation. It applies to the current periodic ``kinetic_table``
    implementation, ``b >= 1``, ``K = 2**b``, either admitted kinetic model,
    and one invocation on one variable. Count each explicit scalar or
    array-element value operation once, including an exact sign change,
    square, sine or cosine. Exact range checks, mask/index bookkeeping,
    allocations, permutations and data movement are excluded. In
    particular, the bound does not count the bit complexity of a Fraction
    used by a boundary range check. Elementary-function units count calls,
    not their internal arithmetic.

    1. ``kinetic_table`` obtains the spacing twice, once inside
       ``kinetic_eigenvalues`` and once for ``kinetic_walsh``. Each periodic
       spacing is ``(upper-lower)/K``, two scalar operations. Charge four
       units in total.
    2. The periodic eigenvalue array performs ``index-K``, absolute value,
       division by K, multiplication by pi, division by h, squaring, and
       multiplication by two. These are seven K-entry operations for the
       spectral model. The finite-difference model adds the sine, for
       eight. Thus the common array charge is ``8K``, separate from the
       four spacing units.
    3. For spectral Walsh coefficients, the Nyquist wave number and energy
       use four scalar operations. In ``signed_square_walsh``, allow b
       additions for ``sum(weights)``, b multiplications and b additions for
       the squared-weight sum, and three operations for the remaining
       identity expression. Allow three operations per single-Z
       coefficient, including its sign, and two per two-Z coefficient.
       Include one further sign for the top signed weight. This totals at
       most ``b + 2b + 3 + 3b + b(b-1) + 1 = b**2 + 5b + 4`` before the four
       Nyquist operations. Normalizing and scaling the coefficient array
       adds ``2K``. The larger, convenient upper bound
       ``2K + b**2 + 6b + 8`` therefore covers the coefficient computation
       for every b>=1.
    4. For finite-difference Walsh coefficients, the four entries of
       ``signs`` use four divisions, four trigonometric calls and two sign
       changes. The spacing square and identity division add two, giving 12
       scalar units. There are K/2 masks. Each mask has b-1 levels, each
       with an angle multiplication, an angle division, a sine or cosine,
       and a product, followed by one final division for the mask. Its loop
       charge is ``(4(b-1)+1)K/2=(4b-3)K/2 <= 2bK``. The total coefficient
       charge is at most ``2bK+12``.
    5. Including spacing, the spectral total is at most
       ``10K+b**2+6b+12``. The finite-difference total is at most
       ``8K+2bK+16``. Both are bounded by ``C(b)=10K+2bK+b**2+6b+16``.
       These are integer inequalities, so no floating-point evaluation of a
       cost is needed.
    6. A model constructs exactly d such kinetic tables. Its setup charge
       is therefore ``d*C(b)``, paid at each model construction and outside
       the evolution's step loop. Bit reversal merely permutes the arrays
       and adds no unit under this convention.

    This gives a bound on the stated explicit operations for both models.
    It is not a runtime bound. Planning (``method.QHD._admit_symbolic_work``),
    the native construction (``method.QHD._select_binary_native``) and the
    classical product (``theory.binary_product_sizes``) each charge ``d*C(b)``
    for their model construction.
    """
    k = 1 << bits
    return 10 * k + 2 * bits * k + bits * bits + 6 * bits + 16


def kinetic_table(grid, var_index, model, relabel):
    """Return the kinetic phase table of variable ``var_index``, bit-reversed for ``relabel``.

    The values are the periodic eigenvalues E_k of ``model`` in Fourier-index
    order (``split_step.kinetic_eigenvalues``), and the Walsh coefficients
    come from their exact Pauli expansion (``kinetic_walsh``).

    Sign convention. Qiskit's QFT is ``F_+ = F_-^dagger`` with
    ``(F_- x)_k = K**(-1/2) sum_n exp(-2 pi i k n/K) x_n``. With J the map
    ``k -> -k mod K``, ``F_+ = J F_-``, so
    ``F_+^dagger D F_+ = F_-^dagger J D J F_-``. Both energies depend on k
    only through ``min(k, K - k)``, the Nyquist mode included, so
    ``E_(-k mod K) = E_k``, ``J D J = D``, and the circuit's
    ``F_+^dagger D F_+`` equals the classical ``F_-^dagger D F_-``. The
    unsigned square ``k**2`` is not symmetric and would break this.

    With W the swap-free QFT, ``W = R F_+`` and ``W^dagger = F_+^dagger R``
    (``qft_gates``). For any diagonal D, symmetric or not,
    ``W^dagger (R D R) W = F_+^dagger R R D R R F_+ = F_+^dagger D F_+``, so
    applying the table ``v[rev_b(k)]`` between the swap-free QFT and its exact
    inverse gives the kinetic step without swap layers, and no permutation
    remains on the position register. Omitting both swap layers while keeping
    D, or permuting the position state instead, gives a different operator.
    The identity also holds for a matched pair of truncated QFTs. Each
    variable is reversed within its own block.
    """
    from .split_step import kinetic_eigenvalues

    bits = register_bits(grid.num_grid_points)
    energies = np.asarray(kinetic_eigenvalues(grid, var_index, model), dtype=float)
    table = PhaseTable(energies, kinetic_walsh(model, bits, grid.spacing(var_index)))
    return table.permuted(bit_reversal(bits)) if relabel else table


class BinaryModel:
    """The phase tables and QFT gates of one binary Plan, rebuilt from its grid and stored support tables.

    Planning, the native construction (``native.append_binary_steps``) and the
    ``ir_product`` kernel (``theory.run_binary_product``) build this model
    from the same Method, grid and ``SupportValues`` and synthesize every
    block through ``construction``, so all three emit the same angles.

    ``held_bytes`` prices the caller's data that stay live while the model
    is used: the evolution states and phase buffers of
    ``theory.run_binary_product``, the rotation census and the Plan's tables
    of resource inspection. The constructor (``__init__``, documented under
    ``construction``) admits them with the model's baseline before building
    any array, and raises ``ValueError`` when they do not fit
    ``QHD.max_bytes``.

    Attributes:
        bits: b with ``K = 2**b``.
        kinetic: One kinetic ``PhaseTable`` per variable, bit-reversed for
            ``qft_bit_reversal="relabel"`` (``kinetic_table``).
        potential: The address-ordered ``PhaseTable`` of each support table,
            keyed by its sorted support, without Walsh coefficients
            (``PhaseTable.dense``) when ``BinarySynthesis.potential`` is
            ``"dense_diagonal"``.
        forward, inverse: The gates of the selected QFT and of its exact
            inverse (``qft_gates``).
        threshold: ``QHD.rotation_threshold``.
        kinetic_model: ``QHD.kinetic_model``.
        spacings: The grid spacing h of each variable.
    """

    def __init__(self, grid, method, support_values, *, held_bytes=0):
        self.bits = register_bits(grid.num_grid_points)
        # The caller's residents, the model, one current construction per table and the largest
        # synthesis or use workspace are admitted before any array is built; the rest funds the
        # optional whole-model cache (_construction_cache_capacity).
        self._cache_capacity = _construction_cache_capacity(
            method.max_bytes, held_bytes=held_bytes, **dict(zip(
                ("model_bytes", "latest_bytes", "build_bytes", "use_bytes"),
                _model_reservation(
                    grid.num_variables, self.bits,
                    [(t.values.array.size, len(t.support)) for t in support_values],
                    cutoff=method.binary_synthesis.aqft_cutoff,
                    swaps=method.binary_synthesis.qft_bit_reversal == "swap",
                    potential_walsh=method.binary_synthesis.potential != "dense_diagonal"),
                strict=True)))
        synthesis = method.binary_synthesis
        relabel = synthesis.qft_bit_reversal == "relabel"
        self.kinetic = tuple(
            kinetic_table(grid, j, method.kinetic_model, relabel) for j in range(grid.num_variables)
        )
        # Only a potential that can select Walsh rotations needs their coefficients and admission.
        table_of = PhaseTable.dense if synthesis.potential == "dense_diagonal" else PhaseTable.from_values
        self.potential = {
            tuple(table.support): table_of(address_table(table.values, len(table.support), grid.num_grid_points),
                                           largest=table.magnitude)
            for table in support_values
        }
        self.forward = qft_gates(self.bits, cutoff=synthesis.aqft_cutoff, swaps=not relabel)
        self.inverse = inverse_gates(self.forward)
        self.threshold = float(method.rotation_threshold)
        self.kinetic_model = method.kinetic_model
        self.spacings = tuple(grid.spacing(j) for j in range(grid.num_variables))
        # One popcount array per qubit count for every table of this model.
        popcounts = {}
        tables = (*self.kinetic, *self.potential.values())
        for table in tables:
            table._popcounts = popcounts
        # Optional whole-model entries only; the baseline above funds the model and current constructions.
        self._cache_bytes = 0
        self._cache = {}
        self._latest = {}
        self.construction_counts = {"hits": 0, "misses": 0, "fallbacks": 0}

    def construction(self, kind, variables, exponent, choice):
        """Return the ``DiagonalConstruction`` of one block's phase diagonal, built once per construction key.

        The construction cache is admitted from distinct model-bound keys,
        their actual phase or mask/angle payloads, each entry's metadata
        allowance h (``_construction_entry_cap``) and the largest
        new-construction workspace. Cache hits avoid construction
        work while every block occurrence still contributes its execution and
        error ledger.

        Within this immutable model, which fixes table content, grid spacing,
        kinetic model and bit-reversal choice, the key is
        ``(kind, variables, exponent.hex(), requested_synthesis, threshold)``;
        the requested synthesis distinguishes a ``min_cx`` trial from a
        forced choice. At equal keys every computed
        array and count is identical, so both second-order halves reuse one
        construction, and each occurrence still counts its CX, rotations,
        omissions and phase charges; the caller forms each occurrence's
        formation allowance.

        Admission. The caller first admits its live data, the model's arrays
        and metadata, one current construction per table, and the largest
        synthesis or use workspace against QHD.max_bytes. For host evolution
        the reservation includes the state buffers and retained complex phase
        tables; for inspection it includes the rotation census. The remaining
        bytes fund optional whole-model cache entries. A miss is built within
        the admitted current-construction envelope and retained globally only
        when its phase or mask/angle payload and key/scalar metadata fit that
        remainder. A fallback still uses admitted bytes and repeats synthesis
        work, which construction_counts records. Every block occurrence
        retains its original execution and error-ledger multiplicity.

        The baseline is ``_model_reservation`` plus the caller's
        ``held_bytes``, checked by ``_construction_cache_capacity`` in
        ``__init__`` before the model's arrays are built, and an entry's
        charge is ``_construction_entry_bytes``. A capacity of zero disables
        only whole-model retention and keeps the admitted fallback.
        """
        table = self.kinetic[variables[0]] if kind == "binary_kinetic" else self.potential[tuple(variables)]
        key = (kind, tuple(variables), float(exponent).hex(), choice, self.threshold)
        built = self._cache.get(key)
        if built is None:
            latest = self._latest.get((kind, tuple(variables)))
            if latest is not None and latest[0] == key:
                built = latest[1]
        if built is not None:
            self.construction_counts["hits"] += 1
            return built
        # The caller has already admitted model + latest + build/use + held bytes.
        # Thus synthesize is funded even if this result cannot be retained globally.
        built = table.synthesize(exponent, choice, self.threshold)
        charge = _construction_entry_bytes(
            built, entries=int(table.values.size), support_size=len(variables))
        if self._cache_bytes + charge <= self._cache_capacity:
            self._cache[key] = built
            self._cache_bytes += charge
            self.construction_counts["misses"] += 1
        else:
            self._latest[(kind, tuple(variables))] = (key, built)
            self.construction_counts["fallbacks"] += 1
        return built


def _scaled_absolute_sum_up(values, divisor):
    """Return an upper binary64 value of ``sum_r |v_r|/divisor`` for a positive divisor.

    The result is exactly zero for empty or all-zero input. Otherwise the
    correctly rounded ``math.fsum`` of the absolute values, stepped one
    number up, bounds their exact sum, and one upward division scales it.
    When that sum overflows, or its step reaches infinity from the largest
    finite number, each absolute value is divided upward and the quotients
    are added upward instead, so a finite scaled sum never needs a
    representable unscaled one (``compile_binary_steps``).
    """
    from .method import _div_up, _sum_up

    # The absolute values as one float64 array, summed without a Python list of its entries.
    magnitudes = np.abs(np.asarray(values, dtype=float)).reshape(-1)
    if not magnitudes.any():
        return 0.0
    try:
        total = math.nextafter(math.fsum(magnitudes), math.inf)
    except OverflowError:
        total = math.inf
    if math.isfinite(total):
        return _div_up(total, divisor)
    return _sum_up(_div_up(magnitude, divisor) for magnitude in magnitudes)


class WalshPhaseTerms(NamedTuple):
    """Upper allowances, in radians, for the identity phases of a Plan's Walsh diagonal blocks.

    Every executed Walsh block occurrence counts, kinetic blocks, both
    second-order potential halves and blocks without kept rotations
    included (``compile_binary_steps``). Kept-state phase admission uses
    them (``method._admit_compiled_phase``).

    Attributes:
        count: B, the number of Walsh blocks, each of which assigns its
            identity phase to the circuit's global phase
            (``native.append_diagonal``).
        identity_sum: Y, the sum of the absolute computed identity phases.
        formation: F_W, the summed formation allowances of the identity
            phases against the exact table or model identity.
        reconstruction: R_W, the summed per-entry allowances of
            ``DiagonalConstruction.emitted_phases`` against the exact phases
            of the same identity phase and kept rotations.
    """

    count: int
    identity_sum: float
    formation: float
    reconstruction: float


def _admit_reconstruction(identity_phase, half_sum, qubits):
    """Admit the finite arithmetic of reconstructing a selected Walsh diagonal (``DiagonalConstruction.emitted_phases``).

    Let A bound the exact sum of the kept absolute angles, n be the number
    of qubits, and ``gamma_n = n u/(1 - n u)``, with ``n u < 1``. Each
    partial butterfly value of ``walsh_sum`` on the kept angles has at most
    n additions on an input path, so its magnitude is at most
    ``G = (1 + gamma_n) A``, and its rounded half is at most
    ``G/2 + 2**-1075``. Require ``G <= sys.float_info.max`` and that bound
    plus ``|identity_phase|`` to be at most ``sys.float_info.max``. The
    caller supplies ``half_sum``, an upper value of ``A/2`` from the
    kept-angle reduction that the phase terms already form, so no inverse
    transform is formed, and the check is evaluated upward as
    ``(1 + gamma_n) A/2 <= max/2`` and
    ``|identity_phase| + (1 + gamma_n) A/2 + 2**-1074 <= max``, with an
    infinite value failing. For ``A = 0`` the reconstruction is the finite
    identity phase exactly. The reconstruction error allowance ``R_W``
    still covers inexact halving underflow (``compile_binary_steps``).

    Raises:
        ValueError: The reconstruction of this block could overflow.
    """
    from .method import _add_up, _gamma_up, _mul_up

    if half_sum == 0:
        return
    bound = _mul_up(_add_up(1.0, _gamma_up(qubits)), half_sum)
    total = _add_up(_add_up(abs(identity_phase), bound), math.ldexp(1.0, -1074))
    if not (bound <= math.ldexp(np.finfo(float).max, -1) and total <= np.finfo(float).max):
        raise ValueError(
            f"a Walsh diagonal on {qubits} qubits has kept rotation angles whose absolute sum, about "
            f"{2 * half_sum!r} rad, fails the sufficient absolute-sum bound, so the reconstruction of its "
            "phases could overflow binary64. The angles lie far "
            "outside any meaningful rotation range, so shorten total_time or rescale the objective")


def compile_binary_steps(model, method, step_weights, constant):
    """Return the binary step blocks and their ledgers.

    Step k applies, in list order, the potential blocks of every support
    table for its full duration dt and then one kinetic block per variable
    (first order, ``U_K(alpha) exp(-i B V)``), or the potential blocks for
    dt/2, the kinetic blocks for dt and the potential blocks for dt/2 again
    (second order, ``exp(-i B/2 V) U_K(alpha) exp(-i B/2 V)``), the ordering
    of the one-hot compiler (``compiler.QHDCompiler.build_step_pauli_ir``).
    Under the ``"integrated"`` coefficient rule the weights are the step
    averages, so both potential halves apply ``B_k/2``, the convention every
    route uses. Equal halves make the step the symmetric split of the first
    Magnus exponent, which keeps its time-ordering and product-formula errors
    separate, as ``method._error_model`` declares them, while unequal halves
    would mix the two (``compiler.QHDCompiler.build_step_pauli_ir`` derives
    this). The support tables are diagonal and commute, so their order
    within a potential factor adds no error. ``U_K(alpha)`` is the product of
    the per-variable conjugations ``F^dagger exp(-i alpha diag(E_j)) F``,
    which act on disjoint registers and apply each variable's periodic
    kinetic operator exactly when the QFTs are exact and no rotation is
    pruned. The binary product therefore differs from the one-hot one,
    whose kinetic term is itself split over overlapping links, and each is
    compared with its own product.

    A block's exponent is its duration times the step weight, so the
    potential diagonal is ``-exponent * v`` and the kinetic one
    ``-exponent * E``. Each kinetic block costs two QFTs (``qft_cx``) plus
    its phase diagonal, and three phase gates per controlled phase in each
    QFT. The objective's constant contributes the physical phase
    ``-dt * weight * constant`` per step, the only entry of the phase ledger,
    because both diagonal syntheses keep each table's identity component
    (``PhaseTable.synthesize``).

    Walsh identity phases. A Walsh block keeps its identity phase
    ``identity_phase = fl(-x c0)`` apart from its rotations, so kept-state
    phase admission charges its formation on both routes and, on the
    ``ir_product`` route, the reconstruction of the emitted phases
    (``WalshPhaseTerms``). The terms come from scalars of the constructions
    that compilation builds anyway. With ``u = 2**-53``,
    ``lambda = 2**-1074`` and n the block's qubits, every operation below is
    evaluated upward (``method._add_up`` and its siblings), and each
    ``lambda/2``, which is not a binary64 number, is charged as lambda. The
    mean absolute value a and the kept-angle half-sum T divide a sum of
    absolute values by a power of two. The correctly rounded ``math.fsum``
    of the absolute values, the premise of the phase ledger, stepped one
    number up and divided upward bounds it when that step is finite.
    Otherwise each absolute value is divided upward and the quotients are
    added upward, so a finite mean never needs a representable unscaled sum.

    - Exponent. ``x = fl(t w)`` for the duration t and step weight w, so
      ``|x - t w| <= eps_x = u |t w| + lambda/2``. The second-order half
      duration ``dt/2`` is exact, because planning admits it as a normal
      number (``schedules.step_weights``). Every exponent is admitted as a
      normal number too (``validation._normal_range``), so the lambda term
      is kept only as a margin of a few multiples of ``2**-1074``.
    - Identity formation. With the exact intended exponent ``X = t w``, the
      admitted identity coefficient ``c0`` of the block's table, a radius
      ``rho_0`` that encloses its distance from the original mathematical
      identity coefficient, and the emitted identity phase ``phi``, zero
      when ``-x c0`` was omitted (``PhaseTable.synthesize``), the identity
      ``phi + X c0* = (phi + x c0) + (X - x) c0 + X (c0* - c0)`` and the
      triangle inequality give
      ``e = |X| rho_0 + eps_x |c0| + |phi + x c0|``, whose last term is an
      exact scalar that includes the product's rounding or the omitted
      phase. For a potential block ``rho_0 = gamma_n a + d_0``, with a the
      table's mean absolute value and ``d_0`` its normalization omission
      (``walsh_admission``). For a kinetic block
      ``rho_0 = E_c = max(|c0 - low|, |c0 - high|)`` with ``[low, high]``
      from ``kinetic_identity_enclosure``.
    - Reconstruction. ``DiagonalConstruction.emitted_phases`` forms
      ``identity_phase - walsh_sum(theta)/2`` over the kept angles theta.
      With ``T = sum_kept |theta|/2``, the n-level butterfly of ``walsh_sum``
      errs by at most ``gamma_n T`` in each half-sum (the pairwise bound of
      ``walsh_coefficients``), the halving by at most ``lambda/2`` more, and
      the final subtraction by at most
      ``u (|identity_phase| + (1 + gamma_n) T + lambda/2) + lambda/2``. Since
      ``gamma_n + u (1 + gamma_n) <= gamma_(n+1)`` and the three lambda terms
      total less than ``2 lambda``, each emitted phase lies within
      ``R = u |identity_phase| + gamma_(n+1) T + 2 lambda`` of
      ``identity_phase`` minus the exact kept-angle half-sum. These errors
      can differ between basis states, so they are relative phases as well as
      a global one (``method._admit_compiled_phase`` bounds their effect on a
      state).

    ``F_W`` sums e and ``R_W`` sums R over the Walsh blocks. A dense diagonal
    keeps its identity phase inside its gate definition, whose formation
    belongs to gate synthesis and execution, and contributes nothing here.

    Range admission. Every exponent ``fl(t w)`` is normal
    (``validation._normal_range``), and each Walsh block's reconstruction has
    finite intermediates (``_admit_reconstruction``), whatever route the
    Plan selects, since any route may later be verified against the
    ``ir_product`` reference. A constant contribution whose ``fl(b_k c)`` or
    ``fl(dt fl(b_k c))`` would be nonzero and below ``2**-1022`` records a
    zero phase, which keeps the contribution count, and its exact
    ``|dt b_k c|`` joins the omitted identity charge of the phase ledger
    (``method._phase_ledger``). Stored table values may be subnormal, and
    the tables stay unchanged. The rotations and dense phase entries that
    the selected constructions omit below the range
    (``PhaseTable.synthesize``) add their exact charges to the pruning bound.

    Returns:
        ``(steps, ledger)``. ``steps`` holds one tuple of ``QHDBinaryBlock``
        per step. ``ledger`` holds ``physical_phase``, the ``math.fsum`` of the
        constant's N contributions (``method._phase_contributions`` counts
        them), ``phase_sources``, ``walsh_phase`` (``WalshPhaseTerms``),
        ``dropped_count``, ``dropped_angle_sum`` (the dropped absolute ``Rz``
        angles summed exactly and rounded to nearest), ``pruning_error_bound``
        and ``aqft_error_bound``, and ``range_omissions``, the lower-range
        omissions of this compilation with their exact charges, the Walsh
        normalization omissions of a table counted only when a selected
        Walsh block uses it. Since
        ``||Rz(theta) - I|| = 2 |sin(theta/4)| <= |theta|/2``, the pruning
        bound is half the exact sum of the absolute angles that the
        threshold dropped, which each block records as an integer count of
        ``2**-1074``, plus the exact range charges of the omitted rotations
        and dense entries. The sum is exact and converted to binary64 once
        upward (``evolution_bounds._upward``), so the recorded bound is never
        below the exact value and is 0.0 exactly when nothing is omitted.
    """
    from math import fsum

    from .evolution_bounds import _upward
    from .method import _add_up, _gamma_up, _mul_up, _sum_up
    from .records import QHDBinaryBlock

    synthesis = method.binary_synthesis
    dt = method.total_time / method.num_steps
    cutoff = synthesis.aqft_cutoff
    swaps = synthesis.qft_bit_reversal == "swap"
    # Two QFTs per conjugation, and three phase gates per controlled phase.
    qft_cost = 2 * qft_cx(model.bits, cutoff, swaps)
    qft_rotations = 2 * 3 * qft_controlled_phases(model.bits, cutoff)
    phases = []
    dropped_count = dropped_units = 0
    # Lower-range omissions of the selected constructions and of the constant's events, with their exact
    # charges, and the potential tables that a selected Walsh block uses.
    omitted = dict(rotations=0, dense_entries=0, identity_phases=0, identity_events=0)
    charges = dict(rotation=Fraction(0), dense=Fraction(0), identity_phase=Fraction(0), identity=Fraction(0))
    walsh_supports = set()
    steps = []
    u, lam = UNIT_ROUNDOFF, math.ldexp(1.0, -1074)

    def distance_up(a, b):
        low, high = min(a, b), max(a, b)
        return 0.0 if low == high else math.nextafter(high - low, math.inf)

    # a, the mean absolute value of each potential table.
    means = {support: _scaled_absolute_sum_up(table.values, float(table.values.size))
             for support, table in model.potential.items()}
    # rho_0 of each kinetic table, E_c = max(|c0 - low|, |c0 - high|).
    kinetic_rho = []
    for table, spacing in zip(model.kinetic, model.spacings, strict=True):
        low, high = kinetic_identity_enclosure(model.kinetic_model, model.bits, spacing)
        c0 = float(table.coefficients[0])
        kinetic_rho.append(max(distance_up(c0, low), distance_up(c0, high)))
    walsh_count = 0
    identity_phases, formations, reconstructions = [], [], []

    def tally(built):
        """Add one selected construction's threshold drops and lower-range omissions."""
        nonlocal dropped_count, dropped_units
        dropped_count += built.dropped_count
        dropped_units += built.dropped_angle_units
        walsh = built.synthesis == "walsh_rotations"
        omitted["rotations" if walsh else "dense_entries"] += built.range_omitted
        charges["rotation" if walsh else "dense"] += built.range_charge
        omitted["identity_phases"] += built.identity_omitted

    def charge(built, duration, weight, exponent, table, rho0):
        """Add one Walsh block's identity-phase allowances (WalshPhaseTerms)."""
        nonlocal walsh_count
        if built.synthesis != "walsh_rotations":
            return
        walsh_count += 1
        y, n = abs(built.identity_phase), built.qubits
        # eps_x = u |t w| + lambda/2, and |X| <= |x| + eps_x.
        eps = _add_up(_mul_up(u, _mul_up(abs(duration), abs(weight))), lam)
        c0 = float(table.coefficients[0])
        # |phi + x c0|, exact, for the emitted identity phase phi, which is |x c0| for an omitted one.
        exact = abs(Fraction(built.identity_phase) + Fraction(exponent) * Fraction(c0))
        if built.identity_omitted:
            charges["identity_phase"] += exact
        mismatch = _upward(exact)
        # e = |X| rho_0 + eps_x |c0| + |phi + x c0|
        error = _sum_up((_mul_up(_add_up(abs(exponent), eps), rho0), _mul_up(eps, abs(c0)), mismatch))
        half_sum = _scaled_absolute_sum_up(built.angles, 2.0)
        _admit_reconstruction(built.identity_phase, half_sum, n)
        identity_phases.append(y)
        formations.append(error)
        # R = u |identity_phase| + gamma_(n+1) T + 2 lambda
        reconstructions.append(_sum_up((_mul_up(u, y), _mul_up(_gamma_up(n + 1), half_sum), 2.0 * lam)))

    # rho_0 = gamma_n a + d_0 of each potential table.
    potential_rho = {support: _add_up(_mul_up(_gamma_up(table.qubits), means[support]),
                                      _upward(table.omission.identity_charge))
                     for support, table in model.potential.items()}

    def potential(duration, weight, time):
        blocks = []
        for support in model.potential:
            exponent = _normal_product(duration, weight, f"the potential exponent t b of support {support}")
            built = model.construction("binary_potential", support, exponent, synthesis.potential)
            charge(built, duration, weight, exponent, model.potential[support], potential_rho[support])
            tally(built)
            if built.synthesis == "walsh_rotations":
                walsh_supports.add(support)
            blocks.append(QHDBinaryBlock(kind="binary_potential", time_step=float(duration), time=float(time),
                                         variables=support, exponent=float(exponent), synthesis=built.synthesis,
                                         cx=built.cx, rotations=built.rotations))
        return blocks

    for time, kinetic_weight, potential_weight in step_weights:
        if constant != 0:
            # exp(-i dt (weight * constant) I), the constant objective as phase, or a zero event.
            coefficient = potential_weight * constant
            skipped = (_underflows(coefficient, "the objective constant's coefficient b c",
                                   lambda: Fraction(potential_weight) * Fraction(constant), zero=potential_weight == 0)
                       or _underflows(-dt * coefficient, "the objective constant's phase contribution -dt b c",
                                      lambda: Fraction(dt) * Fraction(coefficient), zero=coefficient == 0))
            if skipped:
                omitted["identity_events"] += 1
                charges["identity"] += abs(Fraction(dt) * Fraction(potential_weight) * Fraction(constant))
            phases.append(0.0 if skipped else -dt * coefficient)
        kinetic = []
        for j in range(len(model.kinetic)):
            try:
                exponent = _normal_product(dt, kinetic_weight, f"the kinetic exponent dt a of variable {j}")
            except ValueError as error:
                raise _kinetic_range_error(error) from error
            built = model.construction("binary_kinetic", (j,), exponent, synthesis.kinetic_phase)
            charge(built, dt, kinetic_weight, exponent, model.kinetic[j], kinetic_rho[j])
            tally(built)
            kinetic.append(QHDBinaryBlock(kind="binary_kinetic", time_step=float(dt), time=float(time),
                                          variables=(j,), exponent=float(exponent), synthesis=built.synthesis,
                                          cx=qft_cost + built.cx, rotations=qft_rotations + built.rotations))
        if method.trotter_order == 1:
            steps.append(tuple(potential(dt, potential_weight, time) + kinetic))
        else:
            half = dt / 2.0
            steps.append(tuple(potential(half, potential_weight, time) + kinetic
                               + potential(half, potential_weight, time)))
    try:
        physical = fsum(phases)
    except OverflowError:
        # method.QHD.plan rejects a nonfinite physical phase and names the objective constant.
        physical = math.inf
    conjugations = method.num_steps * len(model.kinetic)
    # min(2, 2 N_s d e_F), rounded outward by 1 + 4u for the two products.
    aqft = min(2.0, 2.0 * conjugations * qft_error_bound(model.bits, cutoff) * (1.0 + 4.0 * UNIT_ROUNDOFF))
    ledger = dict(
        physical_phase=physical,
        phase_sources=(("objective_constant", physical),) if phases else (),
        walsh_phase=WalshPhaseTerms(count=walsh_count, identity_sum=_sum_up(identity_phases),
                                    formation=_sum_up(formations), reconstruction=_sum_up(reconstructions)),
        dropped_count=dropped_count,
        # The exact sum of the dropped absolute angles, rounded to nearest.
        dropped_angle_sum=float(Fraction(dropped_units, 1 << 1074)),
        # ||Rz(theta) - I|| <= |theta|/2 per dropped rotation, halved exactly, plus the exact range charges,
        # rounded up once.
        pruning_error_bound=_upward(Fraction(dropped_units, 1 << 1075) + charges["rotation"] + charges["dense"]),
        aqft_error_bound=aqft,
        # A table's normalization omissions count only where a selected Walsh block uses its coefficients.
        range_omissions=dict(
            walsh_tables=tuple(tuple(table.omission) if support in walsh_supports else NO_WALSH_OMISSION
                               for support, table in model.potential.items()),
            **omitted, rotation_charge=charges["rotation"], dense_charge=charges["dense"],
            identity_phase_charge=charges["identity_phase"], identity_charge=charges["identity"]),
    )
    return tuple(steps), ledger
