"""The fermionic evolution generator of QFD from the integrals of chemistry data."""

from fractions import Fraction
from math import isfinite

import numpy as np

from nwqlib._limits import DEFAULT_MAX_BYTES
from .construction import canonical_order, canonical_partner, fermionic_blocks
from .records import QFDEvolutionDiagnostics


def _signed_midpoint(coefficient, partner, sign):
    """Return the exact midpoint ``q = (val(g) + sign val(g_bar))/2`` of a pair and ``m = RN(q)``, rounded once.

    The sum is formed in exact rationals, so two finite values of one sign
    cannot overflow, and the caller gives the partner the exact mirror
    ``sign m`` of the one rounded value.
    """
    exact = (Fraction(coefficient) + sign * Fraction(partner)) / 2
    return exact, float(exact)


def chemistry_evolution_terms(data, evolution_coefficient_cutoff=0.0):
    """Return a Hermitian fermionic evolution generator B for `QFD` from the integrals of chemistry data, with its diagnostics.

    Pass both results to `QFD`:
    `terms, diagnostics = chemistry_evolution_terms(data)` and then
    `QFD(initial_state=..., time_step=..., powers=..., evolution="fermionic_blocks",
    evolution_terms=terms, evolution_diagnostics=diagnostics)`, with the
    target `data.eigenproblem()` and, for example,
    `ingest_occupation(data.reference_occupations, num_qubits=data.num_qubits)`
    as the reference. The helper reads the spatial one-body
    array h, the two-body array ``v[p,q,r,s] = (ps|qr)``, `constant_energy`
    E_c, the `"name"` and `"integral_convention"` entries of `metadata` and
    `reference_occupations`. It runs no electronic-structure calculation,
    and it changes neither `data`, its arrays nor `data.hamiltonian`, which
    remains the target A. B differs from A in general, and the Plan states
    upper bounds on their distance.

    Raw table. Spin modes are interleaved, mode ``2p + sigma`` for spatial
    orbital p with alpha ``sigma = 0`` and beta ``sigma = 1``. The integrals
    are expanded to spin orbitals by the same routine as
    `build_gcim_chemistry_problem`, and every nonzero entry gives one term
    of ``E_c + sum h_ab a_a^dagger a_b + 1/2 sum v_abcd a_a^dagger
    a_b^dagger a_c a_d``, the factor 1/2 applied once as one binary64
    product. The terms are normal ordered with exact anticommutation signs
    and equal monomials summed, removing only exact zeros. The resulting
    coefficients g_T of the normal-ordered monomials T define the table
    C_raw, which need not be Hermitian because binary64 integrals need not
    be exactly symmetric. C_raw holds these binary64 values and is not an
    exact conversion of the integrals.

    Projection. For a pair of distinct adjoint monomials,
    ``T^dagger = s_T T_bar``, where s_T = +1 for every monomial of this
    domain, the returned coefficients are ``b_T = m_T`` and
    ``b_bar = s_T m_T`` with

        m_T = RN((val(g_T) + s_T val(g_bar)) / 2),

    the exact midpoint rounded once, which is the coefficient of the
    Hermitian part ``(C_raw + C_raw^dagger)/2`` in exact arithmetic. For
    two values x and y, ``(z - x)**2 + (z - y)**2`` is smallest at the
    midpoint, so this is the least-squares symmetric correction of the
    pair of C_raw, which says nothing about the exact integrals. For the
    one-body part it gives ``h_B[p,p] = h[p,p]`` and
    ``h_B[p,q] = h_B[q,p] = RN((h[p,q] + h[q,p])/2)``. A self-adjoint
    monomial keeps its real coefficient and E_c is kept. B is then exactly
    Hermitian, as `QFD` requires of `evolution_terms`.

    Cutoff. After the projection a pair with ``|m_T| <= gamma`` is removed
    on both sides, a self-adjoint monomial with ``|b_T| <= gamma`` is
    removed, and E_c is never removed. ``gamma = 0`` removes only exact
    zeros. Removing a number or density term changes the identity
    coefficient c_B of B and never the target's.

    Args:
        data (GCIMChemistryProblemData): The chemistry data, for example
            from `build_gcim_chemistry_problem`, with real finite integrals
            in Hartree.
        evolution_coefficient_cutoff (float): Default `0.0`. The cutoff
            gamma in Hartree, finite and at least 0, compared with the
            projected coefficients, which include the factor 1/2.

    Returns:
        terms (OperatorInput): B as a fermionic input from `ingest_fermion`,
            one row per kept monomial, on `data.num_qubits` modes.
        diagnostics (QFDEvolutionDiagnostics): The source of B, the cutoff,
            the counts of changed and removed pairs, upper bounds on the
            corrections, the one-body asymmetry of h, c_B and the spin
            statement of the blocks.

    Raises:
        ValueError: If the cutoff is negative, nonfinite or not a number, if
            an integral is nonfinite or has a nonzero imaginary part, or if
            the mode count or the reference occupations do not match the
            interleaved spin orbitals.

    Examples:
        For H2 in STO-3G (requires `pyscf` and `openfermion`):

        >>> from nwqlib.algorithms.gcim import build_gcim_chemistry_problem
        >>> from nwqlib.algorithms.qfd import chemistry_evolution_terms
        >>> data = build_gcim_chemistry_problem("H 0 0 0; H 0 0 0.74", basis="sto-3g", reference_methods=())
        >>> terms, diagnostics = chemistry_evolution_terms(data)
        >>> print(terms.basis.dimension, diagnostics.dropped_pairs, diagnostics.evolution_coefficient_cutoff)
        16 0 0.0
    """
    from nwqlib.algorithms.gcim.chemistry import _spin_orbital_integrals
    from nwqlib.algorithms.gcim.pair_reducer import upward
    from nwqlib.operators import ingest_fermion
    from nwqlib.subroutines.fermionic_pool import _normal_ordered_operator

    gamma = evolution_coefficient_cutoff
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not isfinite(gamma) or gamma < 0:
        raise ValueError("evolution_coefficient_cutoff must be a finite number at least 0, in Hartree")
    gamma = float(gamma)
    one, two = np.asarray(data.one_body_integrals), np.asarray(data.two_body_integrals)
    constant = data.constant_energy
    for name, array in (("one_body_integrals", one), ("two_body_integrals", two)):
        if not np.isfinite(array).all() or np.iscomplexobj(array) and np.any(array.imag != 0):
            raise ValueError(f"chemistry {name} must be real and finite for the QFD evolution generator")
    if isinstance(constant, bool) or not isinstance(constant, (int, float)) or not isfinite(constant):
        raise ValueError("chemistry constant_energy must be a finite real number")
    one_spin, two_spin = _spin_orbital_integrals(one.real, two.real)
    q, occupations = one_spin.shape[0], tuple(data.reference_occupations)
    if data.num_qubits != q or len(occupations) != q:
        raise ValueError(f"chemistry data with {one.shape[0]} spatial orbitals needs {q} interleaved spin modes and "
                         f"occupations, not {data.num_qubits} qubits and {len(occupations)} occupations")

    terms = [(float(constant), ())]
    terms += [(float(one_spin[a, b].real), ((int(a), 1), (int(b), 0))) for a, b in np.argwhere(one_spin != 0)]
    terms += [(0.5 * float(two_spin[a, b, c, d].real), ((int(a), 1), (int(b), 1), (int(c), 0), (int(d), 0)))
              for a, b, c, d in np.argwhere(two_spin != 0)]
    raw = {key: value.real for key, value in _normal_ordered_operator(terms, coefficient_cutoff=0).items()}

    kept, paired = {}, set()
    changed = dropped = 0
    symmetrization = rounding = removed = difference = Fraction(0)
    conserving = blocks = 0
    for key in sorted(raw, key=canonical_order):
        if key in paired:
            continue
        partner, sign = canonical_partner(key)
        paired.add(partner)
        if not key:
            kept[key] = raw[key]
            continue
        if partner == key:
            value = raw[key]
            if abs(value) <= gamma:
                dropped += 1
                removed += abs(Fraction(value))
                difference += abs(Fraction(value))
            else:
                kept[key] = value
        else:
            g, g_bar = raw[key], raw.get(partner, 0.0)
            exact, b = _signed_midpoint(g, g_bar, sign)
            b_bar = sign * b
            changed += (b, b_bar) != (g, g_bar)
            symmetrization += abs(exact - Fraction(g)) + abs(sign * exact - Fraction(g_bar))
            rounding += 2 * abs(Fraction(b) - exact)
            if abs(b) <= gamma:
                dropped += 1
                removed += 2 * abs(Fraction(b))
                difference += abs(Fraction(g)) + abs(Fraction(g_bar))
            else:
                kept[key], kept[partner] = b, b_bar
                difference += abs(Fraction(b) - Fraction(g)) + abs(Fraction(b_bar) - Fraction(g_bar))
        if key in kept:
            # A kept self-adjoint monomial or pair is one block of B. With
            # mode 2p + sigma, it conserves N_alpha and N_beta when each spin
            # has as many creations (action 1) as annihilations.
            blocks += 1
            conserving += all(sum(2 * action - 1 for mode, action in key if mode % 2 == spin) == 0 for spin in (0, 1))

    generator = ingest_fermion([(value, key) for key, value in kept.items()], num_modes=q)
    shift, _ = fermionic_blocks(generator, max_bytes=DEFAULT_MAX_BYTES)
    n = one.shape[0]
    asymmetry = max((abs(Fraction(float(one[p, r].real)) - Fraction(float(one[r, p].real)))
                     for p in range(n) for r in range(p + 1, n)), default=Fraction(0))
    correction = sum((abs(Fraction(kept.get(((2 * p, 1), (2 * r, 0)), 0.0)) - Fraction(float(one[p, r].real)))
                      for p in range(n) for r in range(n)), Fraction(0))
    electrons = sum(occupations)
    spin = Fraction(sum(occupations[0::2]) - sum(occupations[1::2]), 2)

    def bound(value):
        result = upward(value)
        return result if isfinite(result) else None

    diagnostics = QFDEvolutionDiagnostics(
        evolution_input=generator.reference.identity,
        source_name=data.metadata.get("name"),
        integral_convention=data.metadata.get("integral_convention"),
        one_body_shape=tuple(int(k) for k in one.shape),
        two_body_shape=tuple(int(k) for k in two.shape),
        evolution_coefficient_cutoff=gamma,
        changed_pairs=changed,
        dropped_pairs=dropped,
        symmetrization_correction=bound(symmetrization),
        rounding_correction=bound(rounding),
        dropped_mass=bound(removed),
        raw_difference=bound(difference),
        max_one_body_asymmetry=bound(asymmetry),
        one_body_correction=bound(correction),
        one_body_correction_bound=bound(2 * correction),
        generator_shift=float(shift),
        spin_statement=(f"{conserving} of {blocks} pairs of adjoint monomials and self-adjoint monomials conserve the "
                        "alpha and beta electron numbers, and hence Sz, with mode 2p + sigma. The reference "
                        f"occupations have N = {electrons} and Sz = {float(spin)!r}."),
    )
    return generator, diagnostics
