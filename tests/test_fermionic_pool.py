"""ADAPT-GCiM fermionic-pool and UCCSD anchors for generator action and symmetry."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pytest
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp

from nwqlib.subroutines.fermionic_pool import (
    apply_generator_exponential,
    enumerate_ceo_ovp_pool,
    enumerate_qeb_sd_pool,
    enumerate_spin_adapted_gsd_pool,
    enumerate_uccsd_sd_pool,
    spin_orbital,
)
from nwqlib.algorithms.gcim import sector_expectations
from _fermionic_references import (
    closed_form_double_generator,
    closed_form_single_generator,
    generator_matrix,
    generator_pauli,
    generator_unitary,
    jw_double_excitation_generator,
    jw_single_excitation_generator,
    spin_squared_operator,
    spin_z_operator,
    total_particle_number_operator,
)


def _matrix(pauli) -> np.ndarray:
    return np.asarray(pauli.to_matrix(), dtype=complex)


def test_spin_orbital_convention() -> None:
    assert spin_orbital(0, "up") == 0
    assert spin_orbital(0, "down") == 1
    assert spin_orbital(3, "up") == 6
    assert spin_orbital(3, "down") == 7


def test_generalized_pool_has_the_signed_operators_of_zheng_table_5():
    # Zheng et al., arXiv:2312.07691v3, Table V (Table 5 in the npj version,
    # doi:10.1038/s41534-024-00916-8). Each tuple is already
    # normally ordered as (creation, creation, annihilation, annihilation),
    # descending within both pairs. Swapping both pairs in its adjoint gives
    # two minus signs, so the adjoint's canonical tuple is (r,s,p,q).
    rows = (
        ("double_singlet", (0, 1, 2, 3), ((1, (2, 1, 6, 5)), (-1, (2, 1, 7, 4)),
                                        (-1, (3, 0, 6, 5)), (1, (3, 0, 7, 4)))),
        ("double_singlet", (1, 1, 2, 2), ((1, (3, 2, 5, 4)),)),
        ("double_singlet", (1, 1, 3, 3), ((1, (3, 2, 7, 6)),)),
        ("double_singlet", (0, 0, 1, 1), ((1, (1, 0, 3, 2)),)),
        ("double_singlet", (0, 0, 3, 3), ((1, (1, 0, 7, 6)),)),
        ("double_singlet", (0, 0, 2, 2), ((1, (1, 0, 5, 4)),)),
        ("double_singlet", (2, 2, 3, 3), ((1, (5, 4, 7, 6)),)),
        ("double_singlet", (1, 3, 2, 2), ((1, (5, 4, 6, 3)), (-1, (5, 4, 7, 2)))),
        ("double_singlet", (1, 3, 3, 3), ((-1, (6, 3, 7, 6)), (1, (7, 2, 7, 6)))),
        ("double_triplet", (0, 1, 0, 3), ((2, (2, 0, 6, 0)), (1, (2, 1, 6, 1)),
                                        (1, (2, 1, 7, 0)), (1, (3, 0, 6, 1)),
                                        (1, (3, 0, 7, 0)), (2, (3, 1, 7, 1)))),
        ("double_singlet", (0, 2, 3, 3), ((-1, (4, 1, 7, 6)), (1, (5, 0, 7, 6)))),
    )
    pool = {(g.family, g.spatial_indices): g for g in enumerate_spin_adapted_gsd_pool(4)}
    for family, indices, terms in rows:
        expected = {}
        norm = np.sqrt(2 * sum(coefficient**2 for coefficient, _ in terms))
        for coefficient, (p, q, r, s) in terms:
            expected[((p, 1), (q, 1), (r, 0), (s, 0))] = coefficient / norm
            expected[((r, 1), (s, 1), (p, 0), (q, 0))] = -coefficient / norm
        actual = {ops: coefficient for coefficient, ops in pool[family, indices].fermion_terms}
        # Four eps covers the short coefficient sums, square root and division.
        assert actual == pytest.approx(expected, rel=4 * np.finfo(float).eps, abs=0)

    # E1 fixes the relative alpha/beta sign for the chosen ordered pair.
    # The pool chooses p=0,q=1, creating on the lower index, to fix its orientation.
    single = pool["single", (0, 1)]
    assert {ops: c for c, ops in single.fermion_terms} == {
        ((0, 1), (2, 0)): .5, ((1, 1), (3, 0)): .5,
        ((2, 1), (0, 0)): -.5, ((3, 1), (1, 0)): -.5,
    }


@pytest.mark.parametrize("cutoff", [0.0, 1e-20])
def test_coalescing_preserves_residues_before_pruning(cutoff):
    from nwqlib.subroutines.fermionic_pool import _normal_ordered_operator, _fermion_terms_to_pauli

    # The large opposite components cancel exactly; fsum must recover the
    # stored small addend, in both parts, regardless of a later cutoff.
    small = complex(2.0**-55, 2.0**-54)
    values = [1 + 1j, small, -1 - 1j]
    assert _normal_ordered_operator([(c, ()) for c in values], coefficient_cutoff=cutoff) == {
        (): small
    }
    terms = {((i, 1), (i, 0)): 2 * c for i, c in enumerate(values)}
    mapped = dict(_fermion_terms_to_pauli(terms, 3, coefficient_cutoff=cutoff).to_list())
    # n_i = (I-Z_i)/2, so only the identity combines across the three terms.
    assert mapped == {"III": small, "IIZ": -values[0], "IZI": -small, "ZII": -values[2]}
    # Individually subthreshold contributions must first coalesce to 4*small.
    assert _normal_ordered_operator([(small, ())] * 4, coefficient_cutoff=abs(2 * small)) == {
        (): 4 * small
    }
    for invalid in (complex(float("nan"), 0), complex(float("inf"), 0)):
        with pytest.raises(ValueError, match="finite"):
            _normal_ordered_operator([(invalid, ())], coefficient_cutoff=cutoff)
        with pytest.raises(ValueError, match="finite"):
            _fermion_terms_to_pauli({(): invalid}, 1, coefficient_cutoff=cutoff)


def test_pauli_action_matches_tensor_factors_with_output_buffer(monkeypatch):
    import nwqlib.operators._pauli as pauli_kernel

    # Two-amplitude tiles make the eight-amplitude action cross tile boundaries.
    monkeypatch.setattr(pauli_kernel, "_PAULI_TILE", 2)
    local = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    entries = [("YZX", 0.25 + 0.5j), ("XYY", -0.5j), ("ZZZ", 0.75), ("III", -0.25)]
    matrix = sum(c * np.kron(np.kron(local[a], local[b]), local[d]) for (a, b, d), c in entries)
    state = np.arange(8, dtype=complex) + 1j * np.arange(8)[::-1]
    # Dyadic coefficients and integer components make every product and sum exact.
    expected = matrix @ state
    out = np.empty_like(state)
    assert pauli_kernel.apply_terms(entries, state, num_qubits=3, out=out) is out
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(pauli_kernel.apply_terms(entries, state, num_qubits=3), expected)
    with pytest.raises(ValueError, match="overlap"):
        pauli_kernel.apply_terms(entries, state, num_qubits=3, out=state)


def test_grouped_pauli_action_on_blocks_stays_within_its_rounding_relation(monkeypatch):
    from fractions import Fraction
    from math import sqrt

    from qiskit.quantum_info import SparsePauliOp

    import nwqlib.operators._pauli as pauli_kernel

    monkeypatch.setattr(pauli_kernel, "_PAULI_TILE", 16)
    rng = np.random.default_rng(11)
    n, columns = 6, 3
    flips = ["".join(rng.choice(list("IX"), n)) for _ in range(4)]
    labels = []
    for _ in range(24):
        flip = flips[rng.integers(len(flips))]
        labels.append("".join(("Y" if rng.random() < .5 else "X") if f == "X" else rng.choice(["I", "Z"])
                              for f in flip))
    coefficients = rng.normal(size=24) + 1j * rng.normal(size=24)
    block = rng.normal(size=(2**n, columns)) + 1j * rng.normal(size=(2**n, columns))
    table = pauli_kernel.PauliTerms._snapshot(
        n, *(np.array([[pauli_kernel._pauli_masks(p)[k]] for p in labels], dtype=np.uint64) for k in (0, 1)),
        coefficients)
    # Exact A V from the independent unit-coefficient Qiskit matrices, whose
    # entries are 0, +-1 or +-i, with rational accumulation.
    exact = [[Fraction(0)] * (2 * columns) for _ in range(2**n)]
    for label, c in zip(labels, coefficients, strict=True):
        matrix = SparsePauliOp(label).to_matrix()
        rows, sources = np.nonzero(matrix)
        for row, source in zip(rows, sources, strict=True):
            factor = complex(c) * complex(matrix[row, source])
            for col in range(columns):
                v = block[source, col]
                re, im = Fraction(factor.real), Fraction(factor.imag)
                exact[row][2 * col] += re * Fraction(v.real) - im * Fraction(v.imag)
                exact[row][2 * col + 1] += re * Fraction(v.imag) + im * Fraction(v.real)
    # The grouped action's relation (apply_terms): ||Y - AV||_F <= e_g*C1*||V||_F
    # with m* the largest flip group and G the number of flip groups.
    u = 2.0**-53
    def gamma(r):
        return r * u / (1 - r * u)

    sizes = [sum(1 for p in labels if "".join("X" if a in "XY" else "I" for a in p) == flip)
             for flip in dict.fromkeys("".join("X" if a in "XY" else "I" for a in p) for p in labels)]
    largest, groups = max(sizes), len(sizes)
    assert largest > 1 and groups > 1
    e_g = (1 + gamma(largest - 1)) * (1 + sqrt(2) * gamma(2)) * (1 + gamma(groups - 1)) - 1
    c1 = float(np.sum(np.abs(coefficients)))
    for state in (block, block[:, 0]):
        for terms in (table, tuple(zip(labels, coefficients, strict=True))):
            result = pauli_kernel.apply_terms(terms, state, num_qubits=n)
            assert result.shape == state.shape
            got = result.reshape(2**n, -1)
            error = sum((Fraction(got[k, col].real) - exact[k][2 * col]) ** 2
                        + (Fraction(got[k, col].imag) - exact[k][2 * col + 1]) ** 2
                        for k in range(2**n) for col in range(got.shape[1]))
            bound = e_g * c1 * float(np.linalg.norm(state))
            assert error <= Fraction(bound) ** 2
    # One block call applies the diagonal shared by all columns. An integer
    # width of a NumPy type, as a generator record may hold, is accepted.
    np.testing.assert_array_equal(
        pauli_kernel.apply_terms(table, block, num_qubits=n)[:, 1],
        pauli_kernel.apply_terms(table, np.ascontiguousarray(block[:, 1]), num_qubits=np.int64(n)))
    # A Y term on |0> gives +i at destination 1; an empty table gives zero.
    np.testing.assert_array_equal(pauli_kernel.apply_terms((("Y", 1),), np.array([1, 0j]), num_qubits=1),
                                  [0, 1j])
    np.testing.assert_array_equal(pauli_kernel.apply_terms((), block, num_qubits=n), np.zeros_like(block))
    with pytest.raises(ValueError, match="column"):
        pauli_kernel.apply_terms(table, block[:, :0], num_qubits=n)
    # A label must spell every qubit in IXYZ; an unknown letter is not I and
    # a short label does not act on the low qubits.
    for label in ("XA", "X"):
        with pytest.raises(ValueError, match=f"'{label}'"):
            pauli_kernel.apply_terms(((label, 1),), np.ones(4, dtype=complex), num_qubits=2)
    # Summing two DBL_MAX identity coefficients overflows, while each ordered
    # product with 1e-308 is finite; the grouped pass retries the term order.
    top = np.finfo(float).max
    tiny = np.array([1e-308, 1e-308], dtype=complex)
    retried = pauli_kernel.apply_terms((("I", top), ("I", top)), tiny, num_qubits=1)
    np.testing.assert_array_equal(retried, [2 * (top * 1e-308)] * 2)


def test_spin_squared_matches_openfermion_s_squared_operator() -> None:
    openfermion = pytest.importorskip("openfermion")
    from openfermion.linalg import get_sparse_operator
    from openfermion.transforms import jordan_wigner

    native = spin_squared_operator(4).toarray()
    reference = get_sparse_operator(
        jordan_wigner(openfermion.s_squared_operator(2)),
        n_qubits=4,
    ).toarray()

    np.testing.assert_allclose(native, reference, rtol=0.0, atol=1.0e-12)


def test_sector_expectations_flag_synthetic_spin_contamination() -> None:
    state = np.zeros(16, dtype=complex)
    state[0b1001] = 1.0 / np.sqrt(2.0)
    state[0b0110] = 1.0 / np.sqrt(2.0)

    qa = sector_expectations(state, num_qubits=4, reference_spin=0.0)

    assert qa["validation_gate"] is False
    assert qa["particle_number"] == pytest.approx(2.0, abs=1.0e-12)
    assert qa["spin_z"] == pytest.approx(0.0, abs=1.0e-12)
    assert qa["spin_squared"] == pytest.approx(2.0, abs=1.0e-12)
    assert qa["spin_contamination"] == pytest.approx(2.0, abs=1.0e-12)


def test_e6_closed_form_matches_hand_rolled_jw_for_empty_and_nonempty_ladders() -> None:
    theta = 0.371
    for p, q, num_qubits in ((1, 0, 2), (3, 0, 4)):
        jw = _matrix(jw_single_excitation_generator(p, q, num_qubits=num_qubits))
        closed = _matrix(closed_form_single_generator(p, q, num_qubits=num_qubits))
        np.testing.assert_allclose(jw, closed, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            la.expm(theta * jw),
            la.expm(theta * closed),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_e7_closed_form_matches_hand_rolled_jw_for_two_index_patterns() -> None:
    theta = -0.219
    for p, r, q, s, num_qubits in ((2, 3, 0, 1, 4), (3, 5, 0, 2, 6)):
        jw = _matrix(jw_double_excitation_generator(p, r, q, s, num_qubits=num_qubits))
        closed = _matrix(closed_form_double_generator(p, r, q, s, num_qubits=num_qubits))
        np.testing.assert_allclose(jw, closed, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            la.expm(theta * jw),
            la.expm(theta * closed),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_spin_adapted_pool_enumeration_order_at_n3() -> None:
    # Pool indices use this complete n_spatial=3 (family, spatial_indices)
    # sequence, with triplet-like before singlet-like for each double pair.
    expected_order = [
        ("single", (0, 1)),
        ("single", (0, 2)),
        ("single", (1, 2)),
        ("double_singlet", (0, 0, 0, 1)),
        ("double_singlet", (0, 0, 0, 2)),
        ("double_singlet", (0, 0, 1, 1)),
        ("double_singlet", (0, 0, 1, 2)),
        ("double_singlet", (0, 0, 2, 2)),
        ("double_triplet", (0, 1, 0, 2)),
        ("double_singlet", (0, 1, 0, 2)),
        ("double_singlet", (0, 1, 1, 1)),
        ("double_triplet", (0, 1, 1, 2)),
        ("double_singlet", (0, 1, 1, 2)),
        ("double_singlet", (0, 1, 2, 2)),
        ("double_singlet", (0, 2, 1, 1)),
        ("double_triplet", (0, 2, 1, 2)),
        ("double_singlet", (0, 2, 1, 2)),
        ("double_singlet", (0, 2, 2, 2)),
        ("double_singlet", (1, 1, 1, 2)),
        ("double_singlet", (1, 1, 2, 2)),
        ("double_singlet", (1, 2, 2, 2)),
    ]
    pool = enumerate_spin_adapted_gsd_pool(3)
    assert [(generator.family, generator.spatial_indices) for generator in pool] == expected_order
    # The two-orbital pool keeps the same order and has no triplet-like double.
    assert [(g.family, g.spatial_indices) for g in enumerate_spin_adapted_gsd_pool(2)] == [
        ("single", (0, 1)),
        ("double_singlet", (0, 0, 0, 1)),
        ("double_singlet", (0, 0, 1, 1)),
        ("double_singlet", (0, 1, 1, 1)),
    ]


def test_pool_family_counts_and_stable_indices() -> None:
    pool = enumerate_spin_adapted_gsd_pool(4)
    assert Counter(generator.family for generator in pool) == {
        "single": 6,
        "double_singlet": 45,
        "double_triplet": 15,
    }
    assert [generator.pool_index for generator in pool] == list(range(len(pool)))
    assert [(generator.family, generator.spatial_indices) for generator in pool[:8]] == [
        ("single", (0, 1)),
        ("single", (0, 2)),
        ("single", (0, 3)),
        ("single", (1, 2)),
        ("single", (1, 3)),
        ("single", (2, 3)),
        ("double_singlet", (0, 0, 0, 1)),
        ("double_singlet", (0, 0, 0, 2)),
    ]


def test_pool_enumeration_law_covers_every_builtin_member(monkeypatch) -> None:
    import nwqlib.subroutines.fermionic_pool as owner
    from nwqlib.algorithms.gcim.adapt_inputs import _pool_enumeration_requirements
    from nwqlib.operators._fermion import mapping_requirements

    # Four spatial orbitals contain every order pattern of four spatial
    # indices, so these members reach each family's largest strings and rows.
    # The bytes that _snapshot_generator charges each member, and the largest
    # numbers of fermion strings and of operators in a string, are measured
    # here.
    reference = (1, 1, 1, 1, 0, 0, 0, 0)
    charged, check = [], owner._check_generator_work

    def recorded(max_bytes, max_products, *, payload_bytes, work):
        charged.append(payload_bytes)
        return check(max_bytes, max_products, payload_bytes=payload_bytes, work=work)

    snapshot_bytes = strings = operators = 0
    with monkeypatch.context() as spy:
        spy.setattr(owner, "_check_generator_work", recorded)
        for pool in (
            enumerate_spin_adapted_gsd_pool(4),
            enumerate_uccsd_sd_pool(reference),
            enumerate_qeb_sd_pool(reference),
            enumerate_ceo_ovp_pool(reference),
        ):
            for generator in pool:
                charged.clear()
                owner._snapshot_generator(generator)
                snapshot_bytes = max(snapshot_bytes, charged[-1])
                strings = max(strings, len(generator.fermion_terms))
                operators = max([operators, *(len(term) for _, term in generator.fermion_terms)])
    # Both normal orderings of one spin-adapted candidate receive the same
    # base terms, so the comparisons are summed per base-term tuple.
    comparisons, per_call, total = Counter(), [], [0]
    compare, normalize = owner._out_of_normal_order, owner._normalized_anti_hermitian_terms

    def counted(left, right):
        total[0] += 1
        return compare(left, right)

    def normal_ordering(base_terms, **options):
        start = total[0]
        result = normalize(base_terms, **options)
        per_call.append(total[0] - start)
        comparisons[tuple(base_terms)] += per_call[-1]
        return result

    monkeypatch.setattr(owner, "_out_of_normal_order", counted)
    monkeypatch.setattr(owner, "_normalized_anti_hermitian_terms", normal_ordering)
    enumerate_spin_adapted_gsd_pool(4)
    # The law charges each candidate the largest member and the mapping of
    # its strings in bytes, and that mapping and the comparisons in work.
    q = len(reference)
    _, mapping_bytes, mapping_work = mapping_requirements((operators,) * strings, num_modes=q, labels=True)
    law_bytes, law_work = _pool_enumeration_requirements(1, q)
    assert law_bytes >= snapshot_bytes + mapping_bytes
    assert law_work >= mapping_work + max(comparisons.values())


@pytest.mark.parametrize(
    "reference",
    (
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 1, 1, 0, 0, 0),
        (1, 1, 1, 1, 0, 0, 0, 0),
        (1, 1, 1, 0, 1, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0),
    ),
)
def test_reference_pool_size_matches_enumeration(reference) -> None:
    from nwqlib.subroutines.fermionic_pool import _reference_pool_size

    # Balanced, spin-polarized, odd and empty references.
    for family, enumerate_pool in (
        ("uccsd_sd", enumerate_uccsd_sd_pool),
        ("qeb_sd", enumerate_qeb_sd_pool),
        ("ceo_ovp", enumerate_ceo_ovp_pool),
    ):
        assert _reference_pool_size(family, reference) == len(enumerate_pool(reference))


def test_generator_invariants_for_every_pool_family() -> None:
    pool = enumerate_spin_adapted_gsd_pool(3)
    assert {generator.family for generator in pool} == {
        "single",
        "double_singlet",
        "double_triplet",
    }
    particle_number = total_particle_number_operator(6).toarray()
    spin_z = spin_z_operator(6).toarray()
    for generator in pool:
        matrix = generator_matrix(generator)
        np.testing.assert_allclose(matrix + matrix.conj().T, 0.0, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            particle_number @ matrix - matrix @ particle_number, 0.0, rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(spin_z @ matrix - matrix @ spin_z, 0.0, rtol=0.0, atol=1.0e-12)
        assert all(abs(complex(coefficient).real) <= 1.0e-12 for _, coefficient in generator.pauli_terms)


def test_uccsd_pool_respects_occ_virt_restriction_and_symmetries() -> None:
    occupations = (1, 1, 0, 0, 0, 0)
    pool = enumerate_uccsd_sd_pool(occupations)
    particle_number = total_particle_number_operator(6).toarray()
    spin_z = spin_z_operator(6).toarray()
    assert [generator.pool_index for generator in pool] == list(range(len(pool)))
    for generator in pool:
        assert generator.num_qubits == 6
        indices = generator.spin_orbital_indices
        assert indices is not None
        if generator.family == "uccsd_single":
            i, a = indices
            # occupied -> virtual only, same spin label
            assert occupations[i] == 1 and occupations[a] == 0
            assert i % 2 == a % 2
            assert generator.spatial_indices == (i // 2, a // 2)
        else:
            assert generator.family == "uccsd_double"
            i, j, a, b = indices
            # occupied pair -> virtual pair only, S_z conserved
            assert i < j and a < b
            assert occupations[i] == 1 and occupations[j] == 1
            assert occupations[a] == 0 and occupations[b] == 0
            assert (i % 2) + (j % 2) == (a % 2) + (b % 2)
            assert generator.spatial_indices == (i // 2, j // 2, a // 2, b // 2)
        matrix = generator_matrix(generator)
        np.testing.assert_allclose(matrix + matrix.conj().T, 0.0, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            particle_number @ matrix - matrix @ particle_number, 0.0, rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(spin_z @ matrix - matrix @ spin_z, 0.0, rtol=0.0, atol=1.0e-12)


def test_uccsd_pool_size_pins_from_hand_combinatorics() -> None:
    # (occupations) -> (singles, doubles) by hand: spin-conserving singles are
    # n_occ_sigma * n_virt_sigma summed over sigma; S_z-conserving doubles are
    # occupied pairs i<j times virtual pairs a<b with matching spin-label sum.
    cases = {
        (1, 0, 0, 0): (1, 0),  # {0}->{2}; no occupied pair
        (1, 1, 0, 0): (2, 1),  # 1*1 + 1*1; (0,1)->(2,3)
        (1, 1, 0, 0, 0, 0): (4, 4),  # 1*2 + 1*2; (0,1)->{(2,3),(2,5),(3,4),(4,5)}
        (1, 1, 1, 1, 0, 0): (4, 4),  # 2*1 + 2*1; alpha-beta occ pairs -> (4,5)
    }
    for occupations, (singles, doubles) in cases.items():
        pool = enumerate_uccsd_sd_pool(occupations)
        families = Counter(generator.family for generator in pool)
        assert families.get("uccsd_single", 0) == singles, occupations
        assert families.get("uccsd_double", 0) == doubles, occupations
        assert len(pool) == singles + doubles, occupations


def test_qeb_and_ceo_pool_size_pins_from_hand_combinatorics() -> None:
    assert {
        occupations: len(enumerate_qeb_sd_pool(occupations))
        for occupations in (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0),
        )
    } == {
        (1, 0, 0, 0): 1,
        (1, 1, 0, 0): 3,
        (1, 1, 0, 0, 0, 0): 8,
        (1, 0, 1, 0, 0, 0, 0, 0): 5,
    }
    # Singles: respectively 1, 2, 4, 4 occupied-to-virtual same-spin moves.
    # Mixed-spin CEO doubles couple direct/crossed exchanges with both signs:
    # 1 or 4 occupied/virtual quartets give 2 or 8 doubles. The same-spin
    # quartet {0,2,4,6} has three partitions into two pairs, coupled in
    # C(3,2)=3 ways with both signs: 4 singles + 6 doubles = 10.
    assert {
        occupations: len(enumerate_ceo_ovp_pool(occupations))
        for occupations in (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0),
        )
    } == {
        (1, 0, 0, 0): 1,
        (1, 1, 0, 0): 4,
        (1, 1, 0, 0, 0, 0): 12,
        (1, 0, 1, 0, 0, 0, 0, 0): 10,
    }


def test_qeb_and_ceo_generators_are_z_ladder_free_and_symmetry_preserving() -> None:
    for pool in (
        enumerate_qeb_sd_pool((1, 1, 0, 0, 0, 0)),
        enumerate_ceo_ovp_pool((1, 1, 0, 0, 0, 0)),
        enumerate_ceo_ovp_pool((1, 0, 1, 0, 0, 0, 0, 0)),
    ):
        particle_number = total_particle_number_operator(pool[0].num_qubits).toarray()
        spin_z = spin_z_operator(pool[0].num_qubits).toarray()
        for generator in pool:
            matrix = generator_matrix(generator)
            np.testing.assert_allclose(matrix + matrix.conj().T, 0.0, rtol=0.0, atol=1.0e-12)
            np.testing.assert_allclose(
                particle_number @ matrix - matrix @ particle_number,
                0.0,
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                spin_z @ matrix - matrix @ spin_z, 0.0, rtol=0.0, atol=1.0e-12
            )
            assert all("Z" not in str(label) for label in generator_pauli(generator).paulis)


def test_qeb_and_ceo_content_pins() -> None:
    """Explicit Pauli label sets distinguish QEB doubles from the two CEO combinations on one
    occupation reference.
    """
    qeb = enumerate_qeb_sd_pool((1, 1, 0, 0))
    ceo = enumerate_ceo_ovp_pool((1, 1, 0, 0))

    assert [(generator.family, generator.spin_orbital_indices) for generator in qeb] == [
        ("qeb_single", (0, 2)),
        ("qeb_single", (1, 3)),
        ("qeb_double", (0, 1, 2, 3)),
    ]
    assert [len(generator_pauli(generator).paulis) for generator in qeb] == [2, 2, 8]
    assert qeb[2].spin_orbital_indices == (0, 1, 2, 3)
    assert {str(label) for label in generator_pauli(qeb[2]).paulis} == {
        "XXYX",
        "XXXY",
        "YXXX",
        "YXYY",
        "XYXX",
        "XYYY",
        "YYYX",
        "YYXY",
    }

    assert [generator.family for generator in ceo] == [
        "ceo_ovp_single",
        "ceo_ovp_single",
        "ceo_ovp_plus",
        "ceo_ovp_minus",
    ]
    assert [len(generator_pauli(generator).paulis) for generator in ceo] == [2, 2, 4, 4]
    assert ceo[2].spin_orbital_indices == (0, 1, 2, 3)
    assert {str(label) for label in generator_pauli(ceo[2]).paulis} == {
        "XXYX",
        "YXXX",
        "XYYY",
        "YYXY",
    }


def test_ceo_ovp_plus_minus_match_independent_dense_direct_crossed_identity() -> None:
    pool = enumerate_ceo_ovp_pool((1, 1, 0, 0))
    (plus,) = (generator for generator in pool if generator.family == "ceo_ovp_plus")
    (minus,) = (generator for generator in pool if generator.family == "ceo_ovp_minus")

    raising = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    def dense_excitation(
        create_a: int,
        create_b: int,
        annihilate_a: int,
        annihilate_b: int,
    ) -> np.ndarray:
        factors = [identity] * 4
        factors[create_a] = factors[create_b] = raising
        factors[annihilate_a] = factors[annihilate_b] = lowering
        forward = np.ones((1, 1), dtype=complex)
        for factor in reversed(factors):
            forward = np.kron(forward, factor)
        return forward - forward.conj().T

    direct = dense_excitation(2, 3, 0, 1)
    crossed = dense_excitation(0, 3, 2, 1)

    np.testing.assert_allclose(
        generator_matrix(plus),
        direct + crossed,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        generator_matrix(minus),
        direct - crossed,
        rtol=0.0,
        atol=1.0e-12,
    )

    # Four active alpha modes in an eight-mode pool; the beta spectators must
    # carry identity. Strip only those identities so the complete operator
    # comparison uses 16x16 matrices, without an eight-qubit dense reference.
    same_spin = [
        generator
        for generator in enumerate_ceo_ovp_pool((1, 0, 1, 0, 0, 0, 0, 0))
        if generator.family in {"ceo_ovp_plus", "ceo_ovp_minus"}
    ]
    exchanges = (
        dense_excitation(2, 3, 0, 1),
        dense_excitation(1, 3, 0, 2),
        dense_excitation(1, 2, 0, 3),
    )
    expected = [
        (family, left + sign * right)
        for left, right in combinations(exchanges, 2)
        for family, sign in (("ceo_ovp_plus", 1), ("ceo_ovp_minus", -1))
    ]
    assert len(same_spin) == len(expected) == 6
    for generator, (family, matrix) in zip(same_spin, expected, strict=True):
        assert generator.family == family
        assert generator.spin_orbital_indices == (0, 2, 4, 6)
        terms = generator_pauli(generator).to_list()
        assert all(label[::2] == "IIII" for label, _ in terms)
        active = SparsePauliOp.from_list([(label[1::2], value) for label, value in terms])
        np.testing.assert_allclose(active.to_matrix(), matrix, rtol=0.0, atol=1.0e-12)


def test_uccsd_pool_members_match_e4_e5_generators_and_closed_forms() -> None:
    pool = enumerate_uccsd_sd_pool((1, 1, 0, 0))
    assert [(generator.family, generator.spin_orbital_indices) for generator in pool] == [
        ("uccsd_single", (0, 2)),
        ("uccsd_single", (1, 3)),
        ("uccsd_double", (0, 1, 2, 3)),
    ]
    scale = 1.0 / np.sqrt(2.0)  # pool members are coefficient-normalized
    np.testing.assert_allclose(
        generator_matrix(pool[0]),
        scale * _matrix(jw_single_excitation_generator(2, 0, num_qubits=4)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        generator_matrix(pool[0]),
        scale * _matrix(closed_form_single_generator(2, 0, num_qubits=4)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        generator_matrix(pool[2]),
        scale * _matrix(jw_double_excitation_generator(2, 3, 0, 1, num_qubits=4)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        generator_matrix(pool[2]),
        scale * _matrix(closed_form_double_generator(2, 3, 0, 1, num_qubits=4)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_uccsd_pool_rejects_bad_occupations_and_handles_empty_sides() -> None:
    for enumerate_pool in (enumerate_uccsd_sd_pool, enumerate_qeb_sd_pool, enumerate_ceo_ovp_pool):
        for occupations in ((1, 2, 0, 0), (1.9, 1.1, 0.9, 0.2)):
            with pytest.raises(ValueError, match="only 0/1"):
                enumerate_pool(occupations)
        with pytest.raises(ValueError, match="must not be empty"):
            enumerate_pool(())
        assert enumerate_pool((1, 1)) == ()
        assert enumerate_pool((0, 0)) == ()
        expected = enumerate_pool((1, 1, 0, 0))
        for occupations in ("1100", np.array([1, 1, 0, 0]), (1.0, 1.0, 0.0, 0.0)):
            actual = enumerate_pool(occupations)
            assert [g.spin_orbital_indices for g in actual] == [
                g.spin_orbital_indices for g in expected
            ]
            assert [g.pauli_terms for g in actual] == [g.pauli_terms for g in expected]
    with pytest.raises(ValueError, match="integer"):
        enumerate_spin_adapted_gsd_pool(2.9)
    assert len(enumerate_spin_adapted_gsd_pool(np.int64(2))) == len(
        enumerate_spin_adapted_gsd_pool(2)
    )


def test_generator_exponential_application_matches_dense_unitary() -> None:
    generator = enumerate_spin_adapted_gsd_pool(3)[4]
    theta = np.pi / 4
    state = np.zeros(2**generator.num_qubits, dtype=complex)
    state[0b001111] = 1.0

    unitary = generator_unitary(generator, theta)
    np.testing.assert_allclose(
        apply_generator_exponential(generator, state, theta),
        unitary @ state,
        rtol=0.0,
        atol=1.0e-12,
    )


def _car_generator_matrix(generator):
    """Independent occupation-basis action of stored fermion strings."""
    dimension = 1 << generator.num_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for coefficient, operators in generator.fermion_terms:
        for ket in range(dimension):
            bra, amplitude = ket, coefficient
            for mode, creation in reversed(operators):
                if ((bra >> mode) & 1) == creation:
                    break
                amplitude *= (-1) ** (bra & ((1 << mode) - 1)).bit_count()
                bra ^= 1 << mode
            else:
                matrix[bra, ket] += amplitude
    return matrix


@pytest.mark.parametrize("theta", (-2.3, 0.23))
def test_compact_gsd_circuits_cover_default_index_classes(theta) -> None:
    from qiskit.quantum_info import Operator
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    # Three spatial labels cover every shared/repeated-index ordering up to an order-preserving relabeling. The six four-distinct rows require the four-orbital pool.
    generators = list(enumerate_spin_adapted_gsd_pool(3)) + [
        g for g in enumerate_spin_adapted_gsd_pool(4) if len(set(g.spatial_indices)) == 4
    ]
    for generator in generators:
        car = _car_generator_matrix(generator)
        np.testing.assert_allclose(car, generator_matrix(generator), rtol=0.0, atol=2.0e-14)
        circuit = build_generator_circuit(generator, theta)
        np.testing.assert_allclose(
            Operator(circuit).data,
            la.expm(theta * car),
            rtol=0.0,
            atol=2.0e-11,
            err_msg=f"{generator.family} {generator.spatial_indices} theta={theta}",
        )


@pytest.mark.parametrize(
    "family, indices, theta",
    (
        ("double_singlet", (0, 0, 1, 3), -2.3),
        ("double_singlet", (0, 1, 1, 3), 0.23),
        ("double_triplet", (0, 1, 1, 3), -0.71),
        ("double_singlet", (0, 1, 2, 3), 3.2),
        ("double_triplet", (0, 2, 1, 3), -2.3),
    ),
)
def test_compact_gsd_control_inverse_and_occupied_spectators(family, indices, theta) -> None:
    from qiskit.quantum_info import Operator
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    generator = next(
        g
        for g in enumerate_spin_adapted_gsd_pool(4)
        if g.family == family and g.spatial_indices == indices
    )
    exact = la.expm(theta * _car_generator_matrix(generator))
    controlled = np.zeros((512, 512), dtype=complex)
    controlled[::2, ::2] = np.eye(256)
    controlled[1::2, 1::2] = exact
    # Entrywise equality includes the relative global phase and every spectator occupation.
    observed = build_generator_circuit(generator, theta, controlled=True)
    np.testing.assert_allclose(Operator(observed).data, controlled, rtol=0.0, atol=2.0e-11)
    inverse = build_generator_circuit(generator, theta, controlled=True, inverse=True)
    np.testing.assert_allclose(Operator(inverse).data, controlled.conj().T, rtol=0.0, atol=2.0e-11)


@pytest.mark.parametrize(
    "family,indices,theta",
    (
        ("double_singlet", (0, 0, 1, 2), 7.3),
        ("double_triplet", (0, 1, 0, 2), -np.pi),
    ),
)
def test_compact_inverse_evaluates_full_map_without_reinverting_controls(
    monkeypatch, family, indices, theta
) -> None:
    from qiskit.circuit import ControlledGate
    from qiskit.quantum_info import Operator
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    generator = next(
        g
        for g in enumerate_spin_adapted_gsd_pool(3)
        if g.family == family and g.spatial_indices == indices
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("generator inverse reentered generic controlled inversion")

    monkeypatch.setattr(ControlledGate, "inverse", forbidden)
    inverse = build_generator_circuit(generator, theta, controlled=True, inverse=True)
    expected = np.eye(128, dtype=complex)
    expected[1::2, 1::2] = la.expm(-theta * _car_generator_matrix(generator))
    np.testing.assert_allclose(Operator(inverse).data, expected, rtol=0.0, atol=2.0e-11)


def test_compact_circuit_preserves_uccsd_qeb_and_ceo_generators() -> None:
    from qiskit.quantum_info import Operator
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    for pool in (enumerate_uccsd_sd_pool, enumerate_qeb_sd_pool, enumerate_ceo_ovp_pool):
        for generator in pool((1, 1, 0, 0)):
            exact = la.expm(-0.37 * generator_matrix(generator))
            np.testing.assert_allclose(
                Operator(build_generator_circuit(generator, -0.37)).data,
                exact,
                rtol=0.0,
                atol=1.0e-12,
            )


def test_compact_four_distinct_circuit_keeps_small_nonzero_angles() -> None:
    from qiskit.quantum_info import Operator
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    generator = next(
        g
        for g in enumerate_spin_adapted_gsd_pool(4)
        if g.family == "double_triplet" and g.spatial_indices == (0, 1, 2, 3)
    )
    car = _car_generator_matrix(generator)
    theta = 1.0e-11
    observed = Operator(build_generator_circuit(generator, theta)).data
    # Compare the leading action where A is nonzero; an absolute whole-unitary tolerance alone would accept the identity circuit here.
    mask = car != 0.0
    np.testing.assert_allclose(observed[mask] / theta, car[mask], rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize("theta", (-7.3, -np.pi, np.pi, 7.3))
def test_compact_closed_forms_across_principal_angle_branches(theta) -> None:
    from qiskit.quantum_info import Operator
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    for n_spatial, family, indices in (
        (3, "double_singlet", (0, 0, 1, 2)),
        (4, "double_singlet", (0, 1, 2, 3)),
        (4, "double_triplet", (0, 1, 2, 3)),
    ):
        generator = next(
            g
            for g in enumerate_spin_adapted_gsd_pool(n_spatial)
            if g.family == family and g.spatial_indices == indices
        )
        exact = la.expm(theta * _car_generator_matrix(generator))
        np.testing.assert_allclose(
            Operator(build_generator_circuit(generator, theta)).data, exact, rtol=0.0, atol=2.0e-11
        )


@pytest.mark.parametrize("theta", (1.0e-11, 0.37))
def test_compact_default_commuting_control_preserves_action_and_phase(theta) -> None:
    from qiskit.quantum_info import Operator
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    for generator in (enumerate_spin_adapted_gsd_pool(2)[0], enumerate_spin_adapted_gsd_pool(2)[2]):
        car = _car_generator_matrix(generator)
        expected = np.eye(32, dtype=complex)
        expected[1::2, 1::2] = la.expm(theta * car)
        observed = Operator(build_generator_circuit(generator, theta, controlled=True)).data
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-12)
        if theta < 1.0e-10:
            mask = car != 0.0
            np.testing.assert_allclose(
                observed[1::2, 1::2][mask] / theta, car[mask], rtol=1.0e-4, atol=1.0e-7
            )


def test_generator_compilation_plan_reuses_local_admission_across_angles(monkeypatch):
    import nwqlib.subroutines.fermionic_circuits as compiler

    generator = enumerate_spin_adapted_gsd_pool(2)[1]
    admissions, local_blocks = [], []
    original_admission = compiler._require_default_double
    original_blocks = compiler._active_occupation_blocks

    def admission(item, **kwargs):
        admissions.append(item.pool_index)
        return original_admission(item, **kwargs)

    def blocks(item, **kwargs):
        local_blocks.append(item.pool_index)
        return original_blocks(item, **kwargs)

    monkeypatch.setattr(compiler, "_require_default_double", admission)
    monkeypatch.setattr(compiler, "_active_occupation_blocks", blocks)
    plan = compiler._plan_generator_circuit(generator)
    for theta in (0.17, -0.41):
        for controlled in (False, True):
            compiler.build_generator_circuit(generator, theta, controlled=controlled, _plan=plan)
    assert admissions == local_blocks == [generator.pool_index]
    assert all(block.shape[0] <= 5 for _, block in plan.occupation_blocks)


def test_native_jw_creation_phase_content():
    from nwqlib.subroutines.fermionic_pool import _fermion_terms_to_pauli

    assert dict(_fermion_terms_to_pauli({((1, 1),): 1}, 2).to_list()) == {"XZ": 0.5, "YZ": -0.5j}
    assert dict(_fermion_terms_to_pauli({((1, 0),): 1}, 2).to_list()) == {"XZ": 0.5, "YZ": 0.5j}


def test_spin_squared_action_zero_mode_without_native_action(monkeypatch):
    import nwqlib.subroutines.fermionic_pool as pool
    from nwqlib.operators import OperatorInput

    monkeypatch.setattr(OperatorInput, "matvec", lambda *a, **k: pytest.fail("native action"))
    # No modes: every spin sum vanishes on the dimension-one space.
    assert pool.apply_spin_squared(np.array([2.0]), num_qubits=0).tolist() == [0]


def test_ladder_adapter_checks_known_storage_before_consumption():
    from nwqlib.subroutines.fermionic_pool import _ladder_expansion_to_pauli

    visits = []

    def rows():
        visits.append(1)
        yield 2j, ()

    with pytest.raises((ValueError, TypeError)):
        _ladder_expansion_to_pauli(rows(), 1, mapping="jw", max_bytes=1)
    assert visits == []
    mapped = _ladder_expansion_to_pauli(rows(), 1, mapping="jw", max_bytes=4096)
    assert mapped.to_list() == [("I", 2j)]
    assert visits == [1]
